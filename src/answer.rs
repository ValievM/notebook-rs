//! Asking with sources and checking what comes back.
//!
//! The model is told to answer only from numbered passages and to quote the words it relied on.
//! A quote that is not really in the passage it names is dropped and reported: a citation that
//! cannot be found is worse than no citation.

use crate::domain::Passage;
use crate::error::{Error, Result};
use crate::llm::Llm;
use serde::Deserialize;

#[derive(Debug, PartialEq)]
pub struct Citation {
    /// 1-based number of the passage, as in the prompt.
    pub n: usize,
    pub source_title: String,
    pub quote: String,
}

#[derive(Debug, PartialEq)]
pub struct Answer {
    pub text: String,
    pub citations: Vec<Citation>,
    /// Quotes the model gave that were not found in the passage it named.
    pub rejected_quotes: usize,
}

#[derive(Deserialize)]
struct Reply {
    answer: String,
    #[serde(default)]
    citations: Vec<ReplyCitation>,
}

#[derive(Deserialize)]
struct ReplyCitation {
    n: usize,
    quote: String,
}

/// A passage together with the number the model sees it under. Numbers need not start at 1 or
/// be consecutive: parts of a long source keep their position in the whole (see `digest`).
#[derive(Debug, Clone, Copy)]
pub struct Numbered<'a> {
    pub n: usize,
    pub passage: &'a Passage,
}

/// The passages numbered 1, 2, 3... in the order given.
pub fn number_from_one(passages: &[Passage]) -> Vec<Numbered<'_>> {
    passages
        .iter()
        .enumerate()
        .map(|(i, passage)| Numbered { n: i + 1, passage })
        .collect()
}

/// A prompt for a task other than a question ("make a summary", "compare"), over numbered passages.
pub fn task_prompt(task: &str, passages: &[Numbered]) -> String {
    render(passages, &format!("Задание: {task}\n"))
}

/// Asks the model to do the task; the citations in the reply are checked against `passages`.
pub fn ask_task(llm: &dyn Llm, task: &str, passages: &[Numbered]) -> Result<Answer> {
    parse_numbered(&llm.complete(&task_prompt(task, passages))?, passages)
}

/// Like `parse`, for passages that carry their own numbers.
pub fn parse_numbered(raw: &str, passages: &[Numbered]) -> Result<Answer> {
    let no_json = || Error::BadReply("no JSON object in the reply".into());
    let start = raw.find('{').ok_or_else(no_json)?;
    let end = raw.rfind('}').ok_or_else(no_json)?;
    let reply: Reply =
        serde_json::from_str(&raw[start..=end]).map_err(|e| Error::BadReply(e.to_string()))?;
    let mut citations = Vec::new();
    let mut rejected_quotes = 0;
    for cited in reply.citations {
        let passage = passages.iter().find(|numbered| numbered.n == cited.n);
        match passage {
            Some(found) if contains_quote(&found.passage.text, &cited.quote) => {
                citations.push(Citation {
                    n: cited.n,
                    source_title: found.passage.source_title.clone(),
                    quote: cited.quote,
                })
            }
            _ => rejected_quotes += 1,
        }
    }
    Ok(Answer {
        text: reply.answer,
        citations,
        rejected_quotes,
    })
}

/// The instructions about the reply format, the numbered passages, then `tail` (the question
/// or the task).
fn render(passages: &[Numbered], tail: &str) -> String {
    let mut text = String::from(
        "Отвечай только по приведённым фрагментам источников. Если ответа в них нет, так и скажи.\n\
         Ответь JSON-объектом без пояснений вокруг: {\"answer\": \"...\", \"citations\": [{\"n\": 1, \"quote\": \"дословная цитата из фрагмента\"}]}.\n\
         В answer ссылайся на фрагменты как [1], [2]. В quote копируй слова фрагмента дословно.\n\n",
    );
    for numbered in passages {
        text.push_str(&format!(
            "[{}] Источник «{}»:\n{}\n\n",
            numbered.n, numbered.passage.source_title, numbered.passage.text
        ));
    }
    text.push_str(tail);
    text
}

pub fn prompt(question: &str, passages: &[Passage]) -> String {
    prompt_guided(question, &[], passages)
}

/// Like `prompt`, with notes on what to look for (the instructions of a deep search plan).
pub fn prompt_guided(question: &str, guidance: &[String], passages: &[Passage]) -> String {
    let mut tail = String::new();
    if !guidance.is_empty() {
        tail.push_str("Что нужно найти и учесть в ответе:\n");
        for note in guidance {
            tail.push_str(&format!("- {note}\n"));
        }
        tail.push('\n');
    }
    tail.push_str(&format!("Вопрос: {question}\n"));
    render(&number_from_one(passages), &tail)
}

pub fn ask(llm: &dyn Llm, question: &str, passages: &[Passage]) -> Result<Answer> {
    ask_guided(llm, question, &[], passages)
}

pub fn ask_guided(
    llm: &dyn Llm,
    question: &str,
    guidance: &[String],
    passages: &[Passage],
) -> Result<Answer> {
    if passages.is_empty() {
        return Ok(Answer {
            text: "В источниках ничего не нашлось по этому вопросу.".into(),
            citations: Vec::new(),
            rejected_quotes: 0,
        });
    }
    parse(
        &llm.complete(&prompt_guided(question, guidance, passages))?,
        passages,
    )
}

pub fn parse(raw: &str, passages: &[Passage]) -> Result<Answer> {
    parse_numbered(raw, &number_from_one(passages))
}

fn normalize(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn contains_quote(passage: &str, quote: &str) -> bool {
    let quote = normalize(quote);
    !quote.is_empty() && normalize(passage).contains(&quote)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::PassageId;

    struct Fake(&'static str);
    impl Llm for Fake {
        fn complete(&self, _prompt: &str) -> Result<String> {
            Ok(self.0.to_string())
        }
    }

    fn passages() -> Vec<Passage> {
        vec![
            Passage {
                id: PassageId(0),
                source_title: "договор.txt".into(),
                ordinal: 0,
                text: "Срок действия договора составляет два года.".into(),
            },
            Passage {
                id: PassageId(0),
                source_title: "заметки.txt".into(),
                ordinal: 0,
                text: "Позвонить научному руководителю.".into(),
            },
        ]
    }

    fn passage_text(source: &str, text: &str) -> Passage {
        Passage {
            id: PassageId(0),
            source_title: source.into(),
            ordinal: 0,
            text: text.into(),
        }
    }

    #[test]
    fn a_task_prompt_shows_the_numbers_it_was_given_and_the_task() {
        let a = passage_text("договор.txt", "Срок два года.");
        let b = passage_text("заметки.txt", "Позвонить руководителю.");
        let prompt = task_prompt(
            "Составь конспект.",
            &[
                Numbered { n: 7, passage: &a },
                Numbered { n: 12, passage: &b },
            ],
        );
        assert!(prompt.contains("[7] Источник «договор.txt»"));
        assert!(prompt.contains("[12] Источник «заметки.txt»"));
        assert!(prompt.contains("Задание: Составь конспект."));
    }

    #[test]
    fn citations_are_checked_against_the_passage_with_that_number() {
        let a = passage_text("договор.txt", "Срок действия договора два года.");
        let b = passage_text("заметки.txt", "Позвонить научному руководителю.");
        let numbered = [
            Numbered { n: 7, passage: &a },
            Numbered { n: 12, passage: &b },
        ];
        let reply = r#"{"answer":"ок","citations":[
            {"n":7,"quote":"договора два года"},
            {"n":12,"quote":"научному руководителю"},
            {"n":12,"quote":"два года"},
            {"n":1,"quote":"договора два года"}]}"#;
        let answer = parse_numbered(reply, &numbered).unwrap();
        let kept: Vec<usize> = answer.citations.iter().map(|c| c.n).collect();
        assert_eq!(kept, vec![7, 12]);
        assert_eq!(answer.rejected_quotes, 2);
    }

    #[test]
    fn the_prompt_numbers_every_passage_and_names_its_source() {
        let text = prompt("Какой срок?", &passages());
        assert!(text.contains("[1] Источник «договор.txt»"));
        assert!(text.contains("[2] Источник «заметки.txt»"));
        assert!(text.contains("Вопрос: Какой срок?"));
    }

    #[test]
    fn keeps_a_citation_whose_quote_is_really_in_the_passage() {
        let reply =
            r#"{"answer":"Два года [1].","citations":[{"n":1,"quote":"составляет  ДВА года"}]}"#;
        let answer = parse(reply, &passages()).unwrap();
        assert_eq!(answer.citations.len(), 1);
        assert_eq!(answer.citations[0].source_title, "договор.txt");
        assert_eq!(answer.rejected_quotes, 0);
    }

    #[test]
    fn drops_an_invented_quote_and_a_citation_of_a_passage_that_does_not_exist() {
        let reply = r#"{"answer":"Пять лет [1][7].","citations":[
            {"n":1,"quote":"составляет пять лет"},
            {"n":7,"quote":"два года"},
            {"n":2,"quote":"научному руководителю"}]}"#;
        let answer = parse(reply, &passages()).unwrap();
        assert_eq!(answer.citations.len(), 1);
        assert_eq!(answer.citations[0].n, 2);
        assert_eq!(answer.rejected_quotes, 2);
    }

    #[test]
    fn reads_json_that_the_model_wrapped_in_text_or_a_fence() {
        let reply = "Вот ответ:\n```json\n{\"answer\":\"ок\"}\n```";
        assert_eq!(parse(reply, &passages()).unwrap().text, "ок");
    }

    #[test]
    fn says_honestly_when_nothing_was_found_and_does_not_ask_the_model() {
        struct Never;
        impl Llm for Never {
            fn complete(&self, _: &str) -> Result<String> {
                panic!("must not be called without passages")
            }
        }
        let answer = ask(&Never, "что-то", &[]).unwrap();
        assert!(answer.text.contains("ничего не нашлось"));
    }

    #[test]
    fn asks_the_model_once_and_returns_its_checked_answer() {
        let fake = Fake(r#"{"answer":"Два года [1].","citations":[{"n":1,"quote":"два года"}]}"#);
        let answer = ask(&fake, "Какой срок?", &passages()).unwrap();
        assert_eq!(answer.text, "Два года [1].");
        assert_eq!(answer.citations.len(), 1);
    }
}
