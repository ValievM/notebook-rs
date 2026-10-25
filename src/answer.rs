//! Asking with sources and checking what comes back.
//!
//! The model is told to answer only from numbered passages and to quote the words it relied on.
//! A quote that is not really in the passage it names is dropped and reported: a citation that
//! cannot be found is worse than no citation.

use crate::llm::Llm;
use crate::store::Passage;
use anyhow::Result;
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

pub fn prompt(question: &str, passages: &[Passage]) -> String {
    let mut text = String::from(
        "Отвечай только по приведённым фрагментам источников. Если ответа в них нет, так и скажи.\n\
         Ответь JSON-объектом без пояснений вокруг: {\"answer\": \"...\", \"citations\": [{\"n\": 1, \"quote\": \"дословная цитата из фрагмента\"}]}.\n\
         В answer ссылайся на фрагменты как [1], [2]. В quote копируй слова фрагмента дословно.\n\n",
    );
    for (i, passage) in passages.iter().enumerate() {
        text.push_str(&format!(
            "[{}] Источник «{}»:\n{}\n\n",
            i + 1,
            passage.source_title,
            passage.text
        ));
    }
    text.push_str(&format!("Вопрос: {question}\n"));
    text
}

pub fn ask(llm: &dyn Llm, question: &str, passages: &[Passage]) -> Result<Answer> {
    if passages.is_empty() {
        return Ok(Answer {
            text: "В источниках ничего не нашлось по этому вопросу.".into(),
            citations: Vec::new(),
            rejected_quotes: 0,
        });
    }
    parse(&llm.complete(&prompt(question, passages))?, passages)
}

pub fn parse(raw: &str, passages: &[Passage]) -> Result<Answer> {
    let start = raw.find('{').ok_or_else(|| anyhow::anyhow!("the model did not return JSON"))?;
    let end = raw.rfind('}').ok_or_else(|| anyhow::anyhow!("the model did not return JSON"))?;
    let reply: Reply = serde_json::from_str(&raw[start..=end])?;
    let mut citations = Vec::new();
    let mut rejected_quotes = 0;
    for cited in reply.citations {
        let passage = cited.n.checked_sub(1).and_then(|i| passages.get(i));
        match passage {
            Some(p) if contains_quote(&p.text, &cited.quote) => citations.push(Citation {
                n: cited.n,
                source_title: p.source_title.clone(),
                quote: cited.quote,
            }),
            _ => rejected_quotes += 1,
        }
    }
    Ok(Answer { text: reply.answer, citations, rejected_quotes })
}

fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

fn contains_quote(passage: &str, quote: &str) -> bool {
    let quote = normalize(quote);
    !quote.is_empty() && normalize(passage).contains(&quote)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake(&'static str);
    impl Llm for Fake {
        fn complete(&self, _prompt: &str) -> Result<String> {
            Ok(self.0.to_string())
        }
    }

    fn passages() -> Vec<Passage> {
        vec![
            Passage { source_title: "договор.txt".into(), ordinal: 0, text: "Срок действия договора составляет два года.".into() },
            Passage { source_title: "заметки.txt".into(), ordinal: 0, text: "Позвонить научному руководителю.".into() },
        ]
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
        let reply = r#"{"answer":"Два года [1].","citations":[{"n":1,"quote":"составляет  ДВА года"}]}"#;
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
