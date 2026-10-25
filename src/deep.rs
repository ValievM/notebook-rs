//! Deep questions: the model first plans what to look for (up to five searches), every search
//! runs on the notebook, the findings are fused into one numbered pool, and the model answers
//! once from that pool. Citations are checked exactly as in a plain question.
//!
//! Open Notebook writes a partial answer per search and then a final one. Here the partial
//! answers are skipped: they cost one model call each (the subscription is the budget) and they
//! break citation numbering, while twelve passages fit one prompt easily (see DECISIONS D-09).

use crate::answer::{self, Answer};
use crate::domain::NotebookId;
use crate::error::{Error, Result};
use crate::llm::Llm;
use crate::parallel::map_ordered;
use crate::retrieve::{Retriever, reciprocal_rank_fusion};
use serde::Deserialize;

const MAX_SEARCHES: usize = 5;
const PASSAGES_PER_SEARCH: usize = 6;
const PASSAGES_IN_ANSWER: usize = 12;

/// One search of the plan: what to look for, and what to take from the findings.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Search {
    pub term: String,
    #[serde(default)]
    pub instructions: String,
}

#[derive(Deserialize)]
struct Plan {
    #[serde(default)]
    searches: Vec<Search>,
}

pub fn plan_prompt(question: &str) -> String {
    format!(
        "Ты планируешь поиск по источникам исследовательского блокнота.\n\
         Разбей вопрос на 1-{MAX_SEARCHES} коротких поисковых запросов, которые вместе покрывают его целиком.\n\
         Ответь JSON-объектом без пояснений вокруг: {{\"searches\": [{{\"term\": \"поисковый запрос\", \"instructions\": \"что именно нужно найти\"}}]}}.\n\n\
         Вопрос: {question}\n"
    )
}

/// The searches of a plan reply: blank ones dropped, at most five kept.
pub fn parse_plan(raw: &str) -> Result<Vec<Search>> {
    let no_json = || Error::BadReply("no JSON object in the plan".into());
    let start = raw.find('{').ok_or_else(no_json)?;
    let end = raw.rfind('}').ok_or_else(no_json)?;
    let plan: Plan =
        serde_json::from_str(&raw[start..=end]).map_err(|e| Error::BadReply(e.to_string()))?;
    Ok(plan
        .searches
        .into_iter()
        .filter(|search| !search.term.trim().is_empty())
        .take(MAX_SEARCHES)
        .collect())
}

/// Plans, searches, fuses and answers. A plan the model cannot produce is not an error: the
/// question itself becomes the only search. A model that is unavailable is.
pub fn research(
    llm: &dyn Llm,
    retriever: &dyn Retriever,
    notebook: NotebookId,
    question: &str,
) -> Result<Answer> {
    let searches = match llm
        .complete(&plan_prompt(question))
        .and_then(|raw| parse_plan(&raw))
    {
        Ok(searches) if !searches.is_empty() => searches,
        Ok(_) | Err(Error::BadReply(_)) => vec![Search {
            term: question.to_string(),
            instructions: String::new(),
        }],
        Err(other) => return Err(other),
    };
    // The searches are independent, so they run at the same time.
    let rankings = map_ordered(&searches, |search| {
        retriever.retrieve(notebook, &search.term, PASSAGES_PER_SEARCH)
    })
    .into_iter()
    .collect::<Result<Vec<_>>>()?;
    let mut passages = reciprocal_rank_fusion(rankings);
    passages.truncate(PASSAGES_IN_ANSWER);
    let guidance: Vec<String> = searches
        .into_iter()
        .map(|search| search.instructions)
        .filter(|instructions| !instructions.trim().is_empty())
        .collect();
    answer::ask_guided(llm, question, &guidance, &passages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Passage, PassageId};
    use crate::parallel::testing::Rendezvous;
    use std::collections::{HashMap, VecDeque};
    use std::sync::Mutex;

    /// Answers from a script (`Err` is an unavailable model) and remembers every prompt.
    struct Script {
        replies: Mutex<VecDeque<std::result::Result<String, String>>>,
        prompts: Mutex<Vec<String>>,
    }

    impl Script {
        fn new(replies: Vec<std::result::Result<&str, &str>>) -> Self {
            Self {
                replies: Mutex::new(
                    replies
                        .into_iter()
                        .map(|r| r.map(String::from).map_err(String::from))
                        .collect(),
                ),
                prompts: Mutex::new(Vec::new()),
            }
        }

        fn calls(&self) -> usize {
            self.prompts.lock().unwrap().len()
        }

        fn prompt(&self, n: usize) -> String {
            self.prompts.lock().unwrap()[n].clone()
        }
    }

    impl Llm for Script {
        fn complete(&self, prompt: &str) -> Result<String> {
            self.prompts.lock().unwrap().push(prompt.to_string());
            self.replies
                .lock()
                .unwrap()
                .pop_front()
                .expect("the script ran out of replies")
                .map_err(Error::ModelUnavailable)
        }
    }

    /// Finds passages by the exact search term and remembers which terms were asked.
    struct ByTerm {
        found: HashMap<&'static str, Vec<Passage>>,
        asked: Mutex<Vec<String>>,
        /// When set, a search only succeeds if the other searches run at the same moment.
        meeting: Option<std::sync::Arc<Rendezvous>>,
    }

    impl ByTerm {
        fn new(found: Vec<(&'static str, Vec<Passage>)>) -> Self {
            Self {
                found: found.into_iter().collect(),
                asked: Mutex::new(Vec::new()),
                meeting: None,
            }
        }

        fn meeting(mut self, rendezvous: std::sync::Arc<Rendezvous>) -> Self {
            self.meeting = Some(rendezvous);
            self
        }

        fn asked(&self) -> Vec<String> {
            self.asked.lock().unwrap().clone()
        }
    }

    impl Retriever for ByTerm {
        fn retrieve(&self, _: NotebookId, question: &str, _: usize) -> Result<Vec<Passage>> {
            self.asked.lock().unwrap().push(question.to_string());
            if let Some(rendezvous) = &self.meeting
                && !rendezvous.meet()
            {
                return Err(Error::ModelUnavailable(
                    "the searches did not run at the same time".into(),
                ));
            }
            Ok(self.found.get(question).cloned().unwrap_or_default())
        }
    }

    fn passage(id: i64, title: &str, text: &str) -> Passage {
        Passage {
            id: PassageId(id),
            source_title: title.into(),
            ordinal: 0,
            text: text.into(),
        }
    }

    const NB: NotebookId = NotebookId(1);
    const TWO_SEARCHES: &str = r#"{"searches":[
        {"term":"срок договора","instructions":"найди срок действия"},
        {"term":"оплата","instructions":"найди порядок оплаты"}]}"#;

    fn lease() -> Passage {
        passage(
            1,
            "договор.txt",
            "Срок действия договора составляет два года.",
        )
    }

    fn payment() -> Passage {
        passage(2, "оплата.txt", "Оплата производится ежемесячно.")
    }

    #[test]
    fn a_plan_keeps_its_searches_drops_blank_ones_and_stops_at_five() {
        let raw = r#"```json
        {"searches":[{"term":"один","instructions":"а"},{"term":"  ","instructions":"б"},
        {"term":"два"},{"term":"три"},{"term":"четыре"},{"term":"пять"},{"term":"шесть"}]}
        ```"#;
        let plan = parse_plan(raw).unwrap();
        let terms: Vec<&str> = plan.iter().map(|s| s.term.as_str()).collect();
        assert_eq!(terms, vec!["один", "два", "три", "четыре", "пять"]);
        assert_eq!(plan[1].instructions, "");
    }

    #[test]
    fn a_reply_without_json_is_a_bad_reply() {
        assert!(matches!(parse_plan("не знаю"), Err(Error::BadReply(_))));
    }

    #[test]
    fn plans_then_searches_each_term_then_answers_once_with_checked_citations() {
        let llm = Script::new(vec![
            Ok(TWO_SEARCHES),
            Ok(
                r#"{"answer":"Два года, платят ежемесячно [1][2].","citations":[
                {"n":1,"quote":"два года"},{"n":2,"quote":"выдуманная цитата"}]}"#,
            ),
        ]);
        let retriever = ByTerm::new(vec![
            ("срок договора", vec![lease()]),
            ("оплата", vec![payment()]),
        ]);

        let answer = research(&llm, &retriever, NB, "Срок и оплата?").unwrap();

        assert_eq!(llm.calls(), 2);
        // The searches run at the same time, so the order in which they arrive is not defined.
        let mut asked = retriever.asked();
        asked.sort();
        assert_eq!(asked, vec!["оплата", "срок договора"]);
        let synthesis = llm.prompt(1);
        assert!(synthesis.contains("Срок действия договора"));
        assert!(synthesis.contains("Оплата производится"));
        assert!(synthesis.contains("найди порядок оплаты"));
        assert!(synthesis.contains("Вопрос: Срок и оплата?"));
        assert_eq!(answer.citations.len(), 1);
        assert_eq!(answer.rejected_quotes, 1);
    }

    #[test]
    fn a_passage_found_by_two_searches_goes_into_the_prompt_once() {
        let llm = Script::new(vec![Ok(TWO_SEARCHES), Ok(r#"{"answer":"ок"}"#)]);
        let retriever = ByTerm::new(vec![
            ("срок договора", vec![lease()]),
            ("оплата", vec![lease()]),
        ]);
        research(&llm, &retriever, NB, "вопрос").unwrap();
        assert_eq!(llm.prompt(1).matches("Срок действия договора").count(), 1);
    }

    #[test]
    fn without_a_usable_plan_the_question_itself_is_the_search() {
        let llm = Script::new(vec![Ok("не JSON"), Ok(r#"{"answer":"ок"}"#)]);
        let retriever = ByTerm::new(vec![("Какой срок?", vec![lease()])]);
        research(&llm, &retriever, NB, "Какой срок?").unwrap();
        assert_eq!(retriever.asked(), vec!["Какой срок?"]);
        assert_eq!(llm.calls(), 2);
    }

    #[test]
    fn an_unavailable_model_is_an_error_not_a_fallback() {
        let llm = Script::new(vec![Err("codex exited")]);
        let retriever = ByTerm::new(vec![]);
        let result = research(&llm, &retriever, NB, "вопрос");
        assert!(matches!(result, Err(Error::ModelUnavailable(_))));
        assert!(retriever.asked().is_empty());
    }

    #[test]
    fn says_so_and_does_not_ask_again_when_no_search_finds_anything() {
        let llm = Script::new(vec![Ok(TWO_SEARCHES)]);
        let retriever = ByTerm::new(vec![]);
        let answer = research(&llm, &retriever, NB, "вопрос").unwrap();
        assert!(answer.text.contains("ничего не нашлось"));
        assert_eq!(llm.calls(), 1);
    }

    #[test]
    fn the_planned_searches_run_at_the_same_time_and_keep_the_plan_order() {
        let llm = Script::new(vec![Ok(TWO_SEARCHES), Ok(r#"{"answer":"ок"}"#)]);
        let retriever = ByTerm::new(vec![
            ("срок договора", vec![lease()]),
            ("оплата", vec![payment()]),
        ])
        .meeting(std::sync::Arc::new(Rendezvous::new(2)));
        research(&llm, &retriever, NB, "вопрос").unwrap();
        let synthesis = llm.prompt(1);
        assert!(
            synthesis.find("Срок действия").unwrap()
                < synthesis.find("Оплата производится").unwrap()
        );
    }

    #[test]
    fn the_answer_sees_at_most_twelve_passages() {
        let many = |from: i64| -> Vec<Passage> {
            (from..from + 8)
                .map(|id| passage(id, "источник", &format!("фрагмент номер {id}")))
                .collect()
        };
        let llm = Script::new(vec![Ok(TWO_SEARCHES), Ok(r#"{"answer":"ок"}"#)]);
        let retriever = ByTerm::new(vec![("срок договора", many(1)), ("оплата", many(9))]);
        research(&llm, &retriever, NB, "вопрос").unwrap();
        assert_eq!(llm.prompt(1).matches("Источник «").count(), 12);
    }

    #[test]
    fn runs_at_most_five_searches() {
        let seven = r#"{"searches":[{"term":"1"},{"term":"2"},{"term":"3"},{"term":"4"},
            {"term":"5"},{"term":"6"},{"term":"7"}]}"#;
        let llm = Script::new(vec![Ok(seven)]);
        let retriever = ByTerm::new(vec![]);
        research(&llm, &retriever, NB, "вопрос").unwrap();
        assert_eq!(retriever.asked().len(), 5);
    }
}
