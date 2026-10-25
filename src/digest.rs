//! Add-ons that work on whole sources instead of a few found passages: a summary of a source (or
//! of the whole notebook) and a comparison of two sources.
//!
//! What fits into one request is limited (`Budget`). Material that fits goes in one call. A
//! longer one is cut into parts in reading order; the parts are summarized at the same time and
//! the summaries are merged. Passages keep their number in the whole material, so a citation
//! from any part means the same passage everywhere, and the merging call sees only the passages
//! the parts cited (whose quotes were already checked), not the whole text again.
//! Every citation, at every step, is checked against its passage.

use crate::answer::{Answer, Numbered, ask_merge, ask_task, number_from_one};
use crate::domain::Passage;
use crate::error::{Error, Result};
use crate::llm::Llm;
use crate::parallel::map_ordered;

/// How much text one model call may get, and how many calls one add-on may make for the parts.
#[derive(Debug, Clone)]
pub struct Budget {
    pub chars_per_call: usize,
    pub max_parts: usize,
}

impl Default for Budget {
    fn default() -> Self {
        // About 13 thousand tokens of Russian per call; six parts cover a short book.
        Self {
            chars_per_call: 40_000,
            max_parts: 6,
        }
    }
}

const SUMMARY_TASK: &str = "Составь конспект по этим фрагментам: ключевые тезисы списком, сгруппированные по разделам (названия разделов видны в начале фрагментов). Каждый тезис со ссылкой [n] и дословной цитатой в citations: короткой, не длиннее 20 слов. Ничего не добавляй от себя.";

const SUMMARY_MERGE_TASK: &str = "Ниже конспекты частей одного материала. Объедини их в один конспект без повторов, не длиннее 8000 знаков: убирай мелкие детали, а не разделы. Сохрани ссылки [n] на фрагменты как есть, ничего не добавляй от себя.";

/// A summary of the passages, in the order they are given.
pub fn summarize(llm: &dyn Llm, passages: &[Passage], budget: &Budget) -> Result<Answer> {
    if passages.is_empty() {
        return Ok(nothing("В источниках нечего конспектировать."));
    }
    summary_of(llm, &number_from_one(passages), budget)
}

/// What two sources have in common, how they differ and where they contradict each other.
///
/// Sources that fit are compared in one call. Longer ones are summarized first, one after the
/// other (each summary already runs its parts at the same time, and twelve model processes at
/// once would be too much for one subscription), and the comparison is made over the summaries
/// and the passages they cite.
pub fn compare(
    llm: &dyn Llm,
    first: &[Passage],
    second: &[Passage],
    budget: &Budget,
) -> Result<Answer> {
    let (Some(a), Some(b)) = (first.first(), second.first()) else {
        return Ok(nothing(
            "В одном из источников нет текста: сравнивать нечего.",
        ));
    };
    let (a, b) = (a.source_title.as_str(), b.source_title.as_str());
    let pool: Vec<Numbered> = first
        .iter()
        .chain(second)
        .enumerate()
        .map(|(i, passage)| Numbered { n: i + 1, passage })
        .collect();
    if cost(&pool) <= budget.chars_per_call {
        return ask_task(llm, &compare_task(a, b, false), &pool);
    }
    let (of_first, of_second) = pool.split_at(first.len());
    let summary_a = summary_of(llm, of_first, budget)?;
    let summary_b = summary_of(llm, of_second, budget)?;
    merge(
        llm,
        &compare_task(a, b, true),
        vec![
            (format!("Конспект «{a}»"), summary_a),
            (format!("Конспект «{b}»"), summary_b),
        ],
        &pool,
    )
}

fn compare_task(first: &str, second: &str, from_summaries: bool) -> String {
    let basis = if from_summaries {
        " Ниже конспекты обоих источников; опирайся на них и на приведённые фрагменты."
    } else {
        ""
    };
    format!(
        "Сравни источники «{first}» и «{second}»: что у них общего, чем они различаются, где противоречат друг другу.{basis} Три раздела: «Общее», «Различия», «Противоречия» (если противоречий нет, так и напиши). Каждое утверждение со ссылкой [n] на фрагмент и дословной цитатой в citations."
    )
}

fn nothing(text: &str) -> Answer {
    Answer {
        text: text.into(),
        citations: Vec::new(),
        rejected_quotes: 0,
    }
}

/// What a passage takes in a request: its text, the source title and the framing around them.
fn cost(passages: &[Numbered]) -> usize {
    passages
        .iter()
        .map(|n| n.passage.text.chars().count() + n.passage.source_title.chars().count() + 20)
        .sum()
}

/// One call if the passages fit, otherwise parts summarized at the same time and merged.
fn summary_of(llm: &dyn Llm, pool: &[Numbered], budget: &Budget) -> Result<Answer> {
    if cost(pool) <= budget.chars_per_call {
        return ask_task(llm, SUMMARY_TASK, pool);
    }
    let parts = split_into_parts(pool, budget)?;
    let summaries = map_ordered(&parts, |part| ask_task(llm, SUMMARY_TASK, part))
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let labelled = summaries
        .into_iter()
        .enumerate()
        .map(|(i, summary)| (format!("Часть {}", i + 1), summary))
        .collect();
    merge_summaries(llm, labelled)
}

/// One summary out of the summaries of the parts. The model sees only their texts: the quotes
/// were checked in the parts, so the merged summary keeps the citations its text refers to.
fn merge_summaries(llm: &dyn Llm, parts: Vec<(String, Answer)>) -> Result<Answer> {
    let mut body = format!("{SUMMARY_MERGE_TASK}\n\n");
    let mut known = Vec::new();
    let mut rejected = 0;
    for (label, answer) in parts {
        body.push_str(&format!("{label}:\n{}\n\n", answer.text));
        known.extend(answer.citations);
        rejected += answer.rejected_quotes;
    }
    ask_merge(llm, &body, known, rejected)
}

/// Cuts the passages into consecutive parts that each fit the budget (a passage bigger than
/// the budget gets a part of its own). Refuses before any call if there would be too many.
fn split_into_parts<'p, 'a>(
    pool: &'p [Numbered<'a>],
    budget: &Budget,
) -> Result<Vec<&'p [Numbered<'a>]>> {
    let mut parts = Vec::new();
    let (mut start, mut used) = (0, 0);
    for (i, numbered) in pool.iter().enumerate() {
        let size = cost(std::slice::from_ref(numbered));
        if i > start && used + size > budget.chars_per_call {
            parts.push(&pool[start..i]);
            (start, used) = (i, 0);
        }
        used += size;
    }
    parts.push(&pool[start..]);
    if parts.len() > budget.max_parts {
        return Err(Error::TooLarge(format!(
            "the text needs {} parts, at most {} are allowed; ask a question instead, or add sources one by one",
            parts.len(),
            budget.max_parts
        )));
    }
    Ok(parts)
}

/// Asks for one answer over what the parts said. The model sees the parts' texts and only the
/// passages they cited (their quotes were checked already); rejected quotes of the parts are
/// added to the count of the final answer.
fn merge(
    llm: &dyn Llm,
    task: &str,
    parts: Vec<(String, Answer)>,
    pool: &[Numbered],
) -> Result<Answer> {
    let mut cited: Vec<usize> = parts
        .iter()
        .flat_map(|(_, answer)| answer.citations.iter().map(|c| c.n))
        .collect();
    cited.sort_unstable();
    cited.dedup();
    let passages: Vec<Numbered> = pool
        .iter()
        .filter(|numbered| cited.binary_search(&numbered.n).is_ok())
        .copied()
        .collect();
    let mut text = format!("{task}\n\n");
    for (label, answer) in &parts {
        text.push_str(&format!("{label}:\n{}\n\n", answer.text));
    }
    let mut merged = ask_task(llm, &text, &passages)?;
    merged.rejected_quotes += parts.iter().map(|(_, a)| a.rejected_quotes).sum::<usize>();
    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::PassageId;
    use crate::parallel::testing::Rendezvous;
    use std::sync::{Arc, Mutex};

    /// A model that answers by looking at the prompt, and remembers every prompt it got.
    struct Fake {
        reply: Box<dyn Fn(&str) -> String + Send + Sync>,
        prompts: Mutex<Vec<String>>,
    }

    impl Fake {
        fn new(reply: impl Fn(&str) -> String + Send + Sync + 'static) -> Self {
            Self {
                reply: Box::new(reply),
                prompts: Mutex::new(Vec::new()),
            }
        }

        fn never() -> Self {
            Self::new(|_| panic!("the model must not be called"))
        }

        fn calls(&self) -> usize {
            self.prompts.lock().unwrap().len()
        }

        fn prompts(&self) -> Vec<String> {
            self.prompts.lock().unwrap().clone()
        }
    }

    impl Llm for Fake {
        fn complete(&self, prompt: &str) -> Result<String> {
            self.prompts.lock().unwrap().push(prompt.to_string());
            Ok((self.reply)(prompt))
        }
    }

    fn passage(source: &str, id: i64) -> Passage {
        Passage {
            id: PassageId(id),
            source_title: source.into(),
            ordinal: id as usize,
            text: format!("Факт номер {id}."),
        }
    }

    fn passages(source: &str, ids: std::ops::RangeInclusive<i64>) -> Vec<Passage> {
        ids.map(|id| passage(source, id)).collect()
    }

    /// The number of the first passage shown in a prompt: "[3] Источник" gives 3.
    fn first_number(prompt: &str) -> usize {
        let start = prompt.find("] Источник").unwrap();
        let open = prompt[..start].rfind('[').unwrap();
        prompt[open + 1..start].parse().unwrap()
    }

    /// One thesis about the first passage of a part, with a real quote.
    fn thesis_about_first_passage(prompt: &str) -> String {
        let n = first_number(prompt);
        format!(
            r#"{{"answer":"Тезис про {n} [{n}].","citations":[{{"n":{n},"quote":"Факт номер {n}."}},{{"n":{n},"quote":"выдумка"}}]}}"#
        )
    }

    /// Each passage costs 13 characters of text + 1 of title + 20 of framing = 34.
    fn two_passages_per_call() -> Budget {
        Budget {
            chars_per_call: 70,
            max_parts: 6,
        }
    }

    #[test]
    fn nothing_to_summarize_says_so_without_asking_the_model() {
        let answer = summarize(&Fake::never(), &[], &Budget::default()).unwrap();
        assert!(answer.text.contains("нечего"), "{}", answer.text);
    }

    #[test]
    fn material_that_fits_is_summarized_in_one_call_with_checked_citations() {
        let llm = Fake::new(thesis_about_first_passage);
        let answer = summarize(&llm, &passages("s", 1..=3), &Budget::default()).unwrap();
        assert_eq!(llm.calls(), 1);
        let prompt = &llm.prompts()[0];
        assert!(prompt.contains("[1] Источник") && prompt.contains("[3] Источник"));
        assert!(prompt.contains("Задание: Составь конспект"));
        assert_eq!(answer.citations.len(), 1);
        assert_eq!(answer.rejected_quotes, 1);
    }

    #[test]
    fn long_material_is_summarized_in_parts_with_numbers_of_the_whole_and_merged() {
        let llm = Fake::new(|prompt| {
            if prompt.contains("Ниже конспекты") {
                r#"{"answer":"Итог [1][3]."}"#.into()
            } else {
                thesis_about_first_passage(prompt)
            }
        });
        let answer = summarize(&llm, &passages("s", 1..=4), &two_passages_per_call()).unwrap();

        let prompts = llm.prompts();
        assert_eq!(prompts.len(), 3, "two parts and one merge");
        let parts: Vec<&String> = prompts
            .iter()
            .filter(|p| p.contains("Задание: Составь конспект"))
            .collect();
        assert_eq!(parts.len(), 2);
        assert!(
            parts
                .iter()
                .any(|p| p.contains("[1] Источник") && p.contains("[2] Источник"))
        );
        assert!(
            parts.iter().any(|p| p.contains("[3] Источник")
                && p.contains("[4] Источник")
                && !p.contains("[1] Источник")),
            "the second part keeps its numbers 3 and 4"
        );
        let merge = prompts
            .iter()
            .find(|p| p.contains("Ниже конспекты"))
            .unwrap();
        assert!(merge.contains("Тезис про 1") && merge.contains("Тезис про 3"));
        assert!(
            !merge.contains("] Источник"),
            "the merge works on the summaries; the quotes of the parts were checked already"
        );
        assert_eq!(answer.text, "Итог [1][3].");
        let cited: Vec<(usize, &str)> = answer
            .citations
            .iter()
            .map(|c| (c.n, c.quote.as_str()))
            .collect();
        assert_eq!(cited, [(1, "Факт номер 1."), (3, "Факт номер 3.")]);
        assert_eq!(answer.rejected_quotes, 2, "one invented quote in each part");
    }

    #[test]
    fn a_merged_summary_keeps_only_the_citations_its_text_still_refers_to() {
        let llm = Fake::new(|prompt| {
            if prompt.contains("Ниже конспекты") {
                r#"{"answer":"Итог [1]."}"#.into()
            } else {
                thesis_about_first_passage(prompt)
            }
        });
        let answer = summarize(&llm, &passages("s", 1..=4), &two_passages_per_call()).unwrap();
        let numbers: Vec<usize> = answer.citations.iter().map(|c| c.n).collect();
        assert_eq!(numbers, [1]);
    }

    #[test]
    fn the_parts_are_summarized_at_the_same_time() {
        let meeting = Arc::new(Rendezvous::new(2));
        let llm = Fake::new(move |prompt| {
            if prompt.contains("Ниже конспекты") {
                return r#"{"answer":"Итог"}"#.into();
            }
            assert!(meeting.meet(), "the parts did not run at the same time");
            thesis_about_first_passage(prompt)
        });
        summarize(&llm, &passages("s", 1..=4), &two_passages_per_call()).unwrap();
    }

    #[test]
    fn material_needing_more_parts_than_allowed_is_refused_without_calls() {
        let budget = Budget {
            chars_per_call: 34,
            max_parts: 3,
        };
        let error = summarize(&Fake::never(), &passages("s", 1..=4), &budget).unwrap_err();
        assert!(matches!(error, Error::TooLarge(_)), "{error}");
    }

    #[test]
    fn two_small_sources_are_compared_in_one_call_naming_both() {
        let llm = Fake::new(|_| r#"{"answer":"Общее: ..."}"#.into());
        let first = passages("договор.txt", 1..=2);
        let second = passages("памятка.txt", 3..=4);
        compare(&llm, &first, &second, &Budget::default()).unwrap();
        assert_eq!(llm.calls(), 1);
        let prompt = &llm.prompts()[0];
        assert!(
            prompt.contains("Сравни источники «договор.txt» и «памятка.txt»"),
            "{prompt}"
        );
        assert!(
            prompt.contains("[1] Источник «договор.txt»")
                && prompt.contains("[4] Источник «памятка.txt»")
        );
    }

    #[test]
    fn long_sources_are_summarized_each_and_then_compared_over_the_cited_passages() {
        let llm = Fake::new(|prompt| {
            if prompt.contains("Сравни источники") {
                r#"{"answer":"Различие [1][3].","citations":[{"n":1,"quote":"Факт номер 1."},{"n":3,"quote":"Факт номер 3."},{"n":3,"quote":"выдумка"}]}"#.into()
            } else {
                thesis_about_first_passage(prompt)
            }
        });
        let first = passages("договор.txt", 1..=2);
        let second = passages("памятка.txt", 3..=4);
        // Titles of 11 characters make a passage cost 44: one source (88) fits, both (176) do not.
        let budget = Budget {
            chars_per_call: 100,
            max_parts: 6,
        };
        let answer = compare(&llm, &first, &second, &budget).unwrap();

        let prompts = llm.prompts();
        assert_eq!(
            prompts.len(),
            3,
            "a summary of each source, then the comparison"
        );
        let merge = prompts
            .iter()
            .find(|p| p.contains("Сравни источники"))
            .unwrap();
        assert!(merge.contains("договор.txt") && merge.contains("памятка.txt"));
        assert!(merge.contains("Тезис про 1") && merge.contains("Тезис про 3"));
        assert!(
            merge.contains("[1] Источник «договор.txt»")
                && merge.contains("[3] Источник «памятка.txt»")
        );
        assert!(!merge.contains("[2] Источник"), "only cited passages");
        assert_eq!(answer.citations.len(), 2);
        assert_eq!(
            answer.rejected_quotes, 3,
            "one invented quote in each summary and in the comparison"
        );
    }

    #[test]
    fn comparing_with_an_empty_source_says_so_and_does_not_ask_the_model() {
        let first = passages("договор.txt", 1..=2);
        let answer = compare(&Fake::never(), &first, &[], &Budget::default()).unwrap();
        assert!(answer.text.contains("нет текста"), "{}", answer.text);
    }
}
