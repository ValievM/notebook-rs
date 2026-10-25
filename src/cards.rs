//! Flashcards for repetition: a question, a short answer, and the place in the source that
//! proves the answer.
//!
//! A card is a fact to learn, so it is kept only when its quote is really in the passage it
//! names: a card with an invented answer is worse than no card. The dropped ones are counted.
//! Long material is cut into parts like in `digest`; every part gives its share of the cards.

use crate::answer::{
    Citation, Numbered, contains_quote, first_json, number_from_one, render_custom,
};
use crate::digest::{Budget, cost, split_into_parts};
use crate::domain::Passage;
use crate::error::Result;
use crate::llm::Llm;
use crate::parallel::map_ordered;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

const MAX_CARDS: usize = 40;
const MAX_QUESTION: usize = 200;
const MAX_ANSWER: usize = 400;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Card {
    pub question: String,
    pub answer: String,
    /// The proof: always there, a card without a checked quote is dropped.
    pub cite: Citation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Deck {
    pub title: String,
    pub cards: Vec<Card>,
    /// Cards the model made whose quote was not found, so they were left out.
    pub dropped: usize,
}

#[derive(Deserialize)]
struct Raw {
    #[serde(default)]
    cards: Vec<RawCard>,
}

#[derive(Deserialize)]
struct RawCard {
    #[serde(default)]
    question: String,
    #[serde(default)]
    answer: String,
    n: Option<usize>,
    quote: Option<String>,
}

const FORMAT: &str = "Составь карточки для повторения по приведённым фрагментам источников, ничего не добавляя от себя: на карточке вопрос и короткий ответ об одном факте.\n\
Ответь JSON-объектом без пояснений вокруг: {\"cards\": [{\"question\": \"вопрос\", \"answer\": \"короткий ответ\", \"n\": 3, \"quote\": \"дословная цитата из фрагмента [3], которая подтверждает ответ\"}]}.\n\
Не больше 12 карточек. quote копируй из фрагмента дословно и коротко; карточки без точной цитаты будут отброшены.\n\n";

/// The cards of the passages. `title` names the deck (the source or the notebook).
pub fn flashcards(
    llm: &dyn Llm,
    passages: &[Passage],
    title: &str,
    budget: &Budget,
) -> Result<Deck> {
    let mut deck = Deck {
        title: title.to_string(),
        cards: Vec::new(),
        dropped: 0,
    };
    if passages.is_empty() {
        return Ok(deck);
    }
    let pool = number_from_one(passages);
    let raws: Vec<Raw> = if cost(&pool) <= budget.chars_per_call {
        vec![ask(llm, &pool)?]
    } else {
        let parts = split_into_parts(&pool, budget)?;
        map_ordered(&parts, |part| ask(llm, part))
            .into_iter()
            .collect::<Result<Vec<_>>>()?
    };
    // The deck is shared out evenly: parts are few (`Budget::max_parts`), so the total stays
    // within MAX_CARDS and a late part is not left without cards.
    let share = (MAX_CARDS / raws.len().max(1)).max(1);
    let mut asked = HashSet::new();
    for raw in raws {
        let mut taken = 0;
        for card in raw.cards {
            if taken >= share {
                break;
            }
            match checked(card, &pool) {
                Some(card) if asked.insert(card.question.to_lowercase()) => {
                    deck.cards.push(card);
                    taken += 1;
                }
                Some(_) => {}
                None => deck.dropped += 1,
            }
        }
    }
    Ok(deck)
}

fn ask(llm: &dyn Llm, pool: &[Numbered]) -> Result<Raw> {
    first_json::<Raw>(&llm.complete(&render_custom(FORMAT, "", pool, ""))?)
}

fn short(text: &str, limit: usize) -> String {
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    match text.char_indices().nth(limit) {
        Some((end, _)) => format!("{}…", &text[..end]),
        None => text,
    }
}

/// The card if it has a question, an answer and a quote that is in its passage.
fn checked(raw: RawCard, pool: &[Numbered]) -> Option<Card> {
    let (question, answer) = (
        short(&raw.question, MAX_QUESTION),
        short(&raw.answer, MAX_ANSWER),
    );
    if question.is_empty() || answer.is_empty() {
        return None;
    }
    let (n, quote) = (raw.n?, raw.quote?);
    let found = pool.iter().find(|numbered| numbered.n == n)?;
    if quote.trim().is_empty() || !contains_quote(&found.passage.text, &quote) {
        return None;
    }
    Some(Card {
        question,
        answer,
        cite: Citation {
            n,
            source_title: found.passage.source_title.clone(),
            ordinal: found.passage.ordinal,
            quote,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::PassageId;
    use std::sync::Mutex;

    struct Fake {
        reply: Box<dyn Fn(&str) -> String + Send + Sync>,
        calls: Mutex<usize>,
    }
    impl Fake {
        fn new(reply: impl Fn(&str) -> String + Send + Sync + 'static) -> Self {
            Self {
                reply: Box::new(reply),
                calls: Mutex::new(0),
            }
        }
    }
    impl Llm for Fake {
        fn complete(&self, prompt: &str) -> Result<String> {
            *self.calls.lock().unwrap() += 1;
            Ok((self.reply)(prompt))
        }
    }

    fn passages(ids: std::ops::RangeInclusive<i64>) -> Vec<Passage> {
        ids.map(|id| Passage {
            id: PassageId(id),
            source_title: "s".into(),
            ordinal: id as usize,
            text: format!("Факт номер {id}."),
        })
        .collect()
    }

    fn big() -> Budget {
        Budget {
            chars_per_call: 10_000,
            max_parts: 6,
        }
    }

    #[test]
    fn only_cards_with_a_quote_found_in_their_passage_are_kept_and_the_rest_are_counted() {
        let llm = Fake::new(|_| {
            r#"Вот: {"cards":[
              {"question":"Какой факт первый?","answer":"Номер 1","n":1,"quote":"факт  НОМЕР 1"},
              {"question":"Выдуманный?","answer":"да","n":2,"quote":"этого нет"},
              {"question":"Без цитаты?","answer":"да"},
              {"question":"Чужой номер?","answer":"да","n":9,"quote":"Факт номер 9."},
              {"question":"  ","answer":"пусто","n":2,"quote":"Факт номер 2."},
              {"question":"Какой второй?","answer":"Номер 2","n":2,"quote":"Факт номер 2."}]}"#
                .into()
        });

        let deck = flashcards(&llm, &passages(1..=2), "Блокнот", &big()).unwrap();

        let questions: Vec<&str> = deck.cards.iter().map(|c| c.question.as_str()).collect();
        assert_eq!(questions, ["Какой факт первый?", "Какой второй?"]);
        assert_eq!(deck.cards[0].cite.n, 1);
        assert_eq!(deck.cards[0].cite.source_title, "s");
        assert_eq!(deck.dropped, 4);
        assert_eq!(deck.title, "Блокнот");
    }

    #[test]
    fn equal_questions_are_not_repeated_and_the_deck_is_bounded() {
        let many: Vec<String> = (0..60)
            .map(|i| {
                format!(
                    r#"{{"question":"Вопрос {i}?","answer":"а","n":1,"quote":"Факт номер 1."}}"#
                )
            })
            .collect();
        let repeated = r#"{"question":"ВОПРОС 0?","answer":"б","n":1,"quote":"Факт номер 1."}"#;
        let reply = format!(r#"{{"cards":[{},{repeated}]}}"#, many.join(","));
        let llm = Fake::new(move |_| reply.clone());

        let deck = flashcards(&llm, &passages(1..=1), "т", &big()).unwrap();

        assert_eq!(deck.cards.len(), MAX_CARDS);
        let unique: HashSet<String> = deck
            .cards
            .iter()
            .map(|c| c.question.to_lowercase())
            .collect();
        assert_eq!(unique.len(), deck.cards.len());
    }

    #[test]
    fn long_material_gives_each_part_its_share_and_keeps_the_numbers_of_the_whole() {
        let llm = Fake::new(|prompt| {
            let first = prompt.find("] Источник").unwrap();
            let open = prompt[..first].rfind('[').unwrap();
            let n: usize = prompt[open + 1..first].parse().unwrap();
            let cards: Vec<String> = (0..20)
                .map(|i| {
                    format!(
                        r#"{{"question":"Часть {n} вопрос {i}?","answer":"а","n":{n},"quote":"Факт номер {n}."}}"#
                    )
                })
                .collect();
            format!(r#"{{"cards":[{}]}}"#, cards.join(","))
        });
        let budget = Budget {
            chars_per_call: 70,
            max_parts: 6,
        };

        let deck = flashcards(&llm, &passages(1..=4), "т", &budget).unwrap();

        assert_eq!(*llm.calls.lock().unwrap(), 2, "one call per part");
        let from_first = deck.cards.iter().filter(|c| c.cite.n == 1).count();
        let from_second = deck.cards.iter().filter(|c| c.cite.n == 3).count();
        assert_eq!(
            (from_first, from_second),
            (20, 20),
            "each part gets MAX_CARDS / 2"
        );
    }

    #[test]
    fn six_parts_share_the_deck_and_the_last_one_is_not_left_out() {
        let llm = Fake::new(|prompt| {
            let first = prompt.find("] Источник").unwrap();
            let open = prompt[..first].rfind('[').unwrap();
            let n: usize = prompt[open + 1..first].parse().unwrap();
            let cards: Vec<String> = (0..20)
                .map(|i| {
                    format!(
                        r#"{{"question":"Часть {n} вопрос {i}?","answer":"а","n":{n},"quote":"Факт номер {n}."}}"#
                    )
                })
                .collect();
            format!(r#"{{"cards":[{}]}}"#, cards.join(","))
        });
        // One passage per part.
        let budget = Budget {
            chars_per_call: 40,
            max_parts: 6,
        };

        let deck = flashcards(&llm, &passages(1..=6), "т", &budget).unwrap();

        assert!(deck.cards.len() <= MAX_CARDS);
        for n in 1..=6 {
            assert_eq!(
                deck.cards.iter().filter(|c| c.cite.n == n).count(),
                6,
                "part {n}"
            );
        }
    }

    #[test]
    fn nothing_to_learn_is_an_empty_deck_without_a_call_and_a_bad_reply_is_an_error() {
        let never = Fake::new(|_| unreachable!());
        let deck = flashcards(&never, &[], "т", &big()).unwrap();
        assert!(deck.cards.is_empty());
        assert_eq!(*never.calls.lock().unwrap(), 0);

        let bad = Fake::new(|_| "Не получилось.".into());
        assert!(flashcards(&bad, &passages(1..=1), "т", &big()).is_err());
    }
}
