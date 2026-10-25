//! A source laid out for reading, with the cited quote marked, so that a citation can be
//! opened at its place in the source.

use crate::answer::locate;
use crate::domain::Passage;

/// One passage of a source. When it holds the cited quote, the text is split around it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassageView {
    pub ordinal: usize,
    pub before: String,
    pub hit: Option<String>,
    pub after: String,
}

/// The passages in reading order. `cited` is the position of the passage that was cited and
/// the quote; the quote is marked in that passage only.
pub fn source_view(passages: &[Passage], cited: Option<(usize, &str)>) -> Vec<PassageView> {
    passages
        .iter()
        .map(|passage| {
            let text = &passage.text;
            let hit = cited
                .filter(|(ordinal, _)| *ordinal == passage.ordinal)
                .and_then(|(_, quote)| locate(text, quote));
            match hit {
                Some(range) => PassageView {
                    ordinal: passage.ordinal,
                    before: text[..range.start].to_string(),
                    hit: Some(text[range.clone()].to_string()),
                    after: text[range.end..].to_string(),
                },
                None => PassageView {
                    ordinal: passage.ordinal,
                    before: text.clone(),
                    hit: None,
                    after: String::new(),
                },
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::PassageId;

    fn passage(ordinal: usize, text: &str) -> Passage {
        Passage {
            id: PassageId(ordinal as i64),
            source_title: "договор.txt".into(),
            ordinal,
            text: text.into(),
        }
    }

    #[test]
    fn the_quote_is_marked_in_the_cited_passage_and_the_text_around_it_is_kept() {
        let passages = [
            passage(0, "Вступление."),
            passage(1, "Срок договора — два года, не больше."),
        ];
        let view = source_view(&passages, Some((1, "ДВА года")));
        assert_eq!(view[0].hit, None);
        assert_eq!(view[0].before, "Вступление.");
        assert_eq!(view[1].before, "Срок договора — ");
        assert_eq!(view[1].hit.as_deref(), Some("два года"));
        assert_eq!(view[1].after, ", не больше.");
    }

    #[test]
    fn the_same_words_in_another_passage_are_not_marked() {
        let passages = [
            passage(0, "Два года назад."),
            passage(1, "Срок — два года."),
        ];
        let view = source_view(&passages, Some((1, "два года")));
        assert_eq!(view[0].hit, None);
        assert_eq!(view[0].before, "Два года назад.");
        assert!(view[1].hit.is_some());
    }

    #[test]
    fn without_a_citation_nothing_is_marked() {
        let passages = [passage(0, "Текст.")];
        let view = source_view(&passages, None);
        assert_eq!(
            view,
            vec![PassageView {
                ordinal: 0,
                before: "Текст.".into(),
                hit: None,
                after: String::new()
            }]
        );
    }

    #[test]
    fn a_quote_missing_from_the_cited_passage_marks_nothing() {
        let passages = [passage(0, "Текст.")];
        let view = source_view(&passages, Some((0, "другое")));
        assert_eq!(view[0].hit, None);
        assert_eq!(view[0].before, "Текст.");
    }
}
