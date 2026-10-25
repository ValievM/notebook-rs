//! What the window sees: plain data in the shape the page wants (camelCase), and error
//! messages in Russian. No logic of the notebook lives here.

use notebook_rs::answer::{Answer, Citation};
use notebook_rs::error::Error;
use notebook_rs::store::{NotebookInfo, SourceInfo};
use notebook_rs::view::PassageView;
use serde::Serialize;

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct NotebookDto {
    pub id: i64,
    pub name: String,
    pub sources: usize,
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SourceDto {
    pub title: String,
    pub passages: usize,
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CitationDto {
    pub n: usize,
    pub source_title: String,
    pub ordinal: usize,
    pub quote: String,
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AnswerDto {
    pub text: String,
    pub citations: Vec<CitationDto>,
    pub rejected_quotes: usize,
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PassageViewDto {
    pub ordinal: usize,
    pub before: String,
    pub hit: Option<String>,
    pub after: String,
}

/// Whether the engine (database and embedding model) is ready, or why it is not.
#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct StatusDto {
    pub ready: bool,
    pub error: Option<String>,
}

impl From<NotebookInfo> for NotebookDto {
    fn from(info: NotebookInfo) -> Self {
        Self {
            id: info.id.0,
            name: info.name,
            sources: info.sources,
        }
    }
}

impl From<SourceInfo> for SourceDto {
    fn from(info: SourceInfo) -> Self {
        Self {
            title: info.title,
            passages: info.passages,
        }
    }
}

impl From<Citation> for CitationDto {
    fn from(citation: Citation) -> Self {
        Self {
            n: citation.n,
            source_title: citation.source_title,
            ordinal: citation.ordinal,
            quote: citation.quote,
        }
    }
}

impl From<Answer> for AnswerDto {
    fn from(answer: Answer) -> Self {
        Self {
            text: answer.text,
            citations: answer.citations.into_iter().map(Into::into).collect(),
            rejected_quotes: answer.rejected_quotes,
        }
    }
}

impl From<PassageView> for PassageViewDto {
    fn from(view: PassageView) -> Self {
        Self {
            ordinal: view.ordinal,
            before: view.before,
            hit: view.hit,
            after: view.after,
        }
    }
}

/// An error explained to the person in the window, with the detail kept at the end.
pub fn user_message(error: &Error) -> String {
    match error {
        Error::ModelUnavailable(detail) => format!(
            "Модель недоступна: {detail}. Проверьте вход командой `codex login status` в терминале."
        ),
        Error::BadReply(detail) => {
            format!(
                "Модель ответила так, что ответ не удалось разобрать ({detail}). Повторите запрос."
            )
        }
        Error::Source(detail) => format!("Не удалось прочитать источник: {detail}"),
        Error::TooLarge(detail) => format!("Слишком много текста для одного запроса: {detail}"),
        Error::NotFound(detail) => format!("Не найдено: {detail}"),
        Error::Usage(detail) => format!("Так нельзя: {detail}"),
        Error::Storage(detail) => format!("Ошибка хранилища: {detail}"),
        Error::Io(detail) => format!("Ошибка чтения или записи: {detail}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use notebook_rs::domain::NotebookId;
    use serde_json::json;

    #[test]
    fn an_answer_reaches_the_page_in_camel_case_with_its_citations() {
        let answer = Answer {
            text: "Два года [1].".into(),
            citations: vec![Citation {
                n: 1,
                source_title: "договор.txt".into(),
                ordinal: 3,
                quote: "два года".into(),
            }],
            rejected_quotes: 2,
        };
        assert_eq!(
            serde_json::to_value(AnswerDto::from(answer)).unwrap(),
            json!({
                "text": "Два года [1].",
                "citations": [{"n": 1, "sourceTitle": "договор.txt", "ordinal": 3, "quote": "два года"}],
                "rejectedQuotes": 2
            })
        );
    }

    #[test]
    fn a_marked_passage_reaches_the_page_with_the_text_around_the_hit() {
        let view = PassageView {
            ordinal: 1,
            before: "Срок — ".into(),
            hit: Some("два года".into()),
            after: ".".into(),
        };
        assert_eq!(
            serde_json::to_value(PassageViewDto::from(view)).unwrap(),
            json!({"ordinal": 1, "before": "Срок — ", "hit": "два года", "after": "."})
        );
    }

    #[test]
    fn notebooks_and_sources_reach_the_page_as_plain_numbers_and_names() {
        let notebook = NotebookDto::from(NotebookInfo {
            id: NotebookId(7),
            name: "Диплом".into(),
            sources: 3,
        });
        assert_eq!(
            serde_json::to_value(notebook).unwrap(),
            json!({"id": 7, "name": "Диплом", "sources": 3})
        );
        let source = SourceDto::from(SourceInfo {
            title: "a.txt".into(),
            passages: 5,
        });
        assert_eq!(
            serde_json::to_value(source).unwrap(),
            json!({"title": "a.txt", "passages": 5})
        );
    }

    #[test]
    fn every_error_is_explained_in_russian_and_keeps_its_detail() {
        let cases = [
            (
                Error::ModelUnavailable("codex exited".into()),
                "codex exited",
            ),
            (Error::BadReply("no JSON".into()), "no JSON"),
            (
                Error::Source("a.pdf: no text layer".into()),
                "a.pdf: no text layer",
            ),
            (Error::TooLarge("needs 9 parts".into()), "needs 9 parts"),
            (Error::NotFound("no source".into()), "no source"),
            (Error::Usage("choose two".into()), "choose two"),
        ];
        for (error, detail) in cases {
            let message = user_message(&error);
            assert!(message.contains(detail), "{message}");
            assert!(
                message
                    .chars()
                    .any(|c| ('а'..='я').contains(&c.to_ascii_lowercase()) || c == 'ё'),
                "not Russian: {message}"
            );
        }
    }

    #[test]
    fn a_model_that_is_down_points_to_the_login_check() {
        let message = user_message(&Error::ModelUnavailable("x".into()));
        assert!(message.contains("codex login status"), "{message}");
    }
}
