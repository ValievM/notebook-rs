//! What the window sees: plain data in the shape the page wants (camelCase), and error
//! messages in Russian. No logic of the notebook lives here.

use notebook_rs::answer::{Answer, Citation};
use notebook_rs::error::Error;
use notebook_rs::metrics::Summary;
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

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct KindStatsDto {
    pub kind: String,
    pub count: u64,
    pub errors: u64,
    pub p50_ms: Option<f64>,
    pub p95_ms: Option<f64>,
    pub max_ms: Option<f64>,
    pub total_ms: f64,
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MemoryDto {
    pub label: String,
    pub last_mb: f64,
    pub max_mb: f64,
    pub samples: u64,
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FailureDto {
    pub at_ms: i64,
    pub kind: String,
    pub subject: Option<String>,
    pub error: String,
    pub detail: String,
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SlowDto {
    pub at_ms: i64,
    pub kind: String,
    pub subject: Option<String>,
    pub ms: f64,
}

/// The metrics as the panel of the window shows them: times in milliseconds, memory in megabytes.
#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SummaryDto {
    pub now_ms: i64,
    pub kinds: Vec<KindStatsDto>,
    pub memory: Vec<MemoryDto>,
    pub failures: Vec<FailureDto>,
    pub slowest: Vec<SlowDto>,
}

impl SummaryDto {
    pub fn from_summary(summary: Summary, now_ms: i64) -> Self {
        let ms = |micros: u64| micros as f64 / 1000.0;
        let mb = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);
        Self {
            now_ms,
            kinds: summary
                .kinds
                .into_iter()
                .map(|k| KindStatsDto {
                    kind: k.kind,
                    count: k.count,
                    errors: k.errors,
                    p50_ms: k.p50_micros.map(ms),
                    p95_ms: k.p95_micros.map(ms),
                    max_ms: k.max_micros.map(ms),
                    total_ms: ms(k.total_micros),
                })
                .collect(),
            memory: summary
                .memory
                .into_iter()
                .map(|m| MemoryDto {
                    label: m.label,
                    last_mb: mb(m.last_bytes),
                    max_mb: mb(m.max_bytes),
                    samples: m.samples,
                })
                .collect(),
            failures: summary
                .failures
                .into_iter()
                .map(|f| FailureDto {
                    at_ms: f.at_ms,
                    kind: f.kind,
                    subject: f.subject,
                    error: f.error,
                    detail: f.detail,
                })
                .collect(),
            slowest: summary
                .slowest
                .into_iter()
                .map(|s| SlowDto {
                    at_ms: s.at_ms,
                    kind: s.kind,
                    subject: s.subject,
                    ms: ms(s.micros),
                })
                .collect(),
        }
    }
}

/// Who answers: the ChatGPT subscription (signed in) or Codex (not).
#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AccountDto {
    pub signed_in: bool,
    pub provider: &'static str,
}

impl AccountDto {
    pub fn new(signed_in: bool) -> Self {
        Self {
            signed_in,
            provider: if signed_in { "chatgpt" } else { "codex" },
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
        Error::SignIn(detail) => format!("Нужен вход через ChatGPT: {detail}"),
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
            (Error::SignIn("run nb login".into()), "run nb login"),
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
    fn the_metrics_reach_the_page_in_milliseconds_and_megabytes() {
        use notebook_rs::metrics::sqlite::{FailureRow, KindStats, MemoryStats, SlowRow};
        let summary = Summary {
            since_ms: 0,
            kinds: vec![
                KindStats {
                    kind: "llm.complete".into(),
                    count: 3,
                    errors: 1,
                    p50_micros: Some(8_200_000),
                    p95_micros: Some(14_000_000),
                    max_micros: Some(15_100_000),
                    total_micros: 30_000_000,
                },
                KindStats {
                    kind: "no.durations".into(),
                    count: 1,
                    errors: 0,
                    p50_micros: None,
                    p95_micros: None,
                    max_micros: None,
                    total_micros: 0,
                },
            ],
            memory: vec![MemoryStats {
                label: "after_start".into(),
                last_bytes: 1_048_576 * 960,
                max_bytes: 1_048_576 * 1100,
                samples: 3,
            }],
            failures: vec![FailureRow {
                at_ms: 5,
                kind: "source.read".into(),
                subject: Some("scan.pdf".into()),
                error: "source".into(),
                detail: "no text layer".into(),
            }],
            slowest: vec![SlowRow {
                at_ms: 7,
                kind: "llm.complete".into(),
                subject: None,
                micros: 15_100_000,
            }],
        };

        assert_eq!(
            serde_json::to_value(SummaryDto::from_summary(summary, 99)).unwrap(),
            json!({
                "nowMs": 99,
                "kinds": [
                    {"kind": "llm.complete", "count": 3, "errors": 1, "p50Ms": 8200.0, "p95Ms": 14000.0, "maxMs": 15100.0, "totalMs": 30000.0},
                    {"kind": "no.durations", "count": 1, "errors": 0, "p50Ms": null, "p95Ms": null, "maxMs": null, "totalMs": 0.0}
                ],
                "memory": [{"label": "after_start", "lastMb": 960.0, "maxMb": 1100.0, "samples": 3}],
                "failures": [{"atMs": 5, "kind": "source.read", "subject": "scan.pdf", "error": "source", "detail": "no text layer"}],
                "slowest": [{"atMs": 7, "kind": "llm.complete", "subject": null, "ms": 15100.0}]
            })
        );
    }

    #[test]
    fn who_answers_reaches_the_page_as_a_flag_and_a_provider_name() {
        assert_eq!(
            serde_json::to_value(AccountDto::new(true)).unwrap(),
            json!({"signedIn": true, "provider": "chatgpt"})
        );
        assert_eq!(
            serde_json::to_value(AccountDto::new(false)).unwrap(),
            json!({"signedIn": false, "provider": "codex"})
        );
    }

    #[test]
    fn a_model_that_is_down_points_to_the_login_check() {
        let message = user_message(&Error::ModelUnavailable("x".into()));
        assert!(message.contains("codex login status"), "{message}");
    }
}
