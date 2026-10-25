//! A summary as text for the terminal (`nb stats`), in Russian.

use super::sqlite::{KindStats, Summary};

/// How long something took, in the unit that reads best: `0.3 мс`, `245 мс`, `8.2 с`, `2 мин 5 с`.
pub fn format_duration(micros: u64) -> String {
    match micros {
        0..1_000 => format!("{:.1} мс", micros as f64 / 1000.0),
        1_000..1_000_000 => format!("{} мс", micros / 1000),
        1_000_000..60_000_000 => format!("{:.1} с", micros as f64 / 1_000_000.0),
        _ => {
            let seconds = micros / 1_000_000;
            format!("{} мин {} с", seconds / 60, seconds % 60)
        }
    }
}

/// A size in bytes: `930 МБ`, `1.0 ГБ`.
pub fn format_bytes(bytes: u64) -> String {
    const MB: u64 = 1024 * 1024;
    if bytes >= 1024 * MB {
        format!("{:.1} ГБ", bytes as f64 / (1024 * MB) as f64)
    } else {
        format!("{} МБ", bytes / MB)
    }
}

/// How long ago something happened: `только что`, `5 мин назад`, `3 ч назад`, `2 дн назад`.
pub fn format_ago(millis: i64) -> String {
    match millis {
        ..60_000 => "только что".to_string(),
        60_000..3_600_000 => format!("{} мин назад", millis / 60_000),
        3_600_000..86_400_000 => format!("{} ч назад", millis / 3_600_000),
        _ => format!("{} дн назад", millis / 86_400_000),
    }
}

/// The whole summary, ready to print. `hours` is the window it was asked for.
pub fn render(summary: &Summary, hours: u64, now_ms: i64) -> String {
    let mut text = format!("Метрики за последние {hours} ч\n\n");
    let nothing = summary.kinds.is_empty()
        && summary.memory.is_empty()
        && summary.failures.is_empty()
        && summary.slowest.is_empty();
    if nothing {
        text.push_str("Пока ничего не записано.\n");
        return text;
    }
    if !summary.kinds.is_empty() {
        text.push_str(&operations_table(&summary.kinds));
    }
    if !summary.memory.is_empty() {
        text.push_str("\nПамять:\n");
        for memory in &summary.memory {
            text.push_str(&format!(
                "  {}: сейчас {}, максимум {} (замеров: {})\n",
                memory.label,
                format_bytes(memory.last_bytes),
                format_bytes(memory.max_bytes),
                memory.samples
            ));
        }
    }
    if !summary.failures.is_empty() {
        text.push_str("\nОшибки (последние):\n");
        for failure in &summary.failures {
            let subject = failure
                .subject
                .as_ref()
                .map(|s| format!("«{s}»  "))
                .unwrap_or_default();
            text.push_str(&format!(
                "  {}  {}  {}{}: {}\n",
                format_ago(now_ms - failure.at_ms),
                failure.kind,
                subject,
                failure.error,
                failure.detail
            ));
        }
    }
    if !summary.slowest.is_empty() {
        text.push_str("\nСамые медленные:\n");
        for slow in &summary.slowest {
            let subject = slow
                .subject
                .as_ref()
                .map(|s| format!("«{s}»  "))
                .unwrap_or_default();
            text.push_str(&format!(
                "  {}  {}  {}{}\n",
                format_duration(slow.micros),
                slow.kind,
                subject,
                format_ago(now_ms - slow.at_ms)
            ));
        }
    }
    text
}

fn operations_table(kinds: &[KindStats]) -> String {
    let width = kinds
        .iter()
        .map(|k| k.kind.chars().count())
        .max()
        .unwrap_or(0)
        .max(9);
    let duration = |micros: Option<u64>| micros.map_or("-".to_string(), format_duration);
    let mut table = String::from("Операции (больше всего времени сверху):\n");
    table.push_str(&format!(
        "  {:<width$}  {:>5}  {:>6}  {:>9}  {:>9}  {:>9}\n",
        "операция", "раз", "ошибок", "p50", "p95", "макс"
    ));
    for k in kinds {
        table.push_str(&format!(
            "  {:<width$}  {:>5}  {:>6}  {:>9}  {:>9}  {:>9}\n",
            k.kind,
            k.count,
            k.errors,
            duration(k.p50_micros),
            duration(k.p95_micros),
            duration(k.max_micros)
        ));
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::sqlite::{FailureRow, MemoryStats, SlowRow};

    #[test]
    fn durations_use_the_unit_that_reads_best() {
        assert_eq!(format_duration(300), "0.3 мс");
        assert_eq!(format_duration(245_000), "245 мс");
        assert_eq!(format_duration(999_000), "999 мс");
        assert_eq!(format_duration(8_240_000), "8.2 с");
        assert_eq!(format_duration(59_900_000), "59.9 с");
        assert_eq!(format_duration(125_000_000), "2 мин 5 с");
        assert_eq!(format_duration(0), "0.0 мс");
    }

    #[test]
    fn sizes_are_in_megabytes_or_gigabytes() {
        assert_eq!(format_bytes(930 * 1024 * 1024), "930 МБ");
        assert_eq!(
            format_bytes(1024 * 1024 * 1024 + 100 * 1024 * 1024),
            "1.1 ГБ"
        );
        assert_eq!(format_bytes(500), "0 МБ");
    }

    #[test]
    fn time_ago_is_rough_and_in_russian() {
        assert_eq!(format_ago(5_000), "только что");
        assert_eq!(format_ago(5 * 60_000), "5 мин назад");
        assert_eq!(format_ago(3 * 3_600_000), "3 ч назад");
        assert_eq!(format_ago(2 * 86_400_000), "2 дн назад");
        assert_eq!(
            format_ago(-5_000),
            "только что",
            "a clock that went back is not an error"
        );
    }

    fn summary() -> Summary {
        Summary {
            since_ms: 0,
            kinds: vec![
                KindStats {
                    kind: "llm.complete".into(),
                    count: 12,
                    errors: 1,
                    p50_micros: Some(8_200_000),
                    p95_micros: Some(14_000_000),
                    max_micros: Some(15_100_000),
                    total_micros: 100_000_000,
                },
                KindStats {
                    kind: "embed.query".into(),
                    count: 30,
                    errors: 0,
                    p50_micros: Some(41_000),
                    p95_micros: Some(90_000),
                    max_micros: Some(120_000),
                    total_micros: 1_300_000,
                },
            ],
            memory: vec![MemoryStats {
                label: "after_start".into(),
                last_bytes: 960 * 1024 * 1024,
                max_bytes: 1100 * 1024 * 1024,
                samples: 3,
            }],
            failures: vec![FailureRow {
                at_ms: 1_000_000 - 5 * 60_000,
                kind: "source.add".into(),
                subject: Some("scan.pdf".into()),
                error: "source".into(),
                detail: "cannot read the source: scan.pdf: the PDF has no text layer".into(),
            }],
            slowest: vec![SlowRow {
                at_ms: 1_000_000 - 3_600_000,
                kind: "llm.complete".into(),
                subject: None,
                micros: 15_100_000,
            }],
        }
    }

    #[test]
    fn the_report_shows_operations_memory_failures_and_the_slowest() {
        let text = render(&summary(), 24, 1_000_000);

        assert!(text.starts_with("Метрики за последние 24 ч\n"), "{text}");
        assert!(text.contains("llm.complete"));
        let llm = text
            .lines()
            .find(|l| l.contains("llm.complete") && l.contains("14.0 с"))
            .expect(&text);
        for part in ["12", "1", "8.2 с", "15.1 с"] {
            assert!(llm.contains(part), "{llm}");
        }
        let embed = text.lines().find(|l| l.contains("embed.query")).unwrap();
        assert!(
            embed.contains("41 мс") && embed.contains("90 мс"),
            "{embed}"
        );
        assert!(
            text.contains("after_start: сейчас 960 МБ, максимум 1.1 ГБ (замеров: 3)"),
            "{text}"
        );
        assert!(
            text.contains("5 мин назад  source.add  «scan.pdf»  source: cannot read the source: scan.pdf: the PDF has no text layer"),
            "{text}"
        );
        assert!(text.contains("15.1 с  llm.complete  1 ч назад"), "{text}");
    }

    #[test]
    fn operations_come_in_the_order_given_longest_total_first() {
        let text = render(&summary(), 24, 1_000_000);
        assert!(text.find("llm.complete").unwrap() < text.find("embed.query").unwrap());
    }

    #[test]
    fn an_empty_summary_says_there_is_nothing_yet() {
        let empty = Summary {
            since_ms: 0,
            kinds: vec![],
            memory: vec![],
            failures: vec![],
            slowest: vec![],
        };
        let text = render(&empty, 1, 0);
        assert!(text.contains("Пока ничего не записано"), "{text}");
        assert!(!text.contains("Память"));
        assert!(!text.contains("Ошибки"));
    }
}
