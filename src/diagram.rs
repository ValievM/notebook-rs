//! A diagram of a topic: boxes joined by arrows (steps of a process, causes and effects, parts
//! and what they depend on). Built from the passages found for the topic.
//!
//! Like a mind map, a box may rest on a passage with a quote, and the quote is checked; the
//! model's reply is untrusted, so the size of the picture is bounded.

use crate::answer::{
    Citation, Numbered, contains_quote, first_json, number_from_one, render_custom,
};
use crate::domain::Passage;
use crate::error::{Error, Result};
use crate::llm::Llm;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

const MAX_NODES: usize = 40;
const MAX_EDGES: usize = 80;
const MAX_LABEL: usize = 70;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagramNode {
    pub id: String,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cite: Option<Citation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub from: String,
    pub to: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Diagram {
    pub title: String,
    pub nodes: Vec<DiagramNode>,
    pub edges: Vec<Edge>,
    /// Quotes of boxes that were not found in their passage.
    pub rejected_quotes: usize,
}

#[derive(Deserialize)]
struct Raw {
    #[serde(default)]
    title: String,
    #[serde(default)]
    nodes: Vec<RawNode>,
    #[serde(default)]
    edges: Vec<RawEdge>,
}

#[derive(Deserialize)]
struct RawNode {
    #[serde(default)]
    id: String,
    #[serde(default)]
    label: String,
    n: Option<usize>,
    quote: Option<String>,
}

#[derive(Deserialize)]
struct RawEdge {
    #[serde(default)]
    from: String,
    #[serde(default)]
    to: String,
    label: Option<String>,
}

const FORMAT: &str = "Нарисуй схему по приведённым фрагментам источников, ничего не добавляя от себя: этапы процесса, причины и следствия, части и связи между ними.\n\
Ответь JSON-объектом без пояснений вокруг: {\"title\": \"название схемы\", \"nodes\": [{\"id\": \"a\", \"label\": \"короткая подпись\", \"n\": 3, \"quote\": \"дословная цитата из фрагмента [3]\"}], \"edges\": [{\"from\": \"a\", \"to\": \"b\", \"label\": \"необязательная подпись стрелки\"}]}.\n\
Не больше 15 блоков; подпись блока до 8 слов. Поля n и quote указывай у блока, который опирается на конкретное место фрагмента; quote копируй из фрагмента дословно и коротко.\n\n";

/// The diagram of `topic` from the passages found for it.
pub fn diagram(llm: &dyn Llm, passages: &[Passage], topic: &str) -> Result<Diagram> {
    if passages.is_empty() {
        return Err(Error::NotFound(
            "nothing was found in the sources for this topic".into(),
        ));
    }
    let pool = number_from_one(passages);
    let tail = format!("Тема схемы: {topic}\n");
    let raw: Raw = first_json(&llm.complete(&render_custom(FORMAT, "", &pool, &tail))?)?;
    let diagram = clean(raw, topic, &pool);
    if diagram.nodes.is_empty() {
        return Err(Error::BadReply("the diagram has no boxes".into()));
    }
    Ok(diagram)
}

fn short(text: &str) -> String {
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    match text.char_indices().nth(MAX_LABEL) {
        Some((end, _)) => format!("{}…", &text[..end]),
        None => text,
    }
}

fn clean(raw: Raw, topic: &str, pool: &[Numbered]) -> Diagram {
    let mut rejected = 0;
    let mut ids = HashSet::new();
    let mut nodes = Vec::new();
    for node in raw.nodes {
        let label = short(&node.label);
        let id = node.id.trim().to_string();
        if label.is_empty() || id.is_empty() || nodes.len() >= MAX_NODES || !ids.insert(id.clone())
        {
            continue;
        }
        let cite = match (node.n, node.quote) {
            (Some(n), Some(quote)) if !quote.trim().is_empty() => {
                match pool.iter().find(|numbered| numbered.n == n) {
                    Some(found) if contains_quote(&found.passage.text, &quote) => Some(Citation {
                        n,
                        source_title: found.passage.source_title.clone(),
                        ordinal: found.passage.ordinal,
                        quote,
                    }),
                    _ => {
                        rejected += 1;
                        None
                    }
                }
            }
            _ => None,
        };
        nodes.push(DiagramNode { id, label, cite });
    }
    let mut seen = HashSet::new();
    let mut edges = Vec::new();
    for edge in raw.edges {
        let (from, to) = (edge.from.trim().to_string(), edge.to.trim().to_string());
        let known = ids.contains(&from) && ids.contains(&to);
        if !known
            || from == to
            || edges.len() >= MAX_EDGES
            || !seen.insert((from.clone(), to.clone()))
        {
            continue;
        }
        let label = edge.label.map(|l| short(&l)).filter(|l| !l.is_empty());
        edges.push(Edge { from, to, label });
    }
    let title = short(&raw.title);
    Diagram {
        title: if title.is_empty() {
            short(topic)
        } else {
            title
        },
        nodes,
        edges,
        rejected_quotes: rejected,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::PassageId;

    struct Fake(String);
    impl Llm for Fake {
        fn complete(&self, _: &str) -> Result<String> {
            Ok(self.0.clone())
        }
    }

    fn passages() -> Vec<Passage> {
        (1..=2)
            .map(|id| Passage {
                id: PassageId(id),
                source_title: "s".into(),
                ordinal: id as usize,
                text: format!("Этап номер {id}."),
            })
            .collect()
    }

    fn run(reply: &str) -> Result<Diagram> {
        diagram(&Fake(reply.into()), &passages(), "Процесс")
    }

    #[test]
    fn boxes_keep_a_checked_citation_and_arrows_join_known_boxes() {
        let result = run(r#"Схема: {"title":"Путь","nodes":[
            {"id":"a","label":"Первый","n":1,"quote":"этап НОМЕР 1"},
            {"id":"b","label":"Второй","n":2,"quote":"выдумка"},
            {"id":"c","label":"Третий"}],
          "edges":[{"from":"a","to":"b","label":"затем"},{"from":"b","to":"c"}]}"#)
        .unwrap();

        assert_eq!(result.title, "Путь");
        let cited: Vec<bool> = result.nodes.iter().map(|n| n.cite.is_some()).collect();
        assert_eq!(cited, [true, false, false]);
        assert_eq!(result.rejected_quotes, 1);
        assert_eq!(result.edges.len(), 2);
        assert_eq!(result.edges[0].label.as_deref(), Some("затем"));
        assert_eq!(result.edges[1].label, None);
    }

    #[test]
    fn broken_arrows_duplicates_and_empty_boxes_are_dropped() {
        let result = run(r#"{"nodes":[
            {"id":"a","label":"Один"},{"id":"a","label":"Копия id"},
            {"id":"b","label":"  "},{"id":"","label":"Без id"},{"id":"c","label":"Три"}],
          "edges":[{"from":"a","to":"a"},{"from":"a","to":"zzz"},{"from":"a","to":"c"},{"from":"a","to":"c"}]}"#)
        .unwrap();

        let ids: Vec<&str> = result.nodes.iter().map(|n| n.id.as_str()).collect();
        assert_eq!(ids, ["a", "c"]);
        assert_eq!(result.edges.len(), 1);
        assert_eq!(result.title, "Процесс", "no title: the topic names it");
    }

    #[test]
    fn the_picture_is_bounded() {
        let nodes: Vec<String> = (0..100)
            .map(|i| format!(r#"{{"id":"n{i}","label":"{}"}}"#, "ы".repeat(200)))
            .collect();
        let edges: Vec<String> = (0..99)
            .flat_map(|i| (i + 1..100).take(5).map(move |j| (i, j)))
            .map(|(i, j)| format!(r#"{{"from":"n{i}","to":"n{j}"}}"#))
            .collect();
        let result = run(&format!(
            r#"{{"nodes":[{}],"edges":[{}]}}"#,
            nodes.join(","),
            edges.join(",")
        ))
        .unwrap();

        assert_eq!(result.nodes.len(), MAX_NODES);
        assert!(result.edges.len() <= MAX_EDGES);
        assert_eq!(result.nodes[0].label.chars().count(), MAX_LABEL + 1);
        assert!(
            result
                .edges
                .iter()
                .all(|e| result.nodes.iter().any(|n| n.id == e.from)
                    && result.nodes.iter().any(|n| n.id == e.to))
        );
    }

    #[test]
    fn nothing_found_or_nothing_drawn_is_an_error() {
        assert!(matches!(
            diagram(&Fake("{}".into()), &[], "т"),
            Err(Error::NotFound(_))
        ));
        assert!(matches!(run(r#"{"nodes":[]}"#), Err(Error::BadReply(_))));
        assert!(run("Не получилось.").is_err());
    }
}
