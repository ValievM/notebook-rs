//! A mind map of a source or of the whole notebook: a tree of short titles, shown as a picture.
//!
//! The model builds the tree from numbered passages and may point a node at the passage it rests
//! on (`n` and a quote). Like every citation in this program, the quote is checked against that
//! passage: a node whose quote is not there stays on the map but loses its link, and is counted.
//! Material that does not fit one request is cut into parts like in `digest`; every part gets its
//! own branch and the branches are put under one root with no extra call.

use crate::answer::{
    Citation, Numbered, contains_quote, first_json, number_from_one, render_custom,
};
use crate::digest::{Budget, cost, split_into_parts};
use crate::domain::Passage;
use crate::error::Result;
use crate::llm::Llm;
use crate::parallel::map_ordered;
use serde::{Deserialize, Serialize};

/// Levels below the root, branches per node, nodes in all and characters of a title: the model's
/// reply is untrusted and a picture must stay readable.
const MAX_DEPTH: usize = 4;
const MAX_CHILDREN: usize = 8;
const MAX_NODES: usize = 150;
const MAX_TITLE: usize = 80;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub title: String,
    /// The passage this node rests on, when the model named one and its quote was found.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cite: Option<Citation>,
    #[serde(default)]
    pub children: Vec<Node>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MindMap {
    pub root: Node,
    /// Quotes of nodes that were not found in their passage.
    pub rejected_quotes: usize,
    /// Nodes in the tree, the root included.
    pub nodes: usize,
}

#[derive(Deserialize)]
struct RawNode {
    #[serde(default)]
    title: String,
    n: Option<usize>,
    quote: Option<String>,
    #[serde(default)]
    children: Vec<RawNode>,
}

const FORMAT: &str = "Построй ментальную карту по приведённым фрагментам источников, ничего не добавляя от себя.\n\
Ответь JSON-объектом без пояснений вокруг: {\"title\": \"тема\", \"children\": [{\"title\": \"ветка\", \"n\": 3, \"quote\": \"дословная цитата из фрагмента [3]\", \"children\": [...]}]}.\n\
Не больше 3 уровней под корнем и 6 веток на уровне; заголовок узла короткий, до 8 слов. \
Поля n и quote указывай у узла, который опирается на конкретное место фрагмента; quote копируй из фрагмента дословно и коротко.\n\n";

/// The mind map of the passages. `title` names the root (the source or the notebook).
pub fn mind_map(
    llm: &dyn Llm,
    passages: &[Passage],
    title: &str,
    budget: &Budget,
) -> Result<MindMap> {
    let empty = || Node {
        title: title.to_string(),
        cite: None,
        children: Vec::new(),
    };
    if passages.is_empty() {
        return Ok(MindMap {
            root: empty(),
            rejected_quotes: 0,
            nodes: 1,
        });
    }
    let pool = number_from_one(passages);
    let mut rejected = 0;
    let mut room = MAX_NODES;
    let root = if cost(&pool) <= budget.chars_per_call {
        let mut raw = ask(llm, &pool, &format!("Корень карты назови: «{title}».\n"))?;
        if raw.title.trim().is_empty() {
            raw.title = title.to_string();
        }
        clean(raw, 0, &pool, &mut room, &mut rejected).unwrap_or_else(empty)
    } else {
        let parts = split_into_parts(&pool, budget)?;
        let hint = format!("Это часть материала «{title}»: корнем карты назови тему этой части.\n");
        let raws = map_ordered(&parts, |part| ask(llm, part, &hint))
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let children = raws
            .into_iter()
            .filter_map(|raw| clean(raw, 1, &pool, &mut room, &mut rejected))
            .collect();
        Node {
            title: title.to_string(),
            cite: None,
            children,
        }
    };
    let nodes = count(&root);
    Ok(MindMap {
        root,
        rejected_quotes: rejected,
        nodes,
    })
}

fn ask(llm: &dyn Llm, pool: &[Numbered], tail: &str) -> Result<RawNode> {
    let prompt = render_custom(FORMAT, "", pool, tail);
    first_json::<RawNode>(&llm.complete(&prompt)?)
}

/// The node as it will be shown: a short title, a checked citation, a bounded number of
/// branches. `None` for a node without a title or beyond the node limit.
fn clean(
    raw: RawNode,
    depth: usize,
    pool: &[Numbered],
    room: &mut usize,
    rejected: &mut usize,
) -> Option<Node> {
    let title = raw.title.split_whitespace().collect::<Vec<_>>().join(" ");
    if title.is_empty() || *room == 0 {
        return None;
    }
    *room -= 1;
    let title = match title.char_indices().nth(MAX_TITLE) {
        Some((end, _)) => format!("{}…", &title[..end]),
        None => title,
    };
    let cite = match (raw.n, raw.quote) {
        (Some(n), Some(quote)) if !quote.trim().is_empty() => {
            match pool.iter().find(|numbered| numbered.n == n) {
                Some(found) if contains_quote(&found.passage.text, &quote) => Some(Citation {
                    n,
                    source_title: found.passage.source_title.clone(),
                    ordinal: found.passage.ordinal,
                    quote,
                }),
                _ => {
                    *rejected += 1;
                    None
                }
            }
        }
        _ => None,
    };
    let children = if depth < MAX_DEPTH {
        raw.children
            .into_iter()
            .take(MAX_CHILDREN)
            .filter_map(|child| clean(child, depth + 1, pool, room, rejected))
            .collect()
    } else {
        Vec::new()
    };
    Some(Node {
        title,
        cite,
        children,
    })
}

fn count(node: &Node) -> usize {
    1 + node.children.iter().map(count).sum::<usize>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::PassageId;
    use std::sync::Mutex;

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
        fn calls(&self) -> usize {
            self.prompts.lock().unwrap().len()
        }
    }
    impl Llm for Fake {
        fn complete(&self, prompt: &str) -> Result<String> {
            self.prompts.lock().unwrap().push(prompt.to_string());
            Ok((self.reply)(prompt))
        }
    }

    fn passage(id: i64) -> Passage {
        Passage {
            id: PassageId(id),
            source_title: "s".into(),
            ordinal: id as usize,
            text: format!("Факт номер {id}."),
        }
    }

    fn passages(ids: std::ops::RangeInclusive<i64>) -> Vec<Passage> {
        ids.map(passage).collect()
    }

    fn big() -> Budget {
        Budget {
            chars_per_call: 10_000,
            max_parts: 6,
        }
    }

    #[test]
    fn a_node_keeps_its_citation_only_if_the_quote_is_in_the_passage() {
        let llm = Fake::new(|_| {
            r#"Вот карта: {"title":"Факты","children":[
                {"title":"Первый","n":1,"quote":"факт  НОМЕР 1"},
                {"title":"Выдуманный","n":2,"quote":"этого нет"},
                {"title":"Чужой номер","n":9,"quote":"Факт номер 9."},
                {"title":"Без ссылки"}]} Готово."#
                .into()
        });

        let map = mind_map(&llm, &passages(1..=2), "Блокнот", &big()).unwrap();

        assert_eq!(map.root.title, "Факты");
        let cites: Vec<Option<usize>> = map
            .root
            .children
            .iter()
            .map(|c| c.cite.as_ref().map(|c| c.n))
            .collect();
        assert_eq!(cites, [Some(1), None, None, None]);
        assert_eq!(
            map.root.children[0].cite.as_ref().unwrap().quote,
            "факт  НОМЕР 1"
        );
        assert_eq!(map.rejected_quotes, 2);
        assert_eq!(map.nodes, 5);
        assert_eq!(llm.calls(), 1);
    }

    #[test]
    fn the_tree_is_bounded_in_depth_branches_titles_and_size() {
        let long = "я".repeat(200);
        let deep = r#"{"title":"1","children":[{"title":"2","children":[{"title":"3","children":[{"title":"4","children":[{"title":"5","children":[{"title":"6"}]}]}]}]}]}"#;
        let llm = Fake::new(move |prompt| {
            if prompt.contains("ширина") {
                let children: Vec<String> = (0..20)
                    .map(|i| format!(r#"{{"title":"ветка {i}"}}"#))
                    .collect();
                format!(
                    r#"{{"title":"корень","children":[{}]}}"#,
                    children.join(",")
                )
            } else if prompt.contains("длинный") {
                format!(r#"{{"title":"{long}"}}"#)
            } else {
                deep.to_string()
            }
        });
        let one = |marker: &str| {
            let mut p = passages(1..=1);
            p[0].text = format!("{marker} факт");
            p
        };

        let deep_map = mind_map(&llm, &one("глубина"), "т", &big()).unwrap();
        fn depth(node: &Node) -> usize {
            1 + node.children.iter().map(depth).max().unwrap_or(0)
        }
        assert_eq!(depth(&deep_map.root), MAX_DEPTH + 1);

        let wide = mind_map(&llm, &one("ширина"), "т", &big()).unwrap();
        assert_eq!(wide.root.children.len(), MAX_CHILDREN);

        let titled = mind_map(&llm, &one("длинный"), "т", &big()).unwrap();
        assert_eq!(titled.root.title.chars().count(), MAX_TITLE + 1);
        assert!(titled.root.title.ends_with('…'));
    }

    #[test]
    fn long_material_gets_a_branch_per_part_under_one_root_without_a_merging_call() {
        let llm = Fake::new(|prompt| {
            let first = prompt.find("] Источник").unwrap();
            let open = prompt[..first].rfind('[').unwrap();
            let n: usize = prompt[open + 1..first].parse().unwrap();
            format!(
                r#"{{"title":"Часть с {n}","children":[{{"title":"Про {n}","n":{n},"quote":"Факт номер {n}."}}]}}"#
            )
        });
        // A passage costs 13 + 1 + 20 = 34 characters: two per call.
        let budget = Budget {
            chars_per_call: 70,
            max_parts: 6,
        };

        let map = mind_map(&llm, &passages(1..=4), "Блокнот", &budget).unwrap();

        assert_eq!(llm.calls(), 2, "one call per part, nothing to merge");
        assert_eq!(map.root.title, "Блокнот");
        let branches: Vec<&str> = map.root.children.iter().map(|c| c.title.as_str()).collect();
        assert_eq!(branches, ["Часть с 1", "Часть с 3"]);
        let leaf = &map.root.children[1].children[0];
        assert_eq!(
            leaf.cite.as_ref().unwrap().n,
            3,
            "numbers are those of the whole"
        );
        assert_eq!(map.rejected_quotes, 0);
        assert_eq!(map.nodes, 5);
    }

    #[test]
    fn a_huge_tree_is_cut_to_the_node_limit() {
        let llm = Fake::new(|_| {
            let leaf = |n: usize| format!(r#"{{"title":"л{n}"}}"#);
            let level = |make: &dyn Fn(usize) -> String| -> String {
                (0..8).map(make).collect::<Vec<_>>().join(",")
            };
            let third = |_| format!(r#"{{"title":"в","children":[{}]}}"#, level(&leaf));
            let second = |_| format!(r#"{{"title":"б","children":[{}]}}"#, level(&third));
            format!(r#"{{"title":"а","children":[{}]}}"#, level(&second))
        });
        let map = mind_map(&llm, &passages(1..=1), "т", &big()).unwrap();
        assert_eq!(map.nodes, MAX_NODES);
    }

    #[test]
    fn branches_of_parts_do_not_make_the_whole_map_deeper_than_the_limit() {
        let llm = Fake::new(|_| {
            r#"{"title":"1","children":[{"title":"2","children":[{"title":"3","children":[{"title":"4","children":[{"title":"5","children":[{"title":"6"}]}]}]}]}]}"#.into()
        });
        let budget = Budget {
            chars_per_call: 70,
            max_parts: 6,
        };
        let map = mind_map(&llm, &passages(1..=4), "Блокнот", &budget).unwrap();
        fn depth(node: &Node) -> usize {
            1 + node.children.iter().map(depth).max().unwrap_or(0)
        }
        assert_eq!(depth(&map.root), MAX_DEPTH + 1);
    }

    #[test]
    fn nothing_to_draw_gives_a_bare_root_without_asking_the_model() {
        let llm = Fake::new(|_| unreachable!());
        let map = mind_map(&llm, &[], "Пусто", &big()).unwrap();
        assert_eq!((map.root.title.as_str(), map.nodes), ("Пусто", 1));
        assert_eq!(llm.calls(), 0);
    }

    #[test]
    fn a_reply_without_a_tree_is_an_error() {
        let llm = Fake::new(|_| "Не могу построить карту.".into());
        assert!(mind_map(&llm, &passages(1..=1), "т", &big()).is_err());
    }

    #[test]
    fn a_root_without_a_title_takes_the_name_of_the_material() {
        let llm = Fake::new(|_| r#"{"title":"  ","children":[{"title":"ветка"}]}"#.into());
        let map = mind_map(&llm, &passages(1..=1), "Договор", &big()).unwrap();
        assert_eq!(map.root.title, "Договор");
        assert_eq!(map.nodes, 2, "its branches stay under the named root");
    }
}
