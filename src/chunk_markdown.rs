//! Markdown-aware splitting: a passage never crosses a heading, carries its heading path
//! ("Договор > Срок"), and starts with the last sentence(s) of the previous passage of the same
//! section, so an answer that sits on a border is whole in at least one passage.

use crate::chunk::{Chunk, Chunker};

pub struct MarkdownChunker {
    /// Wanted passage size in characters (a rough stand-in for tokens: 1000 characters of
    /// Russian is about 350 tokens, inside the 512-token limit of the E5 model).
    pub target: usize,
    /// At most this many characters of the previous passage are repeated at the start.
    pub overlap: usize,
}

impl Default for MarkdownChunker {
    fn default() -> Self {
        Self {
            target: 1000,
            overlap: 150,
        }
    }
}

impl Chunker for MarkdownChunker {
    fn split(&self, text: &str) -> Vec<Chunk> {
        sections(text)
            .iter()
            .flat_map(|section| self.passages_of(section))
            .enumerate()
            .map(|(ordinal, text)| Chunk { ordinal, text })
            .collect()
    }
}

impl MarkdownChunker {
    fn passages_of(&self, section: &Section) -> Vec<String> {
        let pieces: Vec<String> = crate::chunk::split(section.body.trim(), self.target)
            .into_iter()
            .map(|chunk| chunk.text)
            .collect();
        let mut passages = Vec::with_capacity(pieces.len());
        for (i, piece) in pieces.iter().enumerate() {
            let mut text = piece.clone();
            if i > 0 {
                let repeated = tail_sentences(&pieces[i - 1], self.overlap);
                if !repeated.is_empty() {
                    text = format!("{repeated}\n\n{text}");
                }
            }
            if !section.path.is_empty() {
                text = format!("{}\n\n{text}", section.path);
            }
            passages.push(text);
        }
        passages
    }
}

/// The text under one heading, with the path of headings above it ("Договор > Срок").
struct Section {
    path: String,
    body: String,
}

fn sections(text: &str) -> Vec<Section> {
    let mut sections = Vec::new();
    let mut path: Vec<(usize, String)> = Vec::new();
    let mut body = String::new();
    let mut in_code = false;
    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            in_code = !in_code;
        }
        match if in_code { None } else { heading(line) } {
            Some((level, title)) => {
                finish(&mut sections, &path, &mut body);
                path.retain(|(l, _)| *l < level);
                path.push((level, title));
            }
            None => {
                body.push_str(line);
                body.push('\n');
            }
        }
    }
    finish(&mut sections, &path, &mut body);
    sections
}

/// Closes the current section; one with no text makes no passage.
fn finish(sections: &mut Vec<Section>, path: &[(usize, String)], body: &mut String) {
    if !body.trim().is_empty() {
        let path = path
            .iter()
            .map(|(_, t)| t.as_str())
            .collect::<Vec<_>>()
            .join(" > ");
        sections.push(Section {
            path,
            body: std::mem::take(body),
        });
    }
    body.clear();
}

/// `# Title` to `###### Title`: the level and the title.
fn heading(line: &str) -> Option<(usize, String)> {
    let level = line.chars().take_while(|c| *c == '#').count();
    let title = line[level..].strip_prefix(' ')?.trim();
    ((1..=6).contains(&level) && !title.is_empty()).then(|| (level, title.to_string()))
}

/// The longest run of whole sentences at the end of `text` that is at most `max` characters.
fn tail_sentences(text: &str, max: usize) -> &str {
    let mut starts = vec![0];
    for (i, c) in text.char_indices() {
        if matches!(c, '.' | '!' | '?' | '…') {
            let rest = &text[i + c.len_utf8()..];
            let trimmed = rest.trim_start();
            if trimmed.len() < rest.len() && !trimmed.is_empty() {
                starts.push(text.len() - trimmed.len());
            }
        }
    }
    starts
        .into_iter()
        .map(|start| &text[start..])
        .find(|tail| tail.chars().count() <= max)
        .unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(chunks: &[Chunk]) -> Vec<&str> {
        chunks.iter().map(|c| c.text.as_str()).collect()
    }

    fn small() -> MarkdownChunker {
        MarkdownChunker {
            target: 60,
            overlap: 30,
        }
    }

    #[test]
    fn a_passage_carries_the_path_of_its_headings() {
        let chunks = small().split("# Договор\n\n## Срок\n\nДоговор действует два года.");
        assert_eq!(
            texts(&chunks),
            vec!["Договор > Срок\n\nДоговор действует два года."]
        );
    }

    #[test]
    fn a_passage_never_crosses_a_heading() {
        let chunks = small().split("# А\n\nКороткий.\n\n# Б\n\nТоже короткий.");
        assert_eq!(
            texts(&chunks),
            vec!["А\n\nКороткий.", "Б\n\nТоже короткий."]
        );
    }

    #[test]
    fn a_sibling_heading_replaces_the_previous_one_at_its_level() {
        let chunks = small().split("# Д\n\n## Один\n\nТекст.\n\n## Два\n\nДругой.");
        assert_eq!(
            texts(&chunks),
            vec!["Д > Один\n\nТекст.", "Д > Два\n\nДругой."]
        );
    }

    #[test]
    fn a_hash_inside_a_code_block_is_not_a_heading() {
        let chunks = small().split("# Код\n\n```\n# комментарий\n```");
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.starts_with("Код\n\n```"));
    }

    #[test]
    fn text_without_headings_has_no_path() {
        let chunks = small().split("Просто текст.\n\nЕщё абзац.");
        assert_eq!(texts(&chunks), vec!["Просто текст.\n\nЕщё абзац."]);
    }

    #[test]
    fn a_long_section_is_split_and_each_next_passage_repeats_the_last_sentence() {
        let body = "Первое предложение здесь. Второе предложение здесь. Третье предложение здесь. Четвёртое предложение здесь.";
        let chunks = small().split(&format!("# Раздел\n\n{body}"));
        assert_eq!(chunks.len(), 2);
        assert_eq!(
            chunks[0].text,
            "Раздел\n\nПервое предложение здесь. Второе предложение здесь."
        );
        assert_eq!(
            chunks[1].text,
            "Раздел\n\nВторое предложение здесь.\n\nТретье предложение здесь. Четвёртое предложение здесь."
        );
    }

    #[test]
    fn no_overlap_when_the_last_sentence_is_longer_than_allowed() {
        let chunker = MarkdownChunker {
            target: 60,
            overlap: 10,
        };
        let body = "Первое предложение здесь. Второе предложение здесь. Третье предложение здесь. Четвёртое предложение здесь.";
        let chunks = chunker.split(body);
        assert_eq!(
            chunks[1].text,
            "Третье предложение здесь. Четвёртое предложение здесь."
        );
    }

    #[test]
    fn overlap_never_reaches_into_another_section() {
        let chunks = small().split("# А\n\nКонец первого раздела.\n\n# Б\n\nНачало второго.");
        assert!(!chunks[1].text.contains("первого"));
    }

    #[test]
    fn a_heading_without_text_makes_no_passage_but_still_names_its_children() {
        let chunks = small().split("# Пустой\n\n## Полный\n\nТекст.");
        assert_eq!(texts(&chunks), vec!["Пустой > Полный\n\nТекст."]);
    }

    #[test]
    fn ordinals_run_through_the_whole_source() {
        let chunks = small().split("# А\n\nОдин.\n\n# Б\n\nДва.\n\n# В\n\nТри.");
        let ordinals: Vec<usize> = chunks.iter().map(|c| c.ordinal).collect();
        assert_eq!(ordinals, vec![0, 1, 2]);
    }

    #[test]
    fn empty_input_gives_nothing() {
        assert!(small().split("  \n\n  ").is_empty());
        assert!(small().split("# Только заголовок").is_empty());
    }
}
