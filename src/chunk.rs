//! Splits a source into passages that are small enough to cite and big enough to mean something.

/// One citable passage of a source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// 0-based position inside the source.
    pub ordinal: usize,
    pub text: String,
}

/// Packs paragraphs into passages of about `target` characters. A paragraph is never cut in the
/// middle unless it alone is longer than `target`; then it is split on sentence ends.
pub fn split(text: &str, target: usize) -> Vec<Chunk> {
    let mut pieces: Vec<String> = Vec::new();
    for paragraph in text.split("\n\n").map(str::trim).filter(|p| !p.is_empty()) {
        if paragraph.chars().count() <= target {
            pieces.push(paragraph.to_string());
        } else {
            pieces.extend(split_long(paragraph, target));
        }
    }
    let mut chunks: Vec<String> = Vec::new();
    for piece in pieces {
        match chunks.last_mut() {
            Some(last) if last.chars().count() + piece.chars().count() < target => {
                last.push_str("\n\n");
                last.push_str(&piece);
            }
            _ => chunks.push(piece),
        }
    }
    chunks
        .into_iter()
        .enumerate()
        .map(|(ordinal, text)| Chunk { ordinal, text })
        .collect()
}

fn split_long(paragraph: &str, target: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut sentence = String::new();
    for ch in paragraph.chars() {
        sentence.push(ch);
        if matches!(ch, '.' | '!' | '?' | '…') {
            push_sentence(&mut out, &mut current, std::mem::take(&mut sentence), target);
        }
    }
    if !sentence.trim().is_empty() {
        push_sentence(&mut out, &mut current, sentence, target);
    }
    if !current.trim().is_empty() {
        out.push(current.trim().to_string());
    }
    out
}

fn push_sentence(out: &mut Vec<String>, current: &mut String, sentence: String, target: usize) {
    if !current.is_empty() && current.chars().count() + sentence.chars().count() > target {
        out.push(current.trim().to_string());
        current.clear();
    }
    current.push_str(&sentence);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_short_paragraphs_together_up_to_the_target() {
        let chunks = split("Первый абзац.\n\nВторой абзац.\n\nТретий абзац.", 40);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, "Первый абзац.\n\nВторой абзац.");
        assert_eq!(chunks[1].text, "Третий абзац.");
        assert_eq!(chunks[1].ordinal, 1);
    }

    #[test]
    fn splits_one_long_paragraph_on_sentence_ends() {
        let text = "Раз два три. Четыре пять шесть. Семь восемь девять.";
        let chunks = split(text, 20);
        assert!(chunks.len() >= 3);
        assert!(chunks.iter().all(|c| c.text.ends_with('.')));
        assert_eq!(
            chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join(" "),
            text
        );
    }

    #[test]
    fn ignores_empty_input() {
        assert!(split("  \n\n  ", 100).is_empty());
    }
}
