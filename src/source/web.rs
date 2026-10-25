use super::pdf::text_from_bytes;
use super::{Document, Reader};
use crate::error::{Error, Result};
use std::time::Duration;

/// A web page, or a PDF behind a web address. Only `http` and `https`; no cookies or
/// credentials are sent, redirects are followed, the answer is limited in time and size.
pub struct WebReader {
    /// A larger answer is refused instead of being read into memory.
    pub max_bytes: u64,
}

impl Default for WebReader {
    fn default() -> Self {
        Self {
            max_bytes: 10 * 1024 * 1024,
        }
    }
}

impl Reader for WebReader {
    fn accepts(&self, location: &str) -> bool {
        let lower = location.to_lowercase();
        lower.starts_with("http://") || lower.starts_with("https://")
    }

    fn read(&self, location: &str) -> Result<Document> {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(30)))
            .build()
            .into();
        let fail = |what: String| Error::Source(format!("{location}: {what}"));
        let mut response = agent
            .get(location)
            .call()
            .map_err(|e| fail(e.to_string()))?;
        let is_pdf = response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.to_lowercase().contains("pdf"));
        let too_large = |e: ureq::Error| match e {
            ureq::Error::BodyExceedsLimit(_) => fail(format!(
                "the answer is larger than {} bytes",
                self.max_bytes
            )),
            other => fail(other.to_string()),
        };
        let body = response.body_mut().with_config().limit(self.max_bytes);
        if is_pdf {
            let bytes = body.read_to_vec().map_err(too_large)?;
            let text = text_from_bytes(&bytes)
                .map_err(|e| fail(e.to_string().replace("cannot read the source: ", "")))?;
            return Ok(Document {
                title: location.to_string(),
                text,
            });
        }
        let html = body.read_to_string().map_err(too_large)?;
        html_to_markdown(&html, location)
    }
}

/// The main content of an HTML page as Markdown (headings kept, so the chunker can use them),
/// with the page title. Navigation and footers are dropped; a page that does not look like an
/// article is converted whole rather than refused.
pub fn html_to_markdown(html: &str, url: &str) -> Result<Document> {
    let converter = htmd::HtmlToMarkdown::builder()
        .skip_tags(vec!["script", "style", "noscript", "svg", "nav", "footer"])
        .build();
    let convert = |fragment: &str| {
        converter
            .convert(fragment)
            .map_err(|e| Error::Source(format!("{url}: {e}")))
    };
    let article = dom_smoothie::Readability::new(html, Some(url), None)
        .and_then(|mut readability| readability.parse())
        .ok()
        .filter(|article| !article.content.trim().is_empty());
    let (title, mut text) = match article {
        Some(article) => (article.title.trim().to_string(), convert(&article.content)?),
        None => (String::new(), convert(html)?),
    };
    let title = if title.is_empty() {
        url.to_string()
    } else {
        title
    };
    if !text.starts_with("# ") {
        text = format!("# {title}\n\n{text}");
    }
    Ok(Document {
        title,
        text: text.trim().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::super::testing::{minimal_pdf, serve_once};
    use super::*;

    const ARTICLE: &str = r#"<!doctype html><html lang="ru"><head><title>Договор аренды — условия</title></head>
<body>
<nav><a href="/login">Войти в аккаунт</a> <a href="/pricing">Тарифы</a></nav>
<article>
<h1>Договор аренды</h1>
<h2>Срок</h2>
<p>Срок действия договора составляет два года с даты подписания. По истечении срока договор может быть продлён по соглашению сторон, которое оформляется письменно и подписывается обеими сторонами не позднее чем за месяц до окончания срока.</p>
<p>Если ни одна из сторон не заявила о расторжении, договор считается продлённым на тот же срок на тех же условиях, при этом размер арендной платы может быть пересмотрен по взаимному согласию сторон.</p>
<h2>Оплата</h2>
<p>Арендная плата вносится ежемесячно до десятого числа текущего месяца путём перечисления на расчётный счёт арендодателя, указанный в реквизитах настоящего договора, без дополнительных напоминаний со стороны арендодателя.</p>
<p>При просрочке платежа более чем на десять дней арендодатель вправе начислить пени в размере, установленном настоящим договором, и потребовать немедленного погашения образовавшейся задолженности в полном объёме.</p>
</article>
<footer>Все права защищены, политика конфиденциальности</footer>
</body></html>"#;

    #[test]
    fn keeps_the_article_with_its_headings_and_drops_the_page_furniture() {
        let document = html_to_markdown(ARTICLE, "https://example.com/dogovor").unwrap();
        assert_eq!(document.title, "Договор аренды — условия");
        assert!(document.text.contains("## Срок"), "{}", document.text);
        assert!(document.text.contains("## Оплата"), "{}", document.text);
        assert!(
            document
                .text
                .contains("вносится ежемесячно до десятого числа")
        );
        assert!(!document.text.contains("Войти в аккаунт"));
        assert!(!document.text.contains("политика конфиденциальности"));
    }

    #[test]
    fn a_page_that_is_not_an_article_is_converted_whole_not_refused() {
        let html = "<html><head><title>Коротко</title></head><body><h1>Заголовок</h1><p>Одна строка.</p></body></html>";
        let document = html_to_markdown(html, "https://example.com/").unwrap();
        assert!(document.text.contains("Одна строка."), "{}", document.text);
    }

    #[test]
    fn a_page_without_a_title_is_named_by_its_address() {
        let document =
            html_to_markdown("<body><p>Текст.</p></body>", "https://example.com/x").unwrap();
        assert_eq!(document.title, "https://example.com/x");
    }

    #[test]
    fn the_page_title_becomes_the_top_heading_so_passages_carry_it() {
        let html = "<html><head><title>Памятка арендатора</title></head><body><p>Одна строка.</p></body></html>";
        let document = html_to_markdown(html, "https://example.com/").unwrap();
        assert!(
            document.text.starts_with("# Памятка арендатора"),
            "{}",
            document.text
        );
    }

    #[test]
    fn only_http_and_https_addresses_are_accepted() {
        let reader = WebReader::default();
        assert!(reader.accepts("https://example.com/a"));
        assert!(reader.accepts("HTTP://example.com/a"));
        assert!(!reader.accepts("ftp://example.com/a"));
        assert!(!reader.accepts("file:///etc/passwd"));
        assert!(!reader.accepts("/tmp/page.html"));
    }

    #[test]
    fn fetches_a_page_and_reads_it() {
        let url = serve_once(
            "200 OK",
            "text/html; charset=utf-8",
            ARTICLE.as_bytes().to_vec(),
        );
        let document = WebReader::default().read(&url).unwrap();
        assert_eq!(document.title, "Договор аренды — условия");
        assert!(document.text.contains("## Срок"));
    }

    #[test]
    fn a_pdf_behind_an_address_is_read_as_a_pdf() {
        let url = serve_once("200 OK", "application/pdf", minimal_pdf("Hello web pdf"));
        let document = WebReader::default().read(&url).unwrap();
        assert!(
            document.text.contains("Hello web pdf"),
            "{:?}",
            document.text
        );
        assert!(
            !document.text.contains("endobj"),
            "raw PDF syntax leaked: {:?}",
            document.text
        );
    }

    #[test]
    fn an_error_status_is_an_error_that_says_which() {
        let url = serve_once("404 Not Found", "text/html", b"nothing".to_vec());
        let error = WebReader::default().read(&url).unwrap_err().to_string();
        assert!(error.contains("404"), "{error}");
    }

    #[test]
    fn an_answer_larger_than_the_limit_is_refused() {
        let url = serve_once("200 OK", "text/html", vec![b'a'; 5000]);
        let error = WebReader { max_bytes: 100 }
            .read(&url)
            .unwrap_err()
            .to_string();
        assert!(error.contains("larger"), "{error}");
    }
}
