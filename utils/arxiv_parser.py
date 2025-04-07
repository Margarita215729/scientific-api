import feedparser
import requests
import fitz  # PyMuPDF
import tempfile
import os

def download_and_parse_pdf(url):
    response = requests.get(url)
    if response.status_code != 200:
        return "[PDF не доступен]"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name

    text = ""
    try:
        doc = fitz.open(tmp_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    finally:
        os.remove(tmp_path)

    return text or "[Текст из PDF не извлечён]"

def fetch_arxiv_fulltext(keywords, max_results):
    url = f"http://export.arxiv.org/api/query?search_query=all:{keywords}&start=0&max_results={max_results}"
    feed = feedparser.parse(url)
    if not feed.entries:
        return {"error": "No articles found"}

    entry = feed.entries[0]
    title = entry.title
    authors = [author.name for author in entry.authors]
    abstract = entry.summary
    pdf_url = entry.link.replace("abs", "pdf") + ".pdf"

    full_text = download_and_parse_pdf(pdf_url)

    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "text": full_text[:6000],
        "source": pdf_url
    }
