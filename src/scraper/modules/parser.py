from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


def extract_links(html_content: str, base_url: str):
    """
    Extracts all HTML/PDF/DOCX links from the given HTML content within the same domain.
    """

    soup = BeautifulSoup(html_content, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("javascript:"):
            continue
        full_url = urljoin(base_url, href)
        if urlparse(full_url).netloc != urlparse(base_url).netloc:
            continue
        links.add(full_url)
    return links