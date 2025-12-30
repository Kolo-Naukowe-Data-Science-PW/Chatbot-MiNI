import logging
import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from firecrawl import Firecrawl

from src.pipeline.common import CURRENT_VERSION

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScrapedPage:
    url: str
    text: str
    links: list[str]


def clean_headnote(text: str) -> str:
    """
    Removes headnote from scraped content from website.
    Args:
        text (str): text to clean

    Returns:
        str: cleaned text
    """

    marker = "![](https://ww2.mini.pw.edu.pl/wp-content/uploads/WMiNI-01.png)"
    text = text.split(marker, 1)[1]

    return text


def clean_footnote(text: str) -> str:
    """
    Removes footnote from scraped content from website
    Args:
        text (str): text to clean

    Returns:
        str: cleaned text
    """

    marker = "#### Zaloguj się"
    text = text.split(marker, 1)[0]

    return text


def scrap_data() -> list[ScrapedPage]:
    """
    Scrapes data from MiNI PW website using Firecrawl API.

    Args:
        None

    Returns:
        list[ScrapedPage]: list of scraped pages
    """

    FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
    if not FIRECRAWL_API_KEY:
        logger.warning("FIRECRAWL_API_KEY not found in environment variables.")

    app = Firecrawl(api_key=FIRECRAWL_API_KEY)
    output = []

    if CURRENT_VERSION <= 2:

        # first 15 urls to test results
        urls = [
            "https://ww2.mini.pw.edu.pl/studia/dziekanat/informacje-dziekanatu/",
            "https://ww2.mini.pw.edu.pl/wydzial/dziekani/",
            "https://ww2.mini.pw.edu.pl/wydzial/o-nas/",
            "https://ww2.mini.pw.edu.pl/laboratorium/laboratoria/",
            "https://ww2.mini.pw.edu.pl/studia/inzynierskie-i-licencjackie/matematyka-i-analiza-danych/",
            "https://ww2.mini.pw.edu.pl/studia/inzynierskie-i-licencjackie/matematyka-2/",
            "https://ww2.mini.pw.edu.pl/studia/inzynierskie-i-licencjackie/informatyka-2/",
            "https://ww2.mini.pw.edu.pl/studia/inzynierskie-i-licencjackie/computer-science-2/",
            "https://ww2.mini.pw.edu.pl/studia/inzynierskie-i-licencjackie/inzynieria-i-analiza-danych/",
            "https://ww2.mini.pw.edu.pl/studia/magisterskie/matematyka-i-analiza-danych/",
            "https://ww2.mini.pw.edu.pl/studia/magisterskie/matematyka/",
            "https://ww2.mini.pw.edu.pl/studia/magisterskie/informatyka/",
            "https://ww2.mini.pw.edu.pl/studia/magisterskie/inzynieria-i-analiza-danych/",
            "https://ww2.mini.pw.edu.pl/wp-content/uploads/uchwala_rady_21_02_2019.pdf",
            "https://ww2.mini.pw.edu.pl/wydzial/uchwaly-rw/",
        ]

        logger.info(f"V{CURRENT_VERSION}: Scraping limited list of {len(urls)} URLs.")

        for url in urls:
            try:
                result = app.scrape(
                    url,
                    formats=[
                        "markdown",
                        "links",
                    ],  # markdown - for cleaned page content; links - for all links displayed on given url
                    only_main_content=False,
                    timeout=120000,
                )
                text = clean_headnote(result.markdown)
                text = clean_footnote(text)
                output.append(ScrapedPage(url=url, text=text, links=result.links))
                time.sleep(1)
            except Exception:
                logger.warning(f"Couldn't get content from {url}.")

    else:

        logger.info(f"V{CURRENT_VERSION}: Starting full crawl of MiNI PW website.")
        root_urls = ["https://ww2.mini.pw.edu.pl/"]

        if CURRENT_VERSION == 4:
            root_urls.extend(["https://repo.pw.edu.pl/index.seam?lang=pl"])

        logger.info(f"V{CURRENT_VERSION}: Starting crawl for roots: {root_urls}")

        for root in root_urls:

            try:

                crawl_result = app.crawl_url(
                    root,
                    params={"limit": 1000, "scrapeOptions": {"formats": ["markdown"]}},
                )

                for page in crawl_result.get("data", []):
                    raw_text = page.get("markdown", "")
                    clean_text = clean_footnote(clean_headnote(raw_text))
                    output.append(
                        ScrapedPage(url=page.get("url"), text=clean_text, links=[])
                    )

            except Exception as e:

                logger.error(f"Crawl failed for {root}: {e}")

    return output


def main():

    logger.info(f"Starting scraper pipeline V{CURRENT_VERSION}")
    scraped_data = scrap_data()

    output_dir = "src/data/scraped_raw"
    os.makedirs(output_dir, exist_ok=True)

    for _, page in enumerate(scraped_data):
        safe_name = page.url.replace("https://", "").replace("/", "_").strip("_")
        file_path = os.path.join(output_dir, f"{safe_name}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"URL: {page.url}\n\n{page.text}")

    logger.info(f"Successfully saved {len(scraped_data)} pages to {output_dir}")


if __name__ == "__main__":
    main()
