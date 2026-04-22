import logging
import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from firecrawl import Firecrawl

from src.pipeline.common import CURRENT_VERSION
from src.pipeline.links_extended import links

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScrapedPage:
    """
    Data class representing a scraped webpage.

    Attributes
    ----------
    url : str
        The URL of the scraped page.
    text : str
        The cleaned text content of the page.
    links : list[str]
        A list of links found on the page.
    """

    url: str
    text: str
    links: list[str]


def clean_headnote(text: str) -> str:
    """
    Removes the standard headnote from scraped content.

    Parameters
    ----------
    text : str
        The raw text content to clean.

    Returns
    -------
    str
        The text content with the headnote removed.
    """
    marker = "![](https://ww2.mini.pw.edu.pl/wp-content/uploads/WMiNI-01.png)"
    if marker in text:
        text = text.split(marker, 1)[1]

    return text


def clean_footnote(text: str) -> str:
    """
    Removes the standard footnote from scraped content.

    Parameters
    ----------
    text : str
        The raw text content to clean.

    Returns
    -------
    str
        The text content with the footnote removed.
    """
    marker = "#### Zaloguj się"
    if marker in text:
        text = text.split(marker, 1)[0]

    return text


def scrap_data() -> list[ScrapedPage]:
    """
    Scrapes data from the MiNI PW website using the Firecrawl API.

    Depending on the CURRENT_VERSION, it either scrapes a limited list of URLs
    or performs a full crawl of the website.

    Parameters
    ----------
    None

    Returns
    -------
    list[ScrapedPage]
        A list of ScrapedPage objects containing URL, cleaned text, and links.
    """
    firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
    if not firecrawl_api_key:
        logger.warning("FIRECRAWL_API_KEY not found in environment variables.")

    app = Firecrawl(api_key=firecrawl_api_key)
    output: list[ScrapedPage] = []

    if CURRENT_VERSION <= 2:
        # first 15 URLs to test the results
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

        total = len(urls)
        logger.info(f"V{CURRENT_VERSION}: Scraping limited list of {total} URLs.")

        for i, url in enumerate(urls, start=1):
            logger.info(f"[{i}/{total}] Scraping: {url}")
            try:
                result = app.scrape(
                    url,
                    formats=[
                        "markdown",
                        "links",
                    ],  # markdown — for cleaned page content; links — for all links displayed on given url
                    only_main_content=False,
                    timeout=120000,
                )
                text = clean_headnote(result.markdown)
                text = clean_footnote(text)
                output.append(ScrapedPage(url=url, text=text, links=result.links))
                logger.info(f"[{i}/{total}] OK — {len(text)} chars, {len(result.links)} links")
                time.sleep(1)
            except Exception as e:
                logger.warning(f"[{i}/{total}] FAILED: {url} — {e}")

    else:

        urls = links

        total = len(urls)
        logger.info(
            f"V{CURRENT_VERSION}: Scraping list of {total} most important URLs and PDF files."
        )

        for i, url in enumerate(urls, start=1):
            logger.info(f"[{i}/{total}] Scraping: {url}")
            try:
                result = app.scrape(
                    url,
                    formats=[
                        "markdown",
                        "links",
                    ],  # markdown — for cleaned page content; links — for all links displayed on given url
                    only_main_content=False,
                    timeout=120000,
                )
                text = clean_headnote(result.markdown)
                text = clean_footnote(text)
                output.append(ScrapedPage(url=url, text=text, links=result.links))
                logger.info(f"[{i}/{total}] OK — {len(text)} chars, {len(result.links)} links")
                time.sleep(1)
            except Exception as e:
                logger.warning(f"[{i}/{total}] FAILED: {url} — {e}")

        # logger.info(f"V{CURRENT_VERSION}: Starting full crawl of MiNI PW website.")
        # root_urls = ["https://ww2.mini.pw.edu.pl/"]

        # if CURRENT_VERSION >= 4:
        # root_urls.append("https://repo.pw.edu.pl/index.seam?lang=pl")

        # for root_url in root_urls:

        # logger.info(f"Starting crawl for: {root_url}")

        # crawl_response = app.start_crawl(
        # root_url,
        # limit=50,
        # scrape_options=ScrapeOptions(formats=["markdown"]),
        # allow_external_links=False,
        # )

        # job_id = crawl_response.id
        # logger.info(f"Crawl job started: {job_id}")

        # while True:
        # status = app.get_crawl_status(job_id)

        # if status.status == "completed":
        # logger.info(f"Crawl completed for {root_url}")
        # break

        # if status.status == "failed":
        # logger.error(f"Crawl failed for {root_url}: {status}")
        # break

        # logger.info(
        # f"Crawl status: {status.status} "
        # f"({status.completed}/{status.total})"
        # )
        # time.sleep(2)

        # for page in status.data:
        # raw_text = page.markdown or ""
        # clean_text = clean_footnote(clean_headnote(raw_text))

        # if not clean_text.strip():
        # continue

        # output.append(
        # ScrapedPage(
        # url=(
        # page.metadata.url
        # if page.metadata and page.metadata.url
        # else ""
        # ),
        # text=clean_text,
        # links=[],
        # )
        # )

        # logger.info(f"Crawled {len(output)} pages")

    return output


def main() -> None:
    """
    Main function to run the scraper pipeline.

    Scrapes data, creates the output directory, and saves the cleaned
    content to text files.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    logger.info(f"Starting scraper pipeline V{CURRENT_VERSION}")
    scraped_data = scrap_data()

    output_dir = "src/data/scraped_raw"
    os.makedirs(output_dir, exist_ok=True)

    for page in scraped_data:
        safe_name = page.url.replace("https://", "").replace("/", "_").strip("_")[:200]
        file_path = os.path.join(output_dir, f"{safe_name}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"URL: {page.url}\n\n{page.text}")

    logger.info(f"Successfully saved {len(scraped_data)} to {output_dir}.")


if __name__ == "__main__":
    main()
