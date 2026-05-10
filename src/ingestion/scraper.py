import logging
import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from firecrawl import Firecrawl

from src.ingestion.common import CURRENT_VERSION
from src.ingestion.links_curated import links_curated
from src.ingestion.links_extended import links
from src.ingestion.progress import is_scraped, mark_scraped

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
        urls = links_curated

        total = len(urls)
        logger.info(f"V{CURRENT_VERSION}: Scraping curated list of {total} URLs (legal/admin focus).")

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

    # v4: also scrape WRS MiNI Facebook page for events
    if CURRENT_VERSION >= 4:
        facebook_pages = scrape_facebook_page("wrsminipw", n_posts=30)
        if facebook_pages:
            logger.info(f"Facebook: scraped {len(facebook_pages)} posts from WRS MiNI.")
            output.extend(facebook_pages)
        else:
            logger.warning("Facebook scraping returned no posts — skipping.")

    return output


FACEBOOK_PAGE_URL = "https://www.facebook.com/{page_name}"


def scrape_facebook_page(page_name: str, n_posts: int = 20) -> list[ScrapedPage]:
    """
    Scrape recent posts from a public Facebook page using the facebook-scraper library.

    Each post is returned as a ScrapedPage with:
    - url:  direct link to the post (or page URL as fallback)
    - text: post date + text content
    - links: empty list (Facebook doesn't expose in-post links reliably)

    Parameters
    ----------
    page_name : str
        Facebook page name / slug, e.g. ``"wrsminipw"``.
    n_posts : int
        Maximum number of posts to retrieve (library pages through the feed).

    Returns
    -------
    list[ScrapedPage]
        Scraped posts as ScrapedPage objects.  Empty list on failure.
    """
    try:
        from facebook_scraper import get_posts  # optional dep — import lazily
    except ImportError:
        logger.warning(
            "facebook-scraper is not installed. "
            "Install it with: pip install facebook-scraper"
        )
        return []

    page_url = FACEBOOK_PAGE_URL.format(page_name=page_name)
    results: list[ScrapedPage] = []

    try:
        logger.info(f"Scraping Facebook page: {page_url} (up to {n_posts} posts)")
        for post in get_posts(page_name, pages=max(1, n_posts // 10 + 1)):
            text = post.get("post_text") or post.get("text") or ""
            if not text.strip():
                continue

            post_time = post.get("time")
            date_str = post_time.strftime("%Y-%m-%d") if post_time else "unknown date"
            full_text = f"[Facebook WRS MiNI | {date_str}]\n{text.strip()}"

            post_url = post.get("post_url") or page_url
            results.append(ScrapedPage(url=post_url, text=full_text, links=[]))

            if len(results) >= n_posts:
                break

        logger.info(f"Facebook: collected {len(results)} posts from '{page_name}'.")

    except Exception as exc:
        logger.warning(
            "Facebook scraping failed for '%s': %s — "
            "Facebook may be blocking requests. "
            "Try adding cookies via FACEBOOK_COOKIES env var.",
            page_name,
            exc,
        )

    return results


def _save_page(url: str, text: str, output_dir: str) -> None:
    safe_name = url.replace("https://", "").replace("/", "_").strip("_")[:200]
    file_path = os.path.join(output_dir, f"{safe_name}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"URL: {url}\n\n{text}")


def main() -> None:
    """
    Main function to run the scraper pipeline.

    Saves each page immediately after scraping. Skips URLs already recorded
    in the progress tracker so interrupted runs can be safely resumed.
    """
    logger.info(f"Starting scraper pipeline V{CURRENT_VERSION}")

    output_dir = "src/data/scraped_raw"
    os.makedirs(output_dir, exist_ok=True)

    firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
    if not firecrawl_api_key:
        logger.warning("FIRECRAWL_API_KEY not found in environment variables.")
    app = Firecrawl(api_key=firecrawl_api_key)

    urls = links_curated if CURRENT_VERSION <= 2 else links
    total = len(urls)
    saved = 0
    skipped = 0

    for i, url in enumerate(urls, start=1):
        if is_scraped(url):
            logger.info(f"[{i}/{total}] SKIP (already scraped): {url}")
            skipped += 1
            continue

        logger.info(f"[{i}/{total}] Scraping: {url}")
        try:
            result = app.scrape(
                url,
                formats=["markdown", "links"],
                only_main_content=False,
                timeout=120000,
            )
            text = clean_footnote(clean_headnote(result.markdown))
            _save_page(url, text, output_dir)
            mark_scraped(url)
            saved += 1
            logger.info(f"[{i}/{total}] OK — {len(text)} chars, {len(result.links)} links")
            time.sleep(1)
        except Exception as e:
            logger.warning(f"[{i}/{total}] FAILED: {url} — {e}")

    # Facebook (v4+) — treated as a single atomic scrape
    if CURRENT_VERSION >= 4:
        fb_key = "facebook:wrsminipw"
        if is_scraped(fb_key):
            logger.info("Facebook: SKIP (already scraped).")
        else:
            facebook_pages = scrape_facebook_page("wrsminipw", n_posts=30)
            if facebook_pages:
                for page in facebook_pages:
                    _save_page(page.url, page.text, output_dir)
                mark_scraped(fb_key)
                saved += len(facebook_pages)
                logger.info(f"Facebook: saved {len(facebook_pages)} posts.")
            else:
                logger.warning("Facebook scraping returned no posts — skipping.")

    logger.info(f"Scraping done. Saved: {saved}, Skipped: {skipped}/{total}")


if __name__ == "__main__":
    main()
