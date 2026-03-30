import hashlib
import logging
import os
import re

import shutil
import time
from dataclasses import dataclass
from html.parser import HTMLParser

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


class HTMLTextExtractor(HTMLParser):
    """
    HTML parser for extracting text from class schedules.

    Methods
    -------
    handle_data(data)
        Handles text data found in HTML.
    get_text()
        Returns extracted text joined by newlines.
    """

    def __init__(self):
        super().__init__()
        self.result = []

    def handle_data(self, data):
        """
        Handle text data found in HTML.

        Parameters
        ----------
        data : str
            The text data to handle.
        """
        text = data.strip()
        if text:
            self.result.append(text)

    def get_text(self):
        """
        Return extracted text joined by newlines.

        Returns
        -------
        str
            The extracted text.
        """
        return "\n".join(self.result)


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


def sanitize_url(url: str) -> str:
    """
    Remove callback parameters and clean URL.

    Parameters
    ----------
    url : str
        The URL to sanitize.

    Returns
    -------
    str
        The sanitized URL.
    """
    url = re.sub(r"&callback=[^&]*", "", url)
    return url.strip()


def extract_usos_links(html_content: str, keyword: str) -> list[str]:
    """
    Extract links from HTML containing a specific keyword.

    Parameters
    ----------
    html_content : str
        The HTML content to search.
    keyword : str
        The keyword to search for in links.

    Returns
    -------
    list[str]
        A list of unique, valid links containing the keyword.
    """
    if not html_content:
        return []
    pattern = r'href=["\']([^"\']*?' + re.escape(keyword) + r'[^"\']*?)["\']'
    found_urls = re.findall(pattern, html_content)
    valid_links = []
    base_url = "https://usosweb.usos.pw.edu.pl/"
    for url in found_urls:
        url = url.replace("&amp;", "&")
        if url.startswith("kontroler.php"):
            url = base_url + url
        elif url.startswith("/"):
            url = base_url.rstrip("/") + url
        valid_links.append(sanitize_url(url))
    return list(set(valid_links))


def extract_subject_title(markdown_text: str) -> str:
    """
    Extract subject name from Markdown header (line starting with #).

    Parameters
    ----------
    markdown_text : str
        The Markdown text to search.

    Returns
    -------
    str
        The extracted subject title, or "Nieznany Przedmiot" if not found.
    """
    if not markdown_text:
        return "Nieznany Przedmiot"
    match = re.search(r"^#\s+(.+)$", markdown_text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return "Nieznany Przedmiot"


def clean_subject_content(text: str) -> str:
    """
    Aggressive cleaning for SYLLABI content.

    Parameters
    ----------
    text : str
        The raw syllabus text to clean.

    Returns
    -------
    str
        The cleaned syllabus text.
    """
    if not text:
        return ""

    header_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if header_match:
        text = text[header_match.start() :]

    cut_off_markers = ["## Zajęcia w cyklu", "## Class schedule", "Zajęcia w cyklu"]
    found_cut_index = -1
    for marker in cut_off_markers:
        idx = text.find(marker)
        if idx != -1:
            if found_cut_index == -1 or idx < found_cut_index:
                found_cut_index = idx

    if found_cut_index != -1:
        text = text[:found_cut_index]

    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\(.*?\)", r"\1", text)
    text = text.replace("| --- |", "")
    text = re.sub(r"\n\s*\n", "\n\n", text)

    return text.strip()


def clean_schedule_raw_text(text: str, subject_name: str) -> str:
    """
    Intelligent cleaning for CLASS SCHEDULES.

    Adds the subject name at the beginning.

    Parameters
    ----------
    text : str
        The raw schedule text to clean.
    subject_name : str
        The name of the subject for the schedule.

    Returns
    -------
    str
        The cleaned schedule text with subject name header.
    """
    if not text:
        return ""

    separator = "Kliknij, aby wyświetlić okienko z dodatkowymi informacjami"
    parts = text.split(separator)

    cleaned_events = []

    header = f"# PLAN ZAJĘĆ: {subject_name}"

    for part in parts[1:]:
        lines = [line.strip() for line in part.split("\n") if line.strip()]

        if len(lines) < 3:
            continue

        event_info = " | ".join(lines[:8])
        cleaned_events.append(f"## ZAJĘCIA: {event_info}")

    if not cleaned_events:
        return f"{header}\n\nBrak szczegółowych terminów zajęć w pobranym planie."

    return f"{header}\n\n" + "\n\n".join(cleaned_events)


def get_firecrawl_data(page_obj):
    """
    Extract markdown and HTML data from Firecrawl page object.

    Parameters
    ----------
    page_obj : object or dict
        The Firecrawl page object or dictionary.

    Returns
    -------
    dict
        Dictionary containing 'markdown' and 'html' keys.
    """
    if isinstance(page_obj, dict):
        return {"markdown": page_obj.get("markdown", ""), "html": page_obj.get("html", "")}
    else:
        return {"markdown": getattr(page_obj, "markdown", ""), "html": getattr(page_obj, "html", "")}


def scrape_usos_v9(app: Firecrawl) -> list[ScrapedPage]:
    """
    Scrape USOS system for subject syllabi and class schedules.

    Parameters
    ----------
    app : Firecrawl
        The Firecrawl API client instance.

    Returns
    -------
    list[ScrapedPage]
        A list of ScrapedPage objects containing syllabi and schedules.
    """
    output = []

    # Configs in case of non-premium usage (max 5 requests per minute, so sleep_time=12)
    # if sleep_time is set to 1, the program will have to wait and lose data.
    MAX_REQUESTS = 30
    request_count = 0
    SLEEP_TIME = 12

    root_url = "https://usosweb.usos.pw.edu.pl/kontroler.php?_action=katalog2/przedmioty/wybierzGrupePrzedmiotow&jed_org_kod=112000"

    if request_count >= MAX_REQUESTS:
        return output

    try:
        logger.info(f"Step 1: Root URL... (Waiting {SLEEP_TIME}s)")
        root_page = app.scrape(root_url, formats=["markdown", "html"])
        request_count += 1
        data = get_firecrawl_data(root_page)

        group_links = extract_usos_links(data["html"], "szukajPrzedmiotu")
        if not group_links:
            raw_links = re.findall(r"\((https?://[^\s)]*?szukajPrzedmiotu[^\s)]*?)\)", data["markdown"])
            group_links = [sanitize_url(l) for l in raw_links]

        target_groups = list(set(group_links))
        logger.info(f"Found {len(target_groups)} group.")

        for group_url in target_groups:
            if request_count >= MAX_REQUESTS:
                break

            logger.info(f"  Group: {group_url} (Waiting {SLEEP_TIME}s...)")
            time.sleep(SLEEP_TIME)

            try:
                g_page = app.scrape(group_url, formats=["markdown", "html"])
                request_count += 1
                g_data = get_firecrawl_data(g_page)

                subject_links = extract_usos_links(g_data["html"], "pokazPrzedmiot")
                target_subjects = list(set(subject_links))[:2]

                for subj_url in target_subjects:
                    if request_count >= MAX_REQUESTS:
                        break

                    logger.info(f"    📘 Subject: {subj_url}")
                    time.sleep(SLEEP_TIME)

                    try:
                        s_page = app.scrape(subj_url, formats=["markdown", "html"])
                        request_count += 1
                        s_data = get_firecrawl_data(s_page)

                        subject_name = extract_subject_title(s_data["markdown"])
                        logger.info(f"       Name: {subject_name}")

                        clean_md = clean_subject_content(s_data["markdown"])
                        if len(clean_md) > 50:
                            output.append(ScrapedPage(url=subj_url, text=clean_md, links=[]))

                        schedule_links = extract_usos_links(s_data["html"], "pokazPlanZajecPrzedmiotu")

                        if schedule_links:
                            best_schedule_url = schedule_links[0]
                            for sl in schedule_links:
                                if "plan_division=semester" in sl:
                                    best_schedule_url = sl
                                    break

                            if request_count < MAX_REQUESTS:
                                logger.info(f"      🗓️ Downloading SCHEDULE: {best_schedule_url}")
                                time.sleep(SLEEP_TIME)

                                sched_page = app.scrape(best_schedule_url, formats=["html"])
                                request_count += 1
                                sched_data = get_firecrawl_data(sched_page)

                                parser = HTMLTextExtractor()
                                parser.feed(sched_data["html"])
                                raw_text = parser.get_text()

                                clean_sched = clean_schedule_raw_text(raw_text, subject_name)

                                if len(clean_sched) > 20:
                                    output.append(ScrapedPage(url=best_schedule_url, text=clean_sched, links=[]))
                                else:
                                    logger.warning("Empty schedule.")

                    except Exception as e:
                        logger.error(f"Subject error: {e}")

            except Exception as e:
                logger.error(f"Error with group: {e}")

    except Exception as e:
        logger.error(f"Critical error: {e}")

    return output


def scrap_data() -> list[ScrapedPage]:
    """
    Scrapes data from the MiNI PW website using the Firecrawl API.

    Depending on the CURRENT_VERSION, it either scrapes a limited list of URLs,
    performs scraping from extended links, or scrapes USOS system data.

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


    if CURRENT_VERSION >= 5:
        logger.info(f"V{CURRENT_VERSION}: USOS Scrape (NAMED PLANS)")
        return scrape_usos_v9(app)


    if CURRENT_VERSION <= 2:
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
                    ],
                    only_main_content=False,
                    timeout=120000,
                )
                text = clean_headnote(result.markdown)
                text = clean_footnote(text)
                output.append(ScrapedPage(url=url, text=text, links=result.links))
                time.sleep(1)
            except Exception as e:
                logger.warning(f"Couldn't get content from {url}. Error: {e}")

    else:

        urls = links

        logger.info(
            f"V{CURRENT_VERSION}: Scraping list of {len(urls)} most important URLs and PDF files."
        )

        for url in urls:
            try:
                result = app.scrape(
                    url,
                    formats=[
                        "markdown",
                        "links",

                    ],

                    ],  # markdown — for cleaned page content; links — for all links displayed on given url

                    only_main_content=False,
                    timeout=120000,
                )
                text = clean_headnote(result.markdown)
                text = clean_footnote(text)
                output.append(ScrapedPage(url=url, text=text, links=result.links))
                time.sleep(1)
            except Exception as e:
                logger.warning(f"Couldn't get content from {url}. Error: {e}")


        # logger.info(f"V{CURRENT_VERSION}: Starting full crawl of MiNI PW website.")
        # root_urls = ["https://ww2.mini.pw.edu.pl/"]

        # if CURRENT_VERSION >= 4:
        #     root_urls.append("https://repo.pw.edu.pl/index.seam?lang=pl")

        # for root_url in root_urls:

        #     logger.info(f"Starting crawl for: {root_url}")

        #     crawl_response = app.start_crawl(
        #         root_url,
        #         limit=50,
        #         scrape_options=ScrapeOptions(formats=["markdown"]),
        #         allow_external_links=False,
        #     )

        #     job_id = crawl_response.id
        #     logger.info(f"Crawl job started: {job_id}")

        #     while True:
        #         status = app.get_crawl_status(job_id)

        #         if status.status == "completed":
        #             logger.info(f"Crawl completed for {root_url}")
        #             break

        #         if status.status == "failed":
        #             logger.error(f"Crawl failed for {root_url}: {status}")
        #             break

        #         logger.info(
        #             f"Crawl status: {status.status} "
        #             f"({status.completed}/{status.total})"
        #         )
        #         time.sleep(2)

        #     for page in status.data:
        #         raw_text = page.markdown or ""
        #         clean_text = clean_footnote(clean_headnote(raw_text))

        #         if not clean_text.strip():
        #             continue

        #         output.append(
        #             ScrapedPage(
        #                 url=(
        #                     page.metadata.url
        #                     if page.metadata and page.metadata.url
        #                     else ""
        #                 ),
        #                 text=clean_text,
        #                 links=[],
        #             )
        #         )

        # logger.info(f"Crawled {len(output)} pages")

        #logger.info(f"V{CURRENT_VERSION}: Starting full crawl of MiNI PW website.")
        #root_urls = ["https://ww2.mini.pw.edu.pl/"]
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
    content to text files. For USOS data (V5+), uses enhanced file naming.

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

    if CURRENT_VERSION >= 5:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    for page in scraped_data:

        if CURRENT_VERSION >= 5:
            name_hash = hashlib.md5(page.url.encode()).hexdigest()[:10]

            prefix = "unknown"
            if "pokazPlanZajecPrzedmiotu" in page.url:
                prefix = "PLAN_SUBJECT"
                if "prz_kod=" in page.url:
                    prefix = f"PLAN_{page.url.split('prz_kod=')[1].split('&')[0]}"
            elif "pokazPrzedmiot" in page.url:
                prefix = "SYLABUS"
                if "prz_kod=" in page.url:
                    prefix = f"SYLABUS_{page.url.split('prz_kod=')[1].split('&')[0]}"

            safe_name = f"{prefix}_{name_hash}"
            safe_name = re.sub(r'[\\/*?:"<>|]', "", safe_name)
        else:
            safe_name = page.url.replace("https://", "").replace("/", "_").strip("_")


        safe_name = page.url.replace("https://", "").replace("/", "_").strip("_")

        file_path = os.path.join(output_dir, f"{safe_name}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"URL: {page.url}\n\n{page.text}")


    if CURRENT_VERSION >= 5:
        logger.info(f"✅ Finished! Saved {len(scraped_data)} files in {output_dir}")
    else:
        logger.info(f"Successfully saved {len(scraped_data)} to {output_dir}.")

    logger.info(
        f"Successfully saved {len(scraped_data)} to {output_dir}."
    )
    logger.info(f"Successfully saved {len(scraped_data)} to {output_dir}.")



if __name__ == "__main__":
    main()
