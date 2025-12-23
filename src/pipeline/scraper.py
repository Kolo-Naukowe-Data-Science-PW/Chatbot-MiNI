import logging
import os
import time

from dotenv import load_dotenv
from firecrawl import Firecrawl

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_headnote(text: str) -> str:
    """
    Removes headnote from scraped content from website

    :param text: text to clean
    :type text: str
    :return: cleaned text
    :rtype: str
    """

    marker = "![](https://ww2.mini.pw.edu.pl/wp-content/uploads/WMiNI-01.png)"
    text = text.split(marker, 1)[1]

    return text


def clean_footnote(text: str) -> str:
    """
    Removes footnote from scraped content from website

    :param text: text to clean
    :type text: str
    :return: cleaned text
    :rtype: str
    """

    marker = "#### Zaloguj się"
    text = text.split(marker, 1)[0]

    return text


def scrap_data() -> dict:
    """
    Scrap content from mini website.
    For now only for 15 links.

    :return: Description
    :rtype: dict
    """

    FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
    if not FIRECRAWL_API_KEY:
        logger.warning("FIRECRAWL_API_KEY not found in environment variables.")

    app = Firecrawl(api_key=FIRECRAWL_API_KEY)

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

    output = {}

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
            output[url] = {"text": clean_footnote(text), "links": result.links}
            time.wait(1)
        except Exception:
            logger.warning(f"Couldn't get content from {url}.")

    return output
