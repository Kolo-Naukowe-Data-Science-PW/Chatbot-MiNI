import logging
import time
from datetime import datetime

from scraper.main import scrape_new_files

logging.basicConfig(level=logging.INFO)


def weekly_job():
    """
    Scrapes new files every week.
    """

    while True:
        logging.info("Starting weekly scraping job at %s", datetime.now())
        scrape_new_files()
        logging.info("Weekly scraping completed. Next run in 7 days.")
        time.sleep(7 * 24 * 60 * 60)