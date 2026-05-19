from unittest.mock import patch

import pytest

from ingestion.links_curated import links_curated
from ingestion.scraper import (
    ScrapedPage,
    clean_footnote,
    clean_headnote,
    scrap_data,
)

marker = "![](https://ww2.mini.pw.edu.pl/wp-content/uploads/WMiNI-01.png)"


# ============ clean_headnote ============


@pytest.mark.parametrize(
    "input_text_headnote, expected_output_headnote",
    [
        # 1. Brak znacznika w tekście (powinno zwrócić oryginał)
        ("To jest ciekawy artykuł bez stopki.", "To jest ciekawy artykuł bez stopki."),
        # 2. Typowy przypadek - znacznik w środku (ucina wszystko przed znacznikiem)
        (
            f"To jest wstęp. {marker} się aby przeczytać resztę. Jakiś dalszy tekst.",
            " się aby przeczytać resztę. Jakiś dalszy tekst.",
        ),
        # 3. Wiele znaczników - powinno uciąć po pierwszym (zgodnie z .split(marker, 1))
        (
            f"Część 1 {marker} się Część 2 {marker} się Część 3",
            f" się Część 2 {marker} się Część 3",
        ),
        # 4. Znacznik na samym początku
        (f"{marker} się i wykup abonament", " się i wykup abonament"),
        # 5. Znacznik na samym końcu
        (f"Tylko wstęp i nic więcej. {marker}", ""),
        # 6. Pusty ciąg znaków (Edge case)
        ("", ""),
        # 7. Dokładnie sam znacznik
        (f"{marker}", ""),
    ],
)
def test_clean_headnote(input_text_headnote: str, expected_output_headnote: str):
    assert clean_headnote(input_text_headnote) == expected_output_headnote


def test_clean_headnote_error():
    with pytest.raises(TypeError):
        clean_headnote(None)


# ============ clean_footnote ============


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        # 1. Brak znacznika w tekście (powinno zwrócić oryginał)
        ("To jest ciekawy artykuł bez stopki.", "To jest ciekawy artykuł bez stopki."),
        # 2. Typowy przypadek - znacznik w środku (ucina wszystko po znaczniku)
        (
            "To jest wstęp. #### Zaloguj się aby przeczytać resztę. Jakiś dalszy tekst.",
            "To jest wstęp. ",
        ),
        # 3. Wiele znaczników - powinno uciąć po pierwszym (zgodnie z .split(marker, 1))
        ("Część 1 #### Zaloguj się Część 2 #### Zaloguj się Część 3", "Część 1 "),
        # 4. Znacznik na samym początku
        ("#### Zaloguj się i wykup abonament", ""),
        # 5. Znacznik na samym końcu
        ("Tylko wstęp i nic więcej. #### Zaloguj się", "Tylko wstęp i nic więcej. "),
        # 6. Pusty ciąg znaków (Edge case)
        ("", ""),
        # 7. Dokładnie sam znacznik
        ("#### Zaloguj się", ""),
    ],
)
def test_clean_footnote(input_text: str, expected_output: str):
    assert clean_footnote(input_text) == expected_output


def test_clean_footnote_error():
    with pytest.raises(TypeError):
        clean_footnote(None)


# ============ scrap_data ============


# Pomocnicza klasa udająca to, co zwraca app.scrape()
class DummyScrapeResult:
    def __init__(self, markdown, links):
        self.markdown = markdown
        self.links = links


@patch(
    "ingestion.scraper.time.sleep"
)  # Blokujemy time.sleep, żeby testy były błyskawiczne
@patch("ingestion.scraper.Firecrawl")  # Mockujemy klienta API
@patch(
    "ingestion.scraper.clean_footnote", side_effect=lambda x: x
)  # Przepuszczamy tekst bez zmian
@patch(
    "ingestion.scraper.clean_headnote", side_effect=lambda x: x
)  # Przepuszczamy tekst bez zmian
class TestScrapData:

    @patch("ingestion.scraper.CURRENT_VERSION", 2)
    def test_scrap_data_version_2_happy_path(
        self, mock_clean_head, mock_clean_foot, MockFirecrawl, mock_sleep
    ):
        """
        Test dla CURRENT_VERSION <= 2.
        Oczekujemy, że funkcja zescrapuje 15 hardcodowanych URLi.
        """
        mock_app_instance = MockFirecrawl.return_value
        mock_app_instance.scrape.return_value = DummyScrapeResult(
            markdown="Przykładowy tekst strony", links=["http://link1.com"]
        )

        result = scrap_data()

        assert len(result) == len(links_curated)
        assert isinstance(result[0], ScrapedPage)
        assert result[0].text == "Przykładowy tekst strony"
        assert result[0].links == ["http://link1.com"]

        assert mock_app_instance.scrape.call_count == len(links_curated)
        assert mock_sleep.call_count == len(links_curated)

    @patch("ingestion.scraper.CURRENT_VERSION", 3)
    @patch("ingestion.scraper.links", ["http://test.com/1", "http://test.com/2"])
    def test_scrap_data_version_3_custom_links(
        self, mock_clean_head, mock_clean_foot, MockFirecrawl, mock_sleep
    ):
        """
        Test dla CURRENT_VERSION > 2.
        Oczekujemy, że funkcja użyje globalnej zmiennej `links`.
        """
        mock_app_instance = MockFirecrawl.return_value
        mock_app_instance.scrape.return_value = DummyScrapeResult(
            markdown="Inny tekst", links=[]
        )

        result = scrap_data()

        assert len(result) == 2
        assert result[0].url == "http://test.com/1"
        assert result[1].url == "http://test.com/2"
        assert mock_app_instance.scrape.call_count == 2

    @patch("ingestion.scraper.CURRENT_VERSION", 2)
    def test_scrap_data_handles_api_exception(
        self, mock_clean_head, mock_clean_foot, MockFirecrawl, mock_sleep
    ):
        """
        Test sprawdza, czy funkcja nie wywala się całkowicie, gdy API rzuci błędem dla jakiegoś URL-a.
        """
        mock_app_instance = MockFirecrawl.return_value
        mock_app_instance.scrape.side_effect = Exception("API Timeout lub inny błąd")

        result = scrap_data()

        assert result == []
        assert mock_app_instance.scrape.call_count == len(links_curated)

    @patch("ingestion.scraper.os.getenv", return_value=None)
    @patch("ingestion.scraper.logger.warning")
    def test_scrap_data_missing_api_key(
        self,
        mock_logger_warning,
        mock_getenv,
        mock_clean_head,
        mock_clean_foot,
        MockFirecrawl,
        mock_sleep,
    ):
        """
        Testuje czy logger wypluwa ostrzeżenie, gdy brakuje klucza API.
        """
        with (
            patch("ingestion.scraper.CURRENT_VERSION", 3),
            patch("ingestion.scraper.links", []),
        ):
            scrap_data()

        mock_logger_warning.assert_called_once_with(
            "FIRECRAWL_API_KEY not found in environment variables."
        )


@pytest.fixture
def mock_scrap_data():
    # If main is in scraper.py, patch scraper.py's reference to scrap_data
    with patch("src.ingestion.scraper.scrap_data") as mock:
        yield mock


@pytest.fixture
def mock_makedirs():
    # os is imported in scraper.py, so patch it there
    with patch("src.ingestion.scraper.os.makedirs") as mock:
        yield mock


@pytest.fixture
def mock_logger():
    # If scraper.py defines `logger = logging.getLogger(__name__)`
    with patch("src.ingestion.scraper.logger") as mock:
        yield mock
