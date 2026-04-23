import os
from unittest.mock import call, mock_open, patch

import pytest

from pipeline.scraper import (
    ScrapedPage,
    clean_footnote,
    clean_headnote,
    main,
    scrap_data,
)

marker = "![](https://ww2.mini.pw.edu.pl/wp-content/uploads/WMiNI-01.png)"


##test clean_headnote
@pytest.mark.parametrize(
    "input_text_headnote, expected_output_headnote",
    [
        # 1. Brak znacznika w tekście (powinno zwrócić oryginał)
        ("To jest ciekawy artykuł bez stopki.", "To jest ciekawy artykuł bez stopki."),
        # 2. Typowy przypadek - znacznik w środku (ucina wszystko po znaczniku)
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


##test clean_footnote
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


# testy do scrap
# Pomocnicza klasa udająca to, co zwraca app.scrape()
class DummyScrapeResult:
    def __init__(self, markdown, links):
        self.markdown = markdown
        self.links = links


@patch(
    "pipeline.scraper.time.sleep"
)  # Blokujemy time.sleep, żeby testy były błyskawiczne
@patch("pipeline.scraper.Firecrawl")  # Mockujemy klienta API
@patch(
    "pipeline.scraper.clean_footnote", side_effect=lambda x: x
)  # Przepuszczamy tekst bez zmian
@patch(
    "pipeline.scraper.clean_headnote", side_effect=lambda x: x
)  # Przepuszczamy tekst bez zmian
class TestScrapData:

    @patch("pipeline.scraper.CURRENT_VERSION", 2)
    def test_scrap_data_version_2_happy_path(
        self, mock_clean_head, mock_clean_foot, MockFirecrawl, mock_sleep
    ):
        """
        Test dla CURRENT_VERSION <= 2.
        Oczekujemy, że funkcja zescrapuje 15 hardcodowanych URLi.
        """
        # Konfiguracja mocka Firecrawl
        mock_app_instance = MockFirecrawl.return_value
        # Udajemy, że za każdym razem API zwraca poprawny wynik
        mock_app_instance.scrape.return_value = DummyScrapeResult(
            markdown="Przykładowy tekst strony", links=["http://link1.com"]
        )

        # Wywołanie funkcji
        result = scrap_data()

        # Sprawdzenia (Asercje)
        assert len(result) == 15  # W kodzie jest wpisane 15 linków na sztywno
        assert isinstance(result[0], ScrapedPage)
        assert result[0].text == "Przykładowy tekst strony"
        assert result[0].links == ["http://link1.com"]

        # Sprawdzamy, czy scrape i sleep zostały wywołane 15 razy
        assert mock_app_instance.scrape.call_count == 15
        assert mock_sleep.call_count == 15

    @patch("pipeline.scraper.CURRENT_VERSION", 3)
    @patch("pipeline.scraper.links", ["http://test.com/1", "http://test.com/2"])
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

        # Przekazaliśmy 2 linki w zmiennej globalnej 'links', więc wynik powinien mieć dł. 2
        assert len(result) == 2
        assert result[0].url == "http://test.com/1"
        assert result[1].url == "http://test.com/2"
        assert mock_app_instance.scrape.call_count == 2

    @patch("pipeline.scraper.CURRENT_VERSION", 2)
    def test_scrap_data_handles_api_exception(
        self, mock_clean_head, mock_clean_foot, MockFirecrawl, mock_sleep
    ):
        """
        Test sprawdza, czy funkcja nie wywala się całkowicie, gdy API rzuci błędem dla jakiegoś URL-a.
        """
        mock_app_instance = MockFirecrawl.return_value
        # Symulujemy, że metoda scrape rzuca błąd przy każdym wywołaniu
        mock_app_instance.scrape.side_effect = Exception("API Timeout lub inny błąd")

        # Wywołanie funkcji nie powinno rzucić wyjątku, bo jest on łapany w try...except
        result = scrap_data()

        # Skoro wszystkie 15 linków rzuciło błąd, output powinien być pustą listą
        assert result == []
        assert mock_app_instance.scrape.call_count == 15

    @patch("pipeline.scraper.os.getenv", return_value=None)
    @patch("pipeline.scraper.logger.warning")
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
        # Nadpisujemy wersję, żeby skrócić test (wystarczy, że odpali się chociaż dla pustej listy linków)
        with (
            patch("src.pipeline.scraper.CURRENT_VERSION", 3),
            patch("src.pipeline.scraper.links", []),
        ):
            scrap_data()

        # Sprawdzamy czy nasza flaga braku klucza wywołała odpowiedniego loga
        mock_logger_warning.assert_called_once_with(
            "FIRECRAWL_API_KEY not found in environment variables."
        )


@patch("pipeline.scraper.logger")  # Blokujemy loggera, żeby nie śmiecił w konsoli
@patch("pipeline.scraper.os.makedirs")  # Blokujemy tworzenie prawdziwych folderów
@patch("pipeline.scraper.scrap_data")  # Blokujemy prawdziwy scraping
class TestMainPipeline:

    def test_main_happy_path_saves_files(
        self, mock_scrap_data, mock_makedirs, mock_logger
    ):
        """
        Testuje główną ścieżkę: funkcja dostaje dane, tworzy folder
        i poprawnie zapisuje pliki z odpowiednio sformatowaną nazwą.
        """
        # 1. Przygotowujemy sztuczne dane, które rzekomo zwrócił scraper
        mock_scrap_data.return_value = [
            ScrapedPage(
                url="https://ww2.mini.pw.edu.pl/wydzial/",
                text="Tekst wydziału",
                links=[],
            ),
            ScrapedPage(
                url="https://example.com/test_page", text="Inny tekst", links=[]
            ),
        ]

        # 2. Tworzymy mocka dla operacji otwierania plików (mock_open)
        m_open = mock_open()

        # 3. Odpalamy funkcję main(), nakładając mocka na wbudowaną funkcję 'open'
        with patch("builtins.open", m_open):
            main()

        # --- ASERCJE (Sprawdzamy, czy main zachował się poprawnie) ---

        # Czy scraper został wywołany?
        mock_scrap_data.assert_called_once()

        # Czy funkcja próbowała stworzyć prawidłowy folder?
        mock_makedirs.assert_called_once_with("src/data/scraped_raw", exist_ok=True)

        # Czy otworzono dokładnie dwa pliki do zapisu? (bo mamy 2 zescrapowane strony)
        assert m_open.call_count == 2

        # Sprawdzamy czy nazwy plików zostały poprawnie "oczyszczone" przez algorytm w main()
        # "https://ww2.mini.pw.edu.pl/wydzial/" -> "ww2.mini.pw.edu.pl_wydzial"
        m_open.assert_any_call(
            "src/data/scraped_raw/ww2.mini.pw.edu.pl_wydzial.txt", "w", encoding="utf-8"
        )

        # "https://example.com/test_page" -> "example.com_test_page"
        m_open.assert_any_call(
            "src/data/scraped_raw/example.com_test_page.txt", "w", encoding="utf-8"
        )

        # Sprawdzamy, czy do plików wpisano poprawne dane (URL + tekst)
        # m_open() zwraca tzw. file handle, sprawdzamy co do niego wpisano
        handle = m_open()
        expected_calls = [
            call("URL: https://ww2.mini.pw.edu.pl/wydzial/\n\nTekst wydziału"),
            call("URL: https://example.com/test_page\n\nInny tekst"),
        ]
        handle.write.assert_has_calls(expected_calls, any_order=True)

        # Sprawdzamy, czy na koniec zalogowano sukces z poprawną liczbą
        mock_logger.info.assert_called_with(
            "Successfully saved 2 to src/data/scraped_raw."
        )

    def test_main_empty_scraped_data(self, mock_scrap_data, mock_makedirs, mock_logger):
        """
        Testuje sytuację, gdy scraper nic nie znalazł (zwraca pustą listę).
        """
        mock_scrap_data.return_value = []
        m_open = mock_open()

        with patch("builtins.open", m_open):
            main()

        # Folder nadal powinien zostać utworzony (zgodnie z kodem)
        mock_makedirs.assert_called_once_with("src/data/scraped_raw", exist_ok=True)

        # Ale funkcja 'open' nie powinna zostać ani razu wywołana!
        m_open.assert_not_called()

        # Logger powinien zgłosić, że zapisano 0 plików
        mock_logger.info.assert_called_with(
            "Successfully saved 0 to src/data/scraped_raw."
        )

    def test_main_filename_truncation(
        self, mock_scrap_data, mock_makedirs, mock_logger
    ):
        """
        W kodzie jest ucinanie nazwy pliku do 200 znaków ([:200]).
        Ten test sprawdza, czy to faktycznie działa dla bardzo długich URL-i.
        """
        very_long_url = "https://example.com/" + ("a" * 250)
        mock_scrap_data.return_value = [
            ScrapedPage(url=very_long_url, text="Tekst", links=[])
        ]

        m_open = mock_open()
        with patch("builtins.open", m_open):
            main()

        # Oczekiwana nazwa pliku to domena + długi string, ale ucięte do 200 znaków łącznej długości
        expected_filename = (
            very_long_url.replace("https://", "").replace("/", "_").strip("_")[:200]
        )
        expected_path = os.path.join("src/data/scraped_raw", f"{expected_filename}.txt")

        # Sprawdzamy, czy plik został otwarty z tą obciętą ścieżką
        m_open.assert_called_once_with(expected_path, "w", encoding="utf-8")

        # Zabezpieczenie: upewniamy się, że nazwa samego pliku (bez .txt) ma dokładnie 200 znaków
        assert len(expected_filename) == 200
