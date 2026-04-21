import pytest

from pipeline.scraper import clean_footnote, clean_headnote

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
