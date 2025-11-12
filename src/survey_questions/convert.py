import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# --- 1. Import ---
plik = "questions.csv"
df = pd.read_csv(plik, sep=",", encoding="utf-8-sig", quotechar='"')

kolumny = [
    "(1/3) Podaj jedno z pytań, na które odpowiedzi najczęściej szukasz.",
    "(2/3) Podaj jedno z pytań, na które odpowiedzi najczęściej szukasz.",
    "(3/3) Podaj jedno z pytań, na które odpowiedzi najczęściej szukasz."
]

pytania = df[kolumny].melt(value_name="pytanie")["pytanie"]


# --- 2. Cleaning ---
def oczysc_tekst(t):
    t = str(t).lower().strip()
    t = re.sub(r"[^a-z0-9ąćęłńóśźż\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


pytania = (
    pytania.dropna()
    .astype(str)
    .map(oczysc_tekst)
    .drop_duplicates()
    .reset_index(drop=True)
)


# --- 3. Categories ---
kategorie_slowa = {
    "sesja": [
        "sesja", "egzamin", "egzaminy", "kolokwium", "kolos", "test",
        "termin egzaminu", "terminy egzaminow", "plan sesji",
        "termin sesji", "obrona", "zaliczenie", "dzien egzaminu",
        "termin kolokwium", "terminy kolokwiow","testy","sesji"
    ],

    "plan": [
        "plan", "zajecia", "plan zajec", "plan studiow",
        "godzina", "godziny", "grafik", "rozkład", "sala",
        "dzien", "dni", "zmiana planu", "zamiana dni", "dzien zamieniony",
        "plan poniedzialkowy", "plan wtorkowy", "o ktorej mam zajecia",
        "kiedy mam zajecia", "nastepne zajecia", "kolejne zajecia", "sali", "zajęcia", "grupy ćwiczeniowe",
        "godzinach", "ćwiczenia", "wykłady"
    ],

    "sylabus/obieralne": [
        "sylabus", "syllabus", "ects", "kierunek", "semestr", "program studiow", "program studiów"
        "przedmiot", "przedmioty", "przedmioty obieralne", "obieralne",
        "obierak", "przedmioty do wyboru", "realizowac", "obowiazuje",
        "podpiecie", "podpiecia", "podpinac", "plan studiow", "oferta przedmiotow",
        "katalog", "studia magisterskie", "różnice przedmiotowe", "planu studiów", "ectsow",
        "róznice przedmiotowe", "obieralnych"
    ],

    "harmonogram": [
        "harmonogram", "kalendarz", "rok akademicki", "dni wolne", "dzien wolny",
        "wolne", "dni rektorskie", "dzien rektorski", "terminarz",
        "zamiana dni", "zamienione dni", "wolne od zajec", "dni dodatkowo wolne",
        "swieta", "przerwa swiateczna", "dni podmienione", "przerwa świąteczna",
        "dzień","dniem", "zmiany planu", "zmiana planu"
    ],

    "stypendium": [
        "stypendium", "stypendia", "rektora", "wniosek o stypendium",
        "punkty do stypendium", "osiagniecia naukowe", "przyznawanie stypendium",
        "ile wynosi stypendium", "termin stypendium", "punkty rektorskie", "stypendiów"
    ],

    "dziekanat": [
        "dziekanat", "dziekanaty", "sekretariat", "sekretariaty",
        "godziny otwarcia dziekanatu", "otwarte", "godziny pracy",
        "interesant", "przyjecia", "kiedy moge przyjsc",
        "dokument w dziekanacie", "zaswiadczenie", "odbior dyplomu"
    ],

    "pracownik": [
        "pracownik", "prowadzacy", "kontakt", "profesor", "wykladowca",
        "prodziekan", "dziekan", "konsultacje", "godziny konsultacji",
        "numer pokoju", "pokoj", "adres mailowy", "mail do", "u kogo", "tytułów",
        "tytuł"
    ],

    "praktyki": [
        "praktyka", "praktyki", "rozliczenie praktyk", "zaliczenie praktyk",
        "dokumenty do praktyk", "umowa zlecenie", "termin praktyk",
        "czy beda praktyki"
    ],

    "warunek": [
        "warunek", "warunki", "zaliczenie warunkowe", "wznowienie",
        "koszt warunku", "cena warunku", "ile kosztuje", "oplata za warunek",
        "warunkowe zaliczenie", "urlop okolicznosciowy", "wznowienie studiow",
        "warunków", "poprawienie przedmiotów", "poprawianie przedmiotów"
    ],

    "erasmus": [
        "erasmus", "wymiana", "miedzynarodowa", "wymiana studencka",
        "wyjazd", "za granice", "zagraniczna", "program erasmus",
        "regulamin wymian", "wymiana miedzynarodowa", "wymiane"
    ],

    "praca_dyplomowa": [
        "praca dyplomowa", "dyplomowa", "inzynierka", "magisterka",
        "temat pracy", "promotor", "obrona", "tematy prac", "pisanie pracy", "obronę",
        "pracy", "inżynierki"
    ],

    "wydarzenia": [
        "wydarzenie", "wydarzenia", "event", "eventy",
        "juwenalia", "konik", "kucyk", "otrzesiny", "wigilia",
        "kola naukowe", "spotkania", "wyklady otwarte", "wydarzenia politechniczne",
        "dzieje się"
    ],

    "dokumenty": [
        "dokument", "dokumenty", "zaswiadczenie", "wniosek", "podanie",
        "formularz", "druk", "papier", "dostarczyc dokumenty", "badania",
        "badanie","odbiór","ubezpieczenie"
    ],

    "drukarka": [
        "drukarka", "drukarki", "drukowac", "wydrukowac", "druk", "wydruk",
        "jak drukowac", "drukowanie", "wydzialowa drukarka", "drukarkę",
        "drukarce"
    ],

    "parking": [
        "parking", "parkingi", "parkowanie", "miejsce parkingowe", "samochod"
    ],

    "efekty": [
        "efekt", "efekty", "efekty uczenia", "efekty ksztalcenia"
    ],

    "regulamin": [
        "regulamin", "zasady", "regulamin studiow", "przepisy", "zasady zaliczenia", "obowiązkowe"
    ],

    "inne": [
        "internet", "wifi", "sieć", "rezerwacja", "palarnia",
        "wf", "materialy dydaktyczne", "konto", "samorzad", "pokoje do rozmow",
        "oplata", "ogolne informacje", "system oceniania"
    ]
}


# --- 4. Assign category ---
def przypisz_kategorie(pytanie):
    for kategoria, slowa in kategorie_slowa.items():
        for slowo in slowa:
            if re.search(rf"\b{slowo}\b", pytanie):
                return kategoria
    return "inne"


# --- 5. Create base ---
baza = pd.DataFrame({
    "id": range(1, len(pytania) + 1),
    "pytanie": pytania,
})
baza["kategoria"] = baza["pytanie"].map(przypisz_kategorie)

# --- 6. Save as CSV (overwrite) ---
baza.to_csv("questions_cat.csv", index=False, encoding="utf-8-sig")
print(f"Utworzono bazę z {len(baza)} pytaniami i kategoriami: 'questions_cat.csv'.")


# --- 7. Visualisation ---
liczba_pytan = baza["kategoria"].value_counts().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
liczba_pytan.plot(kind="bar")
plt.title("Liczba pytań w poszczególnych kategoriach", fontsize=14)
plt.xlabel("Kategoria", fontsize=12)
plt.ylabel("Liczba pytań", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# --- TF-IDF ---
vectorizer = TfidfVectorizer(stop_words=None)
tfidf_matrix = vectorizer.fit_transform(baza["pytanie"])


# --- 9. Search function---
def znajdz_podobne_pytania(zapytanie, n=5):
    zapytanie = oczysc_tekst(zapytanie)
    zapytanie_tfidf = vectorizer.transform([zapytanie])
    podobienstwa = cosine_similarity(zapytanie_tfidf, tfidf_matrix).flatten()
    indeksy = podobienstwa.argsort()[::-1][:n]

    print("\nNajbardziej podobne pytania i ich kategorie:")
    for i in indeksy:
        print(
            f"• {baza.iloc[i]['pytanie']} (kategoria: {baza.iloc[i]['kategoria']}, podobieństwo: {podobienstwa[i]:.2f})"
        )


# --- 10. Interactive search ---
while True:
    zapytanie = input("\nWpisz słowo klucz lub kategorię (lub 'exit' aby zakończyć): ").strip().lower()
    if zapytanie == "exit":
        print("Zakończono program.")
        break

    dopasowana_kategoria = None
    for kategoria, slowa in kategorie_slowa.items():
        if zapytanie == kategoria or any(re.search(rf"\b{slowo}\b", zapytanie) for slowo in slowa):
            dopasowana_kategoria = kategoria
            break

    if dopasowana_kategoria:
        pytania_z_kategorii = baza[baza["kategoria"] == dopasowana_kategoria]
        if not pytania_z_kategorii.empty:
            print(f"\n Pytania przypisane do kategorii **{dopasowana_kategoria}**:")
            for _, row in pytania_z_kategorii.iterrows():
                print(f"• {row['pytanie']}")
        else:
            print(f"Brak pytań w kategorii '{dopasowana_kategoria}'.")
    else:
        znajdz_podobne_pytania(zapytanie)
