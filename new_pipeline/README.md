This module consists files with new scraping & ingesting logic.

---

chatbot-mini/
├── .gitignore                <-- modified
├── docker-compose.yml        <-- modified
├── ... (old chatbot code)
└── new_pipeline/             <-- this module with new scraping & ingesting
    ├── scraper.py            <-- **TODO** scrapes stuff (at first only 15 URLs, then everything)
    ├── common.py             <-- model config from OpenRouterAI and version config
    ├── parse_html_pdf.py     <-- **TODO** extracts text content from HTML and PDF files
    ├── describe_files.py     <-- extracts text content from XLSX and DOCX files *if needed*
    ├── extract_facts.py      <-- extracts facts from text using the LLM *if needed*
    ├── ingest_facts.py       <-- loads facts from JSON files into the ChromaDB vector database
    └── data/                 <-- NOT TO BE PUSHED, included in the .gitignore
        ├── scraped_raw/      <-- raw scraped files
        ├── processed_text/   <-- text extracted from raw files
        ├── facts/            <-- JSON files containing facts
        └── complex_files/    <-- complex files

---

Version description in the `common.py` file:
1 = 15 URLs + only HTML/PDF + 1 file in 1 chunk + no LLM
2 = 15 URLs + only HTML/PDF + 1 fact in 1 chunk + with LLM
3 = MiNI website + all files + 1 fact in 1 chunk + filtering by hand + with LLM
4 = all sources + all files + 1 fact in 1 chunk + filtering by hand + with LLM

---

**Important note on the XLSX/DOCX files handling**: I think we should extract text from the XLSX/DOCX files without using an LLM. Since we're already using models for fact generation and the final answer, we need to be mindful of token costs.
