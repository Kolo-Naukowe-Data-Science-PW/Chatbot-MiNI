from langchain.text_splitter import RecursiveCharacterTextSplitter

# from pathlib import Path


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """
    Splits a single text string into smaller overlapping chunks.

    Args:
        text : The full text to split into chunks.
        chunk_size : Maximum number of characters in each chunk.
        overlap : Number of overlapping characters between consecutive chunks.

    Returns:
        chunks : A list of text chunks (strings).
    """

    # this splitter uses context/natural boundaries to divide the text into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    chunks = splitter.split_text(text)
    return chunks

    # bardzo fajne, ale jednak main.py zakłada że przetwarzamy string, a nie folder z plikami TXT
    # ale zachowuję, może i zrobimy to tak że będziemy przetwarzać foldery
    #
    # all_chunks = []
    # were only using txt files
    # for file_path in Path(folder_path).glob("*.txt"):
    #    with open(file_path, "r", encoding="utf-8") as f:
    #        text = f.read()
    #    if not text:
    #        print(f"File: {file_path} is empty")
    #        continue
    #    chunks = splitter.split_text(text)
    #    for i,c in enumerate(chunks):
    #        all_chunks.append({
    #            "source" : file_path.name,
    #            "chunk_index" : i,
    #            "text" : c
    #        })
