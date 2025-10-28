from langchain.embeddings import HuggingFaceEmbeddings


class Embedder:

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedder = HuggingFaceEmbeddings(model_name=model_name)

    def generate_embedding(self, text: str):
        """
        Generates a vector embedding for a single text chunk.
        """
        return self.embedder.embed_query(text)

    def generate_embeddings(self, texts: list):
        """
        Generates vector embeddings for a list of text chunks.
        """
        return self.embedder.embed_documents(texts)
