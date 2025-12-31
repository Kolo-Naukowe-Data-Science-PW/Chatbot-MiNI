from langchain_huggingface import HuggingFaceEmbeddings


class Embedder:
    """
    A wrapper class for generating embeddings using HuggingFace models.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the Embedder with a specific HuggingFace model.

        Parameters
        ----------
        model_name : str, optional
            The name or path of the HuggingFace model to use,
            by default "sentence-transformers/all-MiniLM-L6-v2".
        """
        self.model_name = model_name
        self.embedder = HuggingFaceEmbeddings(model_name=model_name)

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generates a vector embedding for a single text chunk.

        Parameters
        ----------
        text : str
            The input text string to be embedded.

        Returns
        -------
        list[float]
            A list of floats representing the vector embedding.
        """
        return self.embedder.embed_query(text)

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generates vector embeddings for a list of text chunks.

        Parameters
        ----------
        texts : list[str]
            A list of text strings to be embedded.

        Returns
        -------
        list[list[float]]
            A list of vectors, where each vector corresponds to a text in the input list.
        """
        return self.embedder.embed_documents(texts)