import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Wraps SentenceTransformer to produce normalised float32 embeddings.

    Usage:
        embedder = Embedder("all-MiniLM-L6-v2")
        vector = embedder.embed("Tomato Leaf Blight")
    """

    def __init__(self, model_name: str):
        """
        Load the SentenceTransformer model.

        Usage:
            embedder = Embedder("all-MiniLM-L6-v2")
        """
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        """
        Encode a single string into a normalised float32 embedding vector.

        Usage:
            vector = embedder.embed("Tomato Leaf Blight Medium")
        """
        embedding = self.model.encode([text])
        embedding = np.array(embedding).astype("float32")
        faiss.normalize_L2(embedding)
        return embedding

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of strings into a matrix of normalised float32 embeddings.

        Usage:
            vectors = embedder.embed_batch(["Tomato Leaf Blight", "Powdery Mildew"])
        """
        embeddings = self.model.encode(texts)
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)
        return embeddings
