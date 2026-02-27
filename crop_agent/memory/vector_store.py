import faiss
import numpy as np
import yaml
from pathlib import Path
from sentence_transformers import SentenceTransformer


class VectorStore:
    """
    FAISS-backed semantic vector store for storing and retrieving crop case embeddings.

    Usage:
        store = VectorStore("config/system_config.yaml")
        store.add("Tomato Leaf Blight Medium High humidity Fungicide")
        store.save()
        indices, scores = store.search("Tomato Leaf Blight")
    """

    def __init__(self, config_path: str):
        """
        Initialise the vector store from a YAML config file.

        Usage:
            store = VectorStore("config/system_config.yaml")
        """
        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        memory_cfg = config["memory"]

        self.index_path = Path(memory_cfg["index_path"])
        self.dimension = memory_cfg["embedding_dim"]
        self.index_type = memory_cfg["index_type"]
        self.top_k = memory_cfg["top_k"]

        # Load embedding model
        self.embed_model = SentenceTransformer(memory_cfg["embedding_model"])

        # Initialize index
        self.index = self._initialize_index()

    def _initialize_index(self):
        """
        Load an existing FAISS index from disk or create a new one based on index_type.

        Usage:
            # Called automatically by __init__; no direct usage needed.
            index = store._initialize_index()
        """
        if self.index_path.exists():
            return faiss.read_index(str(self.index_path))

        if self.index_type == "flat":
            return faiss.IndexFlatIP(self.dimension)

        elif self.index_type == "hnsw":
            return faiss.IndexHNSWFlat(self.dimension, 32)

        else:
            raise ValueError("Unsupported index type")

    def _embed(self, text: str):
        """
        Encode a string into a normalised float32 embedding vector.

        Usage:
            # Called internally by add() and search(); no direct usage needed.
            vector = store._embed("Tomato Leaf Blight")
        """
        embedding = self.embed_model.encode([text])
        embedding = np.array(embedding).astype("float32")
        faiss.normalize_L2(embedding)  # for cosine similarity
        return embedding

    def add(self, text: str):
        """
        Embed a text string and add it to the FAISS index.

        Usage:
            store.add("Tomato Leaf Blight Medium High humidity Fungicide Improved")
        """
        embedding = self._embed(text)
        self.index.add(embedding)

    def search(self, query: str):
        """
        Search the index for the top-k most semantically similar entries.

        Usage:
            indices, scores = store.search("Tomato Leaf Blight Medium High humidity")
            for i, idx in enumerate(indices):
                print(idx, scores[i])
        """
        query_embedding = self._embed(query)
        distances, indices = self.index.search(query_embedding, self.top_k)
        return indices[0], distances[0]

    def save(self):
        """
        Persist the current FAISS index to disk at the configured index_path.

        Usage:
            store.add("Tomato Leaf Blight ...")
            store.save()  # writes to data/memory/crop_memory.index
        """
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
