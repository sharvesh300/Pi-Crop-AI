import sqlite3
from pathlib import Path


class MetadataStore:
    """
    SQLite-backed store for persisting crop case text alongside FAISS index positions.

    Usage:
        store = MetadataStore()                        # default: data/memory/cases.db
        store.add_case("Tomato Leaf Blight Medium")
        case = store.get_case(1)
    """

    def __init__(self, db_path="data/memory/cases.db"):
        """
        Open (or create) the SQLite database and ensure the cases table exists.

        Usage:
            store = MetadataStore()                          # uses default path
            store = MetadataStore("custom/path/cases.db")   # custom path
        """
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        """
        Create the cases table if it does not already exist.

        Usage:
            # Called automatically by __init__; no direct usage needed.
            store._create_table()
        """
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_text TEXT
        )
        """)
        self.conn.commit()

    def add_case(self, text: str):
        """
        Insert a new case text record into the SQLite database.

        Usage:
            store.add_case("Tomato Leaf Blight Medium High humidity Fungicide Improved")
        """
        self.conn.execute("INSERT INTO cases (case_text) VALUES (?)", (text,))
        self.conn.commit()

    def get_case(self, case_id: int):
        """
        Retrieve case text by its auto-incremented integer ID.

        Usage:
            indices, scores = vector_store.search("Tomato Leaf Blight")
            for idx in indices:
                print(store.get_case(idx + 1))  # FAISS is 0-based; SQLite IDs start at 1
        """
        cursor = self.conn.execute("SELECT case_text FROM cases WHERE id=?", (case_id,))
        result = cursor.fetchone()
        return result[0] if result else None
