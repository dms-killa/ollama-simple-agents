"""
Context Manager for Vector Database Integration

This module handles retrieval of relevant context from a vector database
to enhance agent responses with external knowledge.

It supports multiple vector DB backends:
- Chroma (fully implemented)
- Pinecone and Qdrant (stubs that raise NotImplementedError until extended)
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Context Strategy Helper ---
class ContextStrategy:
    """Help determine whether a step should retrieve vector context and
    build the query/collection names."""

    @staticmethod
    def should_retrieve_context(step: Dict[str, Any]) -> bool:
        return bool(step.get("vector_context_enabled", False))

    @staticmethod
    def build_context_query(step: Dict[str, Any], user_request: str, state: Dict[str, Any]) -> str:
        template = step.get("vector_context_query", "")
        if not template:
            return user_request
        try:
            return template.format(user_request=user_request, **state)
        except Exception:
            return user_request

    @staticmethod
    def get_collection_name(step: Dict[str, Any]) -> str:
        return step.get("vector_collection", "default")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manages context retrieval from vector databases.
    """

    def __init__(
        self,
        db_type: str = "chroma",
        embedding_model: Optional[str] = None,
        **kwargs
    ):
        self.db_type = db_type.lower()
        self.client = None
        self.embedding_function = None

        # Initialize embedding model (DB-agnostic)
        self.embedding_model = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )

        # Initialize the appropriate vector DB client
        if self.db_type == "chroma":
            self._init_chroma(**kwargs)
        elif self.db_type == "pinecone":
            self._init_pinecone(**kwargs)
        elif self.db_type == "qdrant":
            self._init_qdrant(**kwargs)
        else:
            raise ValueError(f"Unsupported vector DB type: {db_type}")

    def _init_chroma(self, persist_directory: Optional[str] = None, **kwargs):
        """Initialize ChromaDB client with embedding function."""
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            persist_dir = persist_directory or os.getenv(
                "CHROMA_PERSIST_DIR", "./data/chroma_db"
            )

            Path(persist_dir).mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(path=persist_dir)

            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model
            )
            logger.info(
                f"✓ ChromaDB initialized at {persist_dir} with model {self.embedding_model}"
            )
        except ImportError as e:
            raise ImportError(
                "ChromaDB not installed. Install with: pip install chromadb sentence-transformers"
            ) from e

    def _init_pinecone(self, **kwargs):
        """Initialize Pinecone client (not fully implemented)."""
        try:
            from pinecone import Pinecone

            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY not found in environment")
            self.client = Pinecone(api_key=api_key)
            logger.warning("⚠ Pinecone initialization incomplete - embedding function not configured")
        except ImportError as e:
            raise ImportError(
                "Pinecone not installed. Install with: pip install pinecone-client"
            ) from e

    def _init_qdrant(self, **kwargs):
        """Initialize Qdrant client (not fully implemented)."""
        try:
            from qdrant_client import QdrantClient

            qdrant_url = os.getenv("QDRANT_URL", "localhost")
            qdrant_port = int(os.getenv("QDRANT_PORT", 6333))

            self.client = QdrantClient(host=qdrant_url, port=qdrant_port)
            logger.warning("⚠ Qdrant initialization incomplete - embedding function not configured")
        except Exception as e:
            raise ImportError(
                "Qdrant not installed. Install with: pip install qdrant-client"
            ) from e

    # ---------- Qdrant helper methods ---------------------------------------------------
    def _ensure_qdrant_collection(self, collection_name: str) -> None:
        """Ensure that a Qdrant collection exists.
        If the collection does not exist, it is created with a size inferred from the embedding model.
        """
        try:
            self.client.get_collection(collection_name)
        except Exception:
            from sentence_transformers import SentenceTransformer
            dummy_vec = SentenceTransformer(self.embedding_model).encode(["test"])
            dim = len(dummy_vec[0])
            self.client.create_collection(
                name=collection_name,
                vectors_config={"size": dim, "distance": "Cosine"},
            )
            logger.info(f"✓ Qdrant collection '{collection_name}' created with dimension {dim}")

    def get_or_create_collection(self, collection_name: str) -> Any:
        """Return a Chroma collection or raise for unsupported DBs."""
        if self.db_type == "chroma":
            try:
                collection = self.client.get_or_create_collection(
                    name=collection_name, embedding_function=self.embedding_function
                )
                count = collection.count()
                if count == 0:
                    logger.warning(
                        f"⚠ Collection '{collection_name}' is empty - queries will return no results"
                    )
                else:
                    logger.debug(f"Collection '{collection_name}' has {count} documents")
                return collection
            except Exception as e:
                logger.error(f"Error accessing collection '{collection_name}': {e}")
                raise
        elif self.db_type == "pinecone":
            raise NotImplementedError(
                "Pinecone collection management not yet implemented."
            )
        elif self.db_type == "qdrant":
            raise NotImplementedError(
                "Qdrant collection management not yet implemented."
            )
        return None

    # ---------- Context retrieval -------------------------------------------------------
    def get_relevant_context(
        self,
        query: str,
        collection_name: str,
        top_k: int = 5,
        max_context_length: Optional[int] = None,
    ) -> Optional[str]:
        """Retrieve relevant text snippets from the specified vector DB."""
        if self.db_type == "qdrant":
            self._ensure_qdrant_collection(collection_name)
            if self.embedding_function is None:
                from sentence_transformers import SentenceTransformer
                self.embedding_function = SentenceTransformer(self.embedding_model)
            vector = self.embedding_function.encode([query])[0].tolist()
            hits = self.client.search(
                collection_name=collection_name, query_vector=vector, limit=top_k
            )
            snippets: List[str] = []
            for hit in hits:
                payload = getattr(hit, "payload", {})
                text = payload.get("text") or payload.get("content")
                if text:
                    if max_context_length:
                        text = text[:max_context_length]
                    snippets.append(text)
            return "\n\n---\n\n".join(snippets) if snippets else None
        elif self.db_type == "chroma":
            try:
                collection = self.get_or_create_collection(collection_name)
                results = collection.query(
                    query_texts=[query], n_results=top_k, include=["documents"]
                )
                docs = results.get("documents", [])[0]
                return "\n\n---\n\n".join(docs)
            except Exception as e:
                logger.error(f"Chromadb query error: {e}")
                return None
        else:
            raise NotImplementedError(
                f"Context retrieval not implemented for DB type {self.db_type}"
            )

    # ---------- Document chunking -----------------------------------------------------
    def _chunk_documents(self, documents: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split documents into fixed‑size chunks with optional overlap.
        Returns a list of chunk strings.
        """
        chunks: List[str] = []
        for doc in documents:
            start = 0
            while start < len(doc):
                end = min(start + chunk_size, len(doc))
                chunks.append(doc[start:end])
                start += chunk_size - chunk_overlap
        return chunks
""