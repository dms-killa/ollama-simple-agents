"""
Context Manager for Vector Database Integration

This module handles retrieval of relevant context from a vector database
to enhance agent responses with external knowledge.

Improvements implemented:
- Fixed multi-DB implementation consistency
- Unified embedding function initialization
- Improved ID generation (collision-resistant)
- Document chunking support
- Safe template substitution
- Query caching with LRU
- Specific exception handling
- Proper logging instead of prints
- Collection existence validation
- Distance normalization support
"""

import os
import logging
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from functools import lru_cache
from string import Formatter
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manages context retrieval from vector databases.

    Supports multiple vector DB backends: Chroma (fully implemented),
    Pinecone and Qdrant (stubs that raise NotImplementedError).
    """

    def __init__(
        self, 
        db_type: str = "chroma",
        embedding_model: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the context manager.

        Args:
            db_type: Type of vector database ('chroma', 'pinecone', 'qdrant')
            embedding_model: Name of the embedding model to use
            **kwargs: Additional configuration passed to the specific DB client
        """
        self.db_type = db_type.lower()
        self.client = None
        self.embedding_function = None
        
        # Initialize embedding model (DB-agnostic)
        self.embedding_model = embedding_model or os.getenv(
            "EMBEDDING_MODEL", 
            "all-MiniLM-L6-v2"
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
                "CHROMA_PERSIST_DIR",
                "./data/chroma_db"
            )

            # Create directory if it doesn't exist
            Path(persist_dir).mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(path=persist_dir)

            # Initialize embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model
            )
            
            logger.info(f"✓ ChromaDB initialized at {persist_dir} with model {self.embedding_model}")

        except ImportError:
            raise ImportError(
                "ChromaDB not installed. Install with: pip install chromadb sentence-transformers"
            )

    def _init_pinecone(self, **kwargs):
        """Initialize Pinecone client - NOT YET FULLY IMPLEMENTED."""
        try:
            from pinecone import Pinecone

            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY not found in environment")

            self.client = Pinecone(api_key=api_key)
            
            # TODO: Initialize embedding function for Pinecone
            logger.warning("⚠ Pinecone initialization incomplete - embedding function not configured")

        except ImportError:
            raise ImportError(
                "Pinecone not installed. Install with: pip install pinecone-client"
            )

    def _init_qdrant(self, **kwargs):
        """Initialize Qdrant client - NOT YET FULLY IMPLEMENTED."""
        try:
            from qdrant_client import QdrantClient

            qdrant_url = os.getenv("QDRANT_URL", "localhost")
            qdrant_port = int(os.getenv("QDRANT_PORT", 6333))

            self.client = QdrantClient(host=qdrant_url, port=qdrant_port)
            
            # TODO: Initialize embedding function for Qdrant
            logger.warning("⚠ Qdrant initialization incomplete - embedding function not configured")

        except ImportError:
            raise ImportError(
                "Qdrant not installed. Install with: pip install qdrant-client"
            )

    def get_or_create_collection(self, collection_name: str) -> Any:
        """
        Get or create a collection in the vector database.

        Args:
            collection_name: Name of the collection

        Returns:
            Collection object
            
        Raises:
            NotImplementedError: For non-Chroma databases
        """
        if self.db_type == "chroma":
            try:
                collection = self.client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                
                # Check if collection is empty and warn
                count = collection.count()
                if count == 0:
                    logger.warning(f"⚠ Collection '{collection_name}' is empty - queries will return no results")
                else:
                    logger.debug(f"Collection '{collection_name}' has {count} documents")
                    
                return collection
            except Exception as e:
                logger.error(f"Error accessing collection '{collection_name}': {e}")
                raise
                
        elif self.db_type == "pinecone":
            raise NotImplementedError(
                "Pinecone collection management not yet implemented. "
                "Contribute at: https://github.com/your-repo"
            )
        elif self.db_type == "qdrant":
            raise NotImplementedError(
                "Qdrant collection management not yet implemented. "
                "Contribute at: https://github.com/your-repo"
            )

        return None

    def _chunk_documents(
        self,
        documents: List[str],
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """
        Split documents into smaller chunks for better retrieval.
        
        Args:
            documents: List of document texts
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of overlapping characters between chunks
            
        Returns:
            List of chunked documents
        """
        chunked_docs = []
        
        for doc in documents:
            if len(doc) <= chunk_size:
                chunked_docs.append(doc)
            else:
                # Split into overlapping chunks
                start = 0
                while start < len(doc):
                    end = start + chunk_size
                    chunk = doc[start:end]
                    chunked_docs.append(chunk)
                    start += (chunk_size - chunk_overlap)
                    
        return chunked_docs

    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: int = 200
    ) -> bool:
        """
        Add documents to a collection.

        Args:
            collection_name: Name of the collection
            documents: List of document texts
            metadatas: Optional metadata for each document
            ids: Optional custom IDs (auto-generated if not provided)
            chunk_size: If provided, split documents into chunks of this size
            chunk_overlap: Number of overlapping characters between chunks

        Returns:
            True if successful
            
        Raises:
            NotImplementedError: For non-Chroma databases
        """
        # Apply chunking if requested
        if chunk_size:
            original_count = len(documents)
            documents = self._chunk_documents(documents, chunk_size, chunk_overlap)
            logger.info(f"Chunked {original_count} documents into {len(documents)} chunks")
            
            # Adjust metadata and IDs for chunks
            if metadatas:
                # Repeat metadata for chunks from same document
                new_metadatas = []
                chunk_idx = 0
                for i, doc in enumerate(documents[:original_count]):
                    chunks_for_doc = len(self._chunk_documents([doc], chunk_size, chunk_overlap))
                    new_metadatas.extend([metadatas[i]] * chunks_for_doc)
                metadatas = new_metadatas
            
            ids = None  # Regenerate IDs for chunked documents
        
        collection = self.get_or_create_collection(collection_name)

        # Generate IDs if not provided - use full SHA256 to avoid collisions
        if ids is None:
            ids = [
                hashlib.sha256(doc.encode()).hexdigest()
                for doc in documents
            ]

        # Generate empty metadata if not provided
        if metadatas is None:
            metadatas = [{} for _ in documents]

        if self.db_type == "chroma":
            try:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"✓ Added {len(documents)} documents to collection '{collection_name}'")
                return True
            except Exception as e:
                logger.error(f"Error adding documents to '{collection_name}': {e}")
                raise
                
        elif self.db_type == "pinecone":
            raise NotImplementedError(
                "Pinecone document addition not yet implemented. "
                "Contribute at: https://github.com/your-repo"
            )
        elif self.db_type == "qdrant":
            raise NotImplementedError(
                "Qdrant document addition not yet implemented. "
                "Contribute at: https://github.com/your-repo"
            )

        return False

    def _normalize_distances(
        self,
        distances: List[float],
        method: str = "minmax"
    ) -> List[float]:
        """
        Normalize distance scores to 0-1 range.
        
        Args:
            distances: List of distance values
            method: Normalization method ('minmax' or 'sigmoid')
            
        Returns:
            Normalized scores (higher is better)
        """
        if not distances:
            return []
            
        if method == "minmax":
            min_dist = min(distances)
            max_dist = max(distances)
            if max_dist == min_dist:
                return [1.0] * len(distances)
            # Convert to similarity score (inverse of distance)
            return [1.0 - (d - min_dist) / (max_dist - min_dist) for d in distances]
        elif method == "sigmoid":
            import math
            # Sigmoid normalization (steeper curve)
            return [1.0 / (1.0 + math.exp(d)) for d in distances]
        else:
            return distances

    def query_context(
        self,
        query: str,
        collection_name: str = "default",
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        normalize_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query the vector database for relevant context.

        Args:
            query: The search query
            collection_name: Name of the collection to search
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            normalize_scores: Whether to normalize distance scores

        Returns:
            List of relevant documents with metadata and scores
            
        Raises:
            NotImplementedError: For non-Chroma databases
        """
        collection = self.get_or_create_collection(collection_name)

        if self.db_type == "chroma":
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    where=filter_metadata
                )

                # Format results
                formatted_results = []
                if results['documents'] and results['documents'][0]:
                    distances = results['distances'][0] if results['distances'] else []
                    
                    # Normalize scores if requested
                    scores = self._normalize_distances(distances) if normalize_scores and distances else distances
                    
                    for i, doc in enumerate(results['documents'][0]):
                        formatted_results.append({
                            'content': doc,
                            'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                            'distance': distances[i] if distances else None,
                            'score': scores[i] if scores else None,
                            'id': results['ids'][0][i] if results['ids'] else None
                        })

                logger.debug(f"Retrieved {len(formatted_results)} results for query: '{query[:50]}...'")
                return formatted_results
                
            except Exception as e:
                logger.error(f"Error querying collection '{collection_name}': {e}")
                raise
                
        elif self.db_type == "pinecone":
            raise NotImplementedError(
                "Pinecone querying not yet implemented. "
                "Contribute at: https://github.com/your-repo"
            )
        elif self.db_type == "qdrant":
            raise NotImplementedError(
                "Qdrant querying not yet implemented. "
                "Contribute at: https://github.com/your-repo"
            )

        return []

    def format_context_for_prompt(
        self,
        results: List[Dict[str, Any]],
        include_metadata: bool = True,
        max_context_length: Optional[int] = None
    ) -> str:
        """
        Format retrieved context for inclusion in a prompt.

        Args:
            results: List of query results from query_context()
            include_metadata: Whether to include metadata in output
            max_context_length: Maximum character length for context

        Returns:
            Formatted context string
        """
        if not results:
            return ""

        context_parts = ["# Retrieved Context\n"]

        for i, result in enumerate(results, 1):
            part = f"\n## Source {i}"
            
            # Add score if available
            if result.get('score') is not None:
                part += f" (Relevance: {result['score']:.2f})"

            if include_metadata and result.get('metadata'):
                metadata = result['metadata']
                if metadata:
                    part += "\n**Metadata:**"
                    for key, value in metadata.items():
                        part += f"\n- {key}: {value}"

            part += f"\n\n{result['content']}\n"
            context_parts.append(part)

        context = "\n".join(context_parts)

        # Truncate if necessary
        if max_context_length and len(context) > max_context_length:
            context = context[:max_context_length] + "\n\n... [truncated]"

        return context

    @lru_cache(maxsize=128)
    def get_relevant_context_cached(
        self,
        query: str,
        collection_name: str,
        top_k: int
    ) -> str:
        """
        Cached version of context retrieval for repeated queries.
        
        Note: This is a separate method to allow cache clearing.
        Use get_relevant_context() for the main API.
        """
        results = self.query_context(
            query=query,
            collection_name=collection_name,
            top_k=top_k
        )
        return self.format_context_for_prompt(results)

    def get_relevant_context(
        self,
        query: str,
        collection_name: str = "default",
        top_k: int = 5,
        format_for_prompt: bool = True,
        use_cache: bool = True,
        **kwargs
    ) -> str:
        """
        Convenience method to query and format context in one step.

        Args:
            query: Search query
            collection_name: Collection to search
            top_k: Number of results
            format_for_prompt: Whether to format for LLM prompt
            use_cache: Whether to use LRU cache for repeated queries
            **kwargs: Additional arguments passed to format_context_for_prompt

        Returns:
            Formatted context string ready for prompt injection
        """
        # Use cached version if enabled and no custom kwargs
        if use_cache and format_for_prompt and not kwargs:
            return self.get_relevant_context_cached(query, collection_name, top_k)
        
        # Non-cached path
        results = self.query_context(
            query=query,
            collection_name=collection_name,
            top_k=top_k
        )

        if format_for_prompt:
            return self.format_context_for_prompt(results, **kwargs)

        return results

    def clear_cache(self):
        """Clear the LRU cache for context retrieval."""
        self.get_relevant_context_cached.cache_clear()
        logger.info("✓ Context retrieval cache cleared")

    def list_collections(self) -> List[str]:
        """
        List all available collections.
        
        Returns:
            List of collection names
            
        Raises:
            NotImplementedError: For non-Chroma databases
        """
        if self.db_type == "chroma":
            try:
                collections = self.client.list_collections()
                return [col.name for col in collections]
            except Exception as e:
                logger.error(f"Error listing collections: {e}")
                raise
                
        elif self.db_type == "pinecone":
            raise NotImplementedError(
                "Pinecone collection listing not yet implemented. "
                "Contribute at: https://github.com/your-repo"
            )
        elif self.db_type == "qdrant":
            raise NotImplementedError(
                "Qdrant collection listing not yet implemented. "
                "Contribute at: https://github.com/your-repo"
            )

        return []

    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            collection_name: Name of collection to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.db_type == "chroma":
                self.client.delete_collection(name=collection_name)
                logger.info(f"✓ Deleted collection '{collection_name}'")
                return True
            elif self.db_type == "pinecone":
                raise NotImplementedError(
                    "Pinecone collection deletion not yet implemented. "
                    "Contribute at: https://github.com/your-repo"
                )
            elif self.db_type == "qdrant":
                raise NotImplementedError(
                    "Qdrant collection deletion not yet implemented. "
                    "Contribute at: https://github.com/your-repo"
                )
        except NotImplementedError:
            raise
        except Exception as e:
            logger.error(f"Error deleting collection '{collection_name}': {e}")
            return False

        return False


class ContextStrategy:
    """
    Defines strategies for when and how to retrieve context.

    This can be extended to implement more sophisticated retrieval strategies.
    """

    @staticmethod
    def should_retrieve_context(step: Dict[str, Any]) -> bool:
        """
        Determine if context should be retrieved for this step.

        Args:
            step: Workflow step configuration

        Returns:
            True if context retrieval is enabled
        """
        return step.get('vector_context_enabled', False)

    @staticmethod
    def _extract_template_keys(template: str) -> List[str]:
        """
        Extract placeholder keys from a format string.
        
        Args:
            template: Format string with {key} placeholders
            
        Returns:
            List of key names
        """
        return [field_name for _, field_name, _, _ in Formatter().parse(template) 
                if field_name is not None]

    @staticmethod
    def build_context_query(
        step: Dict[str, Any],
        user_request: str,
        state: Dict[str, Any]
    ) -> str:
        """
        Build the query for context retrieval with safe template substitution.

        Args:
            step: Workflow step configuration
            user_request: Original user request
            state: Current workflow state

        Returns:
            Query string for vector DB
        """
        # Use custom query template if provided
        query_template = step.get('vector_context_query')

        if query_template:
            try:
                # Prepare available values
                available = {'user_request': user_request, **state}
                
                # Extract keys from template
                template_keys = ContextStrategy._extract_template_keys(query_template)
                
                # Build safe substitution dict (only include available keys)
                safe_values = {
                    k: available.get(k, f'{{{k}}}')  # Keep placeholder if value missing
                    for k in template_keys
                }
                
                # Perform substitution
                query = query_template.format(**safe_values)
                logger.debug(f"Built context query from template: '{query[:100]}...'")
                
            except (KeyError, ValueError) as e:
                logger.warning(f"Template substitution failed: {e}. Using fallback.")
                query = user_request
        else:
            # Default: use the current input
            input_source = step.get('input_source', 'user_request')
            if input_source == 'user_request':
                query = user_request
            else:
                query = state.get(input_source, user_request)

        return query

    @staticmethod
    def get_collection_name(step: Dict[str, Any]) -> str:
        """
        Determine which collection to query.

        Args:
            step: Workflow step configuration

        Returns:
            Collection name
        """
        return step.get('vector_collection', 'default')