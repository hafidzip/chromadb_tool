"""
chromadb Tool
=============
Manage vector databases, store document embeddings, and perform semantic search.

Parameters:
  action          - The operation to perform (add, query, get, delete, list_collections, delete_collection) (required).
  collection_name - Name of the collection (required for add, query, get, delete, delete_collection).
  db_path         - Path to the persistent database. If omitted, uses an ephemeral in-memory database.
  ids             - List of document IDs (required for add; optional for get, delete).
  documents       - List of document text strings (required for add).
  metadatas       - List of metadata dictionaries (optional for add).
  embeddings      - List of vector embeddings (optional for add).
  query_texts     - List of text strings to query (required for query).
  n_results       - Number of results to return for query (default 5).
  where           - Metadata filter dictionary (optional for query, get, delete).
  where_document  - Document content filter dictionary (optional for query, get).
  limit           - Max results to return for get (optional).
  offset          - Pagination offset for get (optional).
"""

import asyncio
import os
import logging
import threading
from typing import Any, Dict, List, Mapping, Optional
import chromadb
import chromadb.api
from openchadpy.tool_base import ToolBase

logger = logging.getLogger(__name__)

# Persistent clients cache to reuse client instances per db_path
_client_cache: Dict[str, chromadb.api.ClientAPI] = {}
_cache_lock = threading.Lock()

# Per-database locks to prevent concurrent write/read race conditions
_db_locks: Dict[str, asyncio.Lock] = {}
_db_lock_registry_lock = threading.Lock()

def _get_db_lock(db_path: Optional[str]) -> asyncio.Lock:
    canonical = os.path.normcase(os.path.realpath(db_path)) if db_path else "ephemeral"
    with _db_lock_registry_lock:
        if canonical not in _db_locks:
            _db_locks[canonical] = asyncio.Lock()
        return _db_locks[canonical]

def _get_client(db_path: Optional[str]):
    if not db_path:
        return chromadb.EphemeralClient()
    
    canonical = os.path.normcase(os.path.realpath(db_path))
    with _cache_lock:
        if canonical not in _client_cache:
            os.makedirs(canonical, exist_ok=True)
            _client_cache[canonical] = chromadb.PersistentClient(path=canonical)
        return _client_cache[canonical]

class Tool(ToolBase):
    name = "chromadb"
    description = (
        "Interact with ChromaDB to manage collections, store documents, and perform semantic search queries."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "query", "get", "delete", "list_collections", "delete_collection"],
                "description": "The action to perform.",
            },
            "collection_name": {
                "type": "string",
                "description": "Name of the collection to operate on.",
            },
            "db_path": {
                "type": "string",
                "description": "Absolute path to persistent database. If omitted, uses in-memory ephemeral database.",
            },
            "ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "IDs for the documents. Required for 'add'.",
            },
            "documents": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Document texts. Required for 'add'.",
            },
            "metadatas": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Optional metadata dictionaries for each document.",
            },
            "embeddings": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "number"}
                },
                "description": "Optional custom vector embeddings for the documents.",
            },
            "query_texts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Query texts to perform semantic search. Required for 'query'.",
            },
            "n_results": {
                "type": "integer",
                "description": "Number of query results to return. Defaults to 5.",
            },
            "where": {
                "type": "object",
                "description": "Optional dictionary to filter by metadata (e.g. {'author': 'john'}).",
            },
            "where_document": {
                "type": "object",
                "description": "Optional dictionary to filter by document content (e.g. {'$contains': 'chroma'}).",
            },
            "limit": {
                "type": "integer",
                "description": "Limit results for 'get'.",
            },
            "offset": {
                "type": "integer",
                "description": "Offset results for 'get'.",
            }
        },
        "required": ["action"],
    }
    allowed_callers = ["direct", "code_execution"]

    async def execute(self, **kwargs) -> Dict[str, Any]:
        action: str = kwargs.get("action", "")
        collection_name: Optional[str] = kwargs.get("collection_name")
        db_path: Optional[str] = kwargs.get("db_path")
        
        if db_path:
            db_path = db_path.strip()
            
        if action in ["add", "query", "get", "delete", "delete_collection"] and not collection_name:
            return {"error": f"collection_name is required for action '{action}'"}

        # Get lock to avoid write clashes
        lock = _get_db_lock(db_path)
        await lock.acquire()
        
        try:
            # We run the synchronous ChromaDB calls in an executor to avoid blocking the event loop
            return await asyncio.to_thread(self._run_sync, action, collection_name, db_path, kwargs)
        except Exception as e:
            logger.error(f"[chromadb] Error executing action '{action}': {e}", exc_info=True)
            return {"error": str(e)}
        finally:
            lock.release()

    def _run_sync(self, action: str, collection_name: Optional[str], db_path: Optional[str], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        client = _get_client(db_path)

        if action == "list_collections":
            collections = client.list_collections()
            return {"collections": [c.name for c in collections]}

        if not collection_name:
            return {"error": f"collection_name is required for action '{action}'"}

        if action == "delete_collection":
            try:
                client.delete_collection(name=collection_name)
                return {"success": True, "message": f"Collection '{collection_name}' deleted."}
            except Exception as e:
                return {"error": f"Failed to delete collection: {e}"}

        # Get or create collection
        collection = client.get_or_create_collection(name=collection_name)

        if action == "add":
            ids = kwargs.get("ids")
            documents = kwargs.get("documents")
            metadatas = kwargs.get("metadatas")
            embeddings = kwargs.get("embeddings")

            if not ids or not documents:
                return {"error": "Both 'ids' and 'documents' are required for 'add' action."}
            if len(ids) != len(documents):
                return {"error": "The length of 'ids' and 'documents' must match."}
            if metadatas and len(metadatas) != len(ids):
                return {"error": "The length of 'metadatas' must match 'ids'."}
            if embeddings and len(embeddings) != len(ids):
                return {"error": "The length of 'embeddings' must match 'ids'."}

            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            return {"success": True, "count_added": len(ids)}

        elif action == "query":
            query_texts = kwargs.get("query_texts")
            n_results = kwargs.get("n_results", 5)
            where = kwargs.get("where")
            where_document = kwargs.get("where_document")

            if not query_texts:
                return {"error": "'query_texts' is required for 'query' action."}

            results = collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            
            # Convert non-serializable elements if any
            return {"results": self._serialize_results(results)}

        elif action == "get":
            ids = kwargs.get("ids")
            where = kwargs.get("where")
            where_document = kwargs.get("where_document")
            limit = kwargs.get("limit")
            offset = kwargs.get("offset")

            results = collection.get(
                ids=ids,
                where=where,
                where_document=where_document,
                limit=limit,
                offset=offset
            )
            return {"results": self._serialize_results(results)}

        elif action == "delete":
            ids = kwargs.get("ids")
            where = kwargs.get("where")

            if not ids and not where:
                return {"error": "Either 'ids' or 'where' filter is required for 'delete' action."}

            collection.delete(ids=ids, where=where)
            return {"success": True, "message": "Documents deleted."}

        else:
            return {"error": f"Unknown action '{action}'"}

    def _serialize_results(self, results: Mapping[str, Any]) -> Dict[str, Any]:
        # Handle formatting to ensure correct JSON serialization of numpy arrays or other types
        serialized = {}
        for k, v in results.items():
            if v is None:
                serialized[k] = None
            elif isinstance(v, list):
                serialized[k] = self._serialize_list(v)
            else:
                serialized[k] = v
        return serialized

    def _serialize_list(self, items: List[Any]) -> List[Any]:
        res = []
        for item in items:
            if isinstance(item, list):
                res.append(self._serialize_list(item))
            elif hasattr(item, "tolist"): # numpy array conversion
                res.append(item.tolist())
            else:
                res.append(item)
        return res
