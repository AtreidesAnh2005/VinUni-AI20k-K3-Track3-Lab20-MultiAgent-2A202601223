"""Offline research corpus loader and retriever for ai_agent_offline_research_corpus_v2."""

import json
import logging
from pathlib import Path
from typing import Any

from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)

# Default relative and absolute paths to the dataset
_POSSIBLE_CORPUS_PATHS = [
    Path("ai_agent_offline_research_corpus_v2"),
    Path(__file__).resolve().parent.parent.parent.parent / "ai_agent_offline_research_corpus_v2",
    Path("D:/vin/lab20/VinUni-AI20k-K3-Track3-Lab20-2A202601027/ai_agent_offline_research_corpus_v2"),
]


class OfflineCorpusManager:
    """Manages indexing, retrieval, and topic lookups over ai_agent_offline_research_corpus_v2."""

    def __init__(self, corpus_dir: Path | str | None = None) -> None:
        self.corpus_dir = self._resolve_corpus_dir(corpus_dir)
        self._topics_cache: list[dict[str, Any]] = []
        self._indexed_documents: list[dict[str, Any]] = []
        self._loaded = False

    def _resolve_corpus_dir(self, explicit_path: Path | str | None) -> Path | None:
        if explicit_path:
            p = Path(explicit_path)
            if p.exists() and p.is_dir():
                return p

        for candidate in _POSSIBLE_CORPUS_PATHS:
            if candidate.exists() and candidate.is_dir():
                return candidate
        return None

    def load(self) -> None:
        """Load and index topics from the corpus directory."""
        if self._loaded:
            return

        if not self.corpus_dir:
            logger.warning("ai_agent_offline_research_corpus_v2 directory not found.")
            return

        topics_dir = self.corpus_dir / "topics"
        if not topics_dir.exists():
            logger.warning(f"Topics folder not found at {topics_dir}")
            return

        json_files = sorted(topics_dir.glob("*.json"))
        for f in json_files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                meta = data.get("benchmark_metadata", {})
                topic_info = data.get("topic", {})
                kb = data.get("knowledge_base", {})

                topic_entry = {
                    "topic_id": meta.get("topic_id", f.stem),
                    "topic_number": meta.get("topic_number", 0),
                    "title": topic_info.get("name", f.stem),
                    "research_question": topic_info.get("research_question", ""),
                    "tags": topic_info.get("tags", []),
                    "file_path": str(f),
                    "raw_data": data,
                }
                self._topics_cache.append(topic_entry)

                # Index knowledge articles
                for art in kb.get("knowledge_articles", []):
                    self._indexed_documents.append(
                        {
                            "topic_id": topic_entry["topic_id"],
                            "topic_name": topic_entry["title"],
                            "doc_id": art.get("article_id", "A01"),
                            "title": f"[{art.get('article_id')}] {art.get('title')}",
                            "content": art.get("content", ""),
                            "url": f"corpus://{topic_entry['topic_id']}/article/{art.get('article_id')}",
                            "type": "knowledge_article",
                        }
                    )

                # Index source documents
                for src in kb.get("source_documents", []):
                    self._indexed_documents.append(
                        {
                            "topic_id": topic_entry["topic_id"],
                            "topic_name": topic_entry["title"],
                            "doc_id": src.get("source_id", "S01"),
                            "title": f"[{src.get('source_id')}] {src.get('title')}",
                            "content": src.get("summary") or src.get("snippet") or "",
                            "url": src.get("url") or f"corpus://{topic_entry['topic_id']}/source/{src.get('source_id')}",
                            "type": "source_document",
                        }
                    )

                # Index fact bank entries
                for fact in kb.get("fact_bank", []):
                    self._indexed_documents.append(
                        {
                            "topic_id": topic_entry["topic_id"],
                            "topic_name": topic_entry["title"],
                            "doc_id": fact.get("fact_id", "F01"),
                            "title": f"Fact [{fact.get('fact_id')}]: {topic_entry['title']}",
                            "content": fact.get("statement", ""),
                            "url": f"corpus://{topic_entry['topic_id']}/fact/{fact.get('fact_id')}",
                            "type": "fact",
                        }
                    )

            except Exception as exc:
                logger.warning(f"Failed to load corpus topic {f.name}: {exc}")

        self._loaded = True
        logger.info(
            f"Loaded {len(self._topics_cache)} topics and {len(self._indexed_documents)} "
            "corpus items from ai_agent_offline_research_corpus_v2."
        )

    def list_topics(self) -> list[dict[str, Any]]:
        """Return list of all 30 available topics."""
        self.load()
        return [
            {
                "topic_id": t["topic_id"],
                "topic_number": t["topic_number"],
                "title": t["title"],
                "research_question": t["research_question"],
                "tags": t["tags"],
            }
            for t in self._topics_cache
        ]

    def get_topic_by_id_or_number(self, identifier: str | int) -> dict[str, Any] | None:
        """Find a specific topic by number, ID, or title keyword."""
        self.load()
        id_str = str(identifier).strip().lower()

        for t in self._topics_cache:
            if str(t["topic_number"]) == id_str:
                return t
            if t["topic_id"].lower() == id_str:
                return t
            if id_str in t["title"].lower():
                return t

        return None

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search across indexed corpus documents using keyword scoring."""
        self.load()
        if not self._indexed_documents:
            return []

        query_terms = set(query.lower().split())
        scored: list[tuple[float, dict[str, Any]]] = []

        for doc in self._indexed_documents:
            text = f"{doc['topic_name']} {doc['title']} {doc['content']}".lower()
            score = 0.0

            # Match exact terms
            for term in query_terms:
                if len(term) < 3:
                    continue
                if term in text:
                    score += 2.0
                if term in doc["title"].lower():
                    score += 5.0

            if score > 0:
                scored.append((score, doc))

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[SourceDocument] = []
        for _, doc in scored[:max_results]:
            snippet = doc["content"]
            if len(snippet) > 400:
                snippet = snippet[:400] + "..."

            results.append(
                SourceDocument(
                    title=doc["title"],
                    url=doc["url"],
                    snippet=snippet,
                    metadata={
                        "topic_id": doc["topic_id"],
                        "type": doc["type"],
                        "doc_id": doc["doc_id"],
                        "source": "offline_corpus_v2",
                    },
                )
            )

        return results


# Global singleton instance
_corpus_manager_instance: OfflineCorpusManager | None = None


def get_corpus_manager() -> OfflineCorpusManager:
    """Get or create singleton instance of OfflineCorpusManager."""
    global _corpus_manager_instance
    if _corpus_manager_instance is None:
        _corpus_manager_instance = OfflineCorpusManager()
        _corpus_manager_instance.load()
    return _corpus_manager_instance
