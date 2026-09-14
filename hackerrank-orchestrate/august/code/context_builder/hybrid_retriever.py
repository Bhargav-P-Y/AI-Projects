import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np
import pandas as pd

from data_pipeline.data_loader import DataBundle
from data_pipeline.profile_builder import UserProfile

logger = logging.getLogger(__name__)


@dataclass
class EvidenceCandidate:
    message_id: str
    score: float
    message_text: str
    created_at: str
    conversation_type: str
    sender_user_id: str
    business_id: str
    group_id: str
    user_action: str  # e.g., "opened", "replied", "dismissed", "reported", "no_event"
    dense_sim: float = 0.0
    bm25_sim: float = 0.0
    recency_score: float = 0.0


class BM25Engine:
    """Pure-Python Okapi BM25 engine for fast in-memory keyword similarity scoring."""

    def __init__(self, corpus_texts: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_texts)
        self.doc_tokens: List[List[str]] = [self._tokenize(t) for t in corpus_texts]
        self.doc_lens: List[int] = [len(toks) for toks in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lens) / max(1, self.corpus_size)

        # Inverted index: term -> list of (doc_index, term_frequency)
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}
        self.df: Dict[str, int] = {}

        for doc_idx, toks in enumerate(self.doc_tokens):
            term_counts: Dict[str, int] = {}
            for t in toks:
                term_counts[t] = term_counts.get(t, 0) + 1

            for term, count in term_counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((doc_idx, count))
                self.df[term] = self.df.get(term, 0) + 1

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        if not text:
            return []
        return re.findall(r"\w+", text.lower())

    def get_scores(self, query_text: str) -> List[float]:
        query_tokens = self._tokenize(query_text)
        if not query_tokens or self.corpus_size == 0:
            return [0.0] * self.corpus_size

        scores = [0.0] * self.corpus_size
        for term in set(query_tokens):
            if term not in self.inverted_index:
                continue

            # 1. Inverse Document Frequency (IDF Signal)
            n_q = self.df[term]
            idf = math.log((self.corpus_size - n_q + 0.5) / (n_q + 0.5) + 1.0)

            # Iterate only through documents containing this term via inverted index
            for doc_idx, f_q in self.inverted_index[term]:
                # 2. Term Frequency (TF Signal) & 3. Document Length Normalization Signal
                num = f_q * (self.k1 + 1)
                den = f_q + self.k1 * (1 - self.b + self.b * (self.doc_lens[doc_idx] / max(1.0, self.avg_doc_len)))
                scores[doc_idx] += idf * (num / den)

        return scores


class HybridRetriever:
    """Hybrid Evidence Retriever for Message Notification Router.

    Combines:
    1. Dense Semantic Embedding Similarity (gemini-embedding-001)
    2. Okapi BM25 Keyword Similarity
    3. Exponential Recency Decay
    4. User Engagement Reaction Signals (opened, replied, dismissed, reported)
    5. Sender & Conversation Alignments
    """

    def __init__(
        self,
        data_bundle: DataBundle,
        semantic_detector: Optional[Any] = None,
    ):
        self.data_bundle = data_bundle
        self.semantic_detector = semantic_detector

        # Load message history records
        self.history_records: List[dict] = data_bundle.message_history.to_dict("records")
        self.history_texts: List[str] = [r.get("message_text", "") for r in self.history_records]
        self.history_ids: List[str] = [r["message_id"] for r in self.history_records]

        # Map user_id -> list of indices in history_records
        self.user_to_history_indices: Dict[str, List[int]] = {}
        for idx, rec in enumerate(self.history_records):
            uid = rec["user_id"]
            if uid not in self.user_to_history_indices:
                self.user_to_history_indices[uid] = []
            self.user_to_history_indices[uid].append(idx)

        # Build BM25 engine over all history message texts
        self.bm25_engine = BM25Engine(self.history_texts)

        # Map (user_id, message_id) -> message_event dict
        self.events_map: Dict[Tuple[str, str], dict] = {}
        for row in data_bundle.message_events.to_dict("records"):
            self.events_map[(row["user_id"], row["message_id"])] = row

        # Cache pre-computed dense embeddings for all history messages
        self.history_embeddings: Dict[str, Optional[np.ndarray]] = {}
        if self.semantic_detector and hasattr(self.semantic_detector, "get_text_embedding"):
            logger.info("Pre-embedding historical messages for dense retrieval...")
            for rec in self.history_records:
                msg_id = rec["message_id"]
                txt = rec.get("message_text", "")
                if txt.strip():
                    emb = self.semantic_detector.get_text_embedding(txt)
                    if emb is not None:
                        norm = np.linalg.norm(emb)
                        if norm > 0:
                            emb = emb / norm
                        self.history_embeddings[msg_id] = emb

    def retrieve(
        self,
        message: dict,
        user_profile: Optional[UserProfile] = None,
        top_k: int = 3,
        min_score_threshold: float = 0.35,
        media_text: str = "",
    ) -> Tuple[str, List[EvidenceCandidate]]:
        """Retrieves top-k historical evidence message IDs for an incoming message.

        Returns:
            Tuple[evidence_message_ids_str, candidates_list]
            where evidence_message_ids_str is a ';'-separated string of IDs or "none".
        """
        user_id = message["user_id"]
        candidate_indices = self.user_to_history_indices.get(user_id, [])

        if not candidate_indices:
            return "none", []

        # Prepare query text (use text or extracted media text)
        msg_text = message.get("message_text", "")
        query_text = f"{msg_text} {media_text}".strip()

        if not query_text:
            # If no text at all, rank solely by sender/group match and recency
            query_text = "message"

        # 1. BM25 scores for all candidate indices
        bm25_all = self.bm25_engine.get_scores(query_text)
        max_bm25 = max(bm25_all) if bm25_all else 1.0
        if max_bm25 <= 0:
            max_bm25 = 1.0

        # 2. Dense embedding vector for query_text
        query_emb = None
        if self.semantic_detector and hasattr(self.semantic_detector, "get_text_embedding"):
            raw_emb = self.semantic_detector.get_text_embedding(query_text)
            if raw_emb is not None:
                q_norm = np.linalg.norm(raw_emb)
                if q_norm > 0:
                    query_emb = raw_emb / q_norm

        # 3. Parse incoming message created_at
        msg_dt = message.get("created_at_dt")
        if not isinstance(msg_dt, pd.Timestamp) and not isinstance(msg_dt, datetime):
            try:
                msg_dt = pd.to_datetime(message.get("created_at", ""))
            except Exception:
                msg_dt = datetime.now()

        scored_candidates: List[EvidenceCandidate] = []

        for idx in candidate_indices:
            cand = self.history_records[idx]
            cand_id = cand["message_id"]

            # BM25 normalized score
            bm25_sim = bm25_all[idx] / max_bm25

            # Dense similarity
            dense_sim = 0.0
            if query_emb is not None:
                c_emb = self.history_embeddings.get(cand_id)
                if c_emb is not None:
                    dense_sim = float(np.dot(query_emb, c_emb))
                    dense_sim = max(0.0, min(1.0, dense_sim))

            # Recency decay (exponential decay over 60 days)
            cand_dt = cand.get("created_at_dt")
            if not isinstance(cand_dt, pd.Timestamp) and not isinstance(cand_dt, datetime):
                try:
                    cand_dt = pd.to_datetime(cand.get("created_at", ""))
                except Exception:
                    cand_dt = msg_dt

            days_old = max(0.0, (msg_dt - cand_dt).total_seconds() / 86400.0)
            recency_score = max(0.0, 1.0 - (days_old / 60.0))

            # Reaction / Event engagement signal
            event = self.events_map.get((user_id, cand_id))
            user_action = "no_event"
            engagement_bonus = 0.0

            if event:
                if event.get("message_reported"):
                    user_action = "reported"
                    engagement_bonus = 0.20
                elif event.get("message_replied"):
                    user_action = "replied"
                    engagement_bonus = 0.15
                elif event.get("notification_dismissed"):
                    user_action = "dismissed"
                    engagement_bonus = 0.10
                elif event.get("message_opened"):
                    user_action = "opened"
                    engagement_bonus = 0.05
                elif event.get("muted_after_message"):
                    user_action = "muted_group"
                    engagement_bonus = 0.15

            # Alignment bonuses
            alignment_bonus = 0.0
            msg_sender = message.get("sender_user_id", "")
            cand_sender = cand.get("sender_user_id", "")
            msg_biz = message.get("business_id", "")
            cand_biz = cand.get("business_id", "")
            msg_grp = message.get("group_id", "")
            cand_grp = cand.get("group_id", "")

            if msg_biz and cand_biz and msg_biz == cand_biz:
                alignment_bonus += 0.15
            elif msg_sender and cand_sender and msg_sender == cand_sender:
                alignment_bonus += 0.15

            if msg_grp and cand_grp and msg_grp == cand_grp:
                alignment_bonus += 0.10

            # Content match gating: candidate must have text/semantic similarity
            content_sim = max(dense_sim, bm25_sim)
            if content_sim < 0.20 and not (msg_text == "" and cand.get("message_text") == ""):
                composite_score = 0.0
            else:
                composite_score = (
                    0.55 * dense_sim
                    + 0.35 * bm25_sim
                    + 0.05 * recency_score
                    + 0.05 * (1.0 if alignment_bonus > 0 else 0.0)
                    + engagement_bonus
                )

            scored_candidates.append(
                EvidenceCandidate(
                    message_id=cand_id,
                    score=composite_score,
                    message_text=cand.get("message_text", ""),
                    created_at=str(cand.get("created_at", "")),
                    conversation_type=cand.get("conversation_type", ""),
                    sender_user_id=cand_sender,
                    business_id=cand_biz,
                    group_id=cand_grp,
                    user_action=user_action,
                    dense_sim=dense_sim,
                    bm25_sim=bm25_sim,
                    recency_score=recency_score,
                )
            )

        # Sort candidates by composite score descending
        scored_candidates.sort(key=lambda x: x.score, reverse=True)

        if not scored_candidates or scored_candidates[0].score < min_score_threshold:
            return "none", []

        top_score = scored_candidates[0].score
        top_cand = scored_candidates[0]

        # First candidate must have genuine content similarity or strong relationship signal
        if max(top_cand.dense_sim, top_cand.bm25_sim) < 0.25 and top_score < 0.50:
            return "none", []

        filtered: List[EvidenceCandidate] = [top_cand]

        for c in scored_candidates[1:]:
            # Include secondary candidate ONLY if highly content-relevant AND score is within 0.06 of top_score
            if max(c.dense_sim, c.bm25_sim) >= 0.55 and c.score >= (top_score - 0.06):
                filtered.append(c)
                if len(filtered) == top_k:
                    break

        evidence_ids_str = ";".join([c.message_id for c in filtered])
        return evidence_ids_str, filtered
