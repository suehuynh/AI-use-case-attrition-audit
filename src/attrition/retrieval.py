"""Phase 1 step 2: cross-agency candidate retrieval.

For each dropped 2024 record, return the top-k most similar 2025 records from ANY agency.
These candidates go to the LLM adjudicator next, which decides if a candidate is a true match.

Design notes
- Retrieval favors recall. The adjudicator can reject a bad candidate but can never
  recover one that retrieval left out.
- Agency is a score bonus, not a filter, so cross-agency moves (e.g. USAID work folded
  into State) can still reach the top-k.
- Search against ALL 2025 records, including ones the baseline already matched.
  Otherwise consolidation (several 2024 records -> one 2025 record) is invisible.

Score for a (2024 record a, 2025 record b) pair:
    score = w_tfidf * tfidf_cos(a, b) + w_fuzzy * fuzzy_name(a, b) + w_agency * same_agency(a, b)
All three components are on a 0-1 scale. Weights come from config.yaml -> retrieval.weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import Config


class CandidateRetriever:
    def __init__(self, config: Config, agency_weight: float | None = None):
        p = config.retrieval
        self.top_k: int = p["top_k"]
        self.shortlist_k: int = p.get("shortlist_k", 50)
        self.w_tfidf: float = p["weights"]["tfidf"]
        self.w_fuzzy: float = p["weights"]["fuzzy_name"]
        # Optional override, used by the agency-weight sensitivity check
        self.w_agency: float = p["weights"]["agency"] if agency_weight is None else agency_weight
        self.tfidf_params: dict = dict(p["tfidf"])
        self.tfidf_params["ngram_range"] = tuple(self.tfidf_params["ngram_range"])
        self.vectorizer: TfidfVectorizer | None = None

    # ---- text ---------------------------------------------------------
    @staticmethod
    def build_text(df: pd.DataFrame) -> pd.Series:
        """Name twice (upweights it relative to the longer purpose text) plus purpose."""
        name = df["name_key"].fillna("")
        purpose = df["purpose"].fillna("").str.lower()
        return (name + " " + name + " " + purpose).str.strip()

    def fit(self, prev: pd.DataFrame, nxt: pd.DataFrame) -> "CandidateRetriever":
        """Fit one TF-IDF vocabulary on both years so they share a vector space."""
        t_prev, t_next = self.build_text(prev), self.build_text(nxt)
        self.vectorizer = TfidfVectorizer(**self.tfidf_params)
        self.vectorizer.fit(pd.concat([t_prev, t_next]))
        self._X_prev = self.vectorizer.transform(t_prev)
        self._X_next = self.vectorizer.transform(t_next)
        return self

    # ---- scoring ------------------------------------------------------
    def tfidf_scores(self) -> np.ndarray:
        """Cosine similarity, shape (n_prev, n_next). TF-IDF rows are L2-normalized."""
        return (self._X_prev @ self._X_next.T).toarray()

    @staticmethod
    def fuzzy_name_score(a: str, b: str) -> float:
        return fuzz.token_set_ratio(a, b) / 100.0

    @staticmethod
    def agency_score(agency_a: str, agency_b: str) -> float:
        return 1.0 if agency_a == agency_b else 0.0

    # ---- public -------------------------------------------------------
    def retrieve(self, prev: pd.DataFrame, nxt: pd.DataFrame) -> pd.DataFrame:
        """Top-k candidates per prev record.

        Shortlist per row = top `shortlist_k` by TF-IDF  UNION  top `shortlist_k` by fuzzy
        name score. Fuzzy is computed for all pairs with rapidfuzz.process.cdist (fast C).
        Full weighted score is then computed on the shortlist only.
        """
        from rapidfuzz.process import cdist

        self.fit(prev, nxt)
        tfidf = self.tfidf_scores()
        fuzzy = cdist(prev["name_key"].tolist(), nxt["name_key"].tolist(),
                      scorer=fuzz.token_set_ratio, workers=-1) / 100.0
        same = (prev["agency"].to_numpy()[:, None] == nxt["agency"].to_numpy()[None, :])

        k = min(self.shortlist_k, nxt.shape[0])
        rows = []
        prev_uid, next_uid = prev["uid"].to_numpy(), nxt["uid"].to_numpy()
        for i in range(len(prev)):
            short = np.union1d(np.argpartition(-tfidf[i], k - 1)[:k],
                               np.argpartition(-fuzzy[i], k - 1)[:k])
            score = (self.w_tfidf * tfidf[i, short] + self.w_fuzzy * fuzzy[i, short]
                     + self.w_agency * same[i, short])
            order = np.argsort(-score, kind="stable")[: self.top_k]
            for rank, j in enumerate(order, start=1):
                col = short[j]
                rows.append((prev_uid[i], next_uid[col], rank, float(score[j]),
                             float(tfidf[i, col]), float(fuzzy[i, col]), bool(same[i, col])))
        return pd.DataFrame(rows, columns=["uid_prev", "uid_next", "rank", "score",
                                           "tfidf", "fuzzy_name", "same_agency"])

    # ---- diagnostics --------------------------------------------------
    @staticmethod
    def cross_agency_share(candidates: pd.DataFrame) -> float:
        return float((~candidates["same_agency"].astype(bool)).mean())