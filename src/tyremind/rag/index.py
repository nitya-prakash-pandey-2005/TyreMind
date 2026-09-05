"""Hybrid retrieval over the project's own documentation and results.

The question "why should I believe this?" has answers scattered across a
research audit, a model card, a limitations register and seven experiment result
files. Nobody reads all of that. This makes it answerable in one line, with the
source cited so the answer can be checked rather than trusted.

**Hybrid retrieval, fused with Reciprocal Rank Fusion.** Two retrievers with
complementary failure modes:

  * **BM25** (lexical) nails exact terms -- "CRPS", "C-MAPSS", "0.0044" -- and
    misses paraphrase entirely.
  * **Latent semantic** (TF-IDF then truncated SVD) catches paraphrase -- "how
    sure is it" finding a passage about calibration -- and is weak on rare exact
    tokens.

RRF merges the two ranked lists without needing either to be calibrated against
the other, which is what makes it the standard choice.

**Why LSA and not a transformer encoder.** A sentence-transformer would retrieve
better. It also needs torch, which is roughly two gigabytes, in a product whose
central promise is that the demo runs with the network unplugged. TF-IDF plus SVD
is genuinely a dense semantic embedding -- just a weaker one -- and it ships in
scikit-learn, which is already a dependency. `EMBEDDING_BACKEND` documents the
swap for anyone who wants it.

Nothing here generates text. Retrieval returns passages and their sources; any
narration on top of them goes through `explain/narrate.py`, which cannot invent
a number.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: What a stronger deployment would swap in. Documented rather than hidden so
#: the trade-off is visible: better retrieval, at the cost of a ~2 GB dependency
#: and an offline demo that no longer works on a fresh clone.
EMBEDDING_BACKEND = "tfidf+svd (offline). Swap for sentence-transformers if torch is acceptable."

#: Sources indexed, in the order they are searched. Experiment results are
#: included so a question about a number reaches the JSON that produced it.
DEFAULT_SOURCES = (
    Path("docs"),
    Path("README.md"),
    Path("experiments/results"),
)

_TOKEN = re.compile(r"[a-z0-9][a-z0-9._-]*")

#: Words that carry no retrieval signal. Deliberately short -- an aggressive
#: stoplist removes terms like "not", which inverts the meaning of a limitation.
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this",
    "to", "was", "were", "will", "with", "we", "our", "you", "your",
})


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, keeping numbers and dotted identifiers intact.

    `0.0044` and `exp01_ground_truth_recovery` have to survive as single tokens:
    they are exactly the terms a user searches for, and splitting them would make
    the most specific queries the least answerable.
    """
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


@dataclass
class Passage:
    """One retrievable chunk.

    Attributes:
        text: The chunk itself.
        source: File it came from.
        heading: Nearest enclosing heading, for citation.
        kind: "doc" or "result".
    """

    text: str
    source: str
    heading: str
    kind: str

    @property
    def citation(self) -> str:
        return f"{self.source}{' — ' + self.heading if self.heading else ''}"


@dataclass
class Hit:
    """A retrieved passage with its score and provenance."""

    passage: Passage
    score: float
    lexical_rank: int | None = None
    semantic_rank: int | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.passage.text,
            "source": self.passage.source,
            "heading": self.passage.heading,
            "kind": self.passage.kind,
            "citation": self.passage.citation,
            "score": self.score,
            "lexical_rank": self.lexical_rank,
            "semantic_rank": self.semantic_rank,
        }


#: Markdown that carries no meaning once the text is shown as prose. A table
#: separator or a bare link target retrieves as well as anything else and then
#: reads as line noise in the result panel.
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_EMPHASIS = re.compile(r"[*_`]{1,3}")
_TABLE_SEPARATOR = re.compile(r"^\|?[\s:|-]+\|?$")


def clean_markdown_line(line: str) -> str:
    """Strip markup so a retrieved passage reads as a sentence.

    Links become their text, emphasis markers go, and table pipes become
    separators. The alternative is showing a reader `| | |---|---| | [Research
    audit](docs/...)`, which is technically the right passage and useless.
    """
    text = _MD_LINK.sub(lambda m: m.group(1), line)
    text = _MD_EMPHASIS.sub("", text)
    if text.lstrip().startswith("|"):
        cells = [c.strip() for c in text.strip().strip("|").split("|")]
        text = " · ".join(c for c in cells if c)
    return text.strip()


def chunk_markdown(text: str, source: str, *, target_words: int = 130) -> list[Passage]:
    """Split a document on headings, then on size, keeping the heading attached.

    Splitting on headings rather than a fixed window matters here: these
    documents are structured by argument, and a chunk that straddles the boundary
    between "what we claim" and "what we do not claim" is worse than useless.
    """
    passages: list[Passage] = []
    heading = ""
    buffer: list[str] = []
    in_code = False

    def flush() -> None:
        if not buffer:
            return
        body = " ".join(buffer).strip()
        # Eight words, not twelve: the shortest sentences in these documents
        # are often the most important ones ("It does not measure tread
        # depth"), and a higher floor would drop exactly those.
        if len(body.split()) >= 8:
            passages.append(Passage(body, source, heading, "doc"))
        buffer.clear()

    for raw_line in text.splitlines():
        stripped = raw_line.strip()

        # Code blocks are commands and file trees. They retrieve on incidental
        # tokens and read as noise, so they are skipped entirely.
        if stripped.startswith("```"):
            in_code = not in_code
            flush()
            continue
        if in_code:
            continue

        if stripped.startswith("#"):
            flush()
            heading = clean_markdown_line(stripped.lstrip("#"))
            continue
        if not stripped or _TABLE_SEPARATOR.match(stripped):
            if sum(len(b.split()) for b in buffer) >= target_words:
                flush()
            continue

        cleaned = clean_markdown_line(stripped)
        if not cleaned:
            continue

        buffer.append(cleaned)
        if sum(len(b.split()) for b in buffer) >= target_words * 1.6:
            flush()

    flush()
    return passages


def chunk_result(payload: dict, source: str) -> list[Passage]:
    """Turn an experiment result file into readable, retrievable statements.

    Raw JSON retrieves badly -- a query for "how accurate" will not match
    `{"ssm_mae": 0.0044}`. Rendering each finding as a sentence makes the numbers
    reachable by the language people actually use to ask for them.
    """
    name = payload.get("experiment", source)
    passages: list[Passage] = []

    def add(text: str, heading: str) -> None:
        passages.append(Passage(text, source, heading, "result"))

    summary = payload.get("summary", {}).get("overall")
    if summary:
        add(
            f"Experiment {name}: TyreMind recovers a known degradation rate with mean "
            f"absolute error {summary['ssm_mae']:.4f} s/lap against the naive method's "
            f"{summary['naive_mae']:.4f}, an error reduction of "
            f"{summary['error_reduction_pct']:.1f}%. Interval coverage "
            f"{summary['interval_coverage_95']:.0%}. Naive bias is "
            f"{summary['naive_bias']:+.4f} s/lap, which matches the fuel burn-off slope.",
            "ground truth recovery",
        )

    overall = payload.get("overall")
    if overall and "mae" in overall:
        add(
            f"Experiment {name}: practice-to-race transfer over {overall.get('n_events')} "
            f"events and {overall.get('n_comparisons')} compound comparisons. MAE "
            f"{overall['mae']:.4f} s/lap versus naive {overall.get('naive_mae')}. "
            f"Coverage {overall.get('coverage_95', 0):.0%}. Systematic bias "
            f"{overall.get('bias', 0):+.4f} s/lap -- practice over-predicts race degradation.",
            "practice to race",
        )

    for row in payload.get("lap_time_prediction", []) or []:
        add(
            f"Model ladder, lap-time prediction: {row['model']} scores CRPS "
            f"{row['crps']:.4f}, MAE {row['mae']:.4f}, interval coverage "
            f"{row['coverage_95']:.0%}, bias drift {row['bias_drift']:+.3f}.",
            "model ladder",
        )
    for row in payload.get("degradation_recovery", []) or []:
        add(
            f"Model ladder, degradation recovery: {row['model']} has rate error "
            f"{row['rate_mae']:.4f} s/lap, bias {row['rate_bias']:+.4f}, "
            f"coverage {row['coverage']:.0%}.",
            "model ladder",
        )

    for circuit in payload.get("circuits", []) or []:
        add(
            f"Circuit geometry recovery: {circuit['circuit']} is published as "
            f"{circuit['published_direction']}; the physics layer inferred "
            f"{circuit['predicted_direction']} from GPS traces alone. Left-side energy "
            f"share {circuit['left_side_energy_share']:.1%}, peak lateral "
            f"{circuit['peak_lateral_g']:.1f} g.",
            "circuit asymmetry",
        )

    if "rul_rmse" in payload:
        add(
            f"Cross-domain transfer to {payload.get('dataset')}: remaining-useful-life "
            f"RMSE {payload['rul_rmse']:.1f} cycles over {payload.get('n_engines_scored')} "
            f"engines, {payload.get('fraction_early', 0):.0%} predicted early. "
            f"{payload.get('note', '')}",
            "cross-domain",
        )

    for row in payload.get("synthetic_summary", []) or []:
        add(
            f"Prior sensitivity, variant '{row['variant']}': rate error {row['mae']:.4f} "
            f"s/lap, bias {row['bias']:+.4f}, posterior sd {row['mean_sd']:.4f}, "
            f"coverage {row['coverage']:.0%}.",
            "prior sensitivity",
        )

    if payload.get("verdict"):
        add(
            f"Experiment {name} verdict: {payload['verdict']}. Mean R-squared gain "
            f"{payload.get('mean_r2_gain', 0):+.4f}; per-lap energy varies by "
            f"{payload.get('mean_energy_cv', 0):.1%} within a stint.",
            "energy clock",
        )

    return passages


class HybridIndex:
    """BM25 + latent-semantic retrieval fused with Reciprocal Rank Fusion.

    Args:
        passages: Corpus to index.
        n_components: SVD dimensionality for the semantic side. Small, because
            the corpus is a few hundred passages -- more components would fit
            noise, not meaning.
    """

    #: BM25 term-frequency saturation. 1.5 is the usual choice.
    K1 = 1.5
    #: BM25 length normalisation.
    B = 0.75
    #: RRF damping. 60 is the value from the original RRF paper and behaves well
    #: without tuning, which is the point of using it.
    RRF_K = 60

    #: Relevance floor. A passage sharing no query term scores exactly 0 on BM25;
    #: one with no semantic overlap has near-zero cosine similarity. A hit must
    #: clear one of the two to be worth citing.
    MIN_LEXICAL = 0.0
    MIN_SEMANTIC = 0.08

    def __init__(self, passages: list[Passage], n_components: int = 96) -> None:
        if not passages:
            raise ValueError("cannot build an index over an empty corpus")

        self.passages = passages
        self._tokens = [tokenize(p.text + " " + p.heading) for p in passages]
        self._lengths = np.array([len(t) for t in self._tokens], dtype=float)
        self._avg_length = float(self._lengths.mean()) or 1.0

        self._term_frequencies = [Counter(t) for t in self._tokens]
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            document_frequency.update(set(tokens))

        n = len(passages)
        self._idf = {
            term: math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

        self._build_semantic(n_components)

    def _build_semantic(self, n_components: int) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        corpus = [" ".join(t) for t in self._tokens]
        self._vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=1)
        tfidf = self._vectorizer.fit_transform(corpus)

        # SVD needs strictly fewer components than features; a small corpus can
        # otherwise ask for more dimensions than exist.
        components = max(2, min(n_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1))
        self._svd = TruncatedSVD(n_components=components, random_state=0)
        self._embeddings = normalize(self._svd.fit_transform(tfidf))

    def _bm25(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.passages))
        for i, frequencies in enumerate(self._term_frequencies):
            length_norm = self.K1 * (1 - self.B + self.B * self._lengths[i] / self._avg_length)
            total = 0.0
            for term in query_tokens:
                tf = frequencies.get(term, 0)
                if tf:
                    total += self._idf.get(term, 0.0) * tf * (self.K1 + 1) / (tf + length_norm)
            scores[i] = total
        return scores

    def _semantic(self, query: str) -> np.ndarray:
        from sklearn.preprocessing import normalize

        vector = self._vectorizer.transform([" ".join(tokenize(query))])
        embedded = normalize(self._svd.transform(vector))
        return (self._embeddings @ embedded.T).ravel()

    def search(self, query: str, k: int = 5) -> list[Hit]:
        """Retrieve the k most relevant passages.

        Args:
            query: Natural-language question.
            k: Passages to return.

        Returns:
            Hits ordered by fused score, each carrying the rank it held in both
            retrievers so a reader can see which one found it.

        Raises:
            ValueError: If the query is empty.
        """
        if not query.strip():
            raise ValueError("query is empty")

        query_tokens = tokenize(query)
        lexical = self._bm25(query_tokens)
        semantic = self._semantic(query)

        # Reciprocal Rank Fusion: score by position in each list, not by raw
        # score. BM25 and cosine similarity are on incomparable scales, and RRF
        # sidesteps that without needing either to be calibrated.
        lexical_order = np.argsort(-lexical)
        semantic_order = np.argsort(-semantic)

        lexical_rank = {int(idx): rank for rank, idx in enumerate(lexical_order)}
        semantic_rank = {int(idx): rank for rank, idx in enumerate(semantic_order)}

        fused = {
            i: 1.0 / (self.RRF_K + lexical_rank[i]) + 1.0 / (self.RRF_K + semantic_rank[i])
            for i in range(len(self.passages))
        }

        # RRF ranks everything, so without a relevance floor the worst passage
        # in the corpus comes back with a confident citation attached. The floor
        # is on the underlying scores rather than on rank: a rank-based cutoff is
        # meaningless on a small corpus, where every passage ranks near the top.
        best = sorted(fused, key=lambda i: -fused[i])[:k]
        return [
            Hit(
                passage=self.passages[i],
                score=float(fused[i]),
                lexical_rank=lexical_rank[i],
                semantic_rank=semantic_rank[i],
            )
            for i in best
            if lexical[i] > self.MIN_LEXICAL or semantic[i] > self.MIN_SEMANTIC
        ]


@dataclass
class Corpus:
    """The indexed corpus and its provenance."""

    index: HybridIndex
    n_passages: int
    sources: list[str] = field(default_factory=list)

    def stats(self) -> dict:
        kinds = Counter(p.kind for p in self.index.passages)
        per_source = Counter(p.source for p in self.index.passages)
        return {
            "n_passages": self.n_passages,
            "n_sources": len(self.sources),
            "sources": self.sources,
            # Passage count per file, so a reader can see what the corpus is
            # actually made of rather than trusting a total. A source that
            # contributes two passages is not evidence of much.
            "by_source": [
                {"source": src, "n_passages": n} for src, n in per_source.most_common()
            ],
            "by_kind": dict(kinds),
            "embedding_backend": EMBEDDING_BACKEND,
            "retrieval": "BM25 + latent semantic, fused with Reciprocal Rank Fusion",
        }


def build_corpus(roots: tuple[Path, ...] = DEFAULT_SOURCES) -> Corpus:
    """Index the project's documentation and recorded results.

    Args:
        roots: Files and directories to index.

    Returns:
        A Corpus ready to search.

    Raises:
        ValueError: If nothing could be indexed.
    """
    passages: list[Passage] = []
    sources: list[str] = []

    for root in roots:
        if root.is_file():
            files = [root]
        elif root.is_dir():
            files = sorted(list(root.rglob("*.md")) + list(root.rglob("*.json")))
        else:
            continue

        for path in files:
            try:
                raw = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            name = path.as_posix()
            if path.suffix == ".json":
                try:
                    found = chunk_result(json.loads(raw), name)
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
            else:
                found = chunk_markdown(raw, name)

            if found:
                passages.extend(found)
                sources.append(name)

    if not passages:
        raise ValueError(f"nothing indexable found under {[str(r) for r in roots]}")

    return Corpus(index=HybridIndex(passages), n_passages=len(passages), sources=sources)
