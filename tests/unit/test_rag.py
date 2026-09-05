"""Tests for hybrid retrieval.

The property that matters is not "does it return something" — RRF always returns
something. It is that the two retrievers cover each other's blind spots: exact
tokens the semantic side cannot see, and paraphrase the lexical side cannot see.
Both are tested directly, because a hybrid that has silently collapsed to one
retriever still looks like it works.
"""

from __future__ import annotations

import pytest

from tyremind.rag.index import (
    HybridIndex,
    Passage,
    chunk_markdown,
    chunk_result,
    clean_markdown_line,
    tokenize,
)


def _corpus() -> list[Passage]:
    return [
        Passage(
            "TyreMind recovers a known degradation rate with error 0.0044 s/lap against "
            "the naive method's 0.0966, a reduction of 95.5 percent.",
            "results.json",
            "recovery",
            "result",
        ),
        Passage(
            "The model does not measure tread depth. Public telemetry contains no "
            "physical wear measurement of any kind.",
            "model_card.md",
            "limitations",
            "doc",
        ),
        Passage(
            "How confident the estimate is depends on interval calibration, which is "
            "checked by asking how often the truth falls inside the stated range.",
            "model_card.md",
            "uncertainty",
            "doc",
        ),
        Passage(
            "Race strategy is simulated by playing the remaining laps out thousands of "
            "times and comparing the distribution of finishing times.",
            "strategy.md",
            "simulation",
            "doc",
        ),
        Passage(
            "Turbofan engines from the C-MAPSS benchmark provide run-to-failure ground "
            "truth that motorsport data cannot supply.",
            "cross_domain.md",
            "transfer",
            "doc",
        ),
    ]


@pytest.fixture(scope="module")
def index() -> HybridIndex:
    return HybridIndex(_corpus(), n_components=4)


class TestTokenizer:
    def test_keeps_numbers_and_identifiers_whole(self) -> None:
        """Users search for exactly these. Splitting them makes the most
        specific queries the least answerable."""
        tokens = tokenize("error 0.0044 s/lap from exp01_ground_truth_recovery")
        assert "0.0044" in tokens
        assert "exp01_ground_truth_recovery" in tokens

    def test_stoplist_keeps_negations(self) -> None:
        """Dropping 'not' would make a limitation retrieve as its opposite."""
        assert "not" in tokenize("the model does not measure tread depth")


class TestRetrieval:
    def test_exact_number_is_found_lexically(self, index: HybridIndex) -> None:
        """The lexical half's job. A dense embedding cannot represent a rare
        literal like this."""
        hits = index.search("0.0044", k=2)
        assert hits
        assert "0.0044" in hits[0].passage.text

    def test_paraphrase_is_found_semantically(self, index: HybridIndex) -> None:
        """The semantic half's job: no word of the query appears in the target."""
        hits = index.search("how sure is the model about its answers", k=3)
        assert any("calibration" in h.passage.text for h in hits)

    def test_both_retrievers_rank_the_corpus_differently(self, index: HybridIndex) -> None:
        """Guards against a hybrid that has silently collapsed to one retriever.

        Compares the two rankings directly rather than through `search`, because
        the relevance floor can legitimately leave a single hit — and one hit
        cannot show a difference in ordering even when both retrievers work.

        Reaching into the internals is the point here: this asserts the two
        halves are genuinely independent, which the public API deliberately hides.
        """
        query = "how sure is the model about tread wear"
        lexical = index._bm25(tokenize(query))
        semantic = index._semantic(query)

        import numpy as np

        assert list(np.argsort(-lexical)) != list(np.argsort(-semantic)), (
            "lexical and semantic retrievers ranked the corpus identically, so one "
            "of them is not contributing"
        )

    def test_search_reports_which_retriever_found_each_hit(self, index: HybridIndex) -> None:
        """Provenance is shown in the UI, so it has to be populated."""
        hits = index.search("0.0044", k=2)
        assert hits
        assert all(h.lexical_rank is not None and h.semantic_rank is not None for h in hits)

    def test_irrelevant_query_returns_nothing(self, index: HybridIndex) -> None:
        """RRF always ranks everything, so without a floor the worst passage in
        the corpus comes back with a confident citation attached."""
        assert index.search("quantum entanglement in sourdough starters", k=3) == []

    def test_empty_query_is_rejected(self, index: HybridIndex) -> None:
        with pytest.raises(ValueError, match="empty"):
            index.search("   ")

    def test_empty_corpus_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty corpus"):
            HybridIndex([])


class TestMarkdownCleaning:
    def test_link_keeps_its_text(self) -> None:
        """A regression guard: an earlier version dropped the label and left
        readers with a passage full of holes."""
        assert clean_markdown_line("See [the model card](docs/model_card.md) now.") == (
            "See the model card now."
        )

    def test_table_row_becomes_readable(self) -> None:
        assert clean_markdown_line("| Fuel | 0.081 s/lap |") == "Fuel · 0.081 s/lap"

    def test_emphasis_is_stripped(self) -> None:
        assert clean_markdown_line("**bold** and `code`") == "bold and code"

    def test_code_blocks_are_skipped(self) -> None:
        """Command listings retrieve on incidental tokens and read as noise."""
        text = (
            "# Heading\n\n"
            "This sentence is real prose and should survive the chunker intact.\n\n"
            "```bash\npython -m tyremind.serve --port 8077 --no-browser\n```\n"
        )
        passages = chunk_markdown(text, "x.md")
        assert passages
        assert all("uvicorn" not in p.text and "--no-browser" not in p.text for p in passages)

    def test_heading_is_attached_for_citation(self) -> None:
        text = "## Known limitations\n\nThe model does not measure physical tread depth at all.\n"
        passages = chunk_markdown(text, "x.md")
        assert passages[0].heading == "Known limitations"
        assert passages[0].citation.startswith("x.md")


class TestResultChunking:
    def test_numbers_become_searchable_sentences(self) -> None:
        """Raw JSON retrieves badly: nobody queries `{"ssm_mae": 0.0044}`."""
        payload = {
            "experiment": "exp01",
            "summary": {
                "overall": {
                    "ssm_mae": 0.0044,
                    "naive_mae": 0.0966,
                    "error_reduction_pct": 95.5,
                    "interval_coverage_95": 1.0,
                    "naive_bias": -0.0966,
                }
            },
        }
        passages = chunk_result(payload, "exp01.json")
        assert passages
        assert "0.0044" in passages[0].text
        assert passages[0].kind == "result"

    def test_unknown_shape_yields_nothing_rather_than_crashing(self) -> None:
        assert chunk_result({"something": "unexpected"}, "x.json") == []
