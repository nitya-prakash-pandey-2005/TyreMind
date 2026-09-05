"""Turning model output into sentences, without letting the words invent numbers.

Every explanation in TyreMind is generated from a template that reads structured
model output. The templates are deterministic, work offline, and cannot produce a
number the model did not compute.

An LLM is optional and strictly a *rewriter*. It receives the already-generated
sentences plus the structured facts and may improve the prose. It never sees a
question it could answer from its own knowledge, it is never asked to compute
anything, and its output is checked: any numeric token in the rewrite that does
not appear in the source facts causes the rewrite to be discarded and the
template text used instead.

That check is the whole design. A language model asked to explain a degradation
rate will produce a fluent, plausible, subtly wrong number, and a fluent wrong
number on a pit wall is worse than no explanation at all.
"""

from __future__ import annotations

import contextlib
import os
import re
from dataclasses import dataclass, field

#: Numbers in the generated text must trace back to a computed fact. Matches
#: integers, decimals, and percentages.
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")


@dataclass
class Narration:
    """An explanation and its provenance.

    Attributes:
        text: The prose shown to the user.
        source: "template" or "llm". Always surfaced, so a reader knows whether a
            language model touched the words.
        facts: The structured values the text was generated from.
        rejected_reason: Why an LLM rewrite was discarded, if one was.
    """

    text: str
    source: str
    facts: dict = field(default_factory=dict)
    rejected_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "facts": self.facts,
            "rejected_reason": self.rejected_reason,
        }


def _numbers_in(text: str) -> set[str]:
    """Numeric tokens appearing in a string, normalised for comparison."""
    return {m.group().replace(",", ".").rstrip(".").lstrip("-") for m in _NUMBER.finditer(text)}


def verify_rewrite(rewrite: str, allowed: str) -> tuple[bool, str | None]:
    """Check that a rewrite introduced no numbers the source did not contain.

    The guard against the failure mode that matters. A language model rewriting
    "degradation is 0.113 s/lap" will occasionally produce "roughly 0.11" or
    "about a tenth of a second" -- the first is fine, the second is fine, and
    "0.15" is a fabricated measurement presented with total confidence.

    Rounded forms are accepted, because "0.11" from "0.113" is a legitimate
    rewrite rather than an invention. Anything else is rejected.

    Args:
        rewrite: Candidate text from the language model.
        allowed: The source text whose numbers are permitted.

    Returns:
        ``(ok, reason)``. `reason` is None when the rewrite is acceptable.
    """
    source = _numbers_in(allowed)
    for token in _numbers_in(rewrite):
        if token in source:
            continue
        # Accept a rounding of something in the source.
        if any(
            original.startswith(token) or token.startswith(original[: len(token)])
            for original in source
            if original
        ):
            continue
        return False, f"introduced the number {token!r}, which is not in the model output"
    return True, None


def decomposition_narration(decomposition) -> Narration:
    """Explain a lap-time decomposition in plain language.

    Written for someone who has never seen a degradation curve. The specific
    thing it has to get across is counter-intuitive: a car can get *faster* while
    its tyre gets *worse*, because fuel burn-off is usually larger than
    degradation. Read the stopwatch alone and you miss a dying tyre.
    """
    tyre = decomposition.tyre_seconds
    observed = decomposition.observed_delta
    biggest = max(decomposition.contributions, key=lambda c: abs(c.seconds))

    facts = {
        "driver": decomposition.driver,
        "lap": decomposition.session_lap,
        "reference_lap": decomposition.reference_lap,
        "observed_delta": round(observed, 3),
        "tyre_seconds": round(tyre, 3),
        "compound": decomposition.compound,
        "tyre_age": decomposition.tyre_age,
        "largest_effect": biggest.label,
        "largest_effect_seconds": round(biggest.seconds, 3),
    }

    if observed < -0.05 and tyre > 0.05:
        text = (
            f"Car {decomposition.driver} was {abs(observed):.2f} seconds FASTER on lap "
            f"{decomposition.session_lap} than on lap {decomposition.reference_lap}, so a "
            f"stopwatch says the tyre is fine. It is not. Strip out the things that "
            f"changed around the car and the {decomposition.compound.lower()} tyre has "
            f"actually lost {tyre:.2f} seconds over those laps. The car got faster "
            f"because it burned off fuel, and the fuel gain was bigger than the tyre "
            f"loss. Watching lap times alone would miss a degrading tyre completely."
        )
    elif observed > 0.05:
        share = tyre / observed if observed else 0.0
        text = (
            f"Car {decomposition.driver} lost {observed:.2f} seconds between lap "
            f"{decomposition.reference_lap} and lap {decomposition.session_lap}. Of that, "
            f"{tyre:.2f} seconds is the tyre and the rest comes from conditions around "
            f"the car. The largest single effect was {biggest.label.lower()}, worth "
            f"{abs(biggest.seconds):.2f} seconds. So {share:.0%} of the slowdown is "
            f"the tyre, and pitting would only recover that part."
        )
    else:
        text = (
            f"Car {decomposition.driver} matched its lap {decomposition.reference_lap} "
            f"time, but the underlying causes did not cancel by accident. The tyre "
            f"contributed {tyre:.2f} seconds and everything else "
            f"{decomposition.confounder_seconds:.2f} seconds in the other direction."
        )

    return Narration(text=text, source="template", facts=facts)


def projection_narration(projection) -> Narration:
    """Explain remaining competitive tyre life in plain language."""
    lower, upper = projection.competitive_life_interval()
    facts = {
        "driver": projection.driver,
        "tyre_age": projection.tyre_age,
        "compound": projection.compound,
        "threshold_s": projection.threshold_s if hasattr(projection, "threshold_s") else projection.threshold,
        "expected_life": round(projection.competitive_life(), 1),
        "lower": round(lower, 1),
        "upper": round(upper, 1),
        "applicability": round(float(projection.applicability[-1]), 2),
    }

    threshold = facts["threshold_s"]
    text = (
        f"The {projection.compound.lower()} set on car {projection.driver} has done "
        f"{projection.tyre_age:.0f} laps. Going on how it has degraded so far, it has "
        f"roughly {projection.competitive_life():.0f} more competitive laps in it -- "
        f"somewhere between {lower:.0f} and {upper:.0f}. \"Competitive\" here means "
        f"before it is losing more than {threshold} seconds a lap against a fresh set, "
        f"which is a threshold you can change to suit the situation."
    )

    if float(projection.applicability[-1]) < 0.5:
        text += (
            " Be careful with the far end of that range: it runs past the oldest tyre "
            "this session actually contains, so the model is extrapolating a trend "
            "rather than reporting one."
        )

    return Narration(text=text, source="template", facts=facts)


def strategy_narration(recommendation) -> Narration:
    """Explain a pit recommendation in plain language."""
    facts = {
        "recommended": recommendation.best.strategy.label,
        "margin_s": round(recommendation.margin_s, 2),
        "confidence": round(recommendation.decision_confidence, 2),
        "n_sims": recommendation.best.n_sims,
    }
    text = " ".join(recommendation.reasons)
    return Narration(text=text, source="template", facts=facts)


def rewrite_with_llm(narration: Narration, *, model: str = "gemini-2.0-flash") -> Narration:
    """Optionally improve the prose, keeping the numbers under guard.

    Returns the original narration unchanged if no API key is configured, if the
    call fails, or if the rewrite introduces a number the model did not compute.
    The offline path is the default and is never worse than a missing
    explanation -- it is the same explanation, in slightly plainer prose.

    Args:
        narration: Template-generated narration to improve.
        model: Gemini model name.

    Returns:
        A Narration. `source` says whether the language model was used.
    """
    # Load .env if present. Without this the documented workflow -- copy
    # .env.example, paste a key -- silently does nothing, and narration falls
    # back to templates with no indication that the key was never read.
    with contextlib.suppress(ImportError):
        from dotenv import load_dotenv

        load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return narration

    prompt = (
        "Rewrite the following race-engineering explanation to be clearer for "
        "someone who does not follow motorsport. Keep every number exactly as "
        "given. Do not add any number, figure, or statistic that is not already "
        "present. Do not add caveats or claims of your own. Two to four "
        "sentences.\n\n"
        f"Facts (authoritative): {narration.facts}\n\n"
        f"Text to rewrite:\n{narration.text}"
    )

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model, contents=prompt)
        candidate = (response.text or "").strip()
    except Exception as exc:  # noqa: BLE001 - narration must never break a page
        return Narration(
            text=narration.text,
            source="template",
            facts=narration.facts,
            rejected_reason=f"language model unavailable ({type(exc).__name__})",
        )

    if not candidate:
        return narration

    ok, reason = verify_rewrite(candidate, narration.text + " " + str(narration.facts))
    if not ok:
        return Narration(
            text=narration.text,
            source="template",
            facts=narration.facts,
            rejected_reason=f"rewrite rejected: {reason}",
        )

    return Narration(text=candidate, source="llm", facts=narration.facts)
