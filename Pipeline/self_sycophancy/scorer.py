# scorer.py
from inspect_ai.scorer import scorer  # ✅ correct import
from typing import Dict, Any

# ✅ Simple exact-match scorer
@scorer(name="swe_bench_scorer", metrics={"default": ["exact_match"]})
def swe_bench_scorer(output: Dict[str, Any], target: Dict[str, Any]) -> bool:
    """
    Compare PR output (state.output["pr"]) with ground truth target["patch"].
    Returns True if they are exactly equal after stripping whitespace.
    """
    pr_output = output.get("pr", "").strip()
    expected_patch = target.get("patch", "").strip()

    return pr_output == expected_patch
