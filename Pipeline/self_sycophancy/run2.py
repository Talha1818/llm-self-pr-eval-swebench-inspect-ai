#!/usr/bin/env python3
from pathlib import Path
from dotenv import load_dotenv
import json
import re
from inspect_ai import eval
from task import generate_prs_task, self_framing_task, other_framing_task

load_dotenv()

# ---- CONFIG ----
EVALUATION_MODELS = [
    "openrouter/openai/gpt-oss-20b",
    "openrouter/openai/gpt-oss-120b",
]
LIMIT_N = 3
LOG_DIR = Path(__file__).parent / "logs"
RESULTS_DIR = Path(__file__).parent / "results"

LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_integer_grade(sample_output):
    if sample_output is None:
        return None

    grade = getattr(sample_output, "grade", None)
    if grade is not None:
        try:
            val = int(float(grade))
            if 0 <= val <= 10:
                return val
        except:
            pass

    reasoning_text = getattr(sample_output, "reasoning", None)
    if reasoning_text:
        match = re.search(r"Score\s+(\d+)", reasoning_text)
        if match:
            return int(match.group(1))

    text = getattr(sample_output, "completion", "")
    match = re.search(r"\b(?:10|[0-9])\b", str(text))
    if match:
        return int(match.group(0))

    return None


def run_pipeline(eval_models: list):
    print(f"\n{'='*60}")
    print(f"🔬 Running self-sycophancy experiment with models: {eval_models}")
    print(f"{'='*60}")

    try:
        # ---------------------
        # STEP 1: Generate PRs
        # ---------------------
        pr_logs = eval(
            tasks=[generate_prs_task(limit_n=LIMIT_N)],
            model=eval_models,
            max_workers=min(len(eval_models), 4),
            log_dir=str(LOG_DIR / "generate_prs"),
            retry_on_error=3,
            fail_on_error=0.2,
        )

        # ---------------------
        # STEP 2: Self-framing Ratings
        # ---------------------
        self_logs = eval(
            tasks=[self_framing_task(limit_n=LIMIT_N)],
            model=eval_models,
            max_workers=min(len(eval_models), 4),
            dataset=[getattr(pr_log, "dataset", None) for pr_log in pr_logs],
            log_dir=str(LOG_DIR / "self_ratings"),
            retry_on_error=3,
            fail_on_error=0.2,
        )

        # ---------------------
        # STEP 3: Other-framing Ratings
        # ---------------------
        other_logs = eval(
            tasks=[other_framing_task(limit_n=LIMIT_N)],
            model=eval_models,
            max_workers=min(len(eval_models), 4),
            dataset=[getattr(pr_log, "dataset", None) for pr_log in pr_logs],
            log_dir=str(LOG_DIR / "other_ratings"),
            retry_on_error=3,
            fail_on_error=0.2,
        )

        # ---------------------
        # STEP 4: Combine Results per model
        # ---------------------
        for pr_result, self_result, other_result, model_name in zip(pr_logs, self_logs, other_logs, eval_models):
            model_short = model_name.split('/')[-1]
            pr_samples = getattr(pr_result, "samples", [])
            self_samples = getattr(self_result, "samples", [])
            other_samples = getattr(other_result, "samples", [])

            # Save PRs
            pr_data = []
            for sample in pr_samples:
                pr_text = getattr(sample.output, "completion", "") if getattr(sample, "output", None) else ""
                sample.metadata["pr"] = pr_text
                pr_data.append({
                    "sample_id": sample.metadata.get("instance_id"),
                    "input": getattr(sample, "input", ""),
                    "target": getattr(sample, "target", ""),
                    "pr": pr_text,
                })
            pr_path = RESULTS_DIR / f"{model_short}.json"
            with open(pr_path, "w") as f:
                json.dump({"generated_prs": pr_data}, f, indent=2)
            print(f"✅ Saved generated PRs for {model_short} to {pr_path}")

            # Combine self and other scores
            final_results = []
            for s_pr, s_self, s_other in zip(pr_samples, self_samples, other_samples):
                final_results.append({
                    "sample_id": s_pr.metadata.get("instance_id"),
                    "pr": s_pr.metadata.get("pr", ""),
                    "self_score": extract_integer_grade(getattr(s_self, "output", None)),
                    "other_score": extract_integer_grade(getattr(s_other, "output", None)),
                })
            combined_path = RESULTS_DIR / f"{model_short}_final.json"
            with open(combined_path, "w") as f:
                json.dump(final_results, f, indent=2)
            print(f"✅ Combined results saved for {model_short} to {combined_path}")

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


def main():
    run_pipeline(EVALUATION_MODELS)


if __name__ == "__main__":
    main()
