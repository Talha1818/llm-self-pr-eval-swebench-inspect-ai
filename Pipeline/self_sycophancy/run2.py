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
LIMIT_N = 10  # matches your task dataset limit
LOG_DIR = Path(__file__).parent / "logs"
RESULTS_DIR = Path(__file__).parent / "results"


def extract_integer_grade(sample_output):
    """
    Extracts an integer grade from the model output.
    Supports:
    1. sample_output.grade
    2. numeric in sample_output.completion
    3. 'Score X' pattern inside reasoning text
    """
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
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        for model_name in eval_models:
            model_short = model_name.split('/')[-1]  # e.g., "gpt-oss-20b"
            log_path = LOG_DIR / model_short
            result_path = RESULTS_DIR / f"{model_short}.json"

            log_path.mkdir(parents=True, exist_ok=True)

            # ---------------------
            # STEP 1: Generate PRs
            # ---------------------
            print(f"\n📝 Generating PRs for {model_short}...")
            pr_logs = eval(
                tasks=[generate_prs_task(limit_n=LIMIT_N)],
                model=[model_name],
                max_workers=min(len(eval_models), 4),  # Limit concurrent workers
                log_dir=str(log_path / "generate_prs"),
                retry_on_error=3,
                fail_on_error=0.2,
            )

            if not pr_logs:
                print(f"⚠️ No PRs generated for {model_short}, skipping...")
                continue

            pr_result = pr_logs[0]
            pr_samples = getattr(pr_result, "samples", [])

            generated_data = []
            for sample in pr_samples:
                pr_text = getattr(sample.output, "completion", "") if getattr(sample, "output", None) else ""
                sample.metadata["pr"] = pr_text
                generated_data.append({
                    "sample_id": sample.metadata.get("instance_id"),
                    "input": getattr(sample, "input", ""),
                    "target": getattr(sample, "target", ""),
                    "pr": pr_text,
                })

            with open(result_path, "w") as f:
                json.dump({"generated_prs": generated_data}, f, indent=2)
            print(f"✅ Saved generated PRs to {result_path}")

            # ---------------------
            # STEP 2: Self-framing Ratings
            # ---------------------
            print(f"\n🤖 Running Self-Framing Ratings for {model_short}...")
            self_logs = eval(
                tasks=[self_framing_task(limit_n=LIMIT_N)],
                model=[model_name],
                max_workers=min(len(eval_models), 4),  # Limit concurrent workers
                dataset=getattr(pr_result, "dataset", None),
                log_dir=str(log_path / "self_ratings"),
                retry_on_error=3,
                fail_on_error=0.2,
            )
            self_samples = getattr(self_logs[0], "samples", [])

            # ---------------------
            # STEP 3: Other-framing Ratings
            # ---------------------
            print(f"\n🤖 Running Other-Framing Ratings for {model_short}...")
            other_logs = eval(
                tasks=[other_framing_task(limit_n=LIMIT_N)],
                model=[model_name],
                max_workers=min(len(eval_models), 4),  # Limit concurrent workers
                dataset=getattr(pr_result, "dataset", None),
                log_dir=str(log_path / "other_ratings"),
                retry_on_error=3,
                fail_on_error=0.2,
            )
            other_samples = getattr(other_logs[0], "samples", [])

            # ---------------------
            # STEP 4: Combine Results
            # ---------------------
            final_results = []

            for s_pr, s_self, s_other in zip(pr_samples, self_samples, other_samples):
                pr_text = s_pr.metadata.get("pr", "")
                self_score = extract_integer_grade(getattr(s_self, "output", None))
                other_score = extract_integer_grade(getattr(s_other, "output", None))

                if self_score is None:
                    print(f"⚠️ Missing self_score for sample {s_pr.metadata.get('instance_id')}")
                    print("Raw output:", getattr(s_self, "output", None))

                if other_score is None:
                    print(f"⚠️ Missing other_score for sample {s_pr.metadata.get('instance_id')}")
                    print("Raw output:", getattr(s_other, "output", None))

                final_results.append({
                    "sample_id": s_pr.metadata.get("instance_id"),
                    "pr": pr_text,
                    "self_score": self_score,
                    "other_score": other_score,
                })

            combined_path = RESULTS_DIR / f"{model_short}_final.json"
            with open(combined_path, "w") as f:
                json.dump(final_results, f, indent=2)

            print(f"\n✅ Combined results saved to: {combined_path}")
            print(f"📊 You can now analyze self vs other scores for {model_short}!")

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        print("⚠️ Skipping this run...")


def main():
    LOG_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    run_pipeline(EVALUATION_MODELS)


if __name__ == "__main__":
    main()
