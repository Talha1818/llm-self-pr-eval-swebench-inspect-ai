#!/usr/bin/env python3
from pathlib import Path
from inspect_ai import eval_set, eval
from task import self_sycophancy_task
from dotenv import load_dotenv
import uuid
from inspect_ai.model import get_model

load_dotenv()

# ---- CONFIG ----
EVALUATION_MODELS = [
    "openrouter/openai/gpt-oss-20b",
    "openrouter/openai/gpt-oss-120b",
]
LIMIT_N = 10  # matches your task dataset limit
LOG_DIR = Path(__file__).parent / "logs"

def run_evaluation_for_model(eval_models:list):
    print(f"\n{'='*60}")
    print(f"🔬 Running experiment with model: {eval_models}")
    print(f"{'='*60}")

    try:
        # Unique task ID and log dir
        unique_task_id = str(uuid.uuid4())
        unique_log_dir = LOG_DIR / f"{eval_models[0].split('/')[-1]}_{unique_task_id}"
        unique_log_dir.mkdir(parents=True, exist_ok=True)

        # Create the task instance with solver models
        task_instance = self_sycophancy_task(limit_n=LIMIT_N)

        # success, logs = eval_set(
        #     tasks=[task_instance],
        #     model=[model_id],
        #     log_dir=str(unique_log_dir),
        #     retry_on_error=3,
        #     fail_on_error=0.2,
        # )

        results = eval(
            tasks=[task_instance],
            # model=[model_id],
            model= eval_models,
            max_workers=min(len(eval_models), 4),  # Limit concurrent workers.
            log_dir=str(unique_log_dir),
            retry_on_error=3,
            fail_on_error=0.2,
        )

        print(results)

                # Process results for each model
        if hasattr(results, 'samples') and results.samples:
            # Single result object (if only one model)
            model_results = [results]
        else:
            # Multiple result objects (one per model)
            model_results = list(results) if hasattr(results, '__iter__') else [results]

    # print(f"✅ Eval completed successfully for {model_id}" if success else f"⚠️ Eval completed with some failures for {model_id}")
        # print(f"🗂 Logs saved in: {unique_log_dir}")

        # for sample in logs.samples:
        #     pr_text = sample.outputs[0]           # Generated PR
        #     self_rating = float(sample.outputs[1].splitlines()[-1])
        #     other_rating = float(sample.outputs[2].splitlines()[-1])
        #     print("PR:", pr_text)
        #     print("Self rating:", self_rating)
        #     print("Other rating:", other_rating)

        # print(logs)


    except Exception as e:
        print(f"❌ Error during evaluation for model {eval_models[0]}: {e}")
        import traceback
        traceback.print_exc()
        print("⚠️ Skipping this model and continuing with next...")

def main():
    LOG_DIR.mkdir(exist_ok=True)
    # for model_id in EVALUATION_MODELS:
    run_evaluation_for_model(EVALUATION_MODELS)

if __name__ == "__main__":
    main()
