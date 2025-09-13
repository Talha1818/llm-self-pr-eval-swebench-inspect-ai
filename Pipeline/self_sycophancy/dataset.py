# pipelines/self-sycophancy-swebench/dataset.py

from inspect_ai.dataset import hf_dataset, Sample, Dataset

def record_to_sample(record):
    """
    Convert a single SWE-bench Lite record into an Inspect AI Sample.
    """
    return Sample(
        input=record["problem_statement"],  # what the model sees
        target=record["patch"],             # ground-truth patch
        metadata={
            "instance_id": record["instance_id"],
            "repo": record["repo"],
            "base_commit": record.get("base_commit"),
            "hints_text": record.get("hints_text"),
            "created_at": record.get("created_at"),
            "version": record.get("version"),
            "FAIL_TO_PASS": record.get("FAIL_TO_PASS"),
            "PASS_TO_PASS": record.get("PASS_TO_PASS"),
            "environment_setup_commit": record.get("environment_setup_commit"),
            "test_script": None,  # SWE-bench Lite has no test_script
        },
    )


def load_dataset_subset(limit_n: int = 10) -> Dataset:
    """
    Load SWE-bench Lite using hf_dataset with a custom record-to-sample mapper.
    """
    return hf_dataset(
        path="princeton-nlp/SWE-bench_Lite",
        split="test",
        trust=True,
        limit=limit_n,
        sample_fields=record_to_sample,  # <-- map records to Sample objects
    )


if __name__ == "__main__":
    ds = load_dataset_subset(limit_n=2)
    print(f"✅ Loaded {len(ds.samples)} samples")
    for sample in ds.samples:
        print("Input:", sample.input[:120], "...")
        print("Target (patch):", sample.target[:80], "...")
        print("Metadata:", sample.metadata)
