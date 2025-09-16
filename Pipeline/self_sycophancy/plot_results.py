import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---- Folder containing final JSON results ----
folder = Path("./results")
graph_folder = Path("./graphs")
graph_folder.mkdir(exist_ok=True)  # create folder if not exists

# ---- Load all _final.json files ----
all_records = []
for json_file in folder.glob("*_final.json"):
    model_name = json_file.stem.replace("_final", "")  # extract model name
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    df["model"] = model_name
    all_records.append(df)

# ---- Combine all models into a single DataFrame ----
combined_df = pd.concat(all_records, ignore_index=True)
print(combined_df.head())

# ---- Compute mean self/other scores per model ----
plot_df = combined_df.groupby("model")[["self_score", "other_score"]].mean().reset_index()
print(plot_df)

# ---- Plot: Models vs Self/Other Scores ----
x = np.arange(len(plot_df))  # model positions
width = 0.35  # bar width

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(x - width/2, plot_df["self_score"], width, label="Self Score", color="skyblue")
ax.bar(x + width/2, plot_df["other_score"], width, label="Other Score", color="orange")

# Labels and title
ax.set_xlabel("Model")
ax.set_ylabel("Average Score")
ax.set_title("Average Self vs Other Framing Scores per Model")
ax.set_xticks(x)
ax.set_xticklabels(plot_df["model"], rotation=45, ha="right")
ax.set_ylim(0, 10)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()

# ---- Save figure ----
fig_path = graph_folder / "self_vs_other_scores_per_model.png"
plt.savefig(fig_path, dpi=300)
print(f"✅ Figure saved to {fig_path}")

plt.show()
