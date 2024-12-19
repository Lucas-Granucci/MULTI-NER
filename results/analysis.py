import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Load the data
with open("results/test_models/model_performance.json", "r") as file:
    model_performance = json.load(file)

models = []
language_scores = defaultdict(list)

for model, results in model_performance.items():
    total_model_score = 0
    for language, scores in results.items():
        language_scores[language].append(round(scores["eval_f1"], 3))
        total_model_score += round(scores["eval_f1"], 3)
    print("Model: {}".format(model))
    print("Avg. Score: {}".format(total_model_score / 6))
    print()
    models.append(model)

x = np.arange(len(models))
fig, ax = plt.subplots(layout="constrained", figsize=(14, 8))

width = 0.14
multiplier = -2

# Use a professional color palette
colors = plt.get_cmap("Blues", len(language_scores))

for idx, (language, scores) in enumerate(language_scores.items()):
    offset = width * multiplier
    rects = ax.bar(
        x + offset,
        scores,
        width,
        label=language,
        color=colors(idx),
        edgecolor="black",  # Add border for a cleaner look
    )
    ax.bar_label(rects, padding=2, fontsize=8, rotation=45)
    multiplier += 1

# Enhance the axes
ax.set_ylabel("F1-Score", fontsize=12, weight="bold")
ax.set_xlabel("Models", fontsize=12, weight="bold")
ax.set_title("Model Performance Across Languages", fontsize=16, weight="bold", pad=20)
ax.set_xticks(x + width / 2, labels=models, fontsize=10, rotation=30, ha="right")
ax.set_ylim(0, 1.1)

# Add gridlines for better readability
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Enhance the legend
ax.legend(
    title="Languages",
    title_fontsize=10,
    fontsize=9,
    loc="upper left",
    ncols=5,
    frameon=False,
)

plt.show()
