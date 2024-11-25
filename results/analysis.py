import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

with open("results/model_performance.json", "r") as file:
    model_performance = json.load(file)

models = []
language_scores = defaultdict(list)

for model, results in model_performance.items():
    for language, scores in results.items():
        language_scores[language].append(round(scores['eval_f1'], 3))
    models.append(model)

x = np.arange(len(models))
fig, ax = plt.subplots(layout='constrained', figsize=(14, 7))

width = 0.18
multiplier = -2

for idx, (language, scores) in enumerate(language_scores.items()):
    offset = width * multiplier
    rects = ax.bar(x + offset, scores, width, label=language)
    ax.bar_label(rects, padding=2, fontsize=8)
    multiplier += 1

ax.set_ylabel('F1-Score')
ax.set_title('Models')
ax.set_xticks(x + width, models)
ax.legend(loc='upper left', ncols=5)
ax.set_ylim(0, 1)

plt.show()