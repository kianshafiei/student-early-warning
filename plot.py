import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

W = pd.read_csv("results/by_week.csv")
res = json.load(open("results/results.json"))
INK, MUTE, GRID = "#1f2328", "#6e7781", "#d8dee4"
COL = {"Gradient boosting": "#0969da", "Logistic regression": "#8250df", "Rule": "#cf222e"}

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), dpi=150)
for a in ax:
    a.spines[["top", "right"]].set_visible(False); a.grid(axis="y", color=GRID, lw=0.6)
    a.tick_params(colors=MUTE); a.set_xlabel("Week of the module", color=MUTE)

m = W[W.target == "not_completing"]
for name in ["Gradient boosting", "Logistic regression"]:
    d = m[m.method == name]
    ax[0].plot(d.week, d.auc, color=COL[name], lw=2, label=name)
d = W[(W.target == "withdrew") & (W.method == "Gradient boosting")]
ax[0].plot(d.week, d.auc, color=COL["Gradient boosting"], lw=2, ls="--", label="Gradient boosting, dropout only")
ax[0].set_title("Ranking quality on the unseen cohort (AUC)", color=INK, loc="left", fontsize=11)
ax[0].legend(frameon=False, fontsize=8); ax[0].set_ylim(0.55, 0.9)

r = m[m.method == "Rule"].set_index("week"); g = m[m.method == "Gradient boosting"].set_index("week")
ax[1].plot(r.index, r.precision, color=COL["Rule"], lw=2, label="Rule: inactive 14+ days or missed assessment")
ax[1].plot(g.index, g["prec@rule_volume"], color=COL["Gradient boosting"], lw=2, label="Model, same number of students flagged")
ax[1].plot(g.index, g["prec@10%"], color=COL["Gradient boosting"], lw=2, ls=":", label="Model, top 10% only")
ax[1].plot(g.index, g.base_rate, color=MUTE, lw=1, ls="--", label="Share who don't complete")
ax[1].set_title("Of the students flagged, share who don't complete", color=INK, loc="left", fontsize=11)
ax[1].legend(frameon=False, fontsize=8, loc="lower left"); ax[1].set_ylim(0.0, 1.0)

cal = pd.read_csv("results/calibration_week6.csv")
ax[2].plot([0, 1], [0, 1], color=GRID, lw=1)
ax[2].plot(cal.pred, cal.obs, "o-", color=COL["Gradient boosting"], lw=2, ms=4)
ax[2].set_xlabel("Predicted risk at week 6", color=MUTE); ax[2].set_ylabel("Observed rate", color=MUTE)
ax[2].set_title("Calibration at week 6 (deciles)", color=INK, loc="left", fontsize=11)
fig.tight_layout(); fig.savefig("results/results.png", bbox_inches="tight")
print("saved results/results.png")
