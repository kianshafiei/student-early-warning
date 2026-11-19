"""Train on earlier cohorts, test on the latest cohort (2014J), checkpoint by checkpoint.

Writes results/results.json plus CSVs (by_week, fairness, calibration, examples).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

from earlywarning import data, features
from earlywarning.model import GBM, LogReg, rule_flag
from earlywarning.explain import reasons

OUT = Path("results"); OUT.mkdir(exist_ok=True)
TRAIN, TEST = ["2013B", "2013J", "2014B"], ["2014J"]
WEEKS = list(range(1, 21))          # a checkpoint every week, weeks 1-20
CAPACITY = 0.10                     # the support team can contact 10% of current students a week
MAIN, DROP = "not_completing", "withdrew"
LR, GB, GD = "Logistic regression", "Gradient boosting", "Gradient boosting + demographics"
rng = np.random.default_rng(0)


def top_share(score, share=CAPACITY):
    flag = np.zeros(len(score), int)
    flag[np.argsort(-np.asarray(score), kind="stable")[:int(np.ceil(share * len(score)))]] = 1
    return flag


def top_n(score, n):
    flag = np.zeros(len(score), int); flag[np.argsort(-np.asarray(score), kind="stable")[:n]] = 1
    return flag


def prec_rec(y, flag):
    y, flag = np.asarray(y), np.asarray(flag); tp = (flag & y).sum()
    return tp / max(flag.sum(), 1), tp / max(y.sum(), 1)


def boot(g, fn, n=1000):
    vals = [fn(g.iloc[rng.integers(0, len(g), len(g))]) for _ in range(n)]
    return [round(float(fn(g)), 4), [round(float(np.percentile(vals, q)), 4) for q in (2.5, 97.5)]]


def frame(clean):
    f = pd.concat([features.build(t, 7 * w, clean=clean) for w in WEEKS], ignore_index=True)
    f["week"] = f.cutoff // 7
    return f[f.code_presentation.isin(TRAIN)], f[f.code_presentation.isin(TEST)].copy()


print("building features ...")
t = data.load()
tr, te = frame(clean=True)
te["Rule"] = rule_flag(te)
print(f"train rows {len(tr):,}  test rows {len(te):,}  test students {te.id_student.nunique():,}")

res = {"train_cohorts": TRAIN, "test_cohorts": TEST, "capacity": CAPACITY,
       "train_rows": len(tr), "test_rows": len(te), "test_students": int(te.id_student.nunique())}
tables, fitted = [], {}
for target in [MAIN, DROP]:
    ms = {LR: LogReg(target=target), GB: GBM(target=target), GD: GBM(demographics=True, target=target)}
    for name, m in ms.items():
        te[f"{target}|{name}"] = m.fit(tr).predict(te)
        fitted[(target, name)] = m
    for w, g in te.groupby("week"):
        y = g[target].to_numpy()
        rp, rr = prec_rec(y, g.Rule)
        for name in [*ms, "Rule"]:
            row = {"target": target, "week": w, "students": len(g), "base_rate": y.mean(), "method": name}
            if name == "Rule":
                row.update({"flag_share": g.Rule.mean(), "precision": rp, "recall": rr})
            else:
                s = g[f"{target}|{name}"].to_numpy()
                p10, r10 = prec_rec(y, top_share(s))
                pv, rv = prec_rec(y, top_n(s, int(g.Rule.sum())))
                row.update({"auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s),
                            "prec@10%": p10, "rec@10%": r10, "prec@rule_volume": pv,
                            "rec@rule_volume": rv, "brier": brier_score_loss(y, s)})
            tables.append(row)
W = pd.DataFrame(tables); W.to_csv(OUT / "by_week.csv", index=False)

g6 = te[te.week == 6]
col = lambda target, name: f"{target}|{name}"
ap = lambda g, target, m: average_precision_score(g[target], g[col(target, m)])
res["bootstrap_week6"] = {}
for target in [MAIN, DROP]:
    res["bootstrap_week6"][target] = {
        "pr_auc_gb_minus_lr": boot(g6, lambda g: ap(g, target, GB) - ap(g, target, LR)),
        "pr_auc_gbdemo_minus_gb": boot(g6, lambda g: ap(g, target, GD) - ap(g, target, GB)),
        "precision_gb_minus_rule_at_rule_volume": boot(g6, lambda g: prec_rec(
            g[target], top_n(g[col(target, GB)], int(g.Rule.sum())))[0] - prec_rec(g[target], g.Rule)[0]),
    }

# ---- data-quality check on/off (main target, gradient boosting) ----
tr_raw, te_raw = frame(clean=False)
p_raw = GBM().fit(tr_raw).predict(te_raw)
dq = {}
for label, frm, p in [("with check", te, te[col(MAIN, GB)]), ("without check", te_raw, p_raw)]:
    d = frm.assign(p=np.asarray(p))
    d6 = d[d.week == 6]; b = d6[d6.code_module == "BBB"]
    dq[label] = {"auc_week6_all": roc_auc_score(d6[MAIN], d6.p), "auc_week6_BBB": roc_auc_score(b[MAIN], b.p),
                 "BBB_mean_predicted": float(b.p.mean()), "BBB_actual_rate": float(b[MAIN].mean())}
res["data_quality_check"] = dq

# ---- lead time for students who drop out: weekly lists vs the rule ----
unreg = t["student_registration"].set_index(features.KEY).date_unregistration


def lead(flag_fn):
    flags = pd.concat([g.assign(_f=flag_fn(g)) for _, g in te.groupby("week")])
    wd = flags[flags[DROP] == 1]
    ids = wd.groupby(features.KEY).size().index
    u = unreg.reindex(ids)
    target_ids = u[(u > 7) & (u <= 140)].index          # dropped out in weeks 2-20
    first = wd[wd._f == 1].groupby(features.KEY).cutoff.min().reindex(target_ids)
    ok = first.notna() & (first < u.reindex(target_ids))
    return {"dropouts_weeks_2_20": int(len(target_ids)), "share_flagged_before_dropout": round(float(ok.mean()), 3),
            "median_days_warning": float((u.reindex(target_ids) - first)[ok].median()),
            "flags_raised": int(flags._f.sum())}


res["lead_time"] = {
    "Rule": lead(lambda g: g.Rule.to_numpy()),
    "Dropout model, same weekly volume as rule": lead(lambda g: top_n(g[col(DROP, GB)], int(g.Rule.sum()))),
    "Dropout model, top 10% each week": lead(lambda g: top_share(g[col(DROP, GB)])),
}

# ---- fairness at week 6, main target, same 10% capacity ----
fair = []
for name in [GB, GD]:
    g = g6.assign(flag=top_share(g6[col(MAIN, name)]),
                  disability_s=g6.disability.map({0: "No", 1: "Yes"}),
                  imd_s=g6.imd_band_ord.map(lambda v: "unknown" if v != v else (
                      "most deprived 30%" if v < 30 else ("least deprived 30%" if v >= 70 else "middle 40%"))))
    for attr, c in [("Deprivation (IMD)", "imd_s"), ("Disability", "disability_s"),
                    ("Age", "age_band"), ("Gender", "gender")]:
        for val, h in g.groupby(c):
            p, r = prec_rec(h[MAIN], h.flag)
            fair.append({"model": name, "attribute": attr, "group": val, "students": len(h),
                         "base_rate": h[MAIN].mean(), "flag_rate": h.flag.mean(), "recall": r, "precision": p})
FA = pd.DataFrame(fair); FA.to_csv(OUT / "fairness.csv", index=False)

# ---- calibration at week 6 (main target) ----
cal = pd.DataFrame({"pred": g6[col(MAIN, GB)], "obs": g6[MAIN]})
cal = cal.groupby(pd.qcut(cal.pred, 10), observed=True).mean()
cal.to_csv(OUT / "calibration_week6.csv", index=False)
res["calibration_week6_max_gap"] = round(float((cal.pred - cal.obs).abs().max()), 3)

# ---- what drives the score, and example reasons ----
gb = fitted[(MAIN, GB)]
C = gb.contributions(g6)
res["importance_top10"] = {k: round(float(v), 3) for k, v in C.abs().mean().sort_values(ascending=False).head(10).items()}
flagged = g6[top_share(g6[col(MAIN, GB)]) == 1].sample(10, random_state=1)
ex = pd.DataFrame([{"risk": round(float(r[col(MAIN, GB)]), 2), "outcome": "did not complete" if r[MAIN] else "completed",
                    "reasons": "; ".join(reasons(r, C.loc[i]))} for i, r in flagged.iterrows()])
ex.to_csv(OUT / "examples.csv", index=False)

for target in [MAIN, DROP]:
    w6 = W[(W.target == target) & (W.week == 6)].set_index("method")
    res[f"week6_{target}"] = w6.drop(columns=["target", "week"]).round(4).to_dict("index")
(OUT / "results.json").write_text(json.dumps(res, indent=2, default=float))

pd.set_option("display.width", 220)
for target in [MAIN, DROP]:
    print(f"\n== {target} ==")
    v = W[(W.target == target) & W.week.isin([2, 4, 6, 8, 12, 16])]
    print(v.pivot_table(index="week", columns="method", values=["auc", "prec@10%", "prec@rule_volume"]).round(3).to_string())
    print(v[v.method == "Rule"][["week", "base_rate", "flag_share", "precision", "recall"]].round(3).to_string())
print(json.dumps({k: v for k, v in res.items() if not k.startswith("week6")}, indent=1, default=float))
print(FA.round(3).to_string()); print(cal.round(3).to_string()); print(ex.to_string())
