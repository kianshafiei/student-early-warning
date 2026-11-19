"""Engagement features for one checkpoint, built ONLY from data dated up to the checkpoint.

Each feature is an observable stand-in for something the student-engagement literature
describes (behavioural engagement = activity volume and regularity; disengagement = a
drop or a gap in activity; academic engagement = assessment submission; social
engagement = forum use). Everything is computed per student per module presentation.
"""
import numpy as np
import pandas as pd

KEY = ["code_module", "code_presentation", "id_student"]
COHORT = ["code_module", "code_presentation"]
# activity types with enough sites to be meaningful across modules
TYPES = ["forumng", "quiz", "oucontent", "resource", "homepage", "subpage", "url", "ouwiki"]

BEHAVIOUR = [
    "clicks_total", "clicks_pre_start", "active_days", "active_days_last14", "clicks_last14",
    "clicks_prev14", "trend_14", "days_since_last", "weeks_active_frac", "weekly_cv",
    "n_activity_types", "clicks_pct_in_cohort", "active_days_pct_in_cohort",
    *[f"share_{t}" for t in TYPES],
    "n_due", "n_submitted_due", "n_missed", "n_late", "mean_score", "min_score",
    "days_registered_before_start",
]
DEMOGRAPHIC = ["num_of_prev_attempts", "studied_credits", "highest_education_ord",
               "imd_band_ord", "age_band_ord", "disability", "gender_f"]

EDU = {"No Formal quals": 0, "Lower Than A Level": 1, "A Level or Equivalent": 2,
       "HE Qualification": 3, "Post Graduate Qualification": 4}
AGE = {"0-35": 0, "35-55": 1, "55<=": 2}


def population(t, cutoff):
    """Students still registered at the checkpoint (not unregistered on or before it)."""
    r = t["student_registration"].merge(t["student"][KEY + ["final_result"]], on=KEY)
    still = r.date_unregistration.isna() | (r.date_unregistration > cutoff)
    pop = r.loc[still, KEY + ["date_registration", "final_result"]].copy()
    # main target: the student does not complete the module successfully (withdraws or fails)
    pop["not_completing"] = pop.final_result.isin(["Withdrawn", "Fail"]).astype(int)
    # secondary target: the student goes on to withdraw (drop out)
    pop["withdrew"] = (pop.final_result == "Withdrawn").astype(int)
    return pop.drop(columns="final_result")


def placeholder_assessments(sa, threshold=0.2):
    """Assessments where an implausible share of submitted scores is 0-1%.

    In normal assessments only a few percent of submissions (at most 3% in OULAD) score 0-1. A share above
    `threshold` points to placeholder or unrecorded marks, which a model would read
    as failing students. Uses only scores already submitted, so it works at run time.
    """
    low = sa.assign(low=sa.score <= 1).groupby("id_assessment").low.mean()
    return set(low[low > threshold].index)


def _vle_features(sv, cutoff):
    sv = sv[sv.date <= cutoff]
    g = sv.groupby(KEY)
    f = pd.DataFrame({
        "clicks_total": g.sum_click.sum(),
        "clicks_pre_start": sv[sv.date < 0].groupby(KEY).sum_click.sum(),
        "last_day": g.date.max(),
    })
    days = sv.groupby(KEY + ["date"]).sum_click.sum().reset_index()
    f["active_days"] = days.groupby(KEY).size()
    rec = days[days.date > cutoff - 14]
    f["active_days_last14"] = rec.groupby(KEY).size()
    f["clicks_last14"] = rec.groupby(KEY).sum_click.sum()
    prev = days[(days.date > cutoff - 28) & (days.date <= cutoff - 14)]
    f["clicks_prev14"] = prev.groupby(KEY).sum_click.sum()
    # weekly regularity over the course weeks elapsed so far (weeks from day 0)
    nweeks = max(1, int(np.ceil((cutoff + 1) / 7)))
    wk = days[days.date >= 0].assign(week=lambda d: d.date // 7)
    wk = wk.groupby(KEY + ["week"]).sum_click.sum().reset_index()
    f["weeks_active_frac"] = wk.groupby(KEY).size() / nweeks
    piv = wk.pivot_table(index=KEY, columns="week", values="sum_click", fill_value=0)
    piv = piv.reindex(columns=range(nweeks), fill_value=0)
    f["weekly_cv"] = piv.std(axis=1) / (piv.mean(axis=1) + 1e-9)
    f["n_activity_types"] = sv.groupby(KEY).activity_type.nunique()
    by_t = sv[sv.activity_type.isin(TYPES)].pivot_table(
        index=KEY, columns="activity_type", values="sum_click", aggfunc="sum", fill_value=0)
    for t in TYPES:
        f[f"share_{t}"] = (by_t[t] if t in by_t else 0) / f["clicks_total"]
    return f


def _assessment_features(t, cutoff, clean=True):
    a = t["assessment"]
    a = a[(a.assessment_type != "Exam") & a.date.notna() & (a.date <= cutoff)]
    due = t["student_registration"][KEY].merge(a[COHORT + ["id_assessment", "date"]], on=COHORT)
    sa = t["student_assessment"]
    sa = sa[sa.date_submitted <= cutoff].copy()
    if clean:  # keep the submission, drop the placeholder score
        sa.loc[sa.id_assessment.isin(placeholder_assessments(sa)), "score"] = np.nan
    m = due.merge(sa[["id_assessment", "id_student", "date_submitted", "score"]],
                  on=["id_assessment", "id_student"], how="left")
    m["sub"] = m.date_submitted.notna()
    m["late"] = m["sub"] & (m.date_submitted > m.date)
    g = m.groupby(KEY)
    return pd.DataFrame({
        "n_due": g.size(), "n_submitted_due": g["sub"].sum(), "n_late": g.late.sum(),
        "n_missed": g.size() - g["sub"].sum(),
        "mean_score": g.score.mean(), "min_score": g.score.min(),
    })


def build(t, cutoff, clean=True):
    """Feature table for every student still registered at day `cutoff`."""
    pop = population(t, cutoff).set_index(KEY)
    sv = t["student_vle"]
    f = pop.join(_vle_features(sv, cutoff)).join(_assessment_features(t, cutoff, clean))
    zero = ["clicks_total", "clicks_pre_start", "active_days", "active_days_last14",
            "clicks_last14", "clicks_prev14", "weeks_active_frac", "n_activity_types",
            "n_due", "n_submitted_due", "n_late", "n_missed"]
    f[zero] = f[zero].fillna(0)
    f["weekly_cv"] = f["weekly_cv"].fillna(f["weekly_cv"].max())  # no activity = least regular
    # never active: count the gap from the earliest possible activity (day -25 in this data)
    f["days_since_last"] = cutoff - f.pop("last_day").fillna(-26)
    f["trend_14"] = np.log1p(f.clicks_last14) - np.log1p(f.clicks_prev14)
    # modules differ hugely in how much VLE material they have, so compare within the cohort
    grp = f.groupby(level=COHORT)
    f["clicks_pct_in_cohort"] = grp.clicks_total.rank(pct=True)
    f["active_days_pct_in_cohort"] = grp.active_days.rank(pct=True)
    f["days_registered_before_start"] = -f.pop("date_registration")
    s = t["student"].set_index(KEY)
    f = f.join(pd.DataFrame({
        "num_of_prev_attempts": s.num_of_prev_attempts, "studied_credits": s.studied_credits,
        "highest_education_ord": s.highest_education.map(EDU),
        "imd_band_ord": s.imd_band.str.extract(r"^(\d+)")[0].astype(float),
        "age_band_ord": s.age_band.map(AGE), "disability": s.disability.astype(int),
        "gender_f": (s.gender == "F").astype(int),
        # kept for fairness reporting only, never used as model input
        "imd_band": s.imd_band, "age_band": s.age_band, "gender": s.gender,
    }))
    f["cutoff"] = cutoff
    return f.reset_index()
