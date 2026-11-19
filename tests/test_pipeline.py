import numpy as np
import pandas as pd
import pytest

from earlywarning import features
from earlywarning.explain import reasons
from earlywarning.model import GBM, LogReg, rule_flag


def tables(seed=0):
    """A tiny synthetic module presentation: 40 students, 3 assignments."""
    rng = np.random.default_rng(seed)
    n = 40
    ids = np.arange(1, n + 1)
    cohort = dict(code_module="AAA", code_presentation="2014J")
    outcome = rng.choice(["Pass", "Fail", "Withdrawn", "Distinction"], n)
    unreg = np.where(outcome == "Withdrawn", rng.integers(5, 120, n), np.nan)
    student = pd.DataFrame({**cohort, "id_student": ids, "gender": rng.choice(["M", "F"], n),
                            "region": "x", "highest_education": "A Level or Equivalent",
                            "imd_band": "20-30%", "age_band": "0-35", "num_of_prev_attempts": 0,
                            "studied_credits": 60, "disability": False, "final_result": outcome})
    reg = pd.DataFrame({**cohort, "id_student": ids, "date_registration": -30.0, "date_unregistration": unreg})
    assessment = pd.DataFrame({**cohort, "id_assessment": [1, 2, 3], "assessment_type": "TMA",
                               "date": [20.0, 50.0, 90.0], "weight": [10.0, 30.0, 60.0]})
    sa = pd.DataFrame([(a, s, d + rng.integers(-3, 5), float(rng.integers(30, 100)), False)
                       for a, d in [(1, 20), (2, 50), (3, 90)] for s in ids if rng.random() < 0.8],
                      columns=["id_assessment", "id_student", "date_submitted", "score", "is_banked"])
    rows = [(cohort["code_module"], cohort["code_presentation"], s, rng.integers(1, 4), d, rng.integers(1, 30))
            for s in ids for d in range(-10, 150) if rng.random() < 0.3]
    sv = pd.DataFrame(rows, columns=["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"])
    sv["activity_type"] = sv.id_site.map({1: "forumng", 2: "quiz", 3: "oucontent"})
    return {"student": student, "student_registration": reg, "assessment": assessment,
            "student_assessment": sa, "student_vle": sv}


def test_features_ignore_everything_after_the_checkpoint():
    t = tables()
    before = features.build(t, 42)
    # change the future: add heavy activity and perfect late scores after day 42
    t2 = {k: v.copy() for k, v in t.items()}
    t2["student_vle"].loc[t2["student_vle"].date > 42, "sum_click"] *= 50
    t2["student_assessment"].loc[t2["student_assessment"].date_submitted > 42, "score"] = 100.0
    # ... and late submissions (after the checkpoint) of an assignment that was due before it
    sa = t2["student_assessment"]
    missing = sorted(set(t2["student"].id_student) - set(sa[sa.id_assessment == 1].id_student))
    assert missing
    late = pd.DataFrame({"id_assessment": 1, "id_student": missing, "date_submitted": 60,
                         "score": 100.0, "is_banked": False})
    t2["student_assessment"] = pd.concat([sa, late], ignore_index=True)
    after = features.build(t2, 42)
    cols = features.BEHAVIOUR + features.DEMOGRAPHIC
    pd.testing.assert_frame_equal(before[cols], after[cols])


def test_population_excludes_students_who_already_left():
    t = tables()
    f = features.build(t, 42)
    reg = t["student_registration"].set_index("id_student").date_unregistration
    left = set(reg[reg <= 42].index)
    assert left and not left & set(f.id_student)
    assert set(f.id_student) == set(reg.index) - left


def test_cohort_percentiles_are_within_cohort():
    f = features.build(tables(), 42)
    assert f.clicks_pct_in_cohort.between(0, 1).all()
    assert f.clicks_pct_in_cohort.max() == 1.0


def test_placeholder_marks_are_detected_and_dropped():
    t = tables()
    sa = t["student_assessment"]
    sa.loc[sa.id_assessment == 1, "score"] = 0.0          # assignment 1: placeholder marks
    assert features.placeholder_assessments(sa) == {1}
    clean = features.build(t, 42, clean=True)
    raw = features.build(t, 42, clean=False)
    assert raw.min_score.min() == 0 and (clean.min_score.dropna() > 0).all()
    # the submission itself still counts as submitted
    assert (clean.n_submitted_due == raw.n_submitted_due).all()


def test_rule_flags_inactive_or_missed():
    f = pd.DataFrame({"days_since_last": [0, 20, 3, 3], "n_missed": [0, 0, 1, 0]})
    assert list(rule_flag(f)) == [0, 1, 1, 0]


@pytest.mark.parametrize("Model", [LogReg, GBM])
def test_models_train_and_score(Model):
    f = pd.concat([features.build(tables(s), 42) for s in range(5)], ignore_index=True)
    kw = {"n_estimators": 20, "min_child_samples": 5} if Model is GBM else {}
    p = Model(**kw).fit(f).predict(f)
    assert p.shape == (len(f),) and ((p >= 0) & (p <= 1)).all()


def test_reasons_never_contradict_the_data():
    f = features.build(tables(), 42)
    row = f.iloc[0].copy()
    row["n_missed"], row["mean_score"], row["days_since_last"] = 0, 85.0, 1
    contrib = pd.Series({"n_missed": 1.0, "mean_score": 0.8, "days_since_last": 0.5})
    assert reasons(row, contrib) == []          # pushes up, but nothing concerning to report
    row["n_missed"] = 2
    assert reasons(row, contrib) == ["2 assessment(s) not submitted"]
