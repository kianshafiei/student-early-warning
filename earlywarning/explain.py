"""Turn model contributions into short reasons a tutor can act on.

A reason is shown only if the feature pushes this student's risk up AND its value
is actually concerning, so the text never contradicts what the student did.
"""
import math

# feature -> (is the value concerning?, text)
REASONS = {
    "n_missed": (lambda r: r.n_missed >= 1, lambda r: f"{r.n_missed:.0f} assessment(s) not submitted"),
    "mean_score": (lambda r: math.isnan(r.mean_score) or r.mean_score < 60,
                   lambda r: "no marked assessment yet" if math.isnan(r.mean_score)
                   else f"average assessment score {r.mean_score:.0f}%"),
    "min_score": (lambda r: not math.isnan(r.min_score) and r.min_score < 40,
                  lambda r: f"an assessment scored {r.min_score:.0f}%"),
    "n_late": (lambda r: r.n_late >= 1, lambda r: f"{r.n_late:.0f} late submission(s)"),
    "days_since_last": (lambda r: r.days_since_last >= 7,
                        lambda r: f"no learning-platform activity for {r.days_since_last:.0f} days"),
    "active_days_last14": (lambda r: r.active_days_last14 <= 3,
                           lambda r: f"active on only {r.active_days_last14:.0f} of the last 14 days"),
    "trend_14": (lambda r: r.trend_14 < -0.7, lambda r: "activity dropping compared with the two weeks before"),
    "clicks_pct_in_cohort": (lambda r: r.clicks_pct_in_cohort < 0.25,
                             lambda r: f"less active than {100 - 100 * r.clicks_pct_in_cohort:.0f}% of the cohort"),
    "active_days_pct_in_cohort": (lambda r: r.active_days_pct_in_cohort < 0.25,
                                  lambda r: f"fewer active days than {100 - 100 * r.active_days_pct_in_cohort:.0f}% of the cohort"),
    "weeks_active_frac": (lambda r: r.weeks_active_frac < 0.6,
                          lambda r: f"active in only {100 * r.weeks_active_frac:.0f}% of weeks so far"),
    "days_registered_before_start": (lambda r: r.days_registered_before_start < 14,
                                     lambda r: "registered close to (or after) the start date"),
    "n_activity_types": (lambda r: r.n_activity_types <= 4,
                         lambda r: f"uses only {r.n_activity_types:.0f} types of learning material"),
}


def reasons(row, contrib, k=2):
    """Top-k concerning features pushing this student's risk up, in plain English."""
    out = []
    for c in contrib[contrib > 0].sort_values(ascending=False).index:
        if c in REASONS and REASONS[c][0](row):
            out.append(REASONS[c][1](row))
        if len(out) == k:
            break
    return out
