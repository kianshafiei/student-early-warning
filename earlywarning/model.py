import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from .features import BEHAVIOUR, DEMOGRAPHIC

LGB_PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=0, verbose=-1)
COUNTS = ["clicks_total", "clicks_pre_start", "clicks_last14", "clicks_prev14"]


def columns(demographics=False):
    return BEHAVIOUR + ["cutoff"] + (DEMOGRAPHIC if demographics else [])


def rule_flag(f):
    """A typical dashboard rule: no activity for 14+ days, or a missed assessment."""
    return ((f.days_since_last >= 14) | (f.n_missed >= 1)).astype(int).to_numpy()


class LogReg:
    def __init__(self, demographics=False, target="not_completing"):
        self.cols, self.target = columns(demographics), target
        self.m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               LogisticRegression(max_iter=2000, C=1.0))

    def _x(self, f):
        x = f[self.cols].astype(float).copy()
        x[COUNTS] = np.log1p(x[COUNTS])
        return x

    def fit(self, f):
        self.m.fit(self._x(f), f[self.target])
        return self

    def predict(self, f):
        return self.m.predict_proba(self._x(f))[:, 1]


class GBM:
    def __init__(self, demographics=False, target="not_completing", **kw):
        self.cols, self.target = columns(demographics), target
        self.m = lgb.LGBMClassifier(**{**LGB_PARAMS, **kw})

    def fit(self, f):
        self.m.fit(f[self.cols].astype(float), f[self.target])
        return self

    def predict(self, f):
        return self.m.predict_proba(f[self.cols].astype(float))[:, 1]

    def contributions(self, f):
        """Per-student SHAP-style contributions (log-odds), one column per feature."""
        c = self.m.predict_proba(f[self.cols].astype(float), pred_contrib=True)
        return pd.DataFrame(c[:, :-1], columns=self.cols, index=f.index)
