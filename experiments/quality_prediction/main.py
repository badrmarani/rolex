import pandas as pd
import xgboost as xgb
import numpy as np

from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_validate, cross_val_score, KFold, train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import metrics

# from skopt import gp_minimize
# from skopt.space import Real, Integer
# from skopt.utils import use_named_args
# from skopt.plots import plot_convergence

import hyperopt as hp
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

x12 = pd.read_csv("data/full_no_nans_dataset_x12.csv", sep=",")
x3 = pd.read_csv("data/full_no_nans_dataset_x3.csv", sep=",")

def optimize_params(x, y, params_space, validation_split):
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_split)
    def objective(params):
        reg = xgb.XGBRegressor(
            eval_metric=["rmse"],
            early_stopping_rounds=30,
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=params["learning_rate"],
            min_child_weight=params["min_child_weight"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            gamma=params["gamma"],
        )
        reg.fit(
            x_train, y_train,
            eval_set=[(x_train, y_train), (x_val, y_val)],
            verbose=0,
        )


        y_pred = reg.predict(x_val)
        r2 = metrics.r2_score(y_val, y_pred)

        return {"loss": - r2, "status": STATUS_OK}

    trials = Trials()
    return fmin(
        fn=objective,
        space=params_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
        # verbose=1,
    )

class XGBOPT(BaseEstimator, ClassifierMixin):
    def __init__(self, custom_params_space=None):
        self.custom_params_space = custom_params_space
        self.counter = 1
    
    def fit(self, x, y, validation_split=0.2):
        if self.custom_params_space is None:
            self.custom_params_space = {
                "n_estimators": hp.randint("n_estimators", 50, 300),
                "learning_rate": hp.uniform("learning_rate", 0.0001, 0.05),
                "max_depth": hp.quniform("max_depth", 8, 15, 1),
                "min_child_weight": hp.quniform("min_child_weight", 1, 5, 1),
                "subsample": hp.quniform("subsample", 0.7, 1, 0.05),
                "gamma": hp.quniform("gamma", 0.9, 1, 0.05),
                "colsample_bytree": hp.quniform("colsample_bytree", 0.5, 0.7, 0.05)
            }

        opt = optimize_params(x, y, self.custom_params_space, validation_split)

        self.xgb_model = xgb.XGBRegressor(
            eval_metric=["rmse"],
            n_estimators=int(opt["n_estimators"]),
            max_depth=int(opt["max_depth"]),
            learning_rate=opt["learning_rate"],
            min_child_weight=opt["min_child_weight"],
            subsample=opt["subsample"],
            colsample_bytree=opt["colsample_bytree"],
            gamma=opt["gamma"],
        )

        self.xgb_model.fit(x, y, verbose=0)
        self.best_estimator = self.xgb_model
        return self
    
    def predict(self, x, y=None):
        if not hasattr(self, 'best_estimator'):
            from sklearn.exceptions import NotFittedError
            raise NotFittedError('Call `fit` before `predict`.')
        else:
            return self.best_estimator.predict(x)

kf = KFold(n_splits=4, shuffle=True)
xgb_scores = cross_val_score(
    XGBOPT(),
    x12, x3,
    cv=kf,
    scoring="r2",
    n_jobs=-1,
)

print("R2 = %0.2f (+/- %0.2f)" % (xgb_scores.mean(), xgb_scores.std()))

print("fitting on whole dataset and saving the estimator...")
final = XGBOPT()
final.fit(x12, x3)
final.xgb_model.save_model("experiments/quality_prediction/model.json")