
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from src.config import KRR_PARAMS, SVR_PARAMS, LGBM_PARAMS, MLP_PARAMS

class ModelFactory:
    @staticmethod
    def get_stacking_model():
        estimators = [
            ('krr', KernelRidge(**KRR_PARAMS)),
            ('svr', SVR(**SVR_PARAMS)),
            ('lgbm', LGBMRegressor(**LGBM_PARAMS)),
            ('mlp', MLPRegressor(**MLP_PARAMS))
        ]
        return StackingRegressor(estimators=estimators, final_estimator=RidgeCV(), n_jobs=-1)
