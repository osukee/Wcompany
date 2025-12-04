"""
モデル定義モジュール
各種機械学習モデルの定義とハイパーパラメータチューニングを実装
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import config

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost is not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM is not installed. Install with: pip install lightgbm")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost is not installed. Install with: pip install catboost")

def create_random_forest(**kwargs):
    """RandomForestモデルを作成"""
    params = config.RF_PARAMS.copy()
    params.update(kwargs)
    return RandomForestClassifier(**params)

def create_xgboost(**kwargs):
    """XGBoostモデルを作成"""
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed")
    
    params = config.XGB_PARAMS.copy()
    params.update(kwargs)
    return xgb.XGBClassifier(**params)

def create_lightgbm(**kwargs):
    """LightGBMモデルを作成"""
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is not installed")
    
    params = config.LGBM_PARAMS.copy()
    params.update(kwargs)
    return lgb.LGBMClassifier(**params)

def create_catboost(**kwargs):
    """CatBoostモデルを作成"""
    if not CATBOOST_AVAILABLE:
        raise ImportError("CatBoost is not installed")
    
    params = config.CATBOOST_PARAMS.copy()
    params.update(kwargs)
    return cb.CatBoostClassifier(**params)

def tune_random_forest(X_train, y_train, cv=5, scoring='f1'):
    """RandomForestのハイパーパラメータチューニング"""
    base_model = RandomForestClassifier(
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    
    param_grid = config.RF_GRID_PARAMS
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best {scoring} score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def get_model(model_name='random_forest', **kwargs):
    """モデル名に基づいてモデルを作成"""
    model_name = model_name.lower()
    
    if model_name == 'random_forest' or model_name == 'rf':
        return create_random_forest(**kwargs)
    elif model_name == 'xgboost' or model_name == 'xgb':
        return create_xgboost(**kwargs)
    elif model_name == 'lightgbm' or model_name == 'lgbm':
        return create_lightgbm(**kwargs)
    elif model_name == 'catboost' or model_name == 'cat':
        return create_catboost(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def calculate_class_weight(y_train):
    """不均衡データ用のクラス重みを計算"""
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return dict(zip(classes, class_weights))

