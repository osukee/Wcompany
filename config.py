"""
設定ファイル
モデルのハイパーパラメータや前処理の設定を管理
"""

# データ設定
DATA_FILE = 'data.csv'
TARGET_COL = 'Attrition'
YEAR_COL = 'Year'

# 前処理設定
DROP_COLS = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']

# エンコーディング設定
USE_ONE_HOT_ENCODING = True  # True: One-Hot Encoding, False: Label Encoding
ENCODING_THRESHOLD = 10  # この値以下のユニーク値のカテゴリ変数はOne-Hot Encoding

# 特徴量スケーリング
USE_SCALING = False  # RandomForestはスケーリング不要だが、他のモデル用に設定可能

# 特徴量選択
FEATURE_SELECTION = True
CORRELATION_THRESHOLD = 0.95  # この値以上の相関がある特徴量ペアから1つを削除
FEATURE_IMPORTANCE_THRESHOLD = 0.001  # この値以下の重要度の特徴量を削除

# 不均衡データ対応
USE_SMOTE = False  # True: SMOTEを使用, False: class_weightを使用
SMOTE_RATIO = 0.5  # 少数クラスのサンプル数を多数クラスの何倍にするか

# データ分割設定
TEST_SIZE = 0.2
RANDOM_STATE = 42
TRAIN_YEAR = 2023
TEST_YEAR = 2024

# RandomForest設定
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

# XGBoost設定
XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 1,  # 不均衡データ用（自動計算される）
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss'
}

# LightGBM設定
LGBM_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# CatBoost設定
CATBOOST_PARAMS = {
    'iterations': 200,
    'depth': 6,
    'learning_rate': 0.1,
    'random_state': 42,
    'verbose': False,
    'class_weights': 'Balanced'
}

# ハイパーパラメータチューニング設定
USE_GRID_SEARCH = False  # True: グリッドサーチを使用
GRID_SEARCH_CV = 5  # クロスバリデーションの分割数
GRID_SEARCH_SCORING = 'f1'  # 評価指標

# RandomForest用グリッドサーチパラメータ
RF_GRID_PARAMS = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

# クロスバリデーション設定
USE_CROSS_VALIDATION = False
CV_FOLDS = 5
CV_SCORING = ['accuracy', 'f1', 'roc_auc']

# 評価指標設定
EVALUATION_METRICS = [
    'accuracy',
    'f1',
    'precision',
    'recall',
    'roc_auc',
    'pr_auc',
    'specificity'
]

# 可視化設定
SAVE_PLOTS = True
PLOT_DPI = 150
PLOT_FORMAT = 'png'

# 出力ファイル設定
METRICS_FILE = 'metrics.txt'
CONFUSION_MATRIX_FILE = 'confusion_matrix.png'
FEATURE_IMPORTANCE_FILE = 'feature_importance.png'
ROC_CURVE_FILE = 'roc_curve.png'
PR_CURVE_FILE = 'pr_curve.png'

