"""
離職率予測モデル - メイン学習スクリプト
精度向上のための改善を実装
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import config
import preprocessing as prep
import models
import evaluation as eval_module

# SMOTE用（オプション）
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("SMOTE is not available. Install with: pip install imbalanced-learn")

def apply_smote(X_train, y_train):
    """SMOTEを適用して不均衡データを補正"""
    if not SMOTE_AVAILABLE:
        print("SMOTE is not available. Using class_weight instead.")
        return X_train, y_train
    
    if not config.USE_SMOTE:
        return X_train, y_train
    
    print("SMOTEを適用中...")
    smote = SMOTE(sampling_strategy=config.SMOTE_RATIO, random_state=config.RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"SMOTE適用前: {len(X_train)} samples")
    print(f"SMOTE適用後: {len(X_train_resampled)} samples")
    print(f"  クラス分布: {pd.Series(y_train_resampled).value_counts().to_dict()}")
    
    return X_train_resampled, y_train_resampled


def train_model(X_train, y_train, model_name='random_forest', use_grid_search=False):
    """モデルを学習"""
    print(f"\n{'='*60}")
    print(f"モデル学習: {model_name}")
    print(f"{'='*60}")
    
    # クラス重みの計算（XGBoost用）
    if model_name in ['xgboost', 'xgb']:
        class_weights = models.calculate_class_weight(y_train)
        scale_pos_weight = class_weights[1] / class_weights[0] if 0 in class_weights and 1 in class_weights else 1
        model = models.get_model(model_name, scale_pos_weight=scale_pos_weight)
    else:
        model = models.get_model(model_name)
    
    # ハイパーパラメータチューニング
    if use_grid_search and model_name == 'random_forest':
        print("グリッドサーチを実行中...")
        model = models.tune_random_forest(
            X_train, y_train,
            cv=config.GRID_SEARCH_CV,
            scoring=config.GRID_SEARCH_SCORING
        )
    else:
        # 通常の学習
        print("モデルを学習中...")
        model.fit(X_train, y_train)
    
    return model

def cross_validate_model(model, X_train, y_train):
    """クロスバリデーションを実行"""
    if not config.USE_CROSS_VALIDATION:
        return None
    
    print(f"\n{'='*60}")
    print("クロスバリデーション実行中...")
    print(f"{'='*60}")
    
    cv_scores = {}
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    
    for metric in config.CV_SCORING:
        try:
            scores = cross_val_score(model, X_train, y_train, cv=skf, scoring=metric, n_jobs=-1)
            cv_scores[metric] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
            print(f"{metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        except Exception as e:
            print(f"{metric}の計算に失敗: {e}")
    
    return cv_scores

def main():
    """メイン処理"""
    print("="*60)
    print("離職率予測モデル - 学習開始")
    print("="*60)
    
    # 1. データの読み込み
    print("\n1. データの読み込み...")
    df = prep.load_data()
    print(f"   データ形状: {df.shape}")
    
    # 2. 前処理
    print("\n2. 前処理を実行中...")
    if config.YEAR_COL in df.columns:
        # Yearカラムがある場合: 時系列分割
        # まず前処理を学習データでfit
        df_train_raw = df[df[config.YEAR_COL] == config.TRAIN_YEAR].copy()
        df_test_raw = df[df[config.YEAR_COL] == config.TEST_YEAR].copy()
        
        X_train, y_train, fit_encoders = prep.preprocess_data(df_train_raw, is_train=True)
        X_test, y_test, _ = prep.preprocess_data(df_test_raw, fit_encoders=fit_encoders, is_train=False)
    else:
        # Yearカラムがない場合: ランダム分割
        X, y, fit_encoders = prep.preprocess_data(df, is_train=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
        )
    
    print(f"   学習データ: {X_train.shape}")
    print(f"   テストデータ: {X_test.shape}")
    print(f"   特徴量数: {X_train.shape[1]}")
    
    # 3. SMOTEの適用（オプション）
    if config.USE_SMOTE:
        X_train, y_train = apply_smote(X_train, y_train)
    
    # 4. モデルの学習
    model_name = 'random_forest'  # デフォルトモデル
    use_grid_search = config.USE_GRID_SEARCH
    
    model = train_model(X_train, y_train, model_name=model_name, use_grid_search=use_grid_search)
    
    # 5. クロスバリデーション（オプション）
    cv_scores = cross_validate_model(model, X_train, y_train)
    
    # 6. 評価
    print(f"\n{'='*60}")
    print("モデル評価")
    print(f"{'='*60}")
    
    feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
    metrics, y_pred, y_pred_proba = eval_module.evaluate_model(
        model, X_test, y_test, feature_names=feature_names, save_plots=config.SAVE_PLOTS
    )
    
    # 7. 結果の表示
    print(f"\n{'='*60}")
    print("最終結果")
    print(f"{'='*60}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print(f"\n{'='*60}")
    print("学習完了")
    print(f"{'='*60}")
    
    return model, metrics, cv_scores

if __name__ == '__main__':
    model, metrics, cv_scores = main()
