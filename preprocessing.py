"""
前処理モジュール
データの前処理、特徴量エンジニアリング、特徴量選択を実装
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import config

def load_data(filepath=None):
    """データを読み込む"""
    if filepath is None:
        filepath = config.DATA_FILE
    df = pd.read_csv(filepath)
    return df

def drop_unnecessary_columns(df, drop_cols=None):
    """不要なカラムを削除"""
    if drop_cols is None:
        drop_cols = config.DROP_COLS
    df = df.drop(columns=drop_cols, errors='ignore')
    return df

def encode_target(df, target_col=None):
    """ターゲット変数をエンコーディング"""
    if target_col is None:
        target_col = config.TARGET_COL
    if target_col in df.columns:
        if df[target_col].dtype == 'object':
            df[target_col] = df[target_col].map({'Yes': 1, 'No': 0})
    return df

def create_new_features(df):
    """新規特徴量の作成"""
    df = df.copy()
    
    # 年齢と勤続年数の比率
    if 'Age' in df.columns and 'YearsAtCompany' in df.columns:
        df['AgeToYearsRatio'] = df['Age'] / (df['YearsAtCompany'] + 1)
    
    # 給与とパフォーマンス指標の比率
    if 'MonthlyIncome' in df.columns and 'PerformanceIndex' in df.columns:
        df['IncomeToPerformanceRatio'] = df['MonthlyIncome'] / (df['PerformanceIndex'] + 1)
    
    # 昇進からの年数と勤続年数の比率
    if 'YearsSinceLastPromotion' in df.columns and 'YearsAtCompany' in df.columns:
        df['PromotionRatio'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
    
    # ストレス関連の特徴量
    if 'StressRating' in df.columns and 'StressSelfReported' in df.columns:
        df['StressDifference'] = df['StressRating'] - df['StressSelfReported']
    
    return df

def encode_categorical_features(df, target_col=None, fit_encoders=None):
    """カテゴリ変数のエンコーディング"""
    if target_col is None:
        target_col = config.TARGET_COL
    
    df = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    if fit_encoders is None:
        fit_encoders = {}
    
    if config.USE_ONE_HOT_ENCODING:
        # One-Hot Encoding
        for col in categorical_cols:
            if col in df.columns:
                unique_count = df[col].nunique()
                if unique_count <= config.ENCODING_THRESHOLD:
                    # One-Hot Encoding
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = pd.concat([df, dummies], axis=1)
                    df = df.drop(columns=[col])
                else:
                    # ユニーク値が多い場合はLabel Encoding
                    if col not in fit_encoders:
                        fit_encoders[col] = LabelEncoder()
                        df[col] = fit_encoders[col].fit_transform(df[col].astype(str))
                    else:
                        # テストデータ用
                        df[col] = fit_encoders[col].transform(df[col].astype(str))
    else:
        # Label Encoding
        for col in categorical_cols:
            if col in df.columns:
                if col not in fit_encoders:
                    fit_encoders[col] = LabelEncoder()
                    df[col] = fit_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = fit_encoders[col].transform(df[col].astype(str))
    
    return df, fit_encoders

def remove_high_correlation_features(df, threshold=None):
    """高相関特徴量の削除"""
    if threshold is None:
        threshold = config.CORRELATION_THRESHOLD
    
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        return df, []
    
    corr_matrix = df[numeric_cols].corr().abs()
    
    # 上三角行列を取得
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # 高相関ペアを見つける
    to_drop = []
    for col in upper_triangle.columns:
        if col in df.columns:
            high_corr_cols = upper_triangle.index[upper_triangle[col] > threshold].tolist()
            if high_corr_cols:
                # 最初の1つを削除対象に追加
                to_drop.extend(high_corr_cols[:1])
    
    to_drop = list(set(to_drop))
    df = df.drop(columns=to_drop, errors='ignore')
    
    return df, to_drop

def select_features_by_importance(X, y, model, threshold=None):
    """特徴量重要度に基づく特徴量選択"""
    if threshold is None:
        threshold = config.FEATURE_IMPORTANCE_THRESHOLD
    
    # モデルを学習
    model.fit(X, y)
    
    # 重要度が閾値以下の特徴量を削除
    importances = model.feature_importances_
    feature_names = X.columns if hasattr(X, 'columns') else range(X.shape[1])
    
    important_features = []
    for i, (name, importance) in enumerate(zip(feature_names, importances)):
        if importance > threshold:
            important_features.append(i)
    
    return important_features

def scale_features(X_train, X_test=None, scaler=None):
    """特徴量のスケーリング"""
    if not config.USE_SCALING:
        return X_train, X_test, None
    
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = None
    
    # DataFrameに戻す
    if hasattr(X_train, 'columns'):
        X_train_scaled = pd.DataFrame(
            X_train_scaled,
            columns=X_train.columns,
            index=X_train.index
        )
        if X_test_scaled is not None:
            X_test_scaled = pd.DataFrame(
                X_test_scaled,
                columns=X_test.columns,
                index=X_test.index
            )
    
    return X_train_scaled, X_test_scaled, scaler

def preprocess_data(df, target_col=None, fit_encoders=None, is_train=True):
    """データの前処理を実行"""
    if target_col is None:
        target_col = config.TARGET_COL
    
    # コピーを作成
    df = df.copy()
    
    # 不要なカラムを削除
    df = drop_unnecessary_columns(df)
    
    # ターゲット変数をエンコーディング
    df = encode_target(df, target_col)
    
    # 新規特徴量の作成
    df = create_new_features(df)
    
    # カテゴリ変数のエンコーディング
    df, fit_encoders = encode_categorical_features(df, target_col, fit_encoders)
    
    # 高相関特徴量の削除
    if config.FEATURE_SELECTION:
        df, dropped_cols = remove_high_correlation_features(df)
        if is_train and dropped_cols:
            print(f"高相関特徴量を削除: {len(dropped_cols)}個")
    
    # 特徴量とターゲットに分離
    X = df.drop(columns=[target_col], errors='ignore')
    y = df[target_col] if target_col in df.columns else None
    
    return X, y, fit_encoders

def split_data_by_year(df, target_col=None, year_col=None):
    """Yearカラムに基づいてデータを分割"""
    if target_col is None:
        target_col = config.TARGET_COL
    if year_col is None:
        year_col = config.YEAR_COL
    
    if year_col not in df.columns:
        return None, None, None, None
    
    train_mask = df[year_col] == config.TRAIN_YEAR
    test_mask = df[year_col] == config.TEST_YEAR
    
    X_train = df[train_mask].drop(columns=[target_col, year_col], errors='ignore')
    X_test = df[test_mask].drop(columns=[target_col, year_col], errors='ignore')
    y_train = df.loc[train_mask, target_col]
    y_test = df.loc[test_mask, target_col]
    
    return X_train, X_test, y_train, y_test

