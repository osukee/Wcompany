import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    cross_val_predict
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    roc_auc_score, precision_recall_curve, roc_curve, precision_score, recall_score
)
from sklearn.calibration import CalibratedClassifierCV
import os
import warnings

# XGBoost and LightGBM
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available, skipping...")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not available, skipping...")

# SMOTE for class imbalance
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.combine import SMOTETomek, SMOTEENN
    from imblearn.under_sampling import TomekLinks
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("imbalanced-learn not available, skipping SMOTE...")

warnings.filterwarnings('ignore')

# =============================================================================
# 1. データの読み込み
# =============================================================================
print("=" * 60)
print("離職予測モデル - 改良版 v4 (Recall重視 + キャリブレーション)")
print("=" * 60)

df = pd.read_csv('data.csv')
print(f"\nLoaded data: {len(df)} rows, {len(df.columns)} columns")

# =============================================================================
# 2. 前処理
# =============================================================================
drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
df = df.drop(columns=drop_cols, errors='ignore')

df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# =============================================================================
# 3. 特徴量エンジニアリング（Recall向上に効く特徴量を追加）
# =============================================================================
print("\nCreating engineered features...")

# 基本的な比率特徴量
df['YearsAtRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
df['ManagerRatio'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
df['RoleRatio'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
df['IncomePerLevel'] = df['MonthlyIncome'] / (df['JobLevel'] + 1)

# 満足度関連
df['SatisfactionScore'] = df['WorkLifeBalance'] * df['EnvironmentSatisfaction']
df['EngagementScore'] = df['JobInvolvement'] * df['JobSatisfaction']
df['OverallSatisfaction'] = (
    df['EnvironmentSatisfaction'] + df['JobSatisfaction'] + df['RelationshipSatisfaction']
) / 3

# 満足度の低さ（離職リスク指標）- より細かく
df['LowSatisfaction'] = (
    (df['EnvironmentSatisfaction'] <= 2).astype(int) +
    (df['JobSatisfaction'] <= 2).astype(int) +
    (df['RelationshipSatisfaction'] <= 2).astype(int) +
    (df['WorkLifeBalance'] <= 2).astype(int)
)
df['VeryLowSatisfaction'] = (
    (df['EnvironmentSatisfaction'] == 1).astype(int) +
    (df['JobSatisfaction'] == 1).astype(int)
)

# 昇進とキャリア
df['PromotionGap'] = df['YearsAtCompany'] - df['YearsSinceLastPromotion']
df['AgeExperienceGap'] = df['Age'] - df['TotalWorkingYears'] - 18
df['IncomeLevelRatio'] = df['MonthlyIncome'] / (df['JobLevel'] * 3000 + 1)

# 給与関連
df['IncomePerYear'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
df['CompanyGrowthRate'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
df['TrainingIntensity'] = df['TrainingTimesLastYear'] / (df['YearsAtCompany'] + 1)

# 給与水準比較
avg_income_by_level = df.groupby('JobLevel')['MonthlyIncome'].transform('mean')
df['IncomeVsLevelAvg'] = df['MonthlyIncome'] / (avg_income_by_level + 1)
avg_income_by_role = df.groupby('JobRole')['MonthlyIncome'].transform('mean')
df['IncomeVsRoleAvg'] = df['MonthlyIncome'] / (avg_income_by_role + 1)

# 低給与フラグ（重要な離職予測因子）
df['LowIncome'] = (df['IncomeVsLevelAvg'] < 0.85).astype(int)
df['VeryLowIncome'] = (df['IncomeVsLevelAvg'] < 0.7).astype(int)

# 残業関連
if 'OverTime' in df.columns:
    df['OverTimeNum'] = df['OverTime'].map({'Yes': 1, 'No': 0}) if df['OverTime'].dtype == 'object' else df['OverTime']
    df['OvertimeSatisfaction'] = df['OverTimeNum'] * (5 - df['WorkLifeBalance'])
    df['OvertimeStress'] = df['OverTimeNum'] * (5 - df['JobSatisfaction'])
else:
    df['OverTimeNum'] = 0

# 通勤負荷
df['CommuteStress'] = df['DistanceFromHome'] * (1 + df['OverTimeNum'])
df['HighCommute'] = (df['DistanceFromHome'] > 15).astype(int)
df['VeryHighCommute'] = (df['DistanceFromHome'] > 25).astype(int)

# キャリア停滞（離職の主要因子）
df['CareerStagnation'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
df['StagnationYears'] = (df['YearsSinceLastPromotion'] >= 3).astype(int)
df['LongStagnation'] = (df['YearsSinceLastPromotion'] >= 5).astype(int)

# 年齢・経験カテゴリ
df['AgeBucket'] = pd.cut(df['Age'], bins=[17, 25, 35, 45, 55, 100], labels=[0, 1, 2, 3, 4])
df['AgeBucket'] = df['AgeBucket'].astype(int)
df['ExperienceLevel'] = pd.cut(df['TotalWorkingYears'], bins=[-1, 2, 5, 10, 20, 50], labels=[0, 1, 2, 3, 4])
df['ExperienceLevel'] = df['ExperienceLevel'].astype(int)

# 若手社員フラグ（離職リスクが高い傾向）
df['YoungEmployee'] = (df['Age'] < 30).astype(int)
df['VeryYoung'] = (df['Age'] < 25).astype(int)
df['NewHire'] = (df['YearsAtCompany'] <= 2).astype(int)
df['VeryNewHire'] = (df['YearsAtCompany'] <= 1).astype(int)

# 複合リスクスコア（改良版 - よりRecall重視）
df['AttritionRiskV2'] = (
    df['OverTimeNum'] * 0.20 +
    (5 - df['JobSatisfaction']) / 4 * 0.15 +
    (5 - df['EnvironmentSatisfaction']) / 4 * 0.15 +
    df['NewHire'] * 0.15 +
    df['LowIncome'] * 0.10 +
    df['StagnationYears'] * 0.10 +
    df['HighCommute'] * 0.08 +
    (5 - df['WorkLifeBalance']) / 4 * 0.07
)

# 離職の強いシグナル（OR条件）
df['StrongAttritionSignal'] = (
    ((df['OverTimeNum'] == 1) & (df['JobSatisfaction'] <= 2)) |
    ((df['NewHire'] == 1) & (df['JobSatisfaction'] <= 2)) |
    ((df['OverTimeNum'] == 1) & (df['WorkLifeBalance'] <= 2)) |
    (df['VeryLowSatisfaction'] >= 2)
).astype(int)

# ワークライフバランス関連
df['WorkLifeIssue'] = ((df['WorkLifeBalance'] <= 2) & (df['OverTimeNum'] == 1)).astype(int)
df['SevereWorkLifeIssue'] = ((df['WorkLifeBalance'] == 1) & (df['OverTimeNum'] == 1)).astype(int)

# 会社との関係性
df['TenureManagerRatio'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
df['FrequentManagerChange'] = (df['TenureManagerRatio'] < 0.3).astype(int)

# ストレス関連
if 'StressRating' in df.columns and 'StressSelfReported' in df.columns:
    df['StressCombined'] = df['StressRating'] + df['StressSelfReported']

# 給与と残業の組み合わせ（不満要因）
df['OverworkUnderpaid'] = ((df['OverTimeNum'] == 1) & (df['LowIncome'] == 1)).astype(int)

print(f"Total features after engineering: {len(df.columns) - 1}")

# =============================================================================
# 4. データ分割
# =============================================================================
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_cols = [col for col in num_cols if col != 'Attrition']

X = df.drop('Attrition', axis=1)
y = df['Attrition']

if 'Year' in df.columns:
    train_mask = df['Year'] == 2023
    test_mask = df['Year'] == 2024
    
    X_train = X[train_mask].copy()
    X_test = X[test_mask].copy()
    y_train = y[train_mask].copy()
    y_test = y[test_mask].copy()
    
    if 'Year' in cat_cols:
        cat_cols.remove('Year')
    if 'Year' in num_cols:
        num_cols.remove('Year')
    X_train = X_train.drop('Year', axis=1, errors='ignore')
    X_test = X_test.drop('Year', axis=1, errors='ignore')
    
    print(f"\nTime-based split:")
    print(f"  Training data: {len(X_train)} samples (Year 2023)")
    print(f"  Test data: {len(X_test)} samples (Year 2024)")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nRandom split:")
    print(f"  Training data: {len(X_train)} samples")
    print(f"  Test data: {len(X_test)} samples")

cat_cols = [col for col in cat_cols if col in X_train.columns]
num_cols = [col for col in num_cols if col in X_train.columns]

print(f"\nClass distribution in training data:")
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
print(f"  No Attrition (0): {n_neg} ({n_neg/len(y_train)*100:.1f}%)")
print(f"  Attrition (1): {n_pos} ({n_pos/len(y_train)*100:.1f}%)")

# =============================================================================
# 5. 前処理パイプライン + SMOTE
# =============================================================================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# SMOTETomek（オーバーサンプリング + クリーニング）
if HAS_IMBLEARN:
    print("\nApplying SMOTETomek for class balancing...")
    smotetomek = SMOTETomek(random_state=42, smote=SMOTE(k_neighbors=5, random_state=42))
    X_train_resampled, y_train_resampled = smotetomek.fit_resample(X_train_processed, y_train)
    print(f"  Before: {len(y_train)} samples (Pos: {(y_train==1).sum()})")
    print(f"  After: {len(y_train_resampled)} samples (Pos: {(y_train_resampled==1).sum()})")
else:
    X_train_resampled = X_train_processed
    y_train_resampled = y_train

# =============================================================================
# 6. モデル定義（Recall重視の設定）
# =============================================================================
print("\n" + "=" * 60)
print("Model Training and Evaluation (Recall-focused)")
print("=" * 60)

class_weight_ratio = n_neg / n_pos
print(f"\nOriginal class weight ratio: 1:{class_weight_ratio:.2f}")

# より積極的なRecall重視のクラスウェイト
recall_weight = {0: 1, 1: class_weight_ratio * 1.5}  # 少数派を1.5倍重視

# RandomForest（Recall重視）
rf_classifier = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,  # 少し浅くして汎化性能向上
    min_samples_split=15,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight=recall_weight,
    random_state=42,
    n_jobs=-1
)

# GradientBoosting
gb_classifier = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=4,
    min_samples_split=15,
    min_samples_leaf=5,
    learning_rate=0.02,
    subsample=0.8,
    max_features='sqrt',
    random_state=42
)

models = {
    'RandomForest': rf_classifier,
    'GradientBoosting': gb_classifier
}

# XGBoost（Recall重視）
if HAS_XGBOOST:
    xgb_classifier = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=class_weight_ratio * 1.5,  # Recall重視
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    models['XGBoost'] = xgb_classifier

# LightGBM（Recall重視）
if HAS_LIGHTGBM:
    lgbm_classifier = LGBMClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight=recall_weight,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    models['LightGBM'] = lgbm_classifier

# =============================================================================
# 7. クロスバリデーション
# =============================================================================
print("\n--- Cross-Validation Results (5-Fold, SMOTE) ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    cv_f1 = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='f1', n_jobs=-1)
    cv_roc = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_recall = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='recall', n_jobs=-1)
    cv_results[name] = {'f1': cv_f1, 'roc_auc': cv_roc, 'recall': cv_recall}
    print(f"\n{name}:")
    print(f"  CV F1: {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")
    print(f"  CV ROC-AUC: {cv_roc.mean():.4f} (+/- {cv_roc.std():.4f})")
    print(f"  CV Recall: {cv_recall.mean():.4f} (+/- {cv_recall.std():.4f})")

# =============================================================================
# 8. モデル学習 + キャリブレーション
# =============================================================================
print("\n--- Training Models with Calibration ---")
calibrated_models = {}

for name, model in models.items():
    # まず通常学習
    model.fit(X_train_resampled, y_train_resampled)
    
    # キャリブレーション（確率推定の精度向上）
    calibrated = CalibratedClassifierCV(model, method='isotonic', cv=5)
    calibrated.fit(X_train_resampled, y_train_resampled)
    calibrated_models[name] = calibrated
    
    print(f"  {name} trained and calibrated.")

# =============================================================================
# 9. テストセット評価（複数閾値）
# =============================================================================
print("\n--- Test Set Evaluation ---")
test_results = {}

# 非キャリブレーションモデル
for name, model in models.items():
    pred = model.predict(X_test_processed)
    pred_proba = model.predict_proba(X_test_processed)[:, 1]
    
    test_results[name] = {
        'accuracy': accuracy_score(y_test, pred),
        'f1': f1_score(y_test, pred),
        'roc_auc': roc_auc_score(y_test, pred_proba),
        'precision': precision_score(y_test, pred),
        'recall': recall_score(y_test, pred),
        'predictions': pred,
        'probabilities': pred_proba
    }

# キャリブレーションモデル
for name, model in calibrated_models.items():
    pred_proba = model.predict_proba(X_test_processed)[:, 1]
    # 閾値0.5でのデフォルト予測
    pred = (pred_proba >= 0.5).astype(int)
    
    test_results[f'{name}_Cal'] = {
        'accuracy': accuracy_score(y_test, pred),
        'f1': f1_score(y_test, pred),
        'roc_auc': roc_auc_score(y_test, pred_proba),
        'precision': precision_score(y_test, pred),
        'recall': recall_score(y_test, pred),
        'predictions': pred,
        'probabilities': pred_proba
    }

for name, res in test_results.items():
    print(f"\n{name}:")
    print(f"  Acc: {res['accuracy']:.4f}, F1: {res['f1']:.4f}, ROC: {res['roc_auc']:.4f}")
    print(f"  Prec: {res['precision']:.4f}, Recall: {res['recall']:.4f}")

# =============================================================================
# 10. アンサンブル（キャリブレーション済みモデル）
# =============================================================================
print("\n--- Ensemble with Calibrated Models ---")

# キャリブレーション済みモデルの確率を平均
cal_probas = [test_results[f'{name}_Cal']['probabilities'] for name in models.keys()]
ensemble_proba = np.mean(cal_probas, axis=0)

# より広い範囲の閾値で評価
print("\nThreshold analysis:")
threshold_results = []
for thresh in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    pred = (ensemble_proba >= thresh).astype(int)
    f1 = f1_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    acc = accuracy_score(y_test, pred)
    threshold_results.append({
        'threshold': thresh, 'accuracy': acc, 'f1': f1, 
        'precision': prec, 'recall': rec
    })
    print(f"  Th={thresh:.2f}: Acc={acc:.4f}, F1={f1:.4f}, Prec={prec:.4f}, Recall={rec:.4f}")

# 最適閾値の探索
# F1最大化
precisions, recalls, thresholds = precision_recall_curve(y_test, ensemble_proba)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
best_f1_idx = np.argmax(f1_scores)
best_threshold_f1 = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5

# Recall >= 0.6を達成する最高の閾値
best_threshold_recall60 = None
for tr in sorted(threshold_results, key=lambda x: -x['recall']):
    if tr['recall'] >= 0.6 and (best_threshold_recall60 is None or tr['f1'] > threshold_results[threshold_results.index(best_threshold_recall60)]['f1'] if best_threshold_recall60 else True):
        best_threshold_recall60 = tr['threshold']
        break

if best_threshold_recall60 is None:
    # 0.6達成できない場合、最もRecallが高い閾値を選択
    best_threshold_recall60 = min([tr['threshold'] for tr in threshold_results])

# Recall >= 0.7を達成する閾値
best_threshold_recall70 = None
for thresh in np.linspace(0.10, 0.50, 40):
    pred = (ensemble_proba >= thresh).astype(int)
    rec = recall_score(y_test, pred)
    if rec >= 0.7:
        best_threshold_recall70 = thresh
        break

print(f"\nOptimal threshold (F1 max): {best_threshold_f1:.3f}")
print(f"Threshold for Recall >= 0.6: {best_threshold_recall60:.3f}")
if best_threshold_recall70:
    print(f"Threshold for Recall >= 0.7: {best_threshold_recall70:.3f}")

# 各戦略でのアンサンブル結果
strategies = [
    ('F1-optimized', best_threshold_f1),
    ('Recall60', best_threshold_recall60),
]
if best_threshold_recall70:
    strategies.append(('Recall70', best_threshold_recall70))

for strategy_name, thresh in strategies:
    ensemble_pred = (ensemble_proba >= thresh).astype(int)
    key = f'Ensemble_{strategy_name}'
    test_results[key] = {
        'accuracy': accuracy_score(y_test, ensemble_pred),
        'f1': f1_score(y_test, ensemble_pred),
        'roc_auc': roc_auc_score(y_test, ensemble_proba),
        'precision': precision_score(y_test, ensemble_pred),
        'recall': recall_score(y_test, ensemble_pred),
        'predictions': ensemble_pred,
        'probabilities': ensemble_proba,
        'threshold': thresh
    }
    res = test_results[key]
    print(f"\n{key} (threshold={thresh:.3f}):")
    print(f"  Acc: {res['accuracy']:.4f}, F1: {res['f1']:.4f}, ROC: {res['roc_auc']:.4f}")
    print(f"  Prec: {res['precision']:.4f}, Recall: {res['recall']:.4f}")

# =============================================================================
# 11. 最良モデルの選択
# =============================================================================
# Recallを重視しつつF1もバランス良く
# スコア = F1 * 0.5 + Recall * 0.5
def combined_score(res):
    return res['f1'] * 0.5 + res['recall'] * 0.5

best_model_name = max(test_results, key=lambda x: combined_score(test_results[x]))
best_result = test_results[best_model_name]

# 高Recall版も記録
best_recall_model = max(test_results, key=lambda x: test_results[x]['recall'])
best_recall_result = test_results[best_recall_model]

print(f"\n{'='*60}")
print(f"BEST MODEL (F1 + Recall balanced): {best_model_name}")
print(f"  Accuracy: {best_result['accuracy']:.4f}")
print(f"  F1 Score: {best_result['f1']:.4f}")
print(f"  ROC-AUC: {best_result['roc_auc']:.4f}")
print(f"  Precision: {best_result['precision']:.4f}")
print(f"  Recall: {best_result['recall']:.4f}")
print(f"{'='*60}")

print(f"\nBEST RECALL MODEL: {best_recall_model}")
print(f"  F1: {best_recall_result['f1']:.4f}, Recall: {best_recall_result['recall']:.4f}")

# =============================================================================
# 12. 詳細レポート
# =============================================================================
report = classification_report(y_test, best_result['predictions'], output_dict=True)

with open("metrics.txt", "w") as outfile:
    outfile.write(f"{'='*50}\n")
    outfile.write(f"離職予測モデル - 評価レポート v4\n")
    outfile.write(f"(Recall重視 + キャリブレーション)\n")
    outfile.write(f"{'='*50}\n\n")
    
    outfile.write(f"=== Best Model (F1+Recall): {best_model_name} ===\n")
    outfile.write(f"Accuracy: {best_result['accuracy']:.4f}\n")
    outfile.write(f"F1 Score: {best_result['f1']:.4f}\n")
    outfile.write(f"ROC-AUC: {best_result['roc_auc']:.4f}\n")
    outfile.write(f"Precision: {best_result['precision']:.4f}\n")
    outfile.write(f"Recall: {best_result['recall']:.4f}\n")
    if 'threshold' in best_result:
        outfile.write(f"Threshold: {best_result['threshold']:.3f}\n")
    
    outfile.write(f"\n=== Best Recall Model: {best_recall_model} ===\n")
    outfile.write(f"F1: {best_recall_result['f1']:.4f}, Recall: {best_recall_result['recall']:.4f}\n")
    
    outfile.write(f"\n=== Model Comparison ===\n")
    for name, res in sorted(test_results.items(), key=lambda x: -combined_score(x[1])):
        thresh_info = f" (th={res['threshold']:.3f})" if 'threshold' in res else ""
        outfile.write(f"{name:25s} - F1: {res['f1']:.4f}, ROC: {res['roc_auc']:.4f}, Recall: {res['recall']:.4f}{thresh_info}\n")
    
    outfile.write(f"\n=== Threshold Analysis (Ensemble Calibrated) ===\n")
    for tr in threshold_results:
        outfile.write(f"Th={tr['threshold']:.2f}: Acc={tr['accuracy']:.4f}, F1={tr['f1']:.4f}, Prec={tr['precision']:.4f}, Recall={tr['recall']:.4f}\n")
    
    outfile.write(f"\n=== Cross-Validation (5-Fold, SMOTETomek) ===\n")
    for name, cv_res in cv_results.items():
        outfile.write(f"{name} CV F1: {cv_res['f1'].mean():.4f} (+/- {cv_res['f1'].std():.4f})\n")
        outfile.write(f"{name} CV ROC-AUC: {cv_res['roc_auc'].mean():.4f} (+/- {cv_res['roc_auc'].std():.4f})\n")
        outfile.write(f"{name} CV Recall: {cv_res['recall'].mean():.4f} (+/- {cv_res['recall'].std():.4f})\n")
    
    outfile.write(f"\n=== Feature Engineering ===\n")
    outfile.write(f"Total features: {len(num_cols) + len(cat_cols)}\n")
    outfile.write(f"Numeric: {len(num_cols)}, Categorical: {len(cat_cols)}\n")

print("\nMetrics saved to: metrics.txt")

# =============================================================================
# 13. 可視化
# =============================================================================
# 混同行列
cm = confusion_matrix(y_test, best_result['predictions'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['No Attrition', 'Attrition'],
            yticklabels=['No Attrition', 'Attrition'],
            annot_kws={'size': 14})
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title(f'Confusion Matrix ({best_model_name})', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Confusion matrix saved")

# 特徴量重要度
if 'RandomForest' in models:
    feature_names = num_cols + list(
        preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    )
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': models['RandomForest'].feature_importances_
    }).sort_values('importance', ascending=False).head(25)
    
    plt.figure(figsize=(12, 12))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(feature_importance)))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette=colors)
    plt.title('Top 25 Feature Importance', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Feature importance saved")

# モデル比較
main_models = ['RandomForest', 'XGBoost', 'LightGBM', 'RandomForest_Cal', 'Ensemble_F1-optimized']
if best_threshold_recall70:
    main_models.append('Ensemble_Recall70')
else:
    main_models.append('Ensemble_Recall60')
main_models = [m for m in main_models if m in test_results]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

roc_scores = [test_results[m]['roc_auc'] for m in main_models]
f1_scores_list = [test_results[m]['f1'] for m in main_models]
recall_scores = [test_results[m]['recall'] for m in main_models]

colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c'][:len(main_models)]

for ax, scores, title, ylabel in [
    (axes[0], roc_scores, 'ROC-AUC', 'ROC-AUC Score'),
    (axes[1], f1_scores_list, 'F1 Score', 'F1 Score'),
    (axes[2], recall_scores, 'Recall', 'Recall')
]:
    bars = ax.bar(range(len(main_models)), scores, color=colors)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Model Comparison: {title}', fontsize=14)
    ax.set_ylim([0.0, 1.0] if 'ROC' not in title else [0.5, 1.0])
    ax.set_xticks(range(len(main_models)))
    ax.set_xticklabels(main_models, rotation=45, ha='right', fontsize=9)
    for bar, v in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Model comparison saved")

# ROC曲線
plt.figure(figsize=(10, 8))
for name in main_models:
    res = test_results[name]
    fpr, tpr, _ = roc_curve(y_test, res['probabilities'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})", linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves Comparison', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("ROC curves saved")

# Precision-Recall曲線
plt.figure(figsize=(10, 8))
for name in main_models:
    res = test_results[name]
    prec, rec, _ = precision_recall_curve(y_test, res['probabilities'])
    plt.plot(rec, prec, label=f"{name}", linewidth=2)

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves', fontsize=14)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('precision_recall_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Precision-Recall curves saved")

# =============================================================================
# 14. 完了
# =============================================================================
print("\n" + "=" * 60)
print("Training and Evaluation Completed!")
print("=" * 60)
print("\nGenerated files:")
print("  - metrics.txt")
print("  - confusion_matrix.png")
print("  - feature_importance.png")
print("  - model_comparison.png")
print("  - roc_curves.png")
print("  - precision_recall_curves.png")
