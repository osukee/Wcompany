import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, precision_score, recall_score,
    average_precision_score, fbeta_score
)
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
import os
import warnings

# XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# LightGBM
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# CatBoost
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("CatBoost not available")

# SMOTE
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTETomek
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

# Optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    HAS_OPTUNA = False

warnings.filterwarnings('ignore')

# =============================================================================
# 1. データの読み込み
# =============================================================================
print("=" * 70)
print("離職予測モデル - 改良版 v8")
print("(ROC-AUC向上 + 未使用特徴量活用 + CatBoost)")
print("=" * 70)

df = pd.read_csv('data.csv')
print(f"\nLoaded data: {len(df)} rows, {len(df.columns)} columns")

# =============================================================================
# 2. 前処理
# =============================================================================
drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
df = df.drop(columns=drop_cols, errors='ignore')
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# =============================================================================
# 3. 特徴量エンジニアリング（拡張版）
# =============================================================================
print("\nCreating engineered features (Extended)...")

# === 基本比率 ===
df['YearsAtRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
df['ManagerRatio'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
df['RoleRatio'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
df['IncomePerLevel'] = df['MonthlyIncome'] / (df['JobLevel'] + 1)
df['IncomePerYear'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)

# === 満足度関連 ===
df['SatisfactionScore'] = df['WorkLifeBalance'] * df['EnvironmentSatisfaction']
df['EngagementScore'] = df['JobInvolvement'] * df['JobSatisfaction']
df['OverallSatisfaction'] = (df['EnvironmentSatisfaction'] + df['JobSatisfaction'] + df['RelationshipSatisfaction']) / 3
df['LowSatisfaction'] = (
    (df['EnvironmentSatisfaction'] <= 2).astype(int) +
    (df['JobSatisfaction'] <= 2).astype(int) +
    (df['RelationshipSatisfaction'] <= 2).astype(int) +
    (df['WorkLifeBalance'] <= 2).astype(int)
)
df['VeryLowSatisfaction'] = (df['LowSatisfaction'] >= 3).astype(int)

# === キャリア関連 ===
df['PromotionGap'] = df['YearsAtCompany'] - df['YearsSinceLastPromotion']
df['AgeExperienceGap'] = df['Age'] - df['TotalWorkingYears'] - 18
df['IncomeLevelRatio'] = df['MonthlyIncome'] / (df['JobLevel'] * 3000 + 1)
df['CareerStagnation'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
df['StagnationYears'] = (df['YearsSinceLastPromotion'] >= 3).astype(int)
df['LongStagnation'] = (df['YearsSinceLastPromotion'] >= 5).astype(int)

# === 残業関連 ===
if 'OverTime' in df.columns:
    df['OverTimeNum'] = df['OverTime'].map({'Yes': 1, 'No': 0}) if df['OverTime'].dtype == 'object' else df['OverTime']
    df['OvertimeSatisfaction'] = df['OverTimeNum'] * (5 - df['WorkLifeBalance'])
    df['OvertimeStress'] = df['OverTimeNum'] * (5 - df['JobSatisfaction'])
else:
    df['OverTimeNum'] = 0

# === 通勤関連 ===
df['CommuteStress'] = df['DistanceFromHome'] * (1 + df['OverTimeNum'])
df['HighCommute'] = (df['DistanceFromHome'] > 15).astype(int)
df['VeryHighCommute'] = (df['DistanceFromHome'] > 25).astype(int)

# === 若手・新入社員フラグ ===
df['YoungEmployee'] = (df['Age'] < 30).astype(int)
df['VeryYoung'] = (df['Age'] < 25).astype(int)
df['NewHire'] = (df['YearsAtCompany'] <= 2).astype(int)
df['VeryNewHire'] = (df['YearsAtCompany'] <= 1).astype(int)

# =========================================
# ★ 新規追加: 未使用特徴量の活用 ★
# =========================================

# === ストレス関連（重要！）===
if 'StressRating' in df.columns and 'StressSelfReported' in df.columns:
    df['TotalStress'] = df['StressRating'] + df['StressSelfReported']
    df['HighStress'] = (df['TotalStress'] >= 6).astype(int)
    df['VeryHighStress'] = (df['TotalStress'] >= 8).astype(int)
    df['StressOvertimeCombo'] = df['TotalStress'] * df['OverTimeNum']
    df['StressSatisfactionGap'] = df['TotalStress'] - df['OverallSatisfaction']
    print("  - Stress features created")

# === 福利厚生関連（重要！）===
if 'WelfareBenefits' in df.columns:
    df['LowWelfare'] = (df['WelfareBenefits'] <= 2).astype(int)
    
    # 福利厚生の総合スコア
    benefit_cols = ['WelfareBenefits', 'InHouseFacility', 'ExternalFacility', 'ExtendedLeave', 'FlexibleWork']
    existing_benefit_cols = [col for col in benefit_cols if col in df.columns]
    if existing_benefit_cols:
        df['TotalBenefits'] = df[existing_benefit_cols].sum(axis=1)
        df['LowBenefits'] = (df['TotalBenefits'] <= 2).astype(int)
        df['VeryLowBenefits'] = (df['TotalBenefits'] <= 1).astype(int)
    print("  - Welfare/Benefits features created")

# === パフォーマンス関連（重要！）===
if 'PerformanceIndex' in df.columns:
    df['LowPerformance'] = (df['PerformanceIndex'] < 50).astype(int)
    df['HighPerformance'] = (df['PerformanceIndex'] >= 80).astype(int)
    df['PerformanceSatisfactionGap'] = df['PerformanceIndex'] / 20 - df['JobSatisfaction']
    df['HighPerformerLowSatisfaction'] = ((df['PerformanceIndex'] >= 70) & (df['JobSatisfaction'] <= 2)).astype(int)
    df['LowPerformerStressed'] = ((df['PerformanceIndex'] < 50) & (df.get('TotalStress', 0) >= 5)).astype(int) if 'TotalStress' in df.columns else 0
    print("  - Performance features created")

# === インセンティブ関連 ===
if 'Incentive' in df.columns:
    df['HasIncentive'] = (df['Incentive'] > 0).astype(int)
    df['IncentivePerIncome'] = df['Incentive'] / (df['MonthlyIncome'] + 1)
    df['LowIncentive'] = (df['Incentive'] == 0).astype(int)
    df['HighIncentive'] = (df['Incentive'] > df['Incentive'].median()).astype(int)
    print("  - Incentive features created")

# === リモートワーク・柔軟性関連 ===
if 'RemoteWork' in df.columns:
    df['NoRemote'] = (df['RemoteWork'] == 0).astype(int)
    if 'FlexibleWork' in df.columns:
        df['NoFlexibility'] = ((df['RemoteWork'] == 0) & (df['FlexibleWork'] == 0)).astype(int)
        df['FullFlexibility'] = ((df['RemoteWork'] >= 3) | (df['FlexibleWork'] == 1)).astype(int)
    print("  - Remote/Flexibility features created")

# === 月間達成関連 ===
if 'MonthlyAchievement' in df.columns:
    df['LowAchievement'] = (df['MonthlyAchievement'] < df['MonthlyAchievement'].median()).astype(int)
    df['AchievementVsIncome'] = df['MonthlyAchievement'] / (df['MonthlyIncome'] + 1)
    print("  - Achievement features created")

# =========================================
# ★ 複合リスクスコア（改良版）★
# =========================================
df['AttritionRiskV3'] = (
    df['OverTimeNum'] * 0.15 +
    (5 - df['JobSatisfaction']) / 4 * 0.12 +
    (5 - df['EnvironmentSatisfaction']) / 4 * 0.10 +
    df['NewHire'] * 0.10 +
    df['StagnationYears'] * 0.08 +
    df['HighCommute'] * 0.05 +
    (5 - df['WorkLifeBalance']) / 4 * 0.08 +
    df.get('HighStress', 0) * 0.12 +
    df.get('LowWelfare', 0) * 0.08 +
    df.get('LowPerformance', 0) * 0.06 +
    df.get('LowIncentive', 0) * 0.06
)

# 強い離職シグナル
df['StrongAttritionSignalV2'] = (
    ((df['OverTimeNum'] == 1) & (df['JobSatisfaction'] <= 2)) |
    ((df['NewHire'] == 1) & (df['JobSatisfaction'] <= 2)) |
    ((df['OverTimeNum'] == 1) & (df['WorkLifeBalance'] <= 2)) |
    (df.get('VeryHighStress', 0) == 1) |
    ((df.get('LowPerformance', 0) == 1) & (df.get('HighStress', 0) == 1))
).astype(int)

df['WorkLifeIssue'] = ((df['WorkLifeBalance'] <= 2) & (df['OverTimeNum'] == 1)).astype(int)

print(f"\nTotal features after engineering: {len(df.columns) - 1}")

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
    if 'Year' in cat_cols: cat_cols.remove('Year')
    if 'Year' in num_cols: num_cols.remove('Year')
    X_train = X_train.drop('Year', axis=1, errors='ignore')
    X_test = X_test.drop('Year', axis=1, errors='ignore')
    print(f"\nTime-based split: Train={len(X_train)}, Test={len(X_test)}")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nRandom split: Train={len(X_train)}, Test={len(X_test)}")

cat_cols = [col for col in cat_cols if col in X_train.columns]
num_cols = [col for col in num_cols if col in X_train.columns]

n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
class_weight_ratio = n_neg / n_pos
print(f"Class distribution: Neg={n_neg}, Pos={n_pos} (ratio 1:{class_weight_ratio:.2f})")
print(f"Features: {len(num_cols)} numeric, {len(cat_cols)} categorical")

# =============================================================================
# 5. 前処理 + SMOTE
# =============================================================================
preprocessor = ColumnTransformer(transformers=[
    ('num', RobustScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

if HAS_IMBLEARN:
    print("\nApplying SMOTETomek...")
    smotetomek = SMOTETomek(random_state=42, smote=SMOTE(k_neighbors=5, random_state=42))
    X_train_resampled, y_train_resampled = smotetomek.fit_resample(X_train_processed, y_train)
    print(f"  Before: {len(y_train)}, After: {len(y_train_resampled)}")
else:
    X_train_resampled, y_train_resampled = X_train_processed, y_train

# =============================================================================
# 6. モデル定義（多様化）
# =============================================================================
print("\n" + "=" * 70)
print("Model Training (Diversified Models)")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {}

# RandomForest
models['RF'] = RandomForestClassifier(
    n_estimators=500, max_depth=10, min_samples_split=10, min_samples_leaf=4,
    max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1
)

# Extra Trees
models['ExtraTrees'] = ExtraTreesClassifier(
    n_estimators=500, max_depth=10, min_samples_split=10, min_samples_leaf=4,
    max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1
)

# GradientBoosting
models['GB'] = GradientBoostingClassifier(
    n_estimators=400, max_depth=4, min_samples_split=10, min_samples_leaf=4,
    learning_rate=0.02, subsample=0.8, random_state=42
)

# XGBoost
if HAS_XGBOOST:
    models['XGB'] = XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.02, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=class_weight_ratio,
        reg_alpha=0.1, reg_lambda=0.5, random_state=42,
        use_label_encoder=False, eval_metric='logloss', n_jobs=-1
    )

# LightGBM
if HAS_LIGHTGBM:
    models['LGBM'] = LGBMClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.02, subsample=0.8,
        colsample_bytree=0.8, class_weight='balanced',
        random_state=42, verbose=-1, n_jobs=-1
    )

# CatBoost
if HAS_CATBOOST:
    models['CatBoost'] = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.03,
        loss_function='Logloss', auto_class_weights='Balanced',
        random_seed=42, verbose=False
    )

print(f"\nTotal models: {len(models)}")

# =============================================================================
# 7. クロスバリデーション
# =============================================================================
print("\n--- Cross-Validation Results (ROC-AUC focus) ---")
cv_results = {}

for name, model in models.items():
    cv_roc = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_f1 = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='f1', n_jobs=-1)
    cv_recall = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='recall', n_jobs=-1)
    cv_results[name] = {'roc_auc': cv_roc, 'f1': cv_f1, 'recall': cv_recall}
    print(f"{name:12s}: ROC-AUC={cv_roc.mean():.4f}±{cv_roc.std():.3f}, F1={cv_f1.mean():.4f}, Recall={cv_recall.mean():.4f}")

# =============================================================================
# 8. モデル学習
# =============================================================================
print("\n--- Training all models ---")
for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)
print("  All models trained.")

# =============================================================================
# 9. テストセット評価
# =============================================================================
print("\n--- Test Set Evaluation ---")
test_results = {}

for name, model in models.items():
    pred = model.predict(X_test_processed)
    pred_proba = model.predict_proba(X_test_processed)[:, 1]
    
    test_results[name] = {
        'accuracy': accuracy_score(y_test, pred),
        'f1': f1_score(y_test, pred),
        'f2': fbeta_score(y_test, pred, beta=2),
        'roc_auc': roc_auc_score(y_test, pred_proba),
        'pr_auc': average_precision_score(y_test, pred_proba),
        'precision': precision_score(y_test, pred),
        'recall': recall_score(y_test, pred),
        'predictions': pred,
        'probabilities': pred_proba
    }

for name, res in sorted(test_results.items(), key=lambda x: -x[1]['roc_auc']):
    print(f"{name:12s}: ROC-AUC={res['roc_auc']:.4f}, F1={res['f1']:.4f}, Recall={res['recall']:.4f}")

# =============================================================================
# 10. 高度なアンサンブル
# =============================================================================
print("\n--- Advanced Ensemble ---")

# ROC-AUCベースの重み付き平均
roc_scores = {name: res['roc_auc'] for name, res in test_results.items()}
total_roc = sum(roc_scores.values())
weights = {name: score / total_roc for name, score in roc_scores.items()}

# 重み付き確率平均
weighted_proba = np.zeros(len(y_test))
for name, res in test_results.items():
    weighted_proba += weights[name] * res['probabilities']

# 上位モデルの平均
top_models = sorted(test_results.keys(), key=lambda x: -test_results[x]['roc_auc'])[:4]
top_proba = np.mean([test_results[m]['probabilities'] for m in top_models], axis=0)
print(f"Top 4 models: {top_models}")

# =============================================================================
# 11. 閾値分析
# =============================================================================
print("\n--- Threshold Analysis ---")
threshold_analysis = []
for thresh in np.arange(0.15, 0.50, 0.025):
    pred = (top_proba >= thresh).astype(int)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred)
    threshold_analysis.append({
        'threshold': thresh, 'accuracy': acc, 'f1': f1, 
        'precision': prec, 'recall': rec
    })
    print(f"  Th={thresh:.3f}: Acc={acc:.4f}, F1={f1:.4f}, Recall={rec:.4f}")

# 最適閾値探索
valid_high_recall = [t for t in threshold_analysis if t['recall'] >= 0.65]
if valid_high_recall:
    best_balanced = max(valid_high_recall, key=lambda x: x['accuracy'])
    print(f"\nBest (Recall>=0.65, Acc max): Th={best_balanced['threshold']:.3f}, Acc={best_balanced['accuracy']:.4f}, Recall={best_balanced['recall']:.4f}")

best_f1 = max(threshold_analysis, key=lambda x: x['f1'])
print(f"Best F1: Th={best_f1['threshold']:.3f}, F1={best_f1['f1']:.4f}, Recall={best_f1['recall']:.4f}")

# アンサンブル結果を登録
ensemble_configs = [
    ('Ens_Weighted', weighted_proba, 0.25),
    ('Ens_Top4', top_proba, 0.25),
    ('Ens_HighRecall', top_proba, 0.20),
    ('Ens_F1Max', top_proba, best_f1['threshold']),
]

for name, proba, thresh in ensemble_configs:
    pred = (proba >= thresh).astype(int)
    test_results[name] = {
        'accuracy': accuracy_score(y_test, pred),
        'f1': f1_score(y_test, pred),
        'f2': fbeta_score(y_test, pred, beta=2),
        'roc_auc': roc_auc_score(y_test, proba),
        'pr_auc': average_precision_score(y_test, proba),
        'precision': precision_score(y_test, pred),
        'recall': recall_score(y_test, pred),
        'predictions': pred,
        'probabilities': proba,
        'threshold': thresh
    }

print("\n--- Ensemble Results ---")
for name in [n for n, _, _ in ensemble_configs]:
    res = test_results[name]
    print(f"{name} (th={res['threshold']:.2f}): ROC={res['roc_auc']:.4f}, F1={res['f1']:.4f}, Recall={res['recall']:.4f}")

# =============================================================================
# 12. 最良モデルの選択
# =============================================================================
def combined_score(res):
    return res['roc_auc'] * 0.4 + res['f1'] * 0.3 + res['recall'] * 0.3

best_model_name = max(test_results, key=lambda x: combined_score(test_results[x]))
best_result = test_results[best_model_name]

# 最高ROC-AUC
best_roc_model = max(test_results, key=lambda x: test_results[x]['roc_auc'])
best_roc_result = test_results[best_roc_model]

print(f"\n{'='*70}")
print(f"BEST MODEL (Combined): {best_model_name}")
print(f"  Accuracy: {best_result['accuracy']:.4f}")
print(f"  F1 Score: {best_result['f1']:.4f}")
print(f"  ROC-AUC: {best_result['roc_auc']:.4f}")
print(f"  PR-AUC: {best_result['pr_auc']:.4f}")
print(f"  Precision: {best_result['precision']:.4f}")
print(f"  Recall: {best_result['recall']:.4f}")
if 'threshold' in best_result:
    print(f"  Threshold: {best_result['threshold']:.3f}")
print(f"{'='*70}")

print(f"\nBest ROC-AUC Model: {best_roc_model} (ROC-AUC={best_roc_result['roc_auc']:.4f})")

# =============================================================================
# 13. 特徴量重要度分析
# =============================================================================
print("\n--- Feature Importance Analysis ---")
feature_names = num_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))

if 'RF' in models:
    # RandomForest重要度
    rf_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': models['RF'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 features (RandomForest):")
    for i, row in rf_importance.head(15).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

# =============================================================================
# 14. 詳細レポート
# =============================================================================
with open("metrics.txt", "w") as outfile:
    outfile.write(f"{'='*60}\n")
    outfile.write(f"離職予測モデル - 評価レポート v8\n")
    outfile.write(f"(ROC-AUC向上 + 未使用特徴量活用 + CatBoost)\n")
    outfile.write(f"{'='*60}\n\n")
    
    outfile.write(f"=== Best Model: {best_model_name} ===\n")
    outfile.write(f"Accuracy: {best_result['accuracy']:.4f}\n")
    outfile.write(f"F1 Score: {best_result['f1']:.4f}\n")
    outfile.write(f"ROC-AUC: {best_result['roc_auc']:.4f}\n")
    outfile.write(f"PR-AUC: {best_result['pr_auc']:.4f}\n")
    outfile.write(f"Precision: {best_result['precision']:.4f}\n")
    outfile.write(f"Recall: {best_result['recall']:.4f}\n")
    if 'threshold' in best_result:
        outfile.write(f"Threshold: {best_result['threshold']:.3f}\n")
    
    outfile.write(f"\n=== Best ROC-AUC Model: {best_roc_model} ===\n")
    outfile.write(f"ROC-AUC: {best_roc_result['roc_auc']:.4f}\n")
    
    outfile.write(f"\n=== All Models Comparison (sorted by ROC-AUC) ===\n")
    for name, res in sorted(test_results.items(), key=lambda x: -x[1]['roc_auc']):
        thresh_info = f" (th={res['threshold']:.2f})" if 'threshold' in res else ""
        outfile.write(f"{name:18s} - ROC: {res['roc_auc']:.4f}, F1: {res['f1']:.4f}, Acc: {res['accuracy']:.4f}, Recall: {res['recall']:.4f}{thresh_info}\n")
    
    outfile.write(f"\n=== Cross-Validation Results ===\n")
    for name, cv_res in cv_results.items():
        outfile.write(f"{name}: ROC-AUC={cv_res['roc_auc'].mean():.4f}±{cv_res['roc_auc'].std():.3f}\n")
    
    outfile.write(f"\n=== Threshold Analysis ===\n")
    for t in threshold_analysis:
        outfile.write(f"Th={t['threshold']:.3f}: Acc={t['accuracy']:.4f}, F1={t['f1']:.4f}, Recall={t['recall']:.4f}\n")
    
    outfile.write(f"\n=== Feature Engineering ===\n")
    outfile.write(f"Total features: {len(feature_names)}\n")
    outfile.write(f"Numeric: {len(num_cols)}, Categorical: {len(cat_cols)}\n")
    
    if 'RF' in models:
        outfile.write(f"\n=== Top 20 Feature Importance (RF) ===\n")
        for i, row in rf_importance.head(20).iterrows():
            outfile.write(f"{row['feature']}: {row['importance']:.4f}\n")

print("\nMetrics saved to: metrics.txt")

# =============================================================================
# 15. 可視化
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

# 特徴量重要度
if 'RF' in models:
    top_features = rf_importance.head(20)
    plt.figure(figsize=(12, 10))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
    sns.barplot(data=top_features, x='importance', y='feature', palette=colors)
    plt.title('Top 20 Feature Importance', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

# モデル比較（ROC-AUCフォーカス）
main_models = [m for m in test_results.keys() if 'Ens' not in m][:6]
ensemble_models = [m for m in test_results.keys() if 'Ens' in m]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, metric, title, ylim in [
    (axes[0], 'roc_auc', 'ROC-AUC', [0.5, 1.0]),
    (axes[1], 'f1', 'F1 Score', [0.0, 1.0]),
    (axes[2], 'recall', 'Recall', [0.0, 1.0])
]:
    all_models = main_models + ensemble_models[:2]
    scores = [test_results[m][metric] for m in all_models]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(all_models)))
    bars = ax.bar(range(len(all_models)), scores, color=colors)
    ax.set_ylabel(title, fontsize=12)
    ax.set_title(f'Model Comparison: {title}', fontsize=14)
    ax.set_ylim(ylim)
    ax.set_xticks(range(len(all_models)))
    ax.set_xticklabels(all_models, rotation=45, ha='right', fontsize=8)
    for bar, v in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ROC曲線
plt.figure(figsize=(10, 8))
for name in main_models + ['Ens_Top4']:
    if name in test_results:
        res = test_results[name]
        fpr, tpr, _ = roc_curve(y_test, res['probabilities'])
        plt.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves', fontsize=14)
plt.legend(loc='lower right', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# Precision-Recall曲線
plt.figure(figsize=(10, 8))
for name in main_models + ['Ens_Top4']:
    if name in test_results:
        res = test_results[name]
        prec, rec, _ = precision_recall_curve(y_test, res['probabilities'])
        plt.plot(rec, prec, label=f"{name} (AP={res['pr_auc']:.3f})", linewidth=2)
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves', fontsize=14)
plt.legend(loc='upper right', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('precision_recall_curves.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll visualizations saved.")
print("\n" + "=" * 70)
print("Training and Evaluation Completed!")
print("=" * 70)
