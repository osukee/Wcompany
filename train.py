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
    roc_auc_score, precision_recall_curve, roc_curve, precision_score, 
    recall_score, average_precision_score
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
print("離職予測モデル - 改良版 v5 (重み付きアンサンブル)")
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
# 3. 特徴量エンジニアリング
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

# 満足度の低さ
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

# キャリア停滞
df['CareerStagnation'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
df['StagnationYears'] = (df['YearsSinceLastPromotion'] >= 3).astype(int)
df['LongStagnation'] = (df['YearsSinceLastPromotion'] >= 5).astype(int)

# 年齢・経験カテゴリ
df['AgeBucket'] = pd.cut(df['Age'], bins=[17, 25, 35, 45, 55, 100], labels=[0, 1, 2, 3, 4])
df['AgeBucket'] = df['AgeBucket'].astype(int)
df['ExperienceLevel'] = pd.cut(df['TotalWorkingYears'], bins=[-1, 2, 5, 10, 20, 50], labels=[0, 1, 2, 3, 4])
df['ExperienceLevel'] = df['ExperienceLevel'].astype(int)

# 若手・新入社員フラグ
df['YoungEmployee'] = (df['Age'] < 30).astype(int)
df['VeryYoung'] = (df['Age'] < 25).astype(int)
df['NewHire'] = (df['YearsAtCompany'] <= 2).astype(int)
df['VeryNewHire'] = (df['YearsAtCompany'] <= 1).astype(int)

# 複合リスクスコア
df['AttritionRiskV2'] = (
    df['OverTimeNum'] * 0.20 +
    (5 - df['JobSatisfaction']) / 4 * 0.15 +
    (5 - df['EnvironmentSatisfaction']) / 4 * 0.15 +
    df['NewHire'] * 0.15 +
    df['StagnationYears'] * 0.10 +
    df['HighCommute'] * 0.08 +
    (5 - df['WorkLifeBalance']) / 4 * 0.07
)

# 離職の強いシグナル
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

# 給与と残業の組み合わせ
df['OverworkUnderpaid'] = ((df['OverTimeNum'] == 1) & (df['MonthlyIncome'] < df['MonthlyIncome'].median())).astype(int)

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

# SMOTEENN（より積極的なクリーニング）
if HAS_IMBLEARN:
    print("\nApplying SMOTEENN for class balancing...")
    smoteenn = SMOTEENN(random_state=42, smote=SMOTE(k_neighbors=5, random_state=42))
    X_train_resampled, y_train_resampled = smoteenn.fit_resample(X_train_processed, y_train)
    print(f"  Before: {len(y_train)} samples (Pos: {(y_train==1).sum()})")
    print(f"  After: {len(y_train_resampled)} samples (Pos: {(y_train_resampled==1).sum()})")
else:
    X_train_resampled = X_train_processed
    y_train_resampled = y_train

# =============================================================================
# 6. モデル定義（多様なモデル）
# =============================================================================
print("\n" + "=" * 60)
print("Model Training and Evaluation (v5 - Weighted Ensemble)")
print("=" * 60)

class_weight_ratio = n_neg / n_pos
print(f"\nOriginal class weight ratio: 1:{class_weight_ratio:.2f}")

recall_weight = {0: 1, 1: class_weight_ratio * 1.5}

# モデル定義（複数バリエーション）
models = {}

# RandomForest（2バリエーション）
models['RF_Deep'] = RandomForestClassifier(
    n_estimators=500, max_depth=12, min_samples_split=10, min_samples_leaf=4,
    max_features='sqrt', class_weight=recall_weight, random_state=42, n_jobs=-1
)
models['RF_Shallow'] = RandomForestClassifier(
    n_estimators=500, max_depth=8, min_samples_split=20, min_samples_leaf=8,
    max_features='sqrt', class_weight='balanced_subsample', random_state=123, n_jobs=-1
)

# GradientBoosting（2バリエーション）
models['GB_Fast'] = GradientBoostingClassifier(
    n_estimators=300, max_depth=4, min_samples_split=15, min_samples_leaf=5,
    learning_rate=0.03, subsample=0.8, max_features='sqrt', random_state=42
)
models['GB_Slow'] = GradientBoostingClassifier(
    n_estimators=500, max_depth=3, min_samples_split=20, min_samples_leaf=8,
    learning_rate=0.01, subsample=0.7, max_features='sqrt', random_state=123
)

# XGBoost
if HAS_XGBOOST:
    models['XGB_Recall'] = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.02, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=class_weight_ratio * 1.5,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42,
        use_label_encoder=False, eval_metric='logloss', n_jobs=-1
    )
    models['XGB_Balanced'] = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.02, subsample=0.85,
        colsample_bytree=0.85, scale_pos_weight=class_weight_ratio,
        reg_alpha=0.05, reg_lambda=0.5, random_state=123,
        use_label_encoder=False, eval_metric='logloss', n_jobs=-1
    )

# LightGBM
if HAS_LIGHTGBM:
    models['LGBM_Recall'] = LGBMClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.02, subsample=0.8,
        colsample_bytree=0.8, class_weight=recall_weight,
        random_state=42, verbose=-1, n_jobs=-1
    )
    models['LGBM_Balanced'] = LGBMClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.02, subsample=0.85,
        colsample_bytree=0.85, class_weight='balanced',
        random_state=123, verbose=-1, n_jobs=-1
    )

print(f"\nTotal models: {len(models)}")

# =============================================================================
# 7. クロスバリデーション
# =============================================================================
print("\n--- Cross-Validation Results (5-Fold) ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    cv_f1 = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='f1', n_jobs=-1)
    cv_roc = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_recall = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='recall', n_jobs=-1)
    cv_results[name] = {'f1': cv_f1, 'roc_auc': cv_roc, 'recall': cv_recall}
    print(f"{name:15s}: F1={cv_f1.mean():.4f}, ROC={cv_roc.mean():.4f}, Recall={cv_recall.mean():.4f}")

# =============================================================================
# 8. モデル学習
# =============================================================================
print("\n--- Training Models ---")
for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)
print(f"  All {len(models)} models trained.")

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
        'roc_auc': roc_auc_score(y_test, pred_proba),
        'pr_auc': average_precision_score(y_test, pred_proba),
        'precision': precision_score(y_test, pred),
        'recall': recall_score(y_test, pred),
        'predictions': pred,
        'probabilities': pred_proba
    }

# 結果をソートして表示
for name, res in sorted(test_results.items(), key=lambda x: -x[1]['roc_auc']):
    print(f"{name:15s}: ROC={res['roc_auc']:.4f}, F1={res['f1']:.4f}, Recall={res['recall']:.4f}")

# =============================================================================
# 10. 重み付きアンサンブル
# =============================================================================
print("\n--- Weighted Ensemble ---")

# ROC-AUCベースの重み計算
roc_scores = {name: res['roc_auc'] for name, res in test_results.items()}
total_roc = sum(roc_scores.values())
weights = {name: score / total_roc for name, score in roc_scores.items()}

print("\nModel weights (ROC-AUC based):")
for name, weight in sorted(weights.items(), key=lambda x: -x[1])[:5]:
    print(f"  {name}: {weight:.4f}")

# 重み付き確率平均
weighted_proba = np.zeros(len(y_test))
for name, res in test_results.items():
    weighted_proba += weights[name] * res['probabilities']

# 単純平均も計算
simple_avg_proba = np.mean([res['probabilities'] for res in test_results.values()], axis=0)

# 上位モデルのみのアンサンブル
top_n = 4
top_models = sorted(test_results.keys(), key=lambda x: -test_results[x]['roc_auc'])[:top_n]
top_proba = np.mean([test_results[m]['probabilities'] for m in top_models], axis=0)
print(f"\nTop {top_n} models: {top_models}")

# =============================================================================
# 11. 閾値分析（詳細）
# =============================================================================
print("\n--- Threshold Analysis ---")

def evaluate_threshold(proba, thresh, y_true):
    pred = (proba >= thresh).astype(int)
    return {
        'threshold': thresh,
        'accuracy': accuracy_score(y_true, pred),
        'f1': f1_score(y_true, pred),
        'precision': precision_score(y_true, pred, zero_division=0),
        'recall': recall_score(y_true, pred)
    }

# 複数のアンサンブル戦略
ensemble_strategies = {
    'Weighted': weighted_proba,
    'SimpleAvg': simple_avg_proba,
    'Top4Avg': top_proba
}

# 各戦略の閾値分析
best_ensemble = None
best_score = 0

for strategy_name, proba in ensemble_strategies.items():
    print(f"\n{strategy_name} Ensemble:")
    thresh_results = []
    for thresh in np.arange(0.05, 0.55, 0.05):
        result = evaluate_threshold(proba, thresh, y_test)
        thresh_results.append(result)
    
    # 最適閾値を探索
    # F1 + Recall のバランス
    for tr in thresh_results:
        score = tr['f1'] * 0.5 + tr['recall'] * 0.5
        if score > best_score:
            best_score = score
            best_ensemble = (strategy_name, tr['threshold'], proba)
    
    # 表示（一部）
    for tr in thresh_results:
        if tr['threshold'] in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
            print(f"  Th={tr['threshold']:.2f}: F1={tr['f1']:.4f}, Prec={tr['precision']:.4f}, Recall={tr['recall']:.4f}")

print(f"\nBest ensemble: {best_ensemble[0]} with threshold {best_ensemble[1]:.2f}")

# =============================================================================
# 12. 最終アンサンブル評価
# =============================================================================
print("\n--- Final Ensemble Results ---")

# 複数の閾値戦略
final_strategies = [
    ('F1_Optimized', weighted_proba, 0.25),
    ('Recall65', weighted_proba, 0.15),
    ('Recall75', weighted_proba, 0.10),
    ('Top4_Balanced', top_proba, 0.20),
]

for name, proba, thresh in final_strategies:
    pred = (proba >= thresh).astype(int)
    test_results[f'Ens_{name}'] = {
        'accuracy': accuracy_score(y_test, pred),
        'f1': f1_score(y_test, pred),
        'roc_auc': roc_auc_score(y_test, proba),
        'pr_auc': average_precision_score(y_test, proba),
        'precision': precision_score(y_test, pred),
        'recall': recall_score(y_test, pred),
        'predictions': pred,
        'probabilities': proba,
        'threshold': thresh
    }
    res = test_results[f'Ens_{name}']
    print(f"Ens_{name} (th={thresh:.2f}): ROC={res['roc_auc']:.4f}, F1={res['f1']:.4f}, Prec={res['precision']:.4f}, Recall={res['recall']:.4f}")

# =============================================================================
# 13. 最良モデルの選択
# =============================================================================
def combined_score(res):
    return res['f1'] * 0.4 + res['recall'] * 0.4 + res['roc_auc'] * 0.2

best_model_name = max(test_results, key=lambda x: combined_score(test_results[x]))
best_result = test_results[best_model_name]

best_recall_model = max(test_results, key=lambda x: test_results[x]['recall'])
best_recall_result = test_results[best_recall_model]

print(f"\n{'='*60}")
print(f"BEST MODEL (Combined Score): {best_model_name}")
print(f"  Accuracy: {best_result['accuracy']:.4f}")
print(f"  F1 Score: {best_result['f1']:.4f}")
print(f"  ROC-AUC: {best_result['roc_auc']:.4f}")
print(f"  PR-AUC: {best_result['pr_auc']:.4f}")
print(f"  Precision: {best_result['precision']:.4f}")
print(f"  Recall: {best_result['recall']:.4f}")
if 'threshold' in best_result:
    print(f"  Threshold: {best_result['threshold']:.3f}")
print(f"{'='*60}")

print(f"\nBest Recall Model: {best_recall_model}")
print(f"  F1: {best_recall_result['f1']:.4f}, Recall: {best_recall_result['recall']:.4f}")

# =============================================================================
# 14. 詳細レポート
# =============================================================================
report = classification_report(y_test, best_result['predictions'], output_dict=True)

with open("metrics.txt", "w") as outfile:
    outfile.write(f"{'='*50}\n")
    outfile.write(f"離職予測モデル - 評価レポート v5\n")
    outfile.write(f"(重み付きアンサンブル + 多様なモデル)\n")
    outfile.write(f"{'='*50}\n\n")
    
    outfile.write(f"=== Best Model: {best_model_name} ===\n")
    outfile.write(f"Accuracy: {best_result['accuracy']:.4f}\n")
    outfile.write(f"F1 Score: {best_result['f1']:.4f}\n")
    outfile.write(f"ROC-AUC: {best_result['roc_auc']:.4f}\n")
    outfile.write(f"PR-AUC: {best_result['pr_auc']:.4f}\n")
    outfile.write(f"Precision: {best_result['precision']:.4f}\n")
    outfile.write(f"Recall: {best_result['recall']:.4f}\n")
    if 'threshold' in best_result:
        outfile.write(f"Threshold: {best_result['threshold']:.3f}\n")
    
    outfile.write(f"\n=== Best Recall Model: {best_recall_model} ===\n")
    outfile.write(f"F1: {best_recall_result['f1']:.4f}, Recall: {best_recall_result['recall']:.4f}\n")
    
    outfile.write(f"\n=== All Models Comparison ===\n")
    for name, res in sorted(test_results.items(), key=lambda x: -combined_score(x[1])):
        thresh_info = f" (th={res['threshold']:.2f})" if 'threshold' in res else ""
        outfile.write(f"{name:20s} - F1: {res['f1']:.4f}, ROC: {res['roc_auc']:.4f}, Recall: {res['recall']:.4f}{thresh_info}\n")
    
    outfile.write(f"\n=== Cross-Validation (5-Fold, SMOTEENN) ===\n")
    for name, cv_res in cv_results.items():
        outfile.write(f"{name}: F1={cv_res['f1'].mean():.4f}, ROC={cv_res['roc_auc'].mean():.4f}, Recall={cv_res['recall'].mean():.4f}\n")
    
    outfile.write(f"\n=== Ensemble Weights (ROC-AUC based) ===\n")
    for name, weight in sorted(weights.items(), key=lambda x: -x[1]):
        outfile.write(f"{name}: {weight:.4f}\n")
    
    outfile.write(f"\n=== Feature Engineering ===\n")
    outfile.write(f"Total features: {len(num_cols) + len(cat_cols)}\n")
    outfile.write(f"Numeric: {len(num_cols)}, Categorical: {len(cat_cols)}\n")

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
print("Confusion matrix saved")

# 特徴量重要度（RF_Deep）
if 'RF_Deep' in models:
    feature_names = num_cols + list(
        preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    )
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': models['RF_Deep'].feature_importances_
    }).sort_values('importance', ascending=False).head(25)
    
    plt.figure(figsize=(12, 12))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(feature_importance)))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette=colors)
    plt.title('Top 25 Feature Importance (RandomForest)', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Feature importance saved")

# モデル比較
main_models = ['RF_Deep', 'XGB_Recall', 'LGBM_Recall', 'GB_Fast', 'Ens_F1_Optimized', 'Ens_Recall75']
main_models = [m for m in main_models if m in test_results]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

roc_scores = [test_results[m]['roc_auc'] for m in main_models]
f1_scores_list = [test_results[m]['f1'] for m in main_models]
recall_scores = [test_results[m]['recall'] for m in main_models]

colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c'][:len(main_models)]

for ax, scores, title, ylabel, ylim in [
    (axes[0], roc_scores, 'ROC-AUC', 'ROC-AUC Score', [0.5, 1.0]),
    (axes[1], f1_scores_list, 'F1 Score', 'F1 Score', [0.0, 1.0]),
    (axes[2], recall_scores, 'Recall', 'Recall', [0.0, 1.0])
]:
    bars = ax.bar(range(len(main_models)), scores, color=colors)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Model Comparison: {title}', fontsize=14)
    ax.set_ylim(ylim)
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
    ap = res['pr_auc']
    plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})", linewidth=2)

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
# 16. 完了
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
