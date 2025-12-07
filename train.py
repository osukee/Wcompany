import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier
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

# SMOTE
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTETomek
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

warnings.filterwarnings('ignore')

# =============================================================================
# 1. データの読み込み
# =============================================================================
print("=" * 70)
print("離職予測モデル - 改良版 v9")
print("(特徴量選択 + シンプル化)")
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
# 3. 特徴量エンジニアリング（厳選版）
# =============================================================================
print("\nCreating engineered features (Selected)...")

# === 重要度トップの特徴量のみ作成 ===

# インセンティブ関連（最重要！）
if 'Incentive' in df.columns:
    df['IncentivePerIncome'] = df['Incentive'] / (df['MonthlyIncome'] + 1)
    df['HasIncentive'] = (df['Incentive'] > 0).astype(int)
    df['LowIncentive'] = (df['Incentive'] == 0).astype(int)
    df['HighIncentive'] = (df['Incentive'] > df['Incentive'].median()).astype(int)

# 給与・キャリア関連（重要）
df['IncomePerLevel'] = df['MonthlyIncome'] / (df['JobLevel'] + 1)
df['IncomePerYear'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
df['IncomeLevelRatio'] = df['MonthlyIncome'] / (df['JobLevel'] * 3000 + 1)
df['YearsAtRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
df['ManagerRatio'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
df['PromotionGap'] = df['YearsAtCompany'] - df['YearsSinceLastPromotion']

# ストレス関連（重要！）
if 'StressRating' in df.columns and 'StressSelfReported' in df.columns:
    df['TotalStress'] = df['StressRating'] + df['StressSelfReported']
    df['HighStress'] = (df['TotalStress'] >= 6).astype(int)
    df['OverallSatisfaction'] = (df['EnvironmentSatisfaction'] + df['JobSatisfaction'] + df['RelationshipSatisfaction']) / 3
    df['StressSatisfactionGap'] = df['TotalStress'] - df['OverallSatisfaction']

# 達成度関連
if 'MonthlyAchievement' in df.columns:
    df['AchievementVsIncome'] = df['MonthlyAchievement'] / (df['MonthlyIncome'] + 1)

# 残業関連
if 'OverTime' in df.columns:
    df['OverTimeNum'] = df['OverTime'].map({'Yes': 1, 'No': 0}) if df['OverTime'].dtype == 'object' else df['OverTime']

# 満足度関連（シンプル化）
df['LowSatisfaction'] = (
    (df['EnvironmentSatisfaction'] <= 2).astype(int) +
    (df['JobSatisfaction'] <= 2).astype(int) +
    (df['WorkLifeBalance'] <= 2).astype(int)
)

# キャリア停滞
df['StagnationYears'] = (df['YearsSinceLastPromotion'] >= 3).astype(int)

# 若手・新入社員
df['NewHire'] = (df['YearsAtCompany'] <= 2).astype(int)
df['YoungEmployee'] = (df['Age'] < 30).astype(int)

# 福利厚生（シンプル化）
if 'WelfareBenefits' in df.columns:
    df['LowWelfare'] = (df['WelfareBenefits'] <= 2).astype(int)

# 柔軟性
if 'RemoteWork' in df.columns and 'FlexibleWork' in df.columns:
    df['NoFlexibility'] = ((df['RemoteWork'] == 0) & (df['FlexibleWork'] == 0)).astype(int)

# 複合リスクスコア（シンプル版）
df['AttritionRisk'] = (
    df.get('LowIncentive', 0) * 0.25 +
    df.get('HighStress', 0) * 0.20 +
    (5 - df['JobSatisfaction']) / 4 * 0.15 +
    df['NewHire'] * 0.15 +
    df['StagnationYears'] * 0.10 +
    df.get('LowWelfare', 0) * 0.08 +
    df.get('OverTimeNum', 0) * 0.07
)

print(f"Total features after engineering: {len(df.columns) - 1}")

# =============================================================================
# 4. 重要度の低い特徴量を削除
# =============================================================================
# v8の結果から重要度が低い（<0.01）特徴量を事前に除外
low_importance_cols = [
    'RoleRatio', 'CommuteStress', 'HighCommute', 'VeryHighCommute',
    'VeryLowSatisfaction', 'LongStagnation', 'VeryNewHire', 'VeryYoung',
    'OvertimeSatisfaction', 'OvertimeStress', 'WorkLifeIssue',
    'StrongAttritionSignalV2', 'LowPerformerStressed', 'VeryLowBenefits',
    'VeryHighStress', 'NoRemote', 'FullFlexibility', 'SatisfactionScore',
    'EngagementScore', 'AgeExperienceGap', 'CareerStagnation'
]

for col in low_importance_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

print(f"After removing low-importance features: {len(df.columns) - 1}")

# =============================================================================
# 5. データ分割
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
# 6. 前処理 + SMOTE
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
# 7. モデル（シンプル化）
# =============================================================================
print("\n" + "=" * 70)
print("Model Training (Simplified)")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {}

# トップ4モデルのみ（v8の結果から）
models['GB'] = GradientBoostingClassifier(
    n_estimators=400, max_depth=4, min_samples_split=10,
    learning_rate=0.02, subsample=0.8, random_state=42
)

if HAS_LIGHTGBM:
    models['LGBM'] = LGBMClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.02, subsample=0.8,
        colsample_bytree=0.8, class_weight='balanced',
        random_state=42, verbose=-1, n_jobs=-1
    )

if HAS_XGBOOST:
    models['XGB'] = XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.02, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=class_weight_ratio,
        reg_alpha=0.1, reg_lambda=0.5, random_state=42,
        use_label_encoder=False, eval_metric='logloss', n_jobs=-1
    )

if HAS_CATBOOST:
    models['CatBoost'] = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.03,
        loss_function='Logloss', auto_class_weights='Balanced',
        random_seed=42, verbose=False
    )

print(f"\nTotal models: {len(models)}")

# =============================================================================
# 8. クロスバリデーション
# =============================================================================
print("\n--- Cross-Validation Results ---")
cv_results = {}

for name, model in models.items():
    cv_roc = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_f1 = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='f1', n_jobs=-1)
    cv_results[name] = {'roc_auc': cv_roc, 'f1': cv_f1}
    print(f"{name:12s}: ROC-AUC={cv_roc.mean():.4f}±{cv_roc.std():.3f}, F1={cv_f1.mean():.4f}")

# =============================================================================
# 9. モデル学習
# =============================================================================
print("\n--- Training models ---")
for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)
print("  All models trained.")

# =============================================================================
# 10. テストセット評価
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

for name, res in sorted(test_results.items(), key=lambda x: -x[1]['roc_auc']):
    print(f"{name:12s}: ROC-AUC={res['roc_auc']:.4f}, F1={res['f1']:.4f}, Acc={res['accuracy']:.4f}, Recall={res['recall']:.4f}")

# =============================================================================
# 11. アンサンブル
# =============================================================================
print("\n--- Ensemble ---")

# Top4モデルの平均
top_proba = np.mean([res['probabilities'] for res in test_results.values()], axis=0)

# 閾値分析
print("\nThreshold Analysis:")
threshold_analysis = []
for thresh in np.arange(0.15, 0.50, 0.025):
    pred = (top_proba >= thresh).astype(int)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    rec = recall_score(y_test, pred)
    threshold_analysis.append({'threshold': thresh, 'accuracy': acc, 'f1': f1, 'recall': rec})
    print(f"  Th={thresh:.3f}: Acc={acc:.4f}, F1={f1:.4f}, Recall={rec:.4f}")

# 最適閾値
best_f1 = max(threshold_analysis, key=lambda x: x['f1'])
valid_recall65 = [t for t in threshold_analysis if t['recall'] >= 0.65]
best_recall65 = max(valid_recall65, key=lambda x: x['accuracy']) if valid_recall65 else best_f1

# アンサンブル登録
ensemble_configs = [
    ('Ens_F1Max', top_proba, best_f1['threshold']),
    ('Ens_Recall65', top_proba, best_recall65['threshold']),
    ('Ens_Balanced', top_proba, 0.30),
]

for name, proba, thresh in ensemble_configs:
    pred = (proba >= thresh).astype(int)
    test_results[name] = {
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

print("\n--- Ensemble Results ---")
for name in [n for n, _, _ in ensemble_configs]:
    res = test_results[name]
    print(f"{name} (th={res['threshold']:.2f}): ROC={res['roc_auc']:.4f}, F1={res['f1']:.4f}, Acc={res['accuracy']:.4f}, Recall={res['recall']:.4f}")

# =============================================================================
# 12. 最良モデルの選択
# =============================================================================
def combined_score(res):
    return res['roc_auc'] * 0.4 + res['f1'] * 0.3 + res['recall'] * 0.3

best_model_name = max(test_results, key=lambda x: combined_score(test_results[x]))
best_result = test_results[best_model_name]

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

print(f"\nBest ROC-AUC: {best_roc_model} (ROC-AUC={best_roc_result['roc_auc']:.4f})")

# =============================================================================
# 13. 特徴量重要度
# =============================================================================
feature_names = num_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))

if 'GB' in models:
    gb_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': models['GB'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 features (GradientBoosting):")
    for i, row in gb_importance.head(15).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

# =============================================================================
# 14. レポート
# =============================================================================
with open("metrics.txt", "w") as outfile:
    outfile.write(f"{'='*60}\n")
    outfile.write(f"離職予測モデル - 評価レポート v9\n")
    outfile.write(f"(特徴量選択 + シンプル化)\n")
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
    
    outfile.write(f"\n=== All Models (sorted by ROC-AUC) ===\n")
    for name, res in sorted(test_results.items(), key=lambda x: -x[1]['roc_auc']):
        thresh = f" (th={res['threshold']:.2f})" if 'threshold' in res else ""
        outfile.write(f"{name:15s} - ROC: {res['roc_auc']:.4f}, F1: {res['f1']:.4f}, Acc: {res['accuracy']:.4f}, Recall: {res['recall']:.4f}{thresh}\n")
    
    outfile.write(f"\n=== Feature Engineering ===\n")
    outfile.write(f"Total features: {len(feature_names)}\n")
    outfile.write(f"(Simplified from 115 to {len(feature_names)})\n")
    
    if 'GB' in models:
        outfile.write(f"\n=== Top 15 Features (GB) ===\n")
        for i, row in gb_importance.head(15).iterrows():
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
if 'GB' in models:
    top_features = gb_importance.head(15)
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
    sns.barplot(data=top_features, x='importance', y='feature', palette=colors)
    plt.title('Top 15 Feature Importance (GB)', fontsize=14)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

# モデル比較
all_models = list(test_results.keys())
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, metric, title, ylim in [
    (axes[0], 'roc_auc', 'ROC-AUC', [0.5, 1.0]),
    (axes[1], 'f1', 'F1 Score', [0.0, 1.0]),
    (axes[2], 'accuracy', 'Accuracy', [0.5, 1.0])
]:
    scores = [test_results[m][metric] for m in all_models]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(all_models)))
    bars = ax.bar(range(len(all_models)), scores, color=colors)
    ax.set_ylabel(title, fontsize=12)
    ax.set_title(f'{title}', fontsize=14)
    ax.set_ylim(ylim)
    ax.set_xticks(range(len(all_models)))
    ax.set_xticklabels(all_models, rotation=45, ha='right', fontsize=8)
    for bar, v in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=7)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ROC曲線
plt.figure(figsize=(10, 8))
for name, res in test_results.items():
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

print("\nAll visualizations saved.")
print("\n" + "=" * 70)
print("Training Completed! (Simplified Model)")
print("=" * 70)
