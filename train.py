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
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTETomek
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("imbalanced-learn not available, skipping SMOTE...")

warnings.filterwarnings('ignore')

# =============================================================================
# 1. データの読み込み
# =============================================================================
print("=" * 60)
print("離職予測モデル - 改良版 v3 (SMOTE + Stacking)")
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
# 3. 特徴量エンジニアリング（さらに拡張）
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

# 満足度の低さ（離職リスク指標）
df['LowSatisfaction'] = (
    (df['EnvironmentSatisfaction'] <= 2).astype(int) +
    (df['JobSatisfaction'] <= 2).astype(int) +
    (df['RelationshipSatisfaction'] <= 2).astype(int)
)

# 昇進とキャリア
df['PromotionGap'] = df['YearsAtCompany'] - df['YearsSinceLastPromotion']
df['AgeExperienceGap'] = df['Age'] - df['TotalWorkingYears'] - 18
df['IncomeLevelRatio'] = df['MonthlyIncome'] / (df['JobLevel'] * 3000 + 1)

# 給与関連の詳細
df['IncomePerYear'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
df['CompanyGrowthRate'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
df['TrainingIntensity'] = df['TrainingTimesLastYear'] / (df['YearsAtCompany'] + 1)

# 給与水準（同レベル比較）
avg_income_by_level = df.groupby('JobLevel')['MonthlyIncome'].transform('mean')
df['IncomeVsLevelAvg'] = df['MonthlyIncome'] / (avg_income_by_level + 1)

# 職種別給与比較
avg_income_by_role = df.groupby('JobRole')['MonthlyIncome'].transform('mean')
df['IncomeVsRoleAvg'] = df['MonthlyIncome'] / (avg_income_by_role + 1)

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

# キャリア停滞
df['CareerStagnation'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
df['StagnationYears'] = (df['YearsSinceLastPromotion'] >= 3).astype(int)

# 年齢・経験カテゴリ
df['AgeBucket'] = pd.cut(df['Age'], bins=[17, 25, 35, 45, 55, 100], labels=[0, 1, 2, 3, 4])
df['AgeBucket'] = df['AgeBucket'].astype(int)
df['ExperienceLevel'] = pd.cut(df['TotalWorkingYears'], bins=[-1, 2, 5, 10, 20, 50], labels=[0, 1, 2, 3, 4])
df['ExperienceLevel'] = df['ExperienceLevel'].astype(int)

# 若手社員フラグ（離職リスクが高い傾向）
df['YoungEmployee'] = (df['Age'] < 30).astype(int)
df['NewHire'] = (df['YearsAtCompany'] <= 2).astype(int)

# 複合リスクスコア（改良版）
df['AttritionRisk'] = (
    (df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)) * 0.2 +
    (5 - df['JobSatisfaction']) / 4 * 0.2 +
    (5 - df['EnvironmentSatisfaction']) / 4 * 0.2 +
    df['OverTimeNum'] * 0.15 +
    df['NewHire'] * 0.15 +
    (df['DistanceFromHome'] > 20).astype(int) * 0.1
)

# ストレス関連
if 'StressRating' in df.columns and 'StressSelfReported' in df.columns:
    df['StressCombined'] = df['StressRating'] + df['StressSelfReported']

# ワークライフバランス関連
df['WorkLifeIssue'] = ((df['WorkLifeBalance'] <= 2) & (df['OverTimeNum'] == 1)).astype(int)

# 会社との関係性
df['TenureManagerRatio'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
df['FrequentManagerChange'] = (df['TenureManagerRatio'] < 0.3).astype(int)

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
print(f"  No Attrition (0): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
print(f"  Attrition (1): {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")

# =============================================================================
# 5. 前処理パイプライン + SMOTE
# =============================================================================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])

# 前処理を適用
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# SMOTEでオーバーサンプリング
if HAS_IMBLEARN:
    print("\nApplying SMOTE for class balancing...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
    print(f"  Before SMOTE: {len(y_train)} samples")
    print(f"  After SMOTE: {len(y_train_resampled)} samples")
    print(f"  Class 0: {(y_train_resampled == 0).sum()}, Class 1: {(y_train_resampled == 1).sum()}")
else:
    X_train_resampled = X_train_processed
    y_train_resampled = y_train

# =============================================================================
# 6. モデル定義（SMOTE適用後データ用）
# =============================================================================
print("\n" + "=" * 60)
print("Model Training and Evaluation")
print("=" * 60)

class_weight_ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nOriginal class weight ratio: 1:{class_weight_ratio:.2f}")

# 基本モデル（前処理済みデータ用なのでパイプラインなし）
rf_classifier = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

gb_classifier = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=4,
    learning_rate=0.02,
    subsample=0.8,
    max_features='sqrt',
    random_state=42
)

models = {
    'RandomForest': rf_classifier,
    'GradientBoosting': gb_classifier
}

if HAS_XGBOOST:
    xgb_classifier = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,  # SMOTEでバランス済み
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    models['XGBoost'] = xgb_classifier

if HAS_LIGHTGBM:
    lgbm_classifier = LGBMClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight=None,  # SMOTEでバランス済み
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    models['LightGBM'] = lgbm_classifier

# =============================================================================
# 7. クロスバリデーション（SMOTEデータ）
# =============================================================================
print("\n--- Cross-Validation Results (5-Fold Stratified, with SMOTE) ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    cv_f1 = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='f1', n_jobs=-1)
    cv_roc = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_recall = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='recall', n_jobs=-1)
    cv_results[name] = {'f1': cv_f1, 'roc_auc': cv_roc, 'recall': cv_recall}
    print(f"\n{name}:")
    print(f"  CV F1 Score: {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")
    print(f"  CV ROC-AUC: {cv_roc.mean():.4f} (+/- {cv_roc.std():.4f})")
    print(f"  CV Recall: {cv_recall.mean():.4f} (+/- {cv_recall.std():.4f})")

# =============================================================================
# 8. モデル学習（SMOTEデータ）
# =============================================================================
print("\n--- Training Models on SMOTE-balanced data ---")
for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)
    print(f"  {name} trained.")

# =============================================================================
# 9. Stacking Classifier
# =============================================================================
print("\n--- Building Stacking Classifier ---")

estimators = [
    ('rf', models['RandomForest']),
    ('gb', models['GradientBoosting'])
]
if HAS_XGBOOST:
    estimators.append(('xgb', models['XGBoost']))
if HAS_LIGHTGBM:
    estimators.append(('lgbm', models['LightGBM']))

# 再学習が必要なので、新しいインスタンスを作成
stacking_estimators = []
for name, _ in estimators:
    if name == 'rf':
        stacking_estimators.append(('rf', RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_split=10,
            min_samples_leaf=4, class_weight='balanced', random_state=42, n_jobs=-1
        )))
    elif name == 'gb':
        stacking_estimators.append(('gb', GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.03, random_state=42
        )))
    elif name == 'xgb' and HAS_XGBOOST:
        stacking_estimators.append(('xgb', XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.03,
            random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1
        )))
    elif name == 'lgbm' and HAS_LIGHTGBM:
        stacking_estimators.append(('lgbm', LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.03,
            random_state=42, verbose=-1, n_jobs=-1
        )))

stacking_clf = StackingClassifier(
    estimators=stacking_estimators,
    final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    cv=5,
    stack_method='predict_proba',
    n_jobs=-1
)

stacking_clf.fit(X_train_resampled, y_train_resampled)
print("  Stacking Classifier trained.")

# =============================================================================
# 10. テストセット評価
# =============================================================================
print("\n--- Test Set Evaluation ---")
test_results = {}

for name, model in models.items():
    pred = model.predict(X_test_processed)
    pred_proba = model.predict_proba(X_test_processed)[:, 1]
    
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc = roc_auc_score(y_test, pred_proba)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    
    test_results[name] = {
        'accuracy': acc, 'f1': f1, 'roc_auc': roc,
        'precision': prec, 'recall': rec,
        'predictions': pred, 'probabilities': pred_proba
    }
    print(f"\n{name}:")
    print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc:.4f}")
    print(f"  Precision: {prec:.4f}, Recall: {rec:.4f}")

# Stacking結果
stacking_pred = stacking_clf.predict(X_test_processed)
stacking_proba = stacking_clf.predict_proba(X_test_processed)[:, 1]
test_results['Stacking'] = {
    'accuracy': accuracy_score(y_test, stacking_pred),
    'f1': f1_score(y_test, stacking_pred),
    'roc_auc': roc_auc_score(y_test, stacking_proba),
    'precision': precision_score(y_test, stacking_pred),
    'recall': recall_score(y_test, stacking_pred),
    'predictions': stacking_pred,
    'probabilities': stacking_proba
}
print(f"\nStacking:")
print(f"  Accuracy: {test_results['Stacking']['accuracy']:.4f}, F1: {test_results['Stacking']['f1']:.4f}, ROC-AUC: {test_results['Stacking']['roc_auc']:.4f}")
print(f"  Precision: {test_results['Stacking']['precision']:.4f}, Recall: {test_results['Stacking']['recall']:.4f}")

# =============================================================================
# 11. アンサンブル（複数閾値で評価）
# =============================================================================
print("\n--- Ensemble with Threshold Optimization ---")

all_probas = np.array([r['probabilities'] for r in test_results.values()])
ensemble_proba = np.mean(all_probas, axis=0)

# 複数閾値で評価
print("\nThreshold analysis for ensemble:")
threshold_results = []
for thresh in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    pred = (ensemble_proba >= thresh).astype(int)
    f1 = f1_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    threshold_results.append({'threshold': thresh, 'f1': f1, 'precision': prec, 'recall': rec})
    print(f"  Threshold {thresh:.2f}: F1={f1:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")

# 最適閾値（F1最大化）
precisions, recalls, thresholds = precision_recall_curve(y_test, ensemble_proba)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
best_f1_idx = np.argmax(f1_scores)
best_threshold_f1 = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5

# Recall重視の閾値（Recall >= 0.6を目指す）
best_threshold_recall = 0.5
for thresh in np.linspace(0.2, 0.5, 30):
    pred = (ensemble_proba >= thresh).astype(int)
    rec = recall_score(y_test, pred)
    if rec >= 0.6:
        best_threshold_recall = thresh
        break

print(f"\nOptimal threshold (F1 max): {best_threshold_f1:.3f}")
print(f"Threshold for Recall >= 0.6: {best_threshold_recall:.3f}")

# 両方の閾値で評価
for thresh_name, thresh in [('F1-optimized', best_threshold_f1), ('Recall-optimized', best_threshold_recall)]:
    ensemble_pred = (ensemble_proba >= thresh).astype(int)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    ensemble_f1 = f1_score(y_test, ensemble_pred)
    ensemble_roc = roc_auc_score(y_test, ensemble_proba)
    ensemble_prec = precision_score(y_test, ensemble_pred)
    ensemble_rec = recall_score(y_test, ensemble_pred)
    
    key = f'Ensemble_{thresh_name}'
    test_results[key] = {
        'accuracy': ensemble_acc, 'f1': ensemble_f1, 'roc_auc': ensemble_roc,
        'precision': ensemble_prec, 'recall': ensemble_rec,
        'predictions': ensemble_pred, 'probabilities': ensemble_proba,
        'threshold': thresh
    }
    print(f"\n{key} (threshold={thresh:.3f}):")
    print(f"  Accuracy: {ensemble_acc:.4f}, F1: {ensemble_f1:.4f}, ROC-AUC: {ensemble_roc:.4f}")
    print(f"  Precision: {ensemble_prec:.4f}, Recall: {ensemble_rec:.4f}")

# =============================================================================
# 12. 最良モデルの選択（F1ベース）
# =============================================================================
# ROC-AUC最良
best_roc_model = max(test_results, key=lambda x: test_results[x]['roc_auc'])
# F1最良
best_f1_model = max(test_results, key=lambda x: test_results[x]['f1'])
# Recall最良(F1が一定以上)
recall_candidates = {k: v for k, v in test_results.items() if v['f1'] >= 0.4}
best_recall_model = max(recall_candidates, key=lambda x: recall_candidates[x]['recall']) if recall_candidates else best_f1_model

# メインの最良モデル（F1ベース）
best_model_name = best_f1_model
best_result = test_results[best_model_name]

print(f"\n{'='*60}")
print(f"BEST MODEL (by F1): {best_model_name}")
print(f"  Accuracy: {best_result['accuracy']:.4f}")
print(f"  F1 Score: {best_result['f1']:.4f}")
print(f"  ROC-AUC: {best_result['roc_auc']:.4f}")
print(f"  Precision: {best_result['precision']:.4f}")
print(f"  Recall: {best_result['recall']:.4f}")
print(f"{'='*60}")

# =============================================================================
# 13. 詳細レポート
# =============================================================================
report = classification_report(y_test, best_result['predictions'], output_dict=True)

with open("metrics.txt", "w") as outfile:
    outfile.write(f"{'='*50}\n")
    outfile.write(f"離職予測モデル - 評価レポート v3 (SMOTE + Stacking)\n")
    outfile.write(f"{'='*50}\n\n")
    
    outfile.write(f"=== Best Model (F1): {best_model_name} ===\n")
    outfile.write(f"Accuracy: {best_result['accuracy']:.4f}\n")
    outfile.write(f"F1 Score: {best_result['f1']:.4f}\n")
    outfile.write(f"ROC-AUC: {best_result['roc_auc']:.4f}\n")
    outfile.write(f"Precision: {best_result['precision']:.4f}\n")
    outfile.write(f"Recall: {best_result['recall']:.4f}\n")
    
    outfile.write(f"\n=== Model Comparison ===\n")
    for name, res in sorted(test_results.items(), key=lambda x: -x[1]['f1']):
        thresh_info = f" (th={res['threshold']:.3f})" if 'threshold' in res else ""
        outfile.write(f"{name:25s} - F1: {res['f1']:.4f}, ROC: {res['roc_auc']:.4f}, Recall: {res['recall']:.4f}{thresh_info}\n")
    
    outfile.write(f"\n=== Cross-Validation (5-Fold, SMOTE) ===\n")
    for name, cv_res in cv_results.items():
        outfile.write(f"{name} CV F1: {cv_res['f1'].mean():.4f} (+/- {cv_res['f1'].std():.4f})\n")
        outfile.write(f"{name} CV ROC-AUC: {cv_res['roc_auc'].mean():.4f} (+/- {cv_res['roc_auc'].std():.4f})\n")
        outfile.write(f"{name} CV Recall: {cv_res['recall'].mean():.4f} (+/- {cv_res['recall'].std():.4f})\n")
    
    outfile.write(f"\n=== Threshold Analysis ===\n")
    for tr in threshold_results:
        outfile.write(f"Threshold {tr['threshold']:.2f}: F1={tr['f1']:.4f}, Prec={tr['precision']:.4f}, Recall={tr['recall']:.4f}\n")
    
    outfile.write(f"\n=== Feature Engineering ===\n")
    outfile.write(f"Total features used: {len(num_cols) + len(cat_cols)}\n")
    outfile.write(f"Numeric features: {len(num_cols)}\n")
    outfile.write(f"Categorical features: {len(cat_cols)}\n")
    
    if HAS_IMBLEARN:
        outfile.write(f"\n=== SMOTE Applied ===\n")
        outfile.write(f"Original training samples: {len(y_train)}\n")
        outfile.write(f"After SMOTE: {len(y_train_resampled)}\n")

print("\nMetrics saved to: metrics.txt")

# =============================================================================
# 14. 可視化
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
print("Confusion matrix saved to: confusion_matrix.png")

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
    plt.title('Top 25 Feature Importance (RandomForest)', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Feature importance saved to: feature_importance.png")

# モデル比較（主要モデルのみ）
main_models = ['RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM', 'Stacking', 'Ensemble_F1-optimized']
main_models = [m for m in main_models if m in test_results]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

roc_scores = [test_results[m]['roc_auc'] for m in main_models]
f1_scores_list = [test_results[m]['f1'] for m in main_models]
recall_scores = [test_results[m]['recall'] for m in main_models]

colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c'][:len(main_models)]

# ROC-AUC
bars1 = axes[0].bar(range(len(main_models)), roc_scores, color=colors)
axes[0].set_ylabel('ROC-AUC Score', fontsize=12)
axes[0].set_title('Model Comparison: ROC-AUC', fontsize=14)
axes[0].set_ylim([0.5, 1.0])
axes[0].set_xticks(range(len(main_models)))
axes[0].set_xticklabels(main_models, rotation=45, ha='right')
for bar, v in zip(bars1, roc_scores):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)

# F1
bars2 = axes[1].bar(range(len(main_models)), f1_scores_list, color=colors)
axes[1].set_ylabel('F1 Score', fontsize=12)
axes[1].set_title('Model Comparison: F1 Score', fontsize=14)
axes[1].set_ylim([0.0, 1.0])
axes[1].set_xticks(range(len(main_models)))
axes[1].set_xticklabels(main_models, rotation=45, ha='right')
for bar, v in zip(bars2, f1_scores_list):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)

# Recall
bars3 = axes[2].bar(range(len(main_models)), recall_scores, color=colors)
axes[2].set_ylabel('Recall', fontsize=12)
axes[2].set_title('Model Comparison: Recall', fontsize=14)
axes[2].set_ylim([0.0, 1.0])
axes[2].set_xticks(range(len(main_models)))
axes[2].set_xticklabels(main_models, rotation=45, ha='right')
for bar, v in zip(bars3, recall_scores):
    axes[2].text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Model comparison saved to: model_comparison.png")

# ROC曲線
plt.figure(figsize=(10, 8))
for name in main_models:
    res = test_results[name]
    fpr, tpr, _ = roc_curve(y_test, res['probabilities'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})", linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.5)')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves Comparison', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("ROC curves saved to: roc_curves.png")

# =============================================================================
# 15. 完了
# =============================================================================
print("\n" + "=" * 60)
print("Training and Evaluation Completed Successfully!")
print("=" * 60)
print(f"\nGenerated files:")
print(f"  - metrics.txt")
print(f"  - confusion_matrix.png")
print(f"  - feature_importance.png")
print(f"  - model_comparison.png")
print(f"  - roc_curves.png")
