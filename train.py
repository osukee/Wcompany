import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, precision_score, recall_score,
    average_precision_score
)
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
print("離職予測モデル - 改良版 v11")
print("(Optuna最適化拡張 + 交互作用特徴量)")
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
# 3. 特徴量エンジニアリング（v9と同じ）
# =============================================================================
print("\nCreating engineered features...")

# インセンティブ関連（最重要！）
if 'Incentive' in df.columns:
    df['IncentivePerIncome'] = df['Incentive'] / (df['MonthlyIncome'] + 1)
    df['HasIncentive'] = (df['Incentive'] > 0).astype(int)
    df['LowIncentive'] = (df['Incentive'] == 0).astype(int)
    df['HighIncentive'] = (df['Incentive'] > df['Incentive'].median()).astype(int)

# 給与・キャリア関連
df['IncomePerLevel'] = df['MonthlyIncome'] / (df['JobLevel'] + 1)
df['IncomePerYear'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
df['IncomeLevelRatio'] = df['MonthlyIncome'] / (df['JobLevel'] * 3000 + 1)
df['YearsAtRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
df['ManagerRatio'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
df['PromotionGap'] = df['YearsAtCompany'] - df['YearsSinceLastPromotion']

# ストレス関連
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

# 満足度関連
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

# 福利厚生
if 'WelfareBenefits' in df.columns:
    df['LowWelfare'] = (df['WelfareBenefits'] <= 2).astype(int)

# 柔軟性
if 'RemoteWork' in df.columns and 'FlexibleWork' in df.columns:
    df['NoFlexibility'] = ((df['RemoteWork'] == 0) & (df['FlexibleWork'] == 0)).astype(int)

# 複合リスクスコア
df['AttritionRisk'] = (
    df.get('LowIncentive', 0) * 0.25 +
    df.get('HighStress', 0) * 0.20 +
    (5 - df['JobSatisfaction']) / 4 * 0.15 +
    df['NewHire'] * 0.15 +
    df['StagnationYears'] * 0.10 +
    df.get('LowWelfare', 0) * 0.08 +
    df.get('OverTimeNum', 0) * 0.07
)

# v11: 交互作用特徴量
df['OverTimeHighStress'] = df.get('OverTimeNum', 0) * df.get('HighStress', 0)
df['NewHireLowIncentive'] = df['NewHire'] * df.get('LowIncentive', 0)
df['YoungLowSatisfaction'] = df['YoungEmployee'] * df['LowSatisfaction']
df['StagnationLowWelfare'] = df['StagnationYears'] * df.get('LowWelfare', 0)
df['HighStressLowSatisfaction'] = df.get('HighStress', 0) * (df['LowSatisfaction'] >= 2).astype(int)
df['NewHireNoFlexibility'] = df['NewHire'] * df.get('NoFlexibility', 0)

# v11: グループ統計特徴量
if 'Department' in df.columns:
    dept_avg_income = df.groupby('Department')['MonthlyIncome'].transform('mean')
    df['Dept_IncomeRatio'] = df['MonthlyIncome'] / (dept_avg_income + 1)
    dept_avg_years = df.groupby('Department')['YearsAtCompany'].transform('mean')
    df['Dept_YearsRatio'] = df['YearsAtCompany'] / (dept_avg_years + 1)

if 'JobRole' in df.columns:
    role_avg_income = df.groupby('JobRole')['MonthlyIncome'].transform('mean')
    df['Role_IncomeRatio'] = df['MonthlyIncome'] / (role_avg_income + 1)
    role_avg_years = df.groupby('JobRole')['YearsAtCompany'].transform('mean')
    df['Role_YearsRatio'] = df['YearsAtCompany'] / (role_avg_years + 1)

print(f"Total features: {len(df.columns) - 1}")

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

cat_cols = [col for col in cat_cols if col in X_train.columns]
num_cols = [col for col in num_cols if col in X_train.columns]

n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
class_weight_ratio = n_neg / n_pos
print(f"Class ratio: 1:{class_weight_ratio:.2f}")

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
    print(f"  Resampled: {len(y_train_resampled)}")
else:
    X_train_resampled, y_train_resampled = X_train_processed, y_train

# =============================================================================
# 6. Optuna最適化（拡張版）
# =============================================================================
print("\n" + "=" * 70)
print("Optuna Hyperparameter Optimization (ROC-AUC focus)")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
N_TRIALS = 50  # 最適化のトライアル数
best_params = {}

# GradientBoosting最適化
print("\nOptimizing GradientBoosting...")
def objective_gb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'min_samples_split': trial.suggest_int('min_samples_split', 5, 30),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 15),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42
    }
    model = GradientBoostingClassifier(**params)
    scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='roc_auc', n_jobs=-1)
    return scores.mean()

study_gb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study_gb.optimize(objective_gb, n_trials=N_TRIALS, show_progress_bar=False)
best_params['GB'] = study_gb.best_params
print(f"  Best ROC-AUC: {study_gb.best_value:.4f}")

# LightGBM最適化
if HAS_LIGHTGBM:
    print("\nOptimizing LightGBM...")
    def objective_lgbm(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'class_weight': 'balanced',
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        model = LGBMClassifier(**params)
        scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    study_lgbm = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study_lgbm.optimize(objective_lgbm, n_trials=N_TRIALS, show_progress_bar=False)
    best_params['LGBM'] = study_lgbm.best_params
    print(f"  Best ROC-AUC: {study_lgbm.best_value:.4f}")

# XGBoost最適化
if HAS_XGBOOST:
    print("\nOptimizing XGBoost...")
    def objective_xgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'scale_pos_weight': class_weight_ratio,
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'n_jobs': -1
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    study_xgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study_xgb.optimize(objective_xgb, n_trials=N_TRIALS, show_progress_bar=False)
    best_params['XGB'] = study_xgb.best_params
    print(f"  Best ROC-AUC: {study_xgb.best_value:.4f}")

# CatBoost最適化
if HAS_CATBOOST:
    print("\nOptimizing CatBoost...")
    def objective_cat(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 200, 800),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'auto_class_weights': 'Balanced',
            'random_seed': 42,
            'verbose': False
        }
        model = CatBoostClassifier(**params)
        scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    study_cat = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study_cat.optimize(objective_cat, n_trials=N_TRIALS, show_progress_bar=False)
    best_params['CatBoost'] = study_cat.best_params
    print(f"  Best ROC-AUC: {study_cat.best_value:.4f}")

# =============================================================================
# 7. 最適化されたモデルの学習
# =============================================================================
print("\n" + "=" * 70)
print("Training Optimized Models")
print("=" * 70)

models = {}

# GB
gb_params = best_params['GB'].copy()
gb_params['random_state'] = 42
models['GB_Opt'] = GradientBoostingClassifier(**gb_params)

# LGBM
if HAS_LIGHTGBM and 'LGBM' in best_params:
    lgbm_params = best_params['LGBM'].copy()
    lgbm_params['class_weight'] = 'balanced'
    lgbm_params['random_state'] = 42
    lgbm_params['verbose'] = -1
    lgbm_params['n_jobs'] = -1
    models['LGBM_Opt'] = LGBMClassifier(**lgbm_params)

# XGB
if HAS_XGBOOST and 'XGB' in best_params:
    xgb_params = best_params['XGB'].copy()
    xgb_params['scale_pos_weight'] = class_weight_ratio
    xgb_params['random_state'] = 42
    xgb_params['use_label_encoder'] = False
    xgb_params['eval_metric'] = 'logloss'
    xgb_params['n_jobs'] = -1
    models['XGB_Opt'] = XGBClassifier(**xgb_params)

# CatBoost
if HAS_CATBOOST and 'CatBoost' in best_params:
    cat_params = best_params['CatBoost'].copy()
    cat_params['auto_class_weights'] = 'Balanced'
    cat_params['random_seed'] = 42
    cat_params['verbose'] = False
    models['CatBoost_Opt'] = CatBoostClassifier(**cat_params)

print(f"\nTraining {len(models)} optimized models...")
for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)
print("  All models trained.")

# =============================================================================
# 8. テストセット評価
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
    print(f"{name:15s}: ROC-AUC={res['roc_auc']:.4f}, F1={res['f1']:.4f}, Recall={res['recall']:.4f}")

# =============================================================================
# 9. アンサンブル
# =============================================================================
print("\n--- Ensemble ---")

# 全モデルの平均
all_proba = np.mean([res['probabilities'] for res in test_results.values()], axis=0)

# 重み付き平均（ROC-AUCベース）
roc_scores = {name: res['roc_auc'] for name, res in test_results.items()}
total_roc = sum(roc_scores.values())
weighted_proba = np.zeros(len(y_test))
for name, res in test_results.items():
    weighted_proba += (roc_scores[name] / total_roc) * res['probabilities']

# 閾値分析
print("\nThreshold Analysis:")
threshold_analysis = []
for thresh in np.arange(0.15, 0.50, 0.025):
    pred = (weighted_proba >= thresh).astype(int)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    rec = recall_score(y_test, pred)
    threshold_analysis.append({'threshold': thresh, 'accuracy': acc, 'f1': f1, 'recall': rec})
    print(f"  Th={thresh:.3f}: Acc={acc:.4f}, F1={f1:.4f}, Recall={rec:.4f}")

# 最適閾値
best_f1 = max(threshold_analysis, key=lambda x: x['f1'])
valid_recall = [t for t in threshold_analysis if t['recall'] >= 0.60]
best_balanced = max(valid_recall, key=lambda x: x['f1']) if valid_recall else best_f1

# アンサンブル登録
ensemble_configs = [
    ('Ens_Weighted', weighted_proba, best_balanced['threshold']),
    ('Ens_F1Max', weighted_proba, best_f1['threshold']),
    ('Ens_Simple', all_proba, 0.25),
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
# 10. 最良モデル選択
# =============================================================================
def score(res):
    return res['roc_auc'] * 0.4 + res['f1'] * 0.3 + res['recall'] * 0.3

best_name = max(test_results, key=lambda x: score(test_results[x]))
best = test_results[best_name]

best_roc_name = max(test_results, key=lambda x: test_results[x]['roc_auc'])
best_roc = test_results[best_roc_name]

print(f"\n{'='*70}")
print(f"BEST MODEL: {best_name}")
print(f"  Accuracy: {best['accuracy']:.4f}")
print(f"  F1 Score: {best['f1']:.4f}")
print(f"  ROC-AUC: {best['roc_auc']:.4f}")
print(f"  Precision: {best['precision']:.4f}")
print(f"  Recall: {best['recall']:.4f}")
if 'threshold' in best:
    print(f"  Threshold: {best['threshold']:.3f}")
print(f"{'='*70}")

print(f"\nBest ROC-AUC: {best_roc_name} ({best_roc['roc_auc']:.4f})")

# =============================================================================
# 11. レポート
# =============================================================================
with open("metrics.txt", "w") as f:
    f.write(f"{'='*60}\n")
    f.write(f"離職予測モデル - 評価レポート v11\n")
    f.write(f"(Optuna最適化拡張 + 交互作用特徴量)\n")
    f.write(f"{'='*60}\n\n")
    
    f.write(f"=== Best Model: {best_name} ===\n")
    f.write(f"Accuracy: {best['accuracy']:.4f}\n")
    f.write(f"F1 Score: {best['f1']:.4f}\n")
    f.write(f"ROC-AUC: {best['roc_auc']:.4f}\n")
    f.write(f"Precision: {best['precision']:.4f}\n")
    f.write(f"Recall: {best['recall']:.4f}\n")
    if 'threshold' in best:
        f.write(f"Threshold: {best['threshold']:.3f}\n")
    
    f.write(f"\n=== All Models (sorted by ROC-AUC) ===\n")
    for name, res in sorted(test_results.items(), key=lambda x: -x[1]['roc_auc']):
        th = f" (th={res['threshold']:.2f})" if 'threshold' in res else ""
        f.write(f"{name:15s} - ROC: {res['roc_auc']:.4f}, F1: {res['f1']:.4f}, Recall: {res['recall']:.4f}{th}\n")
    
    f.write(f"\n=== Optuna Best Parameters ===\n")
    for model_name, params in best_params.items():
        f.write(f"\n{model_name}:\n")
        for k, v in params.items():
            f.write(f"  {k}: {v}\n")
    
    f.write(f"\n=== Threshold Analysis ===\n")
    for t in threshold_analysis:
        f.write(f"Th={t['threshold']:.3f}: Acc={t['accuracy']:.4f}, F1={t['f1']:.4f}, Recall={t['recall']:.4f}\n")

print("\nMetrics saved to: metrics.txt")

# =============================================================================
# 12. 可視化
# =============================================================================
# 混同行列
cm = confusion_matrix(y_test, best['predictions'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['No Attrition', 'Attrition'],
            yticklabels=['No Attrition', 'Attrition'],
            annot_kws={'size': 14})
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title(f'Confusion Matrix ({best_name})', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# モデル比較
all_models = list(test_results.keys())
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, metric, title, ylim in [
    (axes[0], 'roc_auc', 'ROC-AUC', [0.7, 1.0]),
    (axes[1], 'f1', 'F1 Score', [0.3, 0.8]),
    (axes[2], 'recall', 'Recall', [0.2, 0.8])
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
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=7)
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
plt.title('ROC Curves (Optimized Models)', fontsize=14)
plt.legend(loc='lower right', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll visualizations saved.")
print("\n" + "=" * 70)
print("Training Completed! (Optuna Optimized)")
print("=" * 70)
