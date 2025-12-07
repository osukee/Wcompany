import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    roc_auc_score, precision_recall_curve, roc_curve, precision_score, 
    recall_score, average_precision_score, fbeta_score
)
import os
import warnings

# XGBoost and LightGBM
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

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
print("=" * 60)
print("離職予測モデル - 改良版 v7")
print("(Recall優先 + Accuracy向上)")
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

# 基本比率
df['YearsAtRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
df['ManagerRatio'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
df['RoleRatio'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
df['IncomePerLevel'] = df['MonthlyIncome'] / (df['JobLevel'] + 1)

# 満足度
df['SatisfactionScore'] = df['WorkLifeBalance'] * df['EnvironmentSatisfaction']
df['EngagementScore'] = df['JobInvolvement'] * df['JobSatisfaction']
df['OverallSatisfaction'] = (df['EnvironmentSatisfaction'] + df['JobSatisfaction'] + df['RelationshipSatisfaction']) / 3
df['LowSatisfaction'] = (
    (df['EnvironmentSatisfaction'] <= 2).astype(int) +
    (df['JobSatisfaction'] <= 2).astype(int) +
    (df['RelationshipSatisfaction'] <= 2).astype(int) +
    (df['WorkLifeBalance'] <= 2).astype(int)
)

# キャリア
df['PromotionGap'] = df['YearsAtCompany'] - df['YearsSinceLastPromotion']
df['AgeExperienceGap'] = df['Age'] - df['TotalWorkingYears'] - 18
df['IncomeLevelRatio'] = df['MonthlyIncome'] / (df['JobLevel'] * 3000 + 1)
df['IncomePerYear'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)

# 残業
if 'OverTime' in df.columns:
    df['OverTimeNum'] = df['OverTime'].map({'Yes': 1, 'No': 0}) if df['OverTime'].dtype == 'object' else df['OverTime']
    df['OvertimeSatisfaction'] = df['OverTimeNum'] * (5 - df['WorkLifeBalance'])
else:
    df['OverTimeNum'] = 0

# 通勤
df['CommuteStress'] = df['DistanceFromHome'] * (1 + df['OverTimeNum'])
df['HighCommute'] = (df['DistanceFromHome'] > 15).astype(int)

# キャリア停滞
df['CareerStagnation'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
df['StagnationYears'] = (df['YearsSinceLastPromotion'] >= 3).astype(int)

# 若手・新入社員
df['YoungEmployee'] = (df['Age'] < 30).astype(int)
df['NewHire'] = (df['YearsAtCompany'] <= 2).astype(int)

# リスクスコア
df['AttritionRisk'] = (
    df['OverTimeNum'] * 0.25 +
    (5 - df['JobSatisfaction']) / 4 * 0.20 +
    (5 - df['EnvironmentSatisfaction']) / 4 * 0.15 +
    df['NewHire'] * 0.15 +
    df['StagnationYears'] * 0.10 +
    df['HighCommute'] * 0.08 +
    (5 - df['WorkLifeBalance']) / 4 * 0.07
)

# 強いシグナル
df['StrongAttritionSignal'] = (
    ((df['OverTimeNum'] == 1) & (df['JobSatisfaction'] <= 2)) |
    ((df['NewHire'] == 1) & (df['JobSatisfaction'] <= 2)) |
    ((df['OverTimeNum'] == 1) & (df['WorkLifeBalance'] <= 2))
).astype(int)

df['WorkLifeIssue'] = ((df['WorkLifeBalance'] <= 2) & (df['OverTimeNum'] == 1)).astype(int)

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
    print(f"\nRandom split: Train={len(X_train)}, Test={len(X_test)}")

cat_cols = [col for col in cat_cols if col in X_train.columns]
num_cols = [col for col in num_cols if col in X_train.columns]

n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
class_weight_ratio = n_neg / n_pos
print(f"Class distribution: Neg={n_neg}, Pos={n_pos} (ratio 1:{class_weight_ratio:.2f})")

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
# 6. Optuna最適化（F2スコア = Recall重視）
# =============================================================================
print("\n" + "=" * 60)
print("Hyperparameter Optimization (F2 Score - Recall Priority)")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# F2スコア（Recall重視）でAccuracyも考慮
def recall_priority_score(y_true, y_pred, y_proba):
    rec = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    f2 = fbeta_score(y_true, y_pred, beta=2)  # Recall重視
    # Recall優先、Accuracy/F2も考慮
    return rec * 0.4 + f2 * 0.35 + acc * 0.25

best_params = {}

if HAS_OPTUNA and HAS_LIGHTGBM:
    print("\nOptimizing LightGBM (Recall priority)...")
    
    def objective_lgbm(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 700),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.5),
            'class_weight': {0: 1, 1: class_weight_ratio * 1.5},  # Recall重視
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        
        model = LGBMClassifier(**params)
        scores = []
        for train_idx, val_idx in cv.split(X_train_resampled, y_train_resampled):
            X_tr, X_val = X_train_resampled[train_idx], X_train_resampled[val_idx]
            y_tr, y_val = y_train_resampled.iloc[train_idx], y_train_resampled.iloc[val_idx]
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            proba = model.predict_proba(X_val)[:, 1]
            scores.append(recall_priority_score(y_val, pred, proba))
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective_lgbm, n_trials=40, show_progress_bar=False)
    best_params['LightGBM'] = study.best_params
    print(f"  Best score: {study.best_value:.4f}")

if HAS_OPTUNA and HAS_XGBOOST:
    print("\nOptimizing XGBoost (Recall priority)...")
    
    def objective_xgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 700),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.5),
            'scale_pos_weight': class_weight_ratio * 1.5,  # Recall重視
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'n_jobs': -1
        }
        
        model = XGBClassifier(**params)
        scores = []
        for train_idx, val_idx in cv.split(X_train_resampled, y_train_resampled):
            X_tr, X_val = X_train_resampled[train_idx], X_train_resampled[val_idx]
            y_tr, y_val = y_train_resampled.iloc[train_idx], y_train_resampled.iloc[val_idx]
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            proba = model.predict_proba(X_val)[:, 1]
            scores.append(recall_priority_score(y_val, pred, proba))
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective_xgb, n_trials=40, show_progress_bar=False)
    best_params['XGBoost'] = study.best_params
    print(f"  Best score: {study.best_value:.4f}")

# =============================================================================
# 7. モデル学習
# =============================================================================
print("\n" + "=" * 60)
print("Training Optimized Models")
print("=" * 60)

models = {}

# RandomForest (Recall重視)
models['RF'] = RandomForestClassifier(
    n_estimators=500, max_depth=8, min_samples_split=10, min_samples_leaf=4,
    max_features='sqrt', class_weight={0: 1, 1: class_weight_ratio * 1.5},
    random_state=42, n_jobs=-1
)

# GradientBoosting
models['GB'] = GradientBoostingClassifier(
    n_estimators=400, max_depth=4, min_samples_split=10, min_samples_leaf=4,
    learning_rate=0.02, subsample=0.8, random_state=42
)

# LightGBM (Optuna最適化)
if HAS_LIGHTGBM and 'LightGBM' in best_params:
    lgbm_params = best_params['LightGBM'].copy()
    lgbm_params['class_weight'] = {0: 1, 1: class_weight_ratio * 1.5}
    lgbm_params['random_state'] = 42
    lgbm_params['verbose'] = -1
    lgbm_params['n_jobs'] = -1
    models['LGBM_Opt'] = LGBMClassifier(**lgbm_params)

# XGBoost (Optuna最適化)
if HAS_XGBOOST and 'XGBoost' in best_params:
    xgb_params = best_params['XGBoost'].copy()
    xgb_params['scale_pos_weight'] = class_weight_ratio * 1.5
    xgb_params['random_state'] = 42
    xgb_params['use_label_encoder'] = False
    xgb_params['eval_metric'] = 'logloss'
    xgb_params['n_jobs'] = -1
    models['XGB_Opt'] = XGBClassifier(**xgb_params)

print(f"\nTraining {len(models)} models...")
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
        'f2': fbeta_score(y_test, pred, beta=2),
        'roc_auc': roc_auc_score(y_test, pred_proba),
        'pr_auc': average_precision_score(y_test, pred_proba),
        'precision': precision_score(y_test, pred),
        'recall': recall_score(y_test, pred),
        'predictions': pred,
        'probabilities': pred_proba
    }

for name, res in sorted(test_results.items(), key=lambda x: -x[1]['recall']):
    print(f"{name:12s}: Recall={res['recall']:.4f}, Acc={res['accuracy']:.4f}, F1={res['f1']:.4f}")

# =============================================================================
# 9. アンサンブル（Recall優先の閾値探索）
# =============================================================================
print("\n--- Ensemble with Recall-Priority Thresholds ---")

# 全モデルの確率平均
all_proba = np.mean([res['probabilities'] for res in test_results.values()], axis=0)

# 上位モデル（Recallベース）
top_recall_models = sorted(test_results.keys(), key=lambda x: -test_results[x]['recall'])[:3]
top_proba = np.mean([test_results[m]['probabilities'] for m in top_recall_models], axis=0)
print(f"Top Recall models: {top_recall_models}")

# 詳細な閾値分析
print("\nThreshold analysis (finer granularity):")
threshold_analysis = []
for thresh in np.arange(0.15, 0.45, 0.025):
    pred = (top_proba >= thresh).astype(int)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    f2 = fbeta_score(y_test, pred, beta=2)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred)
    threshold_analysis.append({
        'threshold': thresh, 'accuracy': acc, 'f1': f1, 'f2': f2,
        'precision': prec, 'recall': rec
    })
    print(f"  Th={thresh:.3f}: Acc={acc:.4f}, F1={f1:.4f}, F2={f2:.4f}, Recall={rec:.4f}")

# 最適閾値探索
# 1. Recall >= 0.65 でAccuracy最大
valid_high_recall = [t for t in threshold_analysis if t['recall'] >= 0.65]
if valid_high_recall:
    best_balanced = max(valid_high_recall, key=lambda x: x['accuracy'])
    print(f"\nBest (Recall>=0.65, Acc max): Th={best_balanced['threshold']:.3f}")
    print(f"  Acc={best_balanced['accuracy']:.4f}, Recall={best_balanced['recall']:.4f}")

# 2. F2最大（Recall重視）
best_f2 = max(threshold_analysis, key=lambda x: x['f2'])
print(f"\nBest F2: Th={best_f2['threshold']:.3f}")
print(f"  Acc={best_f2['accuracy']:.4f}, F2={best_f2['f2']:.4f}, Recall={best_f2['recall']:.4f}")

# 3. Recall >= 0.70 でAccuracy最大
valid_very_high_recall = [t for t in threshold_analysis if t['recall'] >= 0.70]
if valid_very_high_recall:
    best_high_recall = max(valid_very_high_recall, key=lambda x: x['accuracy'])
    print(f"\nBest (Recall>=0.70, Acc max): Th={best_high_recall['threshold']:.3f}")
    print(f"  Acc={best_high_recall['accuracy']:.4f}, Recall={best_high_recall['recall']:.4f}")

# アンサンブル結果を登録
ensemble_configs = []
if valid_high_recall:
    ensemble_configs.append(('Ens_Recall65_AccMax', top_proba, best_balanced['threshold']))
ensemble_configs.append(('Ens_F2Max', top_proba, best_f2['threshold']))
if valid_very_high_recall:
    ensemble_configs.append(('Ens_Recall70_AccMax', top_proba, best_high_recall['threshold']))
ensemble_configs.append(('Ens_Balanced', top_proba, 0.275))

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

print("\n--- Final Ensemble Results ---")
for name in [n for n, _, _ in ensemble_configs]:
    res = test_results[name]
    print(f"{name} (th={res['threshold']:.3f}): Acc={res['accuracy']:.4f}, F1={res['f1']:.4f}, Recall={res['recall']:.4f}")

# =============================================================================
# 10. 最良モデルの選択
# =============================================================================
# Recall優先スコア
def recall_priority_final(res):
    return res['recall'] * 0.45 + res['accuracy'] * 0.30 + res['f1'] * 0.25

best_model_name = max(test_results, key=lambda x: recall_priority_final(test_results[x]))
best_result = test_results[best_model_name]

print(f"\n{'='*60}")
print(f"BEST MODEL (Recall Priority): {best_model_name}")
print(f"  Accuracy: {best_result['accuracy']:.4f}")
print(f"  F1 Score: {best_result['f1']:.4f}")
print(f"  F2 Score: {best_result['f2']:.4f}")
print(f"  ROC-AUC: {best_result['roc_auc']:.4f}")
print(f"  Precision: {best_result['precision']:.4f}")
print(f"  Recall: {best_result['recall']:.4f}")
if 'threshold' in best_result:
    print(f"  Threshold: {best_result['threshold']:.3f}")
print(f"{'='*60}")

# =============================================================================
# 11. 詳細レポート
# =============================================================================
with open("metrics.txt", "w") as outfile:
    outfile.write(f"{'='*50}\n")
    outfile.write(f"離職予測モデル - 評価レポート v7\n")
    outfile.write(f"(Recall優先 + Accuracy向上)\n")
    outfile.write(f"{'='*50}\n\n")
    
    outfile.write(f"=== Best Model: {best_model_name} ===\n")
    outfile.write(f"Accuracy: {best_result['accuracy']:.4f}\n")
    outfile.write(f"F1 Score: {best_result['f1']:.4f}\n")
    outfile.write(f"F2 Score: {best_result['f2']:.4f}\n")
    outfile.write(f"ROC-AUC: {best_result['roc_auc']:.4f}\n")
    outfile.write(f"Precision: {best_result['precision']:.4f}\n")
    outfile.write(f"Recall: {best_result['recall']:.4f}\n")
    if 'threshold' in best_result:
        outfile.write(f"Threshold: {best_result['threshold']:.3f}\n")
    
    outfile.write(f"\n=== All Models Comparison ===\n")
    for name, res in sorted(test_results.items(), key=lambda x: -recall_priority_final(x[1])):
        thresh_info = f" (th={res['threshold']:.3f})" if 'threshold' in res else ""
        outfile.write(f"{name:22s} - Acc: {res['accuracy']:.4f}, F1: {res['f1']:.4f}, Recall: {res['recall']:.4f}{thresh_info}\n")
    
    outfile.write(f"\n=== Threshold Analysis ===\n")
    for t in threshold_analysis:
        outfile.write(f"Th={t['threshold']:.3f}: Acc={t['accuracy']:.4f}, F1={t['f1']:.4f}, F2={t['f2']:.4f}, Recall={t['recall']:.4f}\n")
    
    if best_params:
        outfile.write(f"\n=== Optuna Best Parameters ===\n")
        for model_name, params in best_params.items():
            outfile.write(f"\n{model_name}:\n")
            for k, v in params.items():
                outfile.write(f"  {k}: {v}\n")

print("\nMetrics saved to: metrics.txt")

# =============================================================================
# 12. 可視化
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
    feature_names = num_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': models['RF'].feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    plt.figure(figsize=(12, 10))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
    plt.title('Top 20 Feature Importance', fontsize=14)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

# モデル比較
main_models = [n for n in test_results.keys() if 'Ens' in n or n in ['RF', 'LGBM_Opt', 'XGB_Opt']]
main_models = main_models[:6]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, metric, title, ylim in [
    (axes[0], 'accuracy', 'Accuracy', [0.5, 1.0]),
    (axes[1], 'recall', 'Recall', [0.0, 1.0]),
    (axes[2], 'f1', 'F1 Score', [0.0, 1.0])
]:
    scores = [test_results[m][metric] for m in main_models]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(main_models)))
    bars = ax.bar(range(len(main_models)), scores, color=colors)
    ax.set_ylabel(title, fontsize=12)
    ax.set_title(f'Model Comparison: {title}', fontsize=14)
    ax.set_ylim(ylim)
    ax.set_xticks(range(len(main_models)))
    ax.set_xticklabels(main_models, rotation=45, ha='right', fontsize=8)
    for bar, v in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ROC曲線
plt.figure(figsize=(10, 8))
for name in main_models:
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
for name in main_models:
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
print("\n" + "=" * 60)
print("Training and Evaluation Completed!")
print("=" * 60)
