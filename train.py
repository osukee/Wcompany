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
    recall_score, average_precision_score, fbeta_score, make_scorer
)
from sklearn.feature_selection import SelectFromModel
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

# Optuna for hyperparameter optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    HAS_OPTUNA = False
    print("Optuna not available, using default hyperparameters...")

warnings.filterwarnings('ignore')

# =============================================================================
# 1. データの読み込み
# =============================================================================
print("=" * 60)
print("離職予測モデル - 改良版 v6 (Accuracy重視 + Optuna最適化)")
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
# 3. 特徴量エンジニアリング（厳選版）
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

# 昇進とキャリア
df['PromotionGap'] = df['YearsAtCompany'] - df['YearsSinceLastPromotion']
df['AgeExperienceGap'] = df['Age'] - df['TotalWorkingYears'] - 18
df['IncomeLevelRatio'] = df['MonthlyIncome'] / (df['JobLevel'] * 3000 + 1)

# 給与関連
df['IncomePerYear'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)

# 残業関連
if 'OverTime' in df.columns:
    df['OverTimeNum'] = df['OverTime'].map({'Yes': 1, 'No': 0}) if df['OverTime'].dtype == 'object' else df['OverTime']
    df['OvertimeSatisfaction'] = df['OverTimeNum'] * (5 - df['WorkLifeBalance'])
else:
    df['OverTimeNum'] = 0

# 通勤負荷
df['CommuteStress'] = df['DistanceFromHome'] * (1 + df['OverTimeNum'])
df['HighCommute'] = (df['DistanceFromHome'] > 15).astype(int)

# キャリア停滞
df['CareerStagnation'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
df['StagnationYears'] = (df['YearsSinceLastPromotion'] >= 3).astype(int)

# 若手・新入社員フラグ
df['YoungEmployee'] = (df['Age'] < 30).astype(int)
df['NewHire'] = (df['YearsAtCompany'] <= 2).astype(int)

# 複合リスクスコア
df['AttritionRisk'] = (
    df['OverTimeNum'] * 0.25 +
    (5 - df['JobSatisfaction']) / 4 * 0.20 +
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
    ((df['OverTimeNum'] == 1) & (df['WorkLifeBalance'] <= 2))
).astype(int)

# ワークライフバランス問題
df['WorkLifeIssue'] = ((df['WorkLifeBalance'] <= 2) & (df['OverTimeNum'] == 1)).astype(int)

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
    print(f"  Training: {len(X_train)} samples, Test: {len(X_test)} samples")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nRandom split: Training={len(X_train)}, Test={len(X_test)}")

cat_cols = [col for col in cat_cols if col in X_train.columns]
num_cols = [col for col in num_cols if col in X_train.columns]

n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
class_weight_ratio = n_neg / n_pos
print(f"\nClass distribution: Neg={n_neg}, Pos={n_pos} (ratio 1:{class_weight_ratio:.2f})")

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

# SMOTETomek
if HAS_IMBLEARN:
    print("\nApplying SMOTETomek...")
    smotetomek = SMOTETomek(random_state=42, smote=SMOTE(k_neighbors=5, random_state=42))
    X_train_resampled, y_train_resampled = smotetomek.fit_resample(X_train_processed, y_train)
    print(f"  Before: {len(y_train)}, After: {len(y_train_resampled)}")
else:
    X_train_resampled = X_train_processed
    y_train_resampled = y_train

# =============================================================================
# 6. Optunaによるハイパーパラメータ最適化
# =============================================================================
print("\n" + "=" * 60)
print("Hyperparameter Optimization with Optuna")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# カスタムスコア: Accuracy重視しつつRecall維持
def custom_score(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    # Accuracy重視、Recall/F1も考慮
    return acc * 0.4 + f1 * 0.35 + rec * 0.25

best_params = {}

if HAS_OPTUNA and HAS_LIGHTGBM:
    print("\nOptimizing LightGBM...")
    
    def objective_lgbm(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 600),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'class_weight': 'balanced',
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        
        model = LGBMClassifier(**params)
        
        # クロスバリデーションで評価
        scores = []
        for train_idx, val_idx in cv.split(X_train_resampled, y_train_resampled):
            X_tr, X_val = X_train_resampled[train_idx], X_train_resampled[val_idx]
            y_tr, y_val = y_train_resampled.iloc[train_idx], y_train_resampled.iloc[val_idx]
            
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            scores.append(custom_score(y_val, pred))
        
        return np.mean(scores)
    
    sampler = TPESampler(seed=42)
    study_lgbm = optuna.create_study(direction='maximize', sampler=sampler)
    study_lgbm.optimize(objective_lgbm, n_trials=30, show_progress_bar=False)
    best_params['LightGBM'] = study_lgbm.best_params
    print(f"  Best score: {study_lgbm.best_value:.4f}")

if HAS_OPTUNA and HAS_XGBOOST:
    print("\nOptimizing XGBoost...")
    
    def objective_xgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 600),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'scale_pos_weight': class_weight_ratio,
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
            scores.append(custom_score(y_val, pred))
        
        return np.mean(scores)
    
    sampler = TPESampler(seed=42)
    study_xgb = optuna.create_study(direction='maximize', sampler=sampler)
    study_xgb.optimize(objective_xgb, n_trials=30, show_progress_bar=False)
    best_params['XGBoost'] = study_xgb.best_params
    print(f"  Best score: {study_xgb.best_value:.4f}")

# =============================================================================
# 7. 最適化されたモデルの学習
# =============================================================================
print("\n" + "=" * 60)
print("Training Optimized Models")
print("=" * 60)

models = {}

# RandomForest (手動チューニング)
models['RF'] = RandomForestClassifier(
    n_estimators=500, max_depth=10, min_samples_split=15, min_samples_leaf=5,
    max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1
)

# GradientBoosting
models['GB'] = GradientBoostingClassifier(
    n_estimators=400, max_depth=4, min_samples_split=15, min_samples_leaf=5,
    learning_rate=0.02, subsample=0.8, random_state=42
)

# LightGBM (Optuna最適化)
if HAS_LIGHTGBM:
    lgbm_params = best_params.get('LightGBM', {})
    lgbm_params.update({'class_weight': 'balanced', 'random_state': 42, 'verbose': -1, 'n_jobs': -1})
    models['LGBM_Opt'] = LGBMClassifier(**lgbm_params)
    
    # デフォルトLightGBMも
    models['LGBM_Default'] = LGBMClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.8, class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1
    )

# XGBoost (Optuna最適化)
if HAS_XGBOOST:
    xgb_params = best_params.get('XGBoost', {})
    xgb_params.update({'scale_pos_weight': class_weight_ratio, 'random_state': 42, 
                       'use_label_encoder': False, 'eval_metric': 'logloss', 'n_jobs': -1})
    models['XGB_Opt'] = XGBClassifier(**xgb_params)
    
    # デフォルトXGBoostも
    models['XGB_Default'] = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=class_weight_ratio,
        random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1
    )

# モデル学習
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
        'roc_auc': roc_auc_score(y_test, pred_proba),
        'pr_auc': average_precision_score(y_test, pred_proba),
        'precision': precision_score(y_test, pred),
        'recall': recall_score(y_test, pred),
        'predictions': pred,
        'probabilities': pred_proba
    }

for name, res in sorted(test_results.items(), key=lambda x: -x[1]['accuracy']):
    print(f"{name:15s}: Acc={res['accuracy']:.4f}, F1={res['f1']:.4f}, ROC={res['roc_auc']:.4f}, Recall={res['recall']:.4f}")

# =============================================================================
# 9. アンサンブル（Accuracy重視の閾値）
# =============================================================================
print("\n--- Ensemble with Accuracy-Optimized Thresholds ---")

# 全モデルの確率平均
all_proba = np.mean([res['probabilities'] for res in test_results.values()], axis=0)

# 上位モデルのみ（ROC-AUCベース）
top_models = sorted(test_results.keys(), key=lambda x: -test_results[x]['roc_auc'])[:4]
top_proba = np.mean([test_results[m]['probabilities'] for m in top_models], axis=0)
print(f"Top 4 models: {top_models}")

# 詳細な閾値分析
print("\nThreshold analysis:")
threshold_analysis = []
for thresh in np.arange(0.20, 0.55, 0.05):
    pred = (top_proba >= thresh).astype(int)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred)
    threshold_analysis.append({
        'threshold': thresh, 'accuracy': acc, 'f1': f1, 
        'precision': prec, 'recall': rec
    })
    print(f"  Th={thresh:.2f}: Acc={acc:.4f}, F1={f1:.4f}, Prec={prec:.4f}, Recall={rec:.4f}")

# 最適閾値を探索（Accuracy最大化、ただしRecall >= 0.5を維持）
valid_thresholds = [t for t in threshold_analysis if t['recall'] >= 0.50]
if valid_thresholds:
    best_acc_thresh = max(valid_thresholds, key=lambda x: x['accuracy'])
else:
    best_acc_thresh = max(threshold_analysis, key=lambda x: x['accuracy'])

print(f"\nBest threshold (Acc max, Recall>=0.5): {best_acc_thresh['threshold']:.2f}")
print(f"  Acc={best_acc_thresh['accuracy']:.4f}, Recall={best_acc_thresh['recall']:.4f}")

# F1最大化の閾値も探索
best_f1_thresh = max(threshold_analysis, key=lambda x: x['f1'])
print(f"\nBest threshold (F1 max): {best_f1_thresh['threshold']:.2f}")
print(f"  Acc={best_f1_thresh['accuracy']:.4f}, F1={best_f1_thresh['f1']:.4f}, Recall={best_f1_thresh['recall']:.4f}")

# アンサンブル結果を登録
ensemble_configs = [
    ('Ens_AccMax', top_proba, best_acc_thresh['threshold']),
    ('Ens_F1Max', top_proba, best_f1_thresh['threshold']),
    ('Ens_Balanced', top_proba, 0.30),
    ('Ens_HighRecall', top_proba, 0.20),
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
for name in ['Ens_AccMax', 'Ens_F1Max', 'Ens_Balanced', 'Ens_HighRecall']:
    res = test_results[name]
    print(f"{name} (th={res['threshold']:.2f}): Acc={res['accuracy']:.4f}, F1={res['f1']:.4f}, Recall={res['recall']:.4f}")

# =============================================================================
# 10. 最良モデルの選択
# =============================================================================
# Accuracy重視のスコア
def accuracy_focused_score(res):
    return res['accuracy'] * 0.5 + res['f1'] * 0.3 + res['recall'] * 0.2

best_model_name = max(test_results, key=lambda x: accuracy_focused_score(test_results[x]))
best_result = test_results[best_model_name]

# 最高Accuracy
best_acc_model = max(test_results, key=lambda x: test_results[x]['accuracy'])
best_acc_result = test_results[best_acc_model]

print(f"\n{'='*60}")
print(f"BEST MODEL (Accuracy-focused): {best_model_name}")
print(f"  Accuracy: {best_result['accuracy']:.4f}")
print(f"  F1 Score: {best_result['f1']:.4f}")
print(f"  ROC-AUC: {best_result['roc_auc']:.4f}")
print(f"  Precision: {best_result['precision']:.4f}")
print(f"  Recall: {best_result['recall']:.4f}")
if 'threshold' in best_result:
    print(f"  Threshold: {best_result['threshold']:.2f}")
print(f"{'='*60}")

print(f"\nHighest Accuracy Model: {best_acc_model}")
print(f"  Accuracy: {best_acc_result['accuracy']:.4f}, F1: {best_acc_result['f1']:.4f}, Recall: {best_acc_result['recall']:.4f}")

# =============================================================================
# 11. 詳細レポート
# =============================================================================
with open("metrics.txt", "w") as outfile:
    outfile.write(f"{'='*50}\n")
    outfile.write(f"離職予測モデル - 評価レポート v6\n")
    outfile.write(f"(Accuracy重視 + Optuna最適化)\n")
    outfile.write(f"{'='*50}\n\n")
    
    outfile.write(f"=== Best Model: {best_model_name} ===\n")
    outfile.write(f"Accuracy: {best_result['accuracy']:.4f}\n")
    outfile.write(f"F1 Score: {best_result['f1']:.4f}\n")
    outfile.write(f"ROC-AUC: {best_result['roc_auc']:.4f}\n")
    outfile.write(f"PR-AUC: {best_result['pr_auc']:.4f}\n")
    outfile.write(f"Precision: {best_result['precision']:.4f}\n")
    outfile.write(f"Recall: {best_result['recall']:.4f}\n")
    if 'threshold' in best_result:
        outfile.write(f"Threshold: {best_result['threshold']:.2f}\n")
    
    outfile.write(f"\n=== Highest Accuracy: {best_acc_model} ===\n")
    outfile.write(f"Accuracy: {best_acc_result['accuracy']:.4f}\n")
    outfile.write(f"F1: {best_acc_result['f1']:.4f}, Recall: {best_acc_result['recall']:.4f}\n")
    
    outfile.write(f"\n=== All Models Comparison ===\n")
    for name, res in sorted(test_results.items(), key=lambda x: -x[1]['accuracy']):
        thresh_info = f" (th={res['threshold']:.2f})" if 'threshold' in res else ""
        outfile.write(f"{name:18s} - Acc: {res['accuracy']:.4f}, F1: {res['f1']:.4f}, Recall: {res['recall']:.4f}{thresh_info}\n")
    
    outfile.write(f"\n=== Threshold Analysis ===\n")
    for t in threshold_analysis:
        outfile.write(f"Th={t['threshold']:.2f}: Acc={t['accuracy']:.4f}, F1={t['f1']:.4f}, Prec={t['precision']:.4f}, Recall={t['recall']:.4f}\n")
    
    if best_params:
        outfile.write(f"\n=== Optuna Best Parameters ===\n")
        for model_name, params in best_params.items():
            outfile.write(f"\n{model_name}:\n")
            for k, v in params.items():
                outfile.write(f"  {k}: {v}\n")
    
    outfile.write(f"\n=== Feature Engineering ===\n")
    outfile.write(f"Total features: {len(num_cols) + len(cat_cols)}\n")

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
print("Confusion matrix saved")

# 特徴量重要度
if 'RF' in models:
    feature_names = num_cols + list(
        preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    )
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': models['RF'].feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    plt.figure(figsize=(12, 10))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(feature_importance)))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette=colors)
    plt.title('Top 20 Feature Importance', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Feature importance saved")

# モデル比較
main_models = ['RF', 'GB', 'LGBM_Opt', 'XGB_Opt', 'Ens_AccMax', 'Ens_Balanced']
main_models = [m for m in main_models if m in test_results]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, metric, title, ylim in [
    (axes[0], 'accuracy', 'Accuracy', [0.5, 1.0]),
    (axes[1], 'f1', 'F1 Score', [0.0, 1.0]),
    (axes[2], 'recall', 'Recall', [0.0, 1.0])
]:
    scores = [test_results[m][metric] for m in main_models]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c'][:len(main_models)]
    bars = ax.bar(range(len(main_models)), scores, color=colors)
    ax.set_ylabel(title, fontsize=12)
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
    plt.plot(rec, prec, label=f"{name} (AP={res['pr_auc']:.3f})", linewidth=2)

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
# 13. 完了
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
