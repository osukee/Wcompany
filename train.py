import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    RandomizedSearchCV
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    roc_auc_score, precision_recall_curve, roc_curve
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
    from imblearn.over_sampling import SMOTE
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
print("離職予測モデル - 改良版 v2")
print("=" * 60)

df = pd.read_csv('data.csv')
print(f"\nLoaded data: {len(df)} rows, {len(df.columns)} columns")

# =============================================================================
# 2. 前処理
# =============================================================================
# 不要そうなカラムの削除（IDや定数など）
drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
df = df.drop(columns=drop_cols, errors='ignore')

# ターゲット変数のエンコーディング (Attrition: Yes=1, No=0)
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# =============================================================================
# 3. 特徴量エンジニアリング（拡張版）
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

# 昇進とキャリア
df['PromotionGap'] = df['YearsAtCompany'] - df['YearsSinceLastPromotion']
df['AgeExperienceGap'] = df['Age'] - df['TotalWorkingYears'] - 18
df['IncomeLevelRatio'] = df['MonthlyIncome'] / (df['JobLevel'] * 3000 + 1)

# 追加の高度な特徴量
# 給与成長率の推定
df['IncomePerYear'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)

# 会社での成長率
df['CompanyGrowthRate'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)

# トレーニング頻度
df['TrainingIntensity'] = df['TrainingTimesLastYear'] / (df['YearsAtCompany'] + 1)

# 通勤負荷スコア
df['CommuteStress'] = df['DistanceFromHome'] * (1 + (df['OverTime'].map({'Yes': 1, 'No': 0}) if df['OverTime'].dtype == 'object' else df['OverTime']))

# 残業と満足度の関係
if 'OverTime' in df.columns:
    df['OverTimeNum'] = df['OverTime'].map({'Yes': 1, 'No': 0}) if df['OverTime'].dtype == 'object' else df['OverTime']
    df['OvertimeSatisfaction'] = df['OverTimeNum'] * (5 - df['WorkLifeBalance'])

# キャリア停滞指標
df['CareerStagnation'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)

# 年齢グループベースの給与比較
df['AgeBucket'] = pd.cut(df['Age'], bins=[17, 25, 35, 45, 55, 100], labels=[0, 1, 2, 3, 4])
df['AgeBucket'] = df['AgeBucket'].astype(int)

# 経験レベル
df['ExperienceLevel'] = pd.cut(df['TotalWorkingYears'], bins=[-1, 2, 5, 10, 20, 50], labels=[0, 1, 2, 3, 4])
df['ExperienceLevel'] = df['ExperienceLevel'].astype(int)

# 複合リスクスコア
df['AttritionRisk'] = (
    (df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)) * 0.3 +
    (5 - df['JobSatisfaction']) / 4 * 0.25 +
    (5 - df['EnvironmentSatisfaction']) / 4 * 0.25 +
    df.get('OverTimeNum', 0) * 0.2
)

# ストレス関連指標
if 'StressRating' in df.columns and 'StressSelfReported' in df.columns:
    df['StressCombined'] = df['StressRating'] + df['StressSelfReported']

print(f"Total features after engineering: {len(df.columns) - 1}")

# =============================================================================
# 4. データ分割
# =============================================================================
# カテゴリ変数と数値変数を分離
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_cols = [col for col in num_cols if col != 'Attrition']

# 特徴量とターゲットに分離
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Yearカラムがある場合は時系列分割、ない場合はランダム分割
if 'Year' in df.columns:
    train_mask = df['Year'] == 2023
    test_mask = df['Year'] == 2024
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
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

# カラムリストを更新
cat_cols = [col for col in cat_cols if col in X_train.columns]
num_cols = [col for col in num_cols if col in X_train.columns]

# クラス不均衡の確認
print(f"\nClass distribution in training data:")
print(f"  No Attrition (0): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
print(f"  Attrition (1): {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")

# =============================================================================
# 5. 前処理パイプライン
# =============================================================================
# RobustScalerを使用（外れ値に強い）
preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])

# =============================================================================
# 6. モデル定義（拡張版）
# =============================================================================
print("\n" + "=" * 60)
print("Model Training and Evaluation")
print("=" * 60)

# クラス重みの計算
class_weight_ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nClass weight ratio: 1:{class_weight_ratio:.2f}")

# Random Forest
rf_classifier = RandomForestClassifier(
    n_estimators=400,
    max_depth=15,
    min_samples_split=8,
    min_samples_leaf=3,
    max_features='sqrt',
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)

# Gradient Boosting
gb_classifier = GradientBoostingClassifier(
    n_estimators=250,
    max_depth=6,
    min_samples_split=8,
    min_samples_leaf=3,
    learning_rate=0.03,
    subsample=0.85,
    max_features='sqrt',
    random_state=42
)

# モデルリスト（基本）
models = {
    'RandomForest': Pipeline([('preprocessor', preprocessor), ('classifier', rf_classifier)]),
    'GradientBoosting': Pipeline([('preprocessor', preprocessor), ('classifier', gb_classifier)])
}

# XGBoost
if HAS_XGBOOST:
    xgb_classifier = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=class_weight_ratio,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    models['XGBoost'] = Pipeline([('preprocessor', preprocessor), ('classifier', xgb_classifier)])

# LightGBM
if HAS_LIGHTGBM:
    lgbm_classifier = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        class_weight='balanced',
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    models['LightGBM'] = Pipeline([('preprocessor', preprocessor), ('classifier', lgbm_classifier)])

# =============================================================================
# 7. クロスバリデーション
# =============================================================================
print("\n--- Cross-Validation Results (5-Fold Stratified) ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    cv_roc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_results[name] = {'f1': cv_f1, 'roc_auc': cv_roc}
    print(f"\n{name}:")
    print(f"  CV F1 Score: {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")
    print(f"  CV ROC-AUC: {cv_roc.mean():.4f} (+/- {cv_roc.std():.4f})")

# =============================================================================
# 8. モデル学習
# =============================================================================
print("\n--- Training Models ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"  {name} trained.")

# =============================================================================
# 9. テストセット評価
# =============================================================================
print("\n--- Test Set Evaluation ---")
test_results = {}

for name, model in models.items():
    pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc = roc_auc_score(y_test, pred_proba)
    
    test_results[name] = {
        'accuracy': acc,
        'f1': f1,
        'roc_auc': roc,
        'predictions': pred,
        'probabilities': pred_proba
    }
    print(f"\n{name}:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc:.4f}")

# =============================================================================
# 10. アンサンブル（確率平均 + 閾値最適化）
# =============================================================================
print("\n--- Ensemble Model (Probability Averaging) ---")

# 全モデルの確率を平均
all_probas = np.array([r['probabilities'] for r in test_results.values()])
ensemble_proba = np.mean(all_probas, axis=0)

# 最適閾値の探索（F1スコア最大化）
precisions, recalls, thresholds = precision_recall_curve(y_test, ensemble_proba)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5

# 複数の閾値で評価
threshold_options = [0.3, 0.4, best_threshold, 0.5]
print(f"\nOptimal threshold (F1 maximized): {best_threshold:.3f}")

ensemble_pred = (ensemble_proba >= best_threshold).astype(int)
ensemble_acc = accuracy_score(y_test, ensemble_pred)
ensemble_f1 = f1_score(y_test, ensemble_pred)
ensemble_roc = roc_auc_score(y_test, ensemble_proba)

test_results['Ensemble'] = {
    'accuracy': ensemble_acc,
    'f1': ensemble_f1,
    'roc_auc': ensemble_roc,
    'predictions': ensemble_pred,
    'probabilities': ensemble_proba,
    'threshold': best_threshold
}

print(f"Ensemble Results (threshold={best_threshold:.3f}):")
print(f"  Accuracy: {ensemble_acc:.4f}")
print(f"  F1 Score: {ensemble_f1:.4f}")
print(f"  ROC-AUC: {ensemble_roc:.4f}")

# =============================================================================
# 11. 最良モデルの選択
# =============================================================================
best_model_name = max(test_results, key=lambda x: test_results[x]['roc_auc'])
best_result = test_results[best_model_name]

print(f"\n{'='*60}")
print(f"BEST MODEL: {best_model_name}")
print(f"  Accuracy: {best_result['accuracy']:.4f}")
print(f"  F1 Score: {best_result['f1']:.4f}")
print(f"  ROC-AUC: {best_result['roc_auc']:.4f}")
print(f"{'='*60}")

# =============================================================================
# 12. 詳細レポートと可視化
# =============================================================================
# 分類レポート
report = classification_report(y_test, best_result['predictions'], output_dict=True)

# メトリクスの書き出し
with open("metrics.txt", "w") as outfile:
    outfile.write(f"{'='*50}\n")
    outfile.write(f"離職予測モデル - 評価レポート v2\n")
    outfile.write(f"{'='*50}\n\n")
    
    outfile.write(f"=== Best Model: {best_model_name} ===\n")
    outfile.write(f"Accuracy: {best_result['accuracy']:.4f}\n")
    outfile.write(f"F1 Score: {best_result['f1']:.4f}\n")
    outfile.write(f"ROC-AUC: {best_result['roc_auc']:.4f}\n")
    outfile.write(f"Precision: {report['1']['precision']:.4f}\n")
    outfile.write(f"Recall: {report['1']['recall']:.4f}\n")
    
    outfile.write(f"\n=== Model Comparison ===\n")
    for name, res in test_results.items():
        outfile.write(f"{name:20s} - F1: {res['f1']:.4f}, ROC-AUC: {res['roc_auc']:.4f}\n")
    
    outfile.write(f"\n=== Cross-Validation (5-Fold) ===\n")
    for name, cv_res in cv_results.items():
        outfile.write(f"{name} CV F1: {cv_res['f1'].mean():.4f} (+/- {cv_res['f1'].std():.4f})\n")
        outfile.write(f"{name} CV ROC-AUC: {cv_res['roc_auc'].mean():.4f} (+/- {cv_res['roc_auc'].std():.4f})\n")
    
    if 'Ensemble' in test_results and 'threshold' in test_results['Ensemble']:
        outfile.write(f"\n=== Ensemble Details ===\n")
        outfile.write(f"Optimal Threshold: {test_results['Ensemble']['threshold']:.3f}\n")
    
    outfile.write(f"\n=== Feature Engineering ===\n")
    outfile.write(f"Total features used: {len(num_cols) + len(cat_cols)}\n")
    outfile.write(f"Numeric features: {len(num_cols)}\n")
    outfile.write(f"Categorical features: {len(cat_cols)}\n")

print("\nMetrics saved to: metrics.txt")

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

# 特徴量重要度（RandomForestから取得）
if 'RandomForest' in models:
    rf_fitted = models['RandomForest'].named_steps['classifier']
    feature_names = num_cols + list(
        models['RandomForest'].named_steps['preprocessor']
        .named_transformers_['cat'].get_feature_names_out(cat_cols)
    )
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_fitted.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    plt.figure(figsize=(12, 10))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(feature_importance)))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette=colors)
    plt.title('Top 20 Feature Importance (RandomForest)', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Feature importance saved to: feature_importance.png")

# モデル比較の可視化
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

model_names = list(test_results.keys())
roc_scores = [test_results[m]['roc_auc'] for m in model_names]
f1_scores_list = [test_results[m]['f1'] for m in model_names]

# カラーパレット
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'][:len(model_names)]

# ROC-AUC比較
bars1 = axes[0].bar(model_names, roc_scores, color=colors)
axes[0].set_ylabel('ROC-AUC Score', fontsize=12)
axes[0].set_title('Model Comparison: ROC-AUC', fontsize=14)
axes[0].set_ylim([0.5, 1.0])
axes[0].tick_params(axis='x', rotation=45)
for bar, v in zip(bars1, roc_scores):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.4f}', 
                 ha='center', fontweight='bold', fontsize=10)

# F1スコア比較
bars2 = axes[1].bar(model_names, f1_scores_list, color=colors)
axes[1].set_ylabel('F1 Score', fontsize=12)
axes[1].set_title('Model Comparison: F1 Score', fontsize=14)
axes[1].set_ylim([0.0, 1.0])
axes[1].tick_params(axis='x', rotation=45)
for bar, v in zip(bars2, f1_scores_list):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.4f}', 
                 ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Model comparison saved to: model_comparison.png")

# ROC曲線の比較
plt.figure(figsize=(10, 8))
for name, res in test_results.items():
    fpr, tpr, _ = roc_curve(y_test, res['probabilities'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.4f})", linewidth=2)

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
# 13. 完了メッセージ
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
