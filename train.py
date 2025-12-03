import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import os

# 1. データの読み込み
df = pd.read_csv('data.csv')

# 2. 前処理
# 不要そうなカラムの削除（IDや定数など）
drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
df = df.drop(columns=drop_cols, errors='ignore')

# ターゲット変数のエンコーディング (Attrition: Yes=1, No=0)
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# カテゴリ変数のエンコーディング
cat_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 特徴量とターゲットに分離
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# 3. データ分割（Yearカラムがある場合は時系列分割、ない場合はランダム分割）
if 'Year' in df.columns:
    # Yearカラムがある場合：2023年で学習、2024年でテスト
    train_mask = df['Year'] == 2023
    test_mask = df['Year'] == 2024
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"Training data: {len(X_train)} samples (Year 2023)")
    print(f"Test data: {len(X_test)} samples (Year 2024)")
else:
    # Yearカラムがない場合：ランダム分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training data: {len(X_train)} samples")
    print(f"Test data: {len(X_test)} samples")

# 4. 実行 (Execute): モデル学習
# ※計画(Plan)段階でここのパラメータをいじってPushすることを想定
# 精度向上のための改善:
# - n_estimatorsを200に増加（より多くの決定木で精度向上）
# - max_depthを15に増加（より深い木で複雑なパターンを学習）
# - class_weight='balanced'を追加（不均衡データに対応）
# - min_samples_splitを5に設定（過学習を防ぐ）
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',  # 不均衡データに対応
    random_state=42,
    n_jobs=-1  # 並列処理で高速化
)
model.fit(X_train, y_train)

# 5. 評価 (Check): 予測とスコア算出
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 詳細な分類レポート
report = classification_report(y_test, y_pred, output_dict=True)

# 6. フィードバック用レポートの作成
# メトリクスの書き出し
with open("metrics.txt", "w") as outfile:
    outfile.write(f"Accuracy: {acc:.4f}\n")
    outfile.write(f"F1 Score: {f1:.4f}\n")
    outfile.write(f"Precision: {report['1']['precision']:.4f}\n")
    outfile.write(f"Recall: {report['1']['recall']:.4f}\n")

# 混同行列のグラフ作成
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# 特徴量重要度の可視化（上位10個）
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Top 10 Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

print("Training and Evaluation Completed.")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")

