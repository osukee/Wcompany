# 離職率予測モデル - MLOps ワークフロー

このプロジェクトは、人事データ・離職データを使用して離職率を予測する機械学習モデルを、**Plan-Do-Check-Act (PDCA) サイクル**で自動化するワークフローです。

## 📋 概要

GitHub Actions と CML (Continuous Machine Learning) を組み合わせることで、以下のサイクルを自動化します：

1. **計画 (Plan)**: ハイパーパラメータの変更や特徴量エンジニアリングを行い、ブランチにPush
2. **実行 (Execute)**: GitHub Actionsが自動でモデルを学習
3. **評価 (Check)**: テストデータで精度を検証し、混同行列や特徴量重要度を可視化
4. **フィードバック (Feedback)**: 結果をPRにコメントとして自動投稿

## 🏗️ プロジェクト構成

```
.
├── .github/
│   └── workflows/
│       └── mlops.yml      # GitHub Actionsの設定ファイル
├── data.csv               # 人事データ・離職データ
├── train.py               # 学習・評価を行うスクリプト
├── requirements.txt       # 必要なライブラリ
└── README.md              # このファイル
```

## 🚀 セットアップ

### 1. 必要なライブラリのインストール

```bash
pip install -r requirements.txt
```

### 2. データの準備

`data.csv` ファイルをプロジェクトルートに配置してください。

データには以下のカラムが必要です：
- `Attrition`: 離職フラグ（"Yes" または "No"）
- `Year`: 年（オプション。ある場合は2023年で学習、2024年でテスト）

その他のカラムは特徴量として使用されます。

## 🔄 使い方（改善サイクル）

### ステップ1: 計画 (Plan)

モデルの改善を試みます。例えば：

- ハイパーパラメータの調整（`train.py` の `RandomForestClassifier` のパラメータ）
- 特徴量エンジニアリング（前処理の変更）
- モデルの変更

新しいブランチを作成し、変更をコミット・Pushします：

```bash
git checkout -b improve-model
# train.py を編集
git add train.py
git commit -m "max_depthを15に変更して精度向上を試みる"
git push origin improve-model
```

### ステップ2: Pull Request の作成

GitHub上でPull Requestを作成します。

### ステップ3: 自動実行 (Execute & Check)

PRを作成すると、GitHub Actionsが自動で以下を実行します：

1. コードのチェックアウト
2. 依存関係のインストール
3. モデルの学習と評価
4. メトリクスとグラフの生成

### ステップ4: フィードバック (Feedback)

処理が完了すると、PRのコメント欄に自動で以下のようなレポートが投稿されます：

- **モデル精度メトリクス**: Accuracy, F1 Score, Precision, Recall
- **混同行列**: 予測結果の可視化
- **特徴量重要度**: 上位10個の特徴量

### ステップ5: 次の計画へ

レポートを確認し、結果が良ければマージ、改善が必要であれば再度ステップ1に戻ります。

## 📊 実行例

### ローカルでの実行

```bash
python train.py
```

実行すると以下が生成されます：
- `metrics.txt`: メトリクス（テキスト形式）
- `confusion_matrix.png`: 混同行列のグラフ
- `feature_importance.png`: 特徴量重要度のグラフ

### GitHub Actionsでの実行

1. ブランチにPushするか、PRを作成すると自動実行されます
2. Actionsタブで実行状況を確認できます
3. PRのコメント欄に結果が自動投稿されます

## ⚙️ カスタマイズ

### ハイパーパラメータの調整

`train.py` の以下の部分を編集してください：

```python
model = RandomForestClassifier(
    n_estimators=100,    # 決定木の数
    max_depth=10,        # 最大深度
    random_state=42
)
```

### データ分割方法の変更

`train.py` では、`Year` カラムがある場合は時系列分割（2023年で学習、2024年でテスト）、ない場合はランダム分割を行います。

時系列分割を強制したい場合は、`train.py` の該当部分を編集してください。

### 評価指標の追加

`train.py` の評価部分に、必要なメトリクスを追加できます：

```python
from sklearn.metrics import roc_auc_score, precision_recall_curve

# 例: ROC-AUCスコア
roc_auc = roc_auc_score(y_test, y_pred_proba)
```

## 📝 注意事項

- `data.csv` ファイルはGitにコミットしないことを推奨します（`.gitignore` に追加）
- 大きなデータセットの場合は、GitHub Actionsの実行時間制限に注意してください
- CMLの画像アップロードには、GitHub Tokenが必要です（自動で設定されます）

## 🔗 参考リンク

- [CML Documentation](https://cml.dev/)
- [GitHub Actions Documentation](https://docs.github.com/actions)
- [scikit-learn Documentation](https://scikit-learn.org/)

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

