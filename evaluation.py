"""
評価モジュール
モデルの評価指標の計算と可視化を実装
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import config

def calculate_specificity(y_true, y_pred):
    """Specificity（真陰性率）を計算"""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0
    return specificity

def calculate_all_metrics(y_true, y_pred, y_pred_proba=None):
    """すべての評価指標を計算"""
    metrics = {}
    
    # 基本指標
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['specificity'] = calculate_specificity(y_true, y_pred)
    
    # F1 Score (macro, weighted)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # 確率予測がある場合
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['roc_auc'] = 0.0
        
        try:
            metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
        except:
            metrics['pr_auc'] = 0.0
    else:
        metrics['roc_auc'] = 0.0
        metrics['pr_auc'] = 0.0
    
    return metrics

def save_metrics(metrics, filepath=None):
    """メトリクスをファイルに保存"""
    if filepath is None:
        filepath = config.METRICS_FILE
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    print(f"メトリクスを保存: {filepath}")

def plot_confusion_matrix(y_true, y_pred, filepath=None):
    """混同行列を可視化"""
    if filepath is None:
        filepath = config.CONFUSION_MATRIX_FILE
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(filepath, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"混同行列を保存: {filepath}")

def plot_roc_curve(y_true, y_pred_proba, filepath=None):
    """ROC曲線を可視化"""
    if filepath is None:
        filepath = config.ROC_CURVE_FILE
    
    if y_pred_proba is None:
        print("ROC曲線: 確率予測が利用できません")
        return
    
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filepath, dpi=config.PLOT_DPI, bbox_inches='tight')
        plt.close()
        print(f"ROC曲線を保存: {filepath}")
    except Exception as e:
        print(f"ROC曲線の作成に失敗: {e}")

def plot_pr_curve(y_true, y_pred_proba, filepath=None):
    """Precision-Recall曲線を可視化"""
    if filepath is None:
        filepath = config.PR_CURVE_FILE
    
    if y_pred_proba is None:
        print("PR曲線: 確率予測が利用できません")
        return
    
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filepath, dpi=config.PLOT_DPI, bbox_inches='tight')
        plt.close()
        print(f"PR曲線を保存: {filepath}")
    except Exception as e:
        print(f"PR曲線の作成に失敗: {e}")

def plot_feature_importance(model, feature_names, top_n=20, filepath=None):
    """特徴量重要度を可視化"""
    if filepath is None:
        filepath = config.FEATURE_IMPORTANCE_FILE
    
    try:
        # 特徴量重要度を取得
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            print("特徴量重要度を取得できません")
            return
        
        # DataFrameに変換
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        # 可視化
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance_df, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(filepath, dpi=config.PLOT_DPI, bbox_inches='tight')
        plt.close()
        print(f"特徴量重要度を保存: {filepath}")
    except Exception as e:
        print(f"特徴量重要度の可視化に失敗: {e}")

def evaluate_model(model, X_test, y_test, feature_names=None, save_plots=True):
    """モデルを評価し、すべてのメトリクスと可視化を生成"""
    # 予測
    y_pred = model.predict(X_test)
    
    # 確率予測（可能な場合）
    y_pred_proba = None
    if hasattr(model, 'predict_proba'):
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        except:
            pass
    
    # メトリクスを計算
    metrics = calculate_all_metrics(y_test, y_pred, y_pred_proba)
    
    # メトリクスを保存
    if save_plots:
        save_metrics(metrics)
        plot_confusion_matrix(y_test, y_pred)
        
        if y_pred_proba is not None:
            plot_roc_curve(y_test, y_pred_proba)
            plot_pr_curve(y_test, y_pred_proba)
        
        if feature_names is not None:
            plot_feature_importance(model, feature_names)
    
    # 分類レポート
    print("\n分類レポート:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return metrics, y_pred, y_pred_proba

