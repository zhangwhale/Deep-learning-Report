import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_loss_curve(train_losses, val_losses):
    """训练与验证损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False 
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='训练集 Loss (Train Loss)', marker='o')
    plt.plot(epochs, val_losses, 'r-', label='验证集 Loss (Val Loss)', marker='s')
    
    plt.title('模型训练与验证损失曲线 (Loss Curve)', fontsize=16)
    plt.xlabel('训练轮数 (Epochs)', fontsize=14)
    plt.ylabel('损失值 (Loss)', fontsize=14)
    plt.xticks(epochs)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=300) 
    print("已保存损失曲线图 -> loss_curve.png")
    plt.show()

def plot_confusion_matrix_result(y_true, y_pred):
    """混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    
    # 热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10),
                annot_kws={"size": 12})
    
    plt.title('验证集分类混淆矩阵 (Confusion Matrix)', fontsize=16)
    plt.xlabel('预测类别 (Predicted Label)', fontsize=14)
    plt.ylabel('真实类别 (True Label)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300) 
    print("已保存混淆矩阵图 -> confusion_matrix.png")
    plt.show()