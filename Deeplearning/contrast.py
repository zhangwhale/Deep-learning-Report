import numpy as np
import matplotlib.pyplot as plt
from utils import load_mnist, compute_loss
from model import MLP

def run_single_experiment(X_train, y_train, X_val, y_val, config):
    print(f"\n========== 正在运行实验组: {config['name']} ==========")
    print(f"参数: 结构={config['hidden_dims']}, 激活={config['activations']}, "
          f"Lr={config['lr']}, Batch={config['batch_size']}, L2={config['l2']}")
    
    input_dim = 784
    output_dim = 10
    layer_dims = [input_dim] + config['hidden_dims'] + [output_dim]
    
    model = MLP(layer_dims, config['activations'], l2_lambda=config['l2'])
    
    epochs = 15 # 节省整体运行时间，设为15轮
    val_acc_history = []
    
    for epoch in range(epochs):
        permutation = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]
        
        batch_size = config['batch_size']
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            
            y_pred = model.forward(X_batch)
            grads = model.backward(y_pred, y_batch)
            model.update_parameters(grads, config['lr'])
            
        # 验证集评估
        y_val_pred = model.forward(X_val)
        predictions = np.argmax(y_val_pred, axis=1)
        true_labels = np.argmax(y_val, axis=1)
        val_acc = np.mean(predictions == true_labels)
        val_acc_history.append(val_acc)
        
        print(f"  Epoch {epoch+1:02d}/{epochs} | Val Accuracy: {val_acc*100:.2f}%")
        
    return val_acc_history

def main():
    X_train, X_val, y_train, y_val = load_mnist()
    
    # 配置超参数
    configs = [
        {
            "name": "Baseline",
            "hidden_dims": [256, 128],
            "activations": ['relu', 'relu', 'softmax'],
            "lr": 0.01,
            "batch_size": 64,
            "l2": 1e-4
        },
        {
            "name": "组1",
            "hidden_dims": [512, 256],
            "activations": ['leaky_relu', 'leaky_relu', 'softmax'],
            "lr": 0.005,
            "batch_size": 128,
            "l2": 1e-3
        },
        {
            "name": "组2",
            "hidden_dims": [128],
            "activations": ['tanh', 'softmax'],
            "lr": 0.02,
            "batch_size": 32,
            "l2": 0
        }
    ]
    
    all_results = {}
    for config in configs:
        history = run_single_experiment(X_train, y_train, X_val, y_val, config)
        all_results[config['name']] = history
        
    # --- 绘制对比图 ---
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    
    epochs_range = range(1, 16)
    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']
    
    for i, config in enumerate(configs):
        name = config['name']
        plt.plot(epochs_range, all_results[name], label=name, 
                 color=colors[i], marker=markers[i], linewidth=2)
                 
    plt.title('不同超参数组合的验证集准确率对比 (超参数敏感性分析)', fontsize=15)
    plt.xlabel('训练轮数', fontsize=13)
    plt.ylabel('验证集准确率', fontsize=13)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('hyperparam_tuning.png', dpi=300)
    print("\n已生成参数调优对比图 -> contrast.png")

if __name__ == "__main__":
    main()