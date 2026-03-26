import numpy as np
from utils import load_mnist, compute_loss
from model import MLP
from visual import plot_loss_curve, plot_confusion_matrix_result

def main():
    X_train, X_val, y_train, y_val = load_mnist()
    print(f"训练集形状: {X_train.shape}, 验证集形状: {X_val.shape}")

    input_dim = 784
    hidden_dims = [256, 128] 
    output_dim = 10
    layer_dims = [input_dim] + hidden_dims + [output_dim]
    activations = ['relu', 'relu', 'softmax'] 
    
    epochs = 20
    batch_size = 64      
    learning_rate = 0.1  
    l2_lambda = 1e-4     

    model = MLP(layer_dims, activations, l2_lambda=l2_lambda)

    # --- 记录绘图数据 ---
    train_losses_history = []
    val_losses_history = []

    print("\n开始训练...")
    for epoch in range(epochs):
        permutation = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        epoch_train_losses = [] # 记录当前 epoch 内所有 batch 的 loss

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]

            y_pred = model.forward(X_batch)
            
            # 计算并记录训练集当前的 Batch Loss
            weights = [layer.W for layer in model.layers]
            batch_loss = compute_loss(y_pred, y_batch, weights, l2_lambda)
            epoch_train_losses.append(batch_loss)

            grads = model.backward(y_pred, y_batch)
            model.update_parameters(grads, learning_rate)

        # 计算平均训练 Loss 并记录
        avg_train_loss = np.mean(epoch_train_losses)
        train_losses_history.append(avg_train_loss)

        # 验证集评估
        y_val_pred = model.forward(X_val)
        val_loss = compute_loss(y_val_pred, y_val, weights, l2_lambda)
        val_losses_history.append(val_loss) # 记录验证 Loss
        
        predictions = np.argmax(y_val_pred, axis=1)
        true_labels = np.argmax(y_val, axis=1)
        val_acc = np.mean(predictions == true_labels)

        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc*100:.2f}%")
        
    print("训练结束。")

    # ---生成图表 ---
    print("\n正在生成可视化图表...")
    plot_loss_curve(train_losses_history, val_losses_history)
    plot_confusion_matrix_result(true_labels, predictions)

if __name__ == "__main__":
    main()