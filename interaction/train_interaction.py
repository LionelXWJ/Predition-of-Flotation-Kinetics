# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import argparse
import random
import pandas as pd
from plot import plot_multiple_samples_prediction, plot_full_dataset_comparison

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


ANN_CONFIG = {
    'hidden_dims': [421, 370],
    'dropout': 0.15796,
    'batch_size': 4,
    'epochs': 1000,
    'patience': 100,
    'lr': 0.0005621,
    'weight_decay': 0
}


RNN_CONFIG = {
    'hidden_sizes': [244, 56],
    'dropout': 0.45034145,
    'batch_size': 4,
    'epochs': 1000,
    'patience': 100,
    'lr': 0.007054183,
    'weight_decay': 0
}

LSTM_CONFIG = {
    'hidden_sizes': [240,256],
    'dropout': 0.4639289,
    'batch_size': 4,
    'epochs': 1000,
    'patience': 100,
    'lr': 0.0101468
}

# Define ANN model
class ANNModel(nn.Module):
    def __init__(self, input_size=4, hidden_dims=[64, 32], output_size=1, dropout=0.2):
        super(ANNModel, self).__init__()
        self.input_size = input_size
        self.hidden_dims = hidden_dims
        
        layers = []
        prev_dim = input_size * 6
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_size * 6))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        batch_size, seq_len, features = x.shape

        x_flat = x.view(batch_size, -1)
        
        out_flat = self.network(x_flat)
        
        out = out_flat.view(batch_size, seq_len, -1)
        
        return out

# Define RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size=4, hidden_sizes=[128, 64], output_size=1, dropout=0.2):
        super(RNNModel, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        
        self.rnn_layers = nn.ModuleList()
        
        self.rnn_layers.append(nn.RNN(
            input_size=input_size,
            hidden_size=hidden_sizes[0],
            num_layers=1,
            batch_first=True,
            dropout=0
        ))
        
        for i in range(1, len(hidden_sizes)):
            self.rnn_layers.append(nn.RNN(
                input_size=hidden_sizes[i-1],
                hidden_size=hidden_sizes[i],
                num_layers=1,
                batch_first=True,
                dropout=0
            ))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        self.fc = nn.Linear(hidden_sizes[-1], output_size)
    
    def forward(self, x):
        for i, rnn_layer in enumerate(self.rnn_layers):
            h0 = torch.zeros(1, x.size(0), self.hidden_sizes[i]).to(x.device)
            
            x, _ = rnn_layer(x, h0)
            
            if self.dropout is not None and i < len(self.rnn_layers) - 1:
                x = self.dropout(x)
        
        out = self.fc(x)
        
        return out

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_sizes=[128, 64], output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        
        self.lstm_layers = nn.ModuleList()
        
        self.lstm_layers.append(nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_sizes[0],
            num_layers=1,
            batch_first=True,
            dropout=0
        ))
        
        for i in range(1, len(hidden_sizes)):
            self.lstm_layers.append(nn.LSTM(
                input_size=hidden_sizes[i-1],
                hidden_size=hidden_sizes[i],
                num_layers=1,
                batch_first=True,
                dropout=0
            ))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        self.fc = nn.Linear(hidden_sizes[-1], output_size)
    
    def forward(self, x):
        for i, lstm_layer in enumerate(self.lstm_layers):

            h0 = torch.zeros(1, x.size(0), self.hidden_sizes[i]).to(x.device)
            c0 = torch.zeros(1, x.size(0), self.hidden_sizes[i]).to(x.device)
            
            x, _ = lstm_layer(x, (h0, c0))
            
            if self.dropout is not None and i < len(self.lstm_layers) - 1:
                x = self.dropout(x)
        
        out = self.fc(x)

        return out

def load_and_split_data(data_dir='data_only_interaction', validation_split=0.2, 
                       random_state=42):
    """
    Load data from complete dataset and split into training and validation sets
    
    Args:
        data_dir: Data directory
        validation_split: Validation set ratio
        random_state: Random seed
    
    Returns:
        Training and validation data, along with scalers
    """
    print(f"Loading complete dataset from {data_dir}...")
    
    X_file = f'{data_dir}/X_recovery_full.npy'
    feature_scaler_file = f'{data_dir}/feature_scaler_recovery.pkl'
    print("Using data without time features")
    
    X_full = np.load(X_file)
    y_full = np.load(f'{data_dir}/y_recovery_full.npy')
    
    print(f"Complete dataset shape: X={X_full.shape}, y={y_full.shape}")
    
    with open(feature_scaler_file, 'rb') as f:
        feature_scaler = pickle.load(f)
    with open(f'{data_dir}/target_scaler_recovery.pkl', 'rb') as f:
        target_scaler = pickle.load(f)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, 
        test_size=validation_split, 
        random_state=random_state,
        shuffle=True
    )
    
    print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation set shape: X={X_val.shape}, y={y_val.shape}")
    
    return X_train, X_val, y_train, y_val, feature_scaler, target_scaler

def train_model(model_type='lstm', config=None, device=None, save_path=None, 
                data_dir='data_only_interaction', validation_split=0.2,
                ):
    """
    Main function for training models
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if config is None:
        config_map = {
            'ann': ANN_CONFIG,
            'rnn': RNN_CONFIG,
            'lstm': LSTM_CONFIG,
        }
        config = config_map.get(model_type.lower())
        if config is None:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    print(f"Using configuration: {config}")
    
    X_train, X_val, y_train, y_val, feature_scaler, target_scaler = load_and_split_data(
        data_dir=data_dir, 
        validation_split=validation_split,
        random_state=seed
    )
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, worker_init_fn=worker_init_fn)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], worker_init_fn=worker_init_fn)
    
    input_size = X_train.shape[2]
    output_size = y_train.shape[2]
    
    if model_type.lower() == 'ann':
        model = ANNModel(input_size=input_size, hidden_dims=config['hidden_dims'], 
                         output_size=output_size, dropout=config['dropout'])
    elif model_type.lower() == 'rnn':
        model = RNNModel(input_size=input_size, hidden_sizes=config['hidden_sizes'], 
                         output_size=output_size, dropout=config['dropout'])
    elif model_type.lower() == 'lstm':
        model = LSTMModel(input_size=input_size, hidden_sizes=config['hidden_sizes'], 
                          output_size=output_size, dropout=config['dropout'])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model = model.to(device)
    print(f"Model structure:\n{model}")
    
    criterion = nn.MSELoss()
    weight_decay = config.get('weight_decay', 0.0)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=30)
    
    best_val_loss = float('inf')
    best_val_r2 = -float('inf')
    early_stop_counter = 0
    train_losses = []
    val_losses = []
    val_r2_scores = []
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                # Collect predictions and targets for R² calculation
                val_predictions.append(outputs.cpu().numpy())
                val_targets.append(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        val_pred_flat = np.concatenate([pred.reshape(-1, 1) for pred in val_predictions])
        val_true_flat = np.concatenate([true.reshape(-1, 1) for true in val_targets])
        
        val_pred_inv = target_scaler.inverse_transform(val_pred_flat)
        val_true_inv = target_scaler.inverse_transform(val_true_flat)
        val_r2 = r2_score(val_true_inv, val_pred_inv)
        val_r2_scores.append(val_r2)
        
        scheduler.step(val_r2)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{config["epochs"]}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val R²: {val_r2:.6f}')
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_val_loss = val_loss
            early_stop_counter = 0
            
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_r2': val_r2,
                    'config': config,
                    'input_size': input_size,
                    'output_size': output_size,
                }, save_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= config['patience']:
                print(f"Early stopping: No improvement in validation R² for {config['patience']} epochs")
                break
    
    
    if save_path and best_val_loss == float('inf'):
        torch.save({
            'epoch': config['epochs'] - 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_losses[-1],
            'val_loss': val_losses[-1],
            'config': config,
            'input_size': input_size,
            'output_size': output_size,
        }, save_path)
        print(f"Final model saved to {save_path}")
    
    return model, train_losses, val_losses, feature_scaler, target_scaler

def evaluate_model(model_type='lstm', model_path=None, data_dir='data_only_interaction', validation_split=0.2):
    """Evaluate model performance and generate prediction result visualizations"""
    if model_path is None or not os.path.exists(model_path):
        print(f"Error: model file {model_path} does not exist")
        return
    
    print(f"\n=== Evaluating {model_type.upper()} Model (Recovery Prediction) ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config')
    input_size = checkpoint.get('input_size')
    output_size = checkpoint.get('output_size')

    
    if config is None:
        config_map = {
            'ann': ANN_CONFIG,
            'rnn': RNN_CONFIG,
            'lstm': LSTM_CONFIG,
        }
        config = config_map.get(model_type.lower())
    
    X_train, X_val, y_train, y_val, feature_scaler, target_scaler = load_and_split_data(
        data_dir=data_dir, 
        validation_split=validation_split,
        random_state=seed
    )
    
    if model_type.lower() == 'ann':
        model = ANNModel(input_size=input_size, hidden_dims=config['hidden_dims'], 
                         output_size=output_size, dropout=config['dropout'])
    elif model_type.lower() == 'rnn':
        model = RNNModel(input_size=input_size, hidden_sizes=config['hidden_sizes'], 
                         output_size=output_size, dropout=config['dropout'])
    elif model_type.lower() == 'lstm':
        model = LSTMModel(input_size=input_size, hidden_sizes=config['hidden_sizes'], 
                          output_size=output_size, dropout=config['dropout'])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    results_dir = 'interaction/results'
    plots_dir = f'{results_dir}/training_plots'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    with torch.no_grad():
        y_train_pred = model(X_train_tensor).cpu().numpy()
    
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    with torch.no_grad():
        y_val_pred = model(X_val_tensor).cpu().numpy()
    
    y_train_true_flat = y_train.reshape(-1, 1)
    y_train_pred_flat = y_train_pred.reshape(-1, 1)
    y_val_true_flat = y_val.reshape(-1, 1)
    y_val_pred_flat = y_val_pred.reshape(-1, 1)
    
    y_train_true_inv = target_scaler.inverse_transform(y_train_true_flat)
    y_train_pred_inv = target_scaler.inverse_transform(y_train_pred_flat)
    y_val_true_inv = target_scaler.inverse_transform(y_val_true_flat)
    y_val_pred_inv = target_scaler.inverse_transform(y_val_pred_flat)
    
    train_r2 = r2_score(y_train_true_inv, y_train_pred_inv)
    train_rmse = np.sqrt(mean_squared_error(y_train_true_inv, y_train_pred_inv))
    train_mae = mean_absolute_error(y_train_true_inv, y_train_pred_inv)
    
    val_r2 = r2_score(y_val_true_inv, y_val_pred_inv)
    val_rmse = np.sqrt(mean_squared_error(y_val_true_inv, y_val_pred_inv))
    val_mae = mean_absolute_error(y_val_true_inv, y_val_pred_inv)
    
    print(f"\nTraining set evaluation metrics:")
    print(f"R² Score: {train_r2:.4f}")
    print(f"RMSE: {train_rmse:.4f}")
    print(f"MAE: {train_mae:.4f}")
    
    print(f"\nValidation set evaluation metrics:")
    print(f"R² Score: {val_r2:.4f}")
    print(f"RMSE: {val_rmse:.4f}")
    print(f"MAE: {val_mae:.4f}")
    
    eval_results = {
        'train_r2': train_r2,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'val_r2': val_r2,
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'model_type': model_type,
        'target_type': 'recovery'
    }
    
    results_path = f'{results_dir}/model_metrics_{model_type}_recovery_full.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(eval_results, f)
    
    print(f"Evaluation results saved to: {results_path}")
    
    print(f"\nGenerating prediction visualization...")
    
    plot_full_dataset_comparison(
        y_train_true_inv, y_train_pred_inv, 
        f'{model_type.upper()} Recovery Prediction (Training Set)', 
        f'{plots_dir}/{model_type}_recovery_train_comparison.png'
    )
    
    plot_full_dataset_comparison(
        y_val_true_inv, y_val_pred_inv, 
        f'{model_type.upper()} Recovery Prediction (Validation Set)', 
        f'{plots_dir}/{model_type}_recovery_val_comparison.png'
    )
    
    plot_multiple_samples_prediction(
        X_train, y_train, y_train_pred, feature_scaler, target_scaler,
        f'{model_type.upper()} Recovery Prediction (Training Set Samples)', 
        f'{plots_dir}/{model_type}_recovery_train_samples.png'
    )
    
    plot_multiple_samples_prediction(
        X_val, y_val, y_val_pred, feature_scaler, target_scaler,
        f'{model_type.upper()} Recovery Prediction (Validation Set Samples)', 
        f'{plots_dir}/{model_type}_recovery_val_samples.png'
    )
    
    print(f"Visualization charts saved to: {plots_dir}")
    
    return eval_results
    
def main():
    """Main function, process command line arguments and run training"""
    parser = argparse.ArgumentParser(description='Train ANN/RNN/LSTM models (full dataset, recovery prediction)')
    parser.add_argument('--model', type=str, default='lstm', choices=['ann', 'rnn', 'lstm'], help='Model type: ann, rnn, lstm')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--data-dir', type=str, default='data_only_interaction', help='Data directory')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation set ratio')
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate model, do not train')
    parser.add_argument('--weight-decay', type=float, default=None, help='Weight decay coefficient (L2 regularization)')
    args = parser.parse_args()
    
    device = torch.device('cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda')
    
    save_dir = 'interaction/results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = f'{save_dir}/{args.model}_recovery_full.pth'
    
    config_map = {
        'ann': ANN_CONFIG,
        'rnn': RNN_CONFIG,
        'lstm': LSTM_CONFIG,
    }
    config = config_map.get(args.model.lower()).copy()
    
    if args.weight_decay is not None:
        config['weight_decay'] = args.weight_decay
    
    print("=" * 80)
    print(f"Training {args.model.upper()} Model - Recovery Prediction (Full Dataset)")
    print("=" * 80)
    print(f"Model type: {args.model}")
    print(f"Prediction target: Recovery")
    print(f"Data directory: {args.data_dir}")
    print(f"Validation set ratio: {args.val_split}")
    print(f"Weight decay: {config.get('weight_decay', 0.0)}")
    print(f"Model configuration: {config}")
    print(f"Device: {device}")
    print(f"Model save path: {save_path}")
    print("=" * 80)
    
    if args.eval_only:
        evaluate_model(
            model_type=args.model, 
            model_path=save_path,
            data_dir=args.data_dir,
            validation_split=args.val_split
        )
    else:
        model, train_losses, val_losses, feature_scaler, target_scaler = train_model(
            model_type=args.model,
            config=config,
            device=device,
            save_path=save_path,
            data_dir=args.data_dir,
            validation_split=args.val_split
        )
        
        evaluate_model(
            model_type=args.model, 
            model_path=save_path,
            data_dir=args.data_dir,
            validation_split=args.val_split
        )
    
    print(f"\nCompleted! Model saved at {save_path}")

if __name__ == "__main__":
    main() 