# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import argparse
import random
import json
from datetime import datetime
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')
from train_interaction import ANNModel, RNNModel, LSTMModel

FIXED_SEED = 42
random.seed(FIXED_SEED)
np.random.seed(FIXED_SEED)
torch.manual_seed(FIXED_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(FIXED_SEED)
    torch.cuda.manual_seed_all(FIXED_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    worker_seed = FIXED_SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_and_split_data(data_dir='data_interaction_only', validation_split=0.2, 
                       ):
    """Load and split data"""
    X_file = f'{data_dir}/X_recovery_full.npy'
    feature_scaler_file = f'{data_dir}/feature_scaler_recovery.pkl'
    
    X_full = np.load(X_file)
    y_full = np.load(f'{data_dir}/y_recovery_full.npy')
    
    with open(feature_scaler_file, 'rb') as f:
        feature_scaler = pickle.load(f)
    with open(f'{data_dir}/target_scaler_recovery.pkl', 'rb') as f:
        target_scaler = pickle.load(f)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, 
        test_size=validation_split, 
        random_state=FIXED_SEED,
        shuffle=True
    )
    
    return X_train, X_val, y_train, y_val, feature_scaler, target_scaler

def train_and_evaluate_model(config, model_type='lstm', X_train=None, X_val=None, 
                           y_train=None, y_val=None, feature_scaler=None, target_scaler=None,
                           device=None, verbose=False):
    """Train and evaluate model with pre-loaded data"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if X_train is None or X_val is None or y_train is None or y_val is None:
        raise ValueError("Pre-loaded data must be provided")
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, worker_init_fn=worker_init_fn)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           worker_init_fn=worker_init_fn)
    
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
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=30)
    
    best_val_loss = float('inf')
    best_val_r2 = -float('inf')
    early_stop_counter = 0
    patience = config.get('patience', 50)
    epochs = config.get('epochs', 500)
    
    for epoch in range(epochs):
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
                
                val_predictions.append(outputs.cpu().numpy())
                val_targets.append(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        val_pred_flat = np.concatenate([pred.reshape(-1, 1) for pred in val_predictions])
        val_true_flat = np.concatenate([true.reshape(-1, 1) for true in val_targets])
        
        val_pred_inv = target_scaler.inverse_transform(val_pred_flat)
        val_true_inv = target_scaler.inverse_transform(val_true_flat)
        val_r2 = r2_score(val_true_inv, val_pred_inv)
        
        scheduler.step(val_r2)
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
    
    model.eval()
    with torch.no_grad():
        y_train_pred = model(X_train_tensor.to(device)).cpu().numpy()
        y_val_pred = model(X_val_tensor.to(device)).cpu().numpy()
    
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
    
    return {
        'train_r2': train_r2,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'val_r2': val_r2,
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'epochs_trained': epoch + 1,
        'final_train_loss': train_loss,
        'final_val_loss': val_loss,
        'best_val_r2': best_val_r2
    }

def create_objective_function(model_type, X_train, X_val, y_train, y_val, 
                             feature_scaler, target_scaler, device):
    """Create Optuna objective function with pre-loaded data"""
    
    def objective(trial):
        if model_type.lower() == 'ann':
            hidden_dims = [
                trial.suggest_int('hidden_dim_0', 16, 512),
                trial.suggest_int('hidden_dim_1', 16, 512)
            ]
            
            config = {
                'hidden_dims': hidden_dims,
                'dropout': trial.suggest_float('dropout', 0.05, 0.5),
                'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16, 32]),
                'lr': trial.suggest_float('lr', 1e-5, 1e-1, log=True),
                'epochs': 500,
                'patience': 50
            }
            
        elif model_type.lower() in ['rnn', 'lstm']:
            hidden_sizes = [
                trial.suggest_int('hidden_size_0', 16, 256),
                trial.suggest_int('hidden_size_1', 16, 256)
            ]
            
            config = {
                'hidden_sizes': hidden_sizes,
                'dropout': trial.suggest_float('dropout', 0.05, 0.5),
                'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16, 32]),
                'lr': trial.suggest_float('lr', 1e-5, 1e-1, log=True),
                'epochs': 500,
                'patience': 50
            }
        
        try:

            results = train_and_evaluate_model(
                config=config,
                model_type=model_type,
                X_train=X_train,
                X_val=X_val,
                y_train=y_train,
                y_val=y_val,
                feature_scaler=feature_scaler,
                target_scaler=target_scaler,
                device=device,
                verbose=False
            )
            
            return results['val_r2']
            
        except Exception as e:
            print(f"Trial failed: {str(e)}")
            return 0.0
    
    return objective

def run_hyperparameter_optimization(model_type='lstm', data_dir='data_interaction_only', 
                                   validation_split=0.2, n_trials=100, 
                                   study_name=None, results_dir='interaction/hyperopt_results',
                                   ):
    """Run hyperparameter optimization"""
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if study_name is None:  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"{model_type}_recovery_{timestamp}"
    
    print("=" * 80)
    print(f"Hyperparameter Optimization - {model_type.upper()} Model (No Time Features)")
    print("=" * 80)
    print(f"Model type: {model_type}")
    print(f"Data directory: {data_dir}")
    print(f"Validation split: {validation_split}")
    print(f"Number of trials: {n_trials}")
    print(f"Fixed random seed: {FIXED_SEED}")
    print(f"Study name  : {study_name}")
    print("=" * 80)
    
    print("Loading and preparing data...")
    X_train, X_val, y_train, y_val, feature_scaler, target_scaler = load_and_split_data(
        data_dir=data_dir, 
        validation_split=validation_split,
    )
    print(f"Data loaded: X_train={X_train.shape}, X_val={X_val.shape}, y_train={y_train.shape}, y_val={y_val.shape}")
    
    sampler = TPESampler(seed=FIXED_SEED)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name=study_name
    )
    
    objective = create_objective_function(
        model_type=model_type,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        device=device
    )
    
    print("Starting hyperparameter optimization...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value
    
    print("\n" + "=" * 80)
    print("Optimization completed")
    print("=" * 80)
    print(f"Best validation R²: {best_value:.4f}")
    print(f"Best hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    print("\nRetraining model with best parameters...")
    
    if model_type.lower() == 'ann':

        hidden_dims = [best_params['hidden_dim_0'], best_params['hidden_dim_1']]
        best_config = {
            'hidden_dims': hidden_dims,
            'dropout': best_params['dropout'],
            'batch_size': best_params['batch_size'],
            'lr': best_params['lr'],
            'epochs': 1000,
            'patience': 100
        }
    else:

        hidden_sizes = [best_params['hidden_size_0'], best_params['hidden_size_1']]
        best_config = {
            'hidden_sizes': hidden_sizes,
            'dropout': best_params['dropout'],
            'batch_size': best_params['batch_size'],
            'lr': best_params['lr'],
            'epochs': 1000,
            'patience': 100
        }
    
    final_results = train_and_evaluate_model(
        config=best_config,
        model_type=model_type,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        device=device,
        verbose=True
    )
    
    print(f"\nFinal model performance:")
    print(f"Training R²: {final_results['train_r2']:.4f}")
    print(f"Validation R²: {final_results['val_r2']:.4f}")
    print(f"Validation RMSE: {final_results['val_rmse']:.4f}")
    print(f"Validation MAE: {final_results['val_mae']:.4f}")
    print(f"Training epochs: {final_results['epochs_trained']}")
    
    optimization_results = {
        'study_name': study_name,
        'model_type': model_type,
        'validation_split': validation_split,
        'n_trials': n_trials,
        'fixed_seed': FIXED_SEED,
        'best_trial_number': best_trial.number,
        'best_params': best_params,
        'best_value': best_value,
        'best_config': best_config,
        'final_results': final_results,
        'optimization_history': [
            {
                'trial_number': trial.number,
                'value': trial.value,
                'params': trial.params
            }
            for trial in study.trials
        ]
    }
    
    results_file = f'{results_dir}/hyperopt_{model_type}.json'
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(optimization_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nOptimization results saved to: {results_file}")
    
    study_file = f'{results_dir}/optuna_study_{model_type}.pkl'
    with open(study_file, 'wb') as f:
        pickle.dump(study, f)
    
    print(f"Optuna study object saved to: {study_file}")
    
    return study, optimization_results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization - LSTM Model Recovery Prediction')
    parser.add_argument('--model', type=str, default='ann', choices=['ann', 'rnn', 'lstm'], 
                       help='Model type')
    parser.add_argument('--data-dir', type=str, default='data_only_interaction', 
                       help='Data directory')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--n-trials', type=int, default=10, help='Number of optimization trials')
    parser.add_argument('--study-name', type=str, help='Study name')
    parser.add_argument('--results-dir', type=str, default='interaction/hyperopt_results', 
                       help='Results save directory')
    
    args = parser.parse_args()
    
    study, results = run_hyperparameter_optimization(
        model_type=args.model,
        data_dir=args.data_dir,
        validation_split=args.val_split,
        n_trials=args.n_trials,
        study_name=args.study_name,
        results_dir=args.results_dir,
    )
    
    print("\nHyperparameter optimization completed!")

if __name__ == "__main__":
    main() 