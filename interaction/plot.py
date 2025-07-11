import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns
import os
from pathlib import Path

seed = 42
random.seed(seed)
np.random.seed(seed)

def plot_multiple_samples_prediction(X_data, y_true, y_pred, feature_scaler, target_scaler, title, save_path):
    """
    Plot time series prediction comparison charts for multiple samples
    Prioritize original data samples (first 85 samples) with recovery rate >= 25% at the 5min time point
    """
    batch_size, seq_len, _ = y_pred.shape
    y_pred_flat = y_pred.reshape(-1, 1)
    y_true_flat = y_true.reshape(-1, 1)
    
    y_true_inv = target_scaler.inverse_transform(y_true_flat)
    y_pred_inv = target_scaler.inverse_transform(y_pred_flat)
    
    y_true_inv = y_true_inv.reshape(batch_size, seq_len)
    y_pred_inv = y_pred_inv.reshape(batch_size, seq_len)
    
    time_points = [0, 0.5, 1, 2, 3, 5]
    
    # 加载原始数据以获取正确的操作条件
    import pickle
    with open('data_only_interaction/raw_data.pkl', 'rb') as f:
        raw_data = pickle.load(f)
    
    # 5min时间点对应的索引位置
    last_time_point_idx = 5  # 5min对应的索引位置
    
    # 首先尝试从原始数据样本（前85个）中找出5min回收率不低于25%的样本
    original_data_limit = len(raw_data)  # 原始数据样本数量
    original_valid_samples = []
    
    for i in range(min(batch_size, original_data_limit)):
        if y_true_inv[i, last_time_point_idx] >= 25.0:
            original_valid_samples.append(i)
    
    print(f"在原始数据样本中找到 {len(original_valid_samples)} 个5min回收率不低于25%的样本")
    
    # 如果原始数据中的有效样本不足9个，再从所有样本中补充
    if len(original_valid_samples) < 9:
        print(f"原始数据中有效样本不足9个，从所有样本中补充")
        all_valid_samples = []
        for i in range(batch_size):
            if y_true_inv[i, last_time_point_idx] >= 25.0:
                all_valid_samples.append(i)
        
        # 优先使用原始数据样本，然后补充其他样本
        valid_samples = original_valid_samples.copy()
        for sample_idx in all_valid_samples:
            if sample_idx not in valid_samples and len(valid_samples) < 9:
                valid_samples.append(sample_idx)
        
        print(f"总共找到 {len(all_valid_samples)} 个5min回收率不低于25%的样本")
    else:
        valid_samples = original_valid_samples
    
    # 如果仍然不足9个，降低条件要求
    if len(valid_samples) < 9:
        print(f"有效样本仍不足9个，降低条件门槛")
        # 优先从原始数据中按5min回收率排序选择
        original_samples_with_recovery = [(i, y_true_inv[i, last_time_point_idx]) 
                                        for i in range(min(batch_size, original_data_limit))]
        original_samples_sorted = sorted(original_samples_with_recovery, key=lambda x: -x[1])
        
        valid_samples = [idx for idx, _ in original_samples_sorted[:min(9, len(original_samples_sorted))]]
        
        # 如果原始数据样本还是不够9个，再从所有样本中补充
        if len(valid_samples) < 9:
            all_samples_with_recovery = [(i, y_true_inv[i, last_time_point_idx]) for i in range(batch_size)]
            all_samples_sorted = sorted(all_samples_with_recovery, key=lambda x: -x[1])
            
            for idx, recovery in all_samples_sorted:
                if idx not in valid_samples and len(valid_samples) < 9:
                    valid_samples.append(idx)
        
        min_recovery = min([y_true_inv[i, last_time_point_idx] for i in valid_samples])
        print(f"选取回收率最高的9个样本，最低回收率: {min_recovery:.2f}%")
    elif len(valid_samples) > 9:
        # 如果有效样本超过9个，优先选择原始数据中回收率最高的样本
        if len(original_valid_samples) >= 9:
            # 从原始数据的有效样本中选择回收率最高的9个
            original_samples_sorted = sorted(original_valid_samples, 
                                           key=lambda i: -y_true_inv[i, last_time_point_idx])
            valid_samples = original_samples_sorted[:9]
            print(f"从原始数据的有效样本中选择回收率最高的9个")
        else:
            # 先选择所有原始数据的有效样本，再从其他样本中补充
            other_valid_samples = [i for i in valid_samples if i >= original_data_limit]
            other_samples_sorted = sorted(other_valid_samples, 
                                        key=lambda i: -y_true_inv[i, last_time_point_idx])
            needed_count = 9 - len(original_valid_samples)
            valid_samples = original_valid_samples + other_samples_sorted[:needed_count]
            print(f"选择所有{len(original_valid_samples)}个原始数据有效样本，补充{needed_count}个其他样本")
    
    # 标记哪些是原始数据样本
    original_flags = [i < original_data_limit for i in valid_samples]
    original_count = sum(original_flags)
    print(f"最终选择的9个样本中，{original_count}个来自原始数据，{9-original_count}个来自增强数据")
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    
    for plot_idx, sample_idx in enumerate(valid_samples):
        row = plot_idx // 3
        col = plot_idx % 3
        ax = axes[row, col]
        
        # 从原始数据中获取操作条件，而不是从缩放后的特征中获取
        if sample_idx < len(raw_data):
            # 原始数据样本
            condition = raw_data[sample_idx]['condition']
            aeration = condition[0]
            speed = condition[1] 
            reagent = condition[2]
        else:
            # 增强数据样本，使用默认值或从其他地方获取
            aeration = "N/A"
            speed = "N/A"
            reagent = "N/A"
        
        # 确保是数值类型才格式化
        if isinstance(aeration, (int, float)):
            aeration_str = f"{aeration:.2f} m³/(m²·min)"
        else:
            aeration_str = "N/A"
            
        if isinstance(speed, (int, float)):
            speed_str = f"{speed:.0f} rpm"
        else:
            speed_str = "N/A"
            
        if isinstance(reagent, (int, float)):
            reagent_str = f"{reagent:.2f} kg/t"
        else:
            reagent_str = "N/A"
        
        condition_str = f"Aeration Rate: {aeration_str}\nImpeller Speed: {speed_str}\nReagent Dosage: {reagent_str}"
        
        ax.plot(time_points, y_true_inv[sample_idx], 'bo-', label='True Value', linewidth=2, markersize=6)
        ax.plot(time_points, y_pred_inv[sample_idx], 'r--o', label='Predicted Value', linewidth=2, markersize=6)
        
        # 添加5min回收率信息和数据来源标识
        recovery_5min = y_true_inv[sample_idx, last_time_point_idx]
        data_source = "Original" if sample_idx < original_data_limit else "Augmented"
        combined_str = f"Sample {sample_idx+1}\n{condition_str}"
        
        # Sample信息上移到更高位置
        ax.text(0.98, 0.05, combined_str, transform=ax.transAxes, 
               fontsize=14, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))
        
        if row == 2:
            ax.set_xlabel('Time (min)', fontsize=18, fontweight='bold')           
        else:                
            ax.set_xlabel('')
        
        if col == 0:
            ax.set_ylabel('Recovery (%)', fontsize=18, fontweight='bold')           
        else:
            ax.set_ylabel('')
        
        # 设置坐标轴格式和样式
        from matplotlib.ticker import FormatStrFormatter
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # 纵坐标不使用一位小数，使用默认格式
        ax.tick_params(axis='both', which='major', labelsize=14, direction='in', 
                      length=6, width=1.5, pad=8)
        
        # 设置刻度标签字体加粗
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
            
        ax.set_yticks(range(0, 81, 20))
        ax.set_ylim(0, 65)
        
        # 加粗子图边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)
        
        if plot_idx == 0:
            # True Value图例显示于左上角
            ax.legend(fontsize=14, loc='upper left', bbox_to_anchor=(0.02, 1))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_full_dataset_comparison(y_true_flat, y_pred_flat, title, save_path):
    """
    Plot comparison between predictions and actual results for the entire dataset
    Generate two separate plots: scatter plot and time series plot
    """
    # 计算R²分数
    r2 = r2_score(y_true_flat, y_pred_flat)
    
    # 第一张图：散点图 (True vs Predicted Values)
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true_flat, y_pred_flat, alpha=0.6, edgecolors='w', s=100)
    
    min_val = min(np.min(y_true_flat), np.min(y_pred_flat))
    max_val = max(np.max(y_true_flat), np.max(y_pred_flat))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3)
    
    plt.xlabel('True Value (%)', fontsize=28, fontweight='bold')
    plt.ylabel('Predicted Value (%)', fontsize=28, fontweight='bold')
    
    # 设置坐标轴格式和样式
    ax = plt.gca()
    # 不再保留一位小数，使用默认格式
    plt.tick_params(axis='both', which='major', labelsize=24, direction='in', 
                   length=8, width=2, pad=8)
    
    # 设置刻度标签字体加粗
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # 设置坐标轴刻度间隔为20，从-2开始，最大值设定为65
    plt.xticks(range(0, 81, 20))
    plt.yticks(range(0, 81, 20))
    plt.xlim(-2, 65)
    plt.ylim(-2, 65)
    
    # 加粗坐标轴边框
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
    
    # 保存散点图
    scatter_path = save_path.replace('.png', '_scatter.png')
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 第二张图：时间序列图 (只显示前100个点)
    plt.figure(figsize=(12, 6))
    display_points = min(100, len(y_true_flat))
    plt.plot(range(display_points), y_true_flat[:display_points], 'bo-', 
             label='True Value', alpha=0.7, markersize=6, linewidth=2)
    plt.plot(range(display_points), y_pred_flat[:display_points], 'r--o', 
             label='Predicted Value', alpha=0.7, markersize=6, linewidth=2)
    
    plt.xlabel('Sample Points', fontsize=22)
    plt.ylabel('Recovery (%)', fontsize=22)
    plt.title(f'{title} - Time Series Comparison (First {display_points} Points)', fontsize=24)
    plt.legend(fontsize=18, loc='best')
    plt.tick_params(axis='both', which='major', labelsize=18)
    
    # 保存时间序列图
    timeseries_path = save_path.replace('.png', '_timeseries.png')
    plt.tight_layout()
    plt.savefig(timeseries_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Scatter plot saved to: {scatter_path}")
    print(f"Time series plot saved to: {timeseries_path}")