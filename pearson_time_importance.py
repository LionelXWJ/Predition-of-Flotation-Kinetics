import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 16

class PearsonTimeImportanceAnalyzer:
    """Pearson correlation-based feature importance analyzer"""
    
    def __init__(self, data_folder='data', results_folder='pearson'):
        self.data_folder = data_folder
        self.results_folder = results_folder
        
        self.selected_features = ['A²', 'R²', 'S²', 'S×R', 'A×R', 'A×S', 'A×S×R']
        self.original_features = ['A', 'S', 'R']
        self.feature_names = ['A', 'S', 'R', 'Time Point', 'A²', 'S²', 'R²', 'A×S', 'A×R', 'S×R', 'A×S×R']
    
    def load_data(self):
        """Load recovery data"""
        X_all = np.load(f'{self.data_folder}/X_recovery_enhanced.npy')
        y_all = np.load(f'{self.data_folder}/y_recovery_enhanced.npy')
        return X_all, y_all
    
    def pearson_correlation_analysis_over_time(self, X, y):
        """Calculate Pearson correlation for different time points"""
        time_correlations = {}
        time_map = {0.5: 1, 1: 2, 2: 3, 3: 4, 5: 5}
        
        X_flat = X.reshape(-1, X.shape[-1])
        
        for t_display, t_idx in time_map.items():
            if t_idx >= y.shape[1]:
                t_idx = y.shape[1] - 1
            
            y_time = y[:, t_idx]
            y_target = y_time.repeat(X.shape[1])
            
            correlations = []
            for i in range(X.shape[-1]):
                try:
                    corr, _ = pearsonr(X_flat[:, i], y_target)
                    correlations.append(abs(corr))
                except:
                    correlations.append(0.0)
            
            time_correlations[f'{t_display:.1f}min'] = np.array(correlations)
        
        return time_correlations
    
    def plot_heatmap(self, time_correlations, save_path=None, analysis_type='selected'):
        """Plot feature importance heatmap"""
        features = self.selected_features if analysis_type == 'selected' else self.original_features
        feature_indices = {feature: i for i, feature in enumerate(self.feature_names)}
        
        selected_indices = [feature_indices[f] for f in features if f in feature_indices]
        time_points = list(time_correlations.keys())
        
        heatmap_data = []
        for idx in selected_indices:
            row_data = [time_correlations[t][idx] if idx < len(time_correlations[t]) else 0.0 
                       for t in time_points]
            heatmap_data.append(row_data)
        
        df = pd.DataFrame(heatmap_data, 
                         index=[self.feature_names[i] for i in selected_indices],
                         columns=time_points)
        
        fig_height = 6 if analysis_type == 'original' else 8
        plt.figure(figsize=(10, fig_height))
        
        sns.heatmap(df, annot=True, fmt='.3f', cmap='rocket_r', 
                   vmin=0.2, vmax=1, linewidths=0,
                   annot_kws={'size': 14, 'weight': 'bold', 'color': 'white'})
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_analysis(self, include_original=True):
        """Execute analysis"""
        print("Starting feature importance analysis...")
        
        X_all, y_all = self.load_data()
        
        analysis_folder = f'{self.results_folder}/pearson_time_importance'
        os.makedirs(analysis_folder, exist_ok=True)
        
        time_correlations = self.pearson_correlation_analysis_over_time(X_all, y_all)
        
        # Plot selected features
        self.plot_heatmap(
            time_correlations,
            save_path=f'{analysis_folder}/recovery_time_importance_selected.png',
            analysis_type='selected'
        )
        
        # Plot original features if requested
        if include_original:
            self.plot_heatmap(
                time_correlations,
                save_path=f'{analysis_folder}/recovery_time_importance_original.png',
                analysis_type='original'
            )
        
        # Save results
        results = {
            'time_correlations': time_correlations,
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'original_features': self.original_features
        }
        
        with open(f'{analysis_folder}/recovery_time_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Analysis completed! Results saved to: {analysis_folder}")
        return results

def main():
    """Main function"""
    print("Recovery Data Feature Importance Analysis")
    print("="*50)
    
    analyzer = PearsonTimeImportanceAnalyzer()
    results = analyzer.run_analysis(include_original=True)
    
    print("Analysis completed successfully!")

if __name__ == "__main__":
    main()