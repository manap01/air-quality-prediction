import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os
warnings.filterwarnings('ignore')

class Visualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        # Pastikan folder reports/figures ada
        os.makedirs("reports/figures", exist_ok=True)
    
    def plot_correlation_matrix(self, df, save_path=None):
        """Plot correlation matrix heatmap"""
        plt.figure(figsize=(12, 10))
        correlation_matrix = df.corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f')
        
        plt.title('Correlation Matrix of Air Quality Features', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_feature_distributions(self, df, features, save_path=None):
        """Plot feature distributions"""
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, feature in enumerate(features):
            if i < len(axes):
                sns.histplot(data=df, x=feature, kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {feature}')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_aqi_over_time(self, df, date_column='date', aqi_column='AQI', save_path=None):
        """Plot AQI trend over time"""
        plt.figure(figsize=(15, 6))
        
        df_sorted = df.sort_values(date_column)
        plt.plot(df_sorted[date_column], df_sorted[aqi_column], 
                color='steelblue', alpha=0.7, linewidth=1)
        
        # Add moving average
        window = min(30, len(df) // 10)
        if window > 1:
            moving_avg = df_sorted[aqi_column].rolling(window=window).mean()
            plt.plot(df_sorted[date_column], moving_avg, 
                    color='red', linewidth=2, label=f'{window}-day Moving Average')
            plt.legend()
        
        plt.xlabel('Date')
        plt.ylabel('Air Quality Index (AQI)')
        plt.title('Air Quality Index Over Time')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_pollutant_relationships(self, df, pollutants, aqi_column='AQI', save_path=None):
        """Plot relationships between pollutants and AQI"""
        n_pollutants = len(pollutants)
        n_cols = 2
        n_rows = (n_pollutants + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, pollutant in enumerate(pollutants):
            if i < len(axes) and pollutant in df.columns:
                sns.scatterplot(data=df, x=pollutant, y=aqi_column, 
                              alpha=0.6, ax=axes[i])
                
                # Add trend line
                z = np.polyfit(df[pollutant].dropna(), 
                             df.loc[df[pollutant].notna(), aqi_column], 1)
                p = np.poly1d(z)
                axes[i].plot(df[pollutant], p(df[pollutant]), 
                           "r--", alpha=0.8, linewidth=2)
                
                axes[i].set_title(f'{pollutant} vs AQI')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(pollutants), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_feature_importance(self, feature_names, importances, title="Feature Importance", save_path=None):
        """Plot feature importance (dari array importances)"""
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(importances)), importances[indices])
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title(title)
        plt.xticks(range(len(importances)), 
                  [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_feature_importance_from_model(self, model, feature_names, model_name="Model", top_n=20, save_path=None):
        """Plot feature importance dari model (jika ada)"""
        # Cek apakah model memiliki atribut feature_importances_
        if not hasattr(model, 'feature_importances_'):
            print(f"Model {model_name} tidak memiliki feature_importances_.")
            return
            
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[indices], color='b', align='center')
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.gca().invert_yaxis()  # Urutkan dari paling penting di atas
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_model_comparison(self, models_performance, save_path=None):
        """Plot model performance comparison"""
        metrics = ['MAE', 'RMSE', 'R2_Score']
        models = list(models_performance.keys())
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, metric in enumerate(metrics):
            values = [models_performance[model][metric] for model in models]
            bars = axes[i].bar(models, values, alpha=0.7)
            axes[i].set_title(f'{metric} Comparison')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_prediction_vs_actual(self, y_true, y_pred, model_name="Model", save_path=None):
        """Plot predictions vs actual values"""
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.6, color='steelblue')
        
        # Perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        plt.xlabel('Actual AQI')
        plt.ylabel('Predicted AQI')
        plt.title(f'{model_name} - Predictions vs Actual\n'
                 f'MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_residuals(self, y_true, y_pred, model_name="Model", save_path=None):
        """Plot residuals analysis"""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title(f'{model_name} - Residuals vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals distribution
        sns.histplot(residuals, kde=True, ax=axes[1])
        axes[1].axvline(x=0, color='r', linestyle='--')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{model_name} - Residuals Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()