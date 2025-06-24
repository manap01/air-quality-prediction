"""
Main script untuk menjalankan pipeline prediksi kualitas udara
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.train_model import ModelTrainer
from src.models.predict_model import ModelPredictor
from src.visualization.visualize import Visualizer

def generate_report(metrics, model_name, report_path='reports/laporan_proyek.md'):
    """
    Generate laporan proyek dalam format markdown
    
    Args:
        metrics (dict): Dictionary berisi metrik evaluasi
        model_name (str): Nama model yang digunakan
        report_path (str): Path untuk menyimpan laporan
    """
    # Pastikan folder reports ada
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    # Konten laporan
    report_content = f"""
# Laporan Proyek Prediksi Kualitas Udara

## Hasil Evaluasi Model

**Model yang digunakan**: {model_name}

**Metrik Evaluasi**:
- MAE (Mean Absolute Error): {metrics['mae']:.4f}
- RMSE (Root Mean Squared Error): {metrics['rmse']:.4f}
- R² (R-Squared): {metrics['r2']:.4f}

## Visualisasi Kinerja Model

### 1. Actual vs Predicted
![Actual vs Predicted](figures/{model_name}_actual_vs_predicted.png)

### 2. Feature Importance
![Feature Importance](figures/{model_name}_feature_importance.png)

### 3. Residual Analysis
![Residuals](figures/{model_name}_residuals.png)

## Interpretasi Hasil
Model {model_name} menunjukkan kinerja sebagai berikut:
- **MAE**: Rata-rata kesalahan absolut prediksi adalah {metrics['mae']:.2f} poin AQI
- **RMSE**: Kesalahan prediksi standar adalah {metrics['rmse']:.2f} poin AQI
- **R²**: Model menjelaskan {metrics['r2']*100:.1f}% variasi dalam data
"""

    # Simpan ke file
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"✅ Laporan proyek disimpan di: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Air Quality Prediction Pipeline')
    parser.add_argument('--mode', choices=['train', 'predict', 'evaluate'], 
                       default='train', help='Mode to run the pipeline')
    parser.add_argument('--data_path', type=str, 
                       default='data/raw/air_quality_raw.csv', 
                       help='Path to input data')
    parser.add_argument('--model_name', type=str, 
                       default='random_forest', 
                       choices=['linear_regression', 'random_forest', 'xgboost'],
                       help='Model to use')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models/trained_models', exist_ok=True)
    os.makedirs('models/model_artifacts', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)
    
    print("🌬️ Air Quality Prediction Pipeline")
    print("=" * 50)
    
    if args.mode == 'train':
        print("📊 Loading and preprocessing data...")
        
        # Load data
        loader = DataLoader()
        try:
            data = loader.load_data(args.data_path)
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            return
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.preprocess(data)
        
        # Feature engineering
        feature_engineer = FeatureEngineer()
        engineered_data = feature_engineer.create_features(processed_data)
        
        # Save processed data
        processed_path = 'data/processed/air_quality_processed.csv'
        engineered_data.to_csv(processed_path, index=False)
        print(f"✅ Processed data saved to {processed_path}")
        
        # Train model
        print(f"🤖 Training {args.model_name} model...")
        trainer = ModelTrainer()
        result = trainer.train_model(
            engineered_data, 
            args.model_name,
            return_test_data=True
        )
        
        if len(result) == 3:
            model, metrics, (X_test, y_test, y_pred) = result
        else:
            model, metrics = result
            X_test, y_test, y_pred = None, None, None
        
        print("📈 Training Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Save model
        model_path = f'models/trained_models/{args.model_name}_model.pkl'
        joblib.dump(model, model_path)
        print(f"💾 Model saved to {model_path}")
        
        # Visualize results
        if X_test is not None and y_test is not None and y_pred is not None:
            visualizer = Visualizer()
            
            # Plot feature importance
            if hasattr(model, 'feature_importances_'):
                visualizer.plot_feature_importance_from_model(
                    model=model,
                    feature_names=X_test.columns,
                    model_name=args.model_name,
                    save_path=f'reports/figures/{args.model_name}_feature_importance.png'
                )
            
            # Plot actual vs predicted
            visualizer.plot_prediction_vs_actual(
                y_true=y_test,
                y_pred=y_pred,
                model_name=args.model_name,
                save_path=f'reports/figures/{args.model_name}_actual_vs_predicted.png'
            )
            
            # Plot residuals
            visualizer.plot_residuals(
                y_true=y_test,
                y_pred=y_pred,
                model_name=args.model_name,
                save_path=f'reports/figures/{args.model_name}_residuals.png'
            )
            
            print(f"📊 Visualizations saved to reports/figures")
        
        # Generate project report
        print("📝 Generating project report...")
        generate_report(metrics, args.model_name)
        
    elif args.mode == 'predict':
        print("🔮 Making predictions...")
        
        # Load model and make predictions
        predictor = ModelPredictor()
        predictions = predictor.predict(args.data_path, args.model_name)
        
        # Save predictions
        pred_path = 'data/processed/predictions.csv'
        pd.DataFrame(predictions, columns=['Predicted_AQI']).to_csv(pred_path, index=False)
        print(f"✅ Predictions saved to {pred_path}")
        
    elif args.mode == 'evaluate':
        print("📊 Evaluating model performance...")
        # Implement evaluation logic
        pass

if __name__ == "__main__":
    main()