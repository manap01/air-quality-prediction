import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        pass

    def train_model(self, data, model_name='random_forest', return_test_data=False):
        """
        Melatih model machine learning untuk prediksi AQI.
        
        Args:
            data (pd.DataFrame): Data yang sudah dipreprocess dan di-feature engineering.
            model_name (str): Nama model yang akan digunakan.
            return_test_data (bool): Jika True, kembalikan juga data test (X_val, y_val, y_pred)
            
        Returns:
            model: Model yang sudah dilatih.
            metrics: Dictionary berisi metrik evaluasi.
            (optional) (X_val, y_val, y_pred) jika return_test_data=True
        """
        logger.info(f"Memulai training model {model_name}...")
        
        # Hapus kolom non-numerik (datetime, kategori, dll)
        data = self._remove_non_numeric_columns(data)
        
        # Pisahkan fitur dan target
        X = data.drop('AQI', axis=1)
        y = data['AQI']
        
        # Bagi data menjadi training dan validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Inisialisasi model
        if model_name == 'linear_regression':
            model = LinearRegression()
        elif model_name == 'ridge':
            model = Ridge()
        elif model_name == 'lasso':
            model = Lasso()
        elif model_name == 'random_forest':
            model = RandomForestRegressor(random_state=42)
        elif model_name == 'gradient_boosting':
            model = GradientBoostingRegressor(random_state=42)
        elif model_name == 'xgboost':
            model = XGBRegressor(random_state=42)
        elif model_name == 'svr':
            model = SVR()
        else:
            raise ValueError(f"Model tidak dikenal: {model_name}")
        
        # Latih model
        model.fit(X_train, y_train)
        
        # Prediksi di validation set
        y_pred = model.predict(X_val)
        
        # Hitung metrik evaluasi
        metrics = {
            'mse': mean_squared_error(y_val, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'mae': mean_absolute_error(y_val, y_pred),
            'r2': r2_score(y_val, y_pred)
        }
        
        logger.info(f"Training model {model_name} selesai.")
        logger.info(f"Metrics: {metrics}")
        
        if return_test_data:
            return model, metrics, (X_val, y_val, y_pred)
        else:
            return model, metrics
    
    def _remove_non_numeric_columns(self, data):
        """
        Menghapus kolom non-numerik dari dataset
        
        Args:
            data (pd.DataFrame): Data asli
            
        Returns:
            pd.DataFrame: Data dengan hanya kolom numerik
        """
        # Identifikasi kolom numerik
        numeric_cols = data.select_dtypes(include=['int', 'float', 'int32', 'float32', 'int64', 'float64']).columns
        
        # Pastikan target column (AQI) tetap ada
        if 'AQI' not in numeric_cols:
            numeric_cols = numeric_cols.append(pd.Index(['AQI']))
        
        # Simpan hanya kolom numerik
        numeric_data = data[numeric_cols]
        
        logger.info(f"Shape setelah menghapus kolom non-numerik: {numeric_data.shape}")
        return numeric_data
    
    def hyperparameter_tuning(self, model, param_grid, X_train, y_train):
        """
        Melakukan hyperparameter tuning menggunakan GridSearchCV.
        
        Args:
            model: Model machine learning.
            param_grid (dict): Grid parameter untuk tuning.
            X_train (pd.DataFrame): Data training fitur.
            y_train (pd.Series): Data training target.
            
        Returns:
            best_model: Model terbaik hasil tuning.
        """
        logger.info("Memulai hyperparameter tuning...")
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        logger.info(f"Hyperparameter tuning selesai. Best parameters: {grid_search.best_params_}")
        
        return best_model
    
    def save_model(self, model, file_path):
        """
        Menyimpan model ke file.
        
        Args:
            model: Model machine learning.
            file_path (str): Path untuk menyimpan model.
        """
        joblib.dump(model, file_path)
        logger.info(f"Model disimpan di: {file_path}")