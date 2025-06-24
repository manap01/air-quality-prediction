import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPredictor:
    """
    Class untuk melakukan prediksi menggunakan trained model
    """
    
    def __init__(self):
        self.model = None
        self.model_name = None
        self.scaler = None
        self.feature_columns = None
    
    def load_model(self, model_name='random_forest'):
        """
        Load trained model dari disk
        
        Args:
            model_name (str): Nama model yang akan di-load
        """
        model_path = f'models/trained_models/{model_name}_model.pkl'
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file tidak ditemukan: {model_path}")
        
        self.model = joblib.load(model_path)
        self.model_name = model_name
        
        # Load scaler if exists
        scaler_path = f'models/model_artifacts/scaler.pkl'
        if Path(scaler_path).exists():
            self.scaler = joblib.load(scaler_path)
        
        logger.info(f"Model {model_name} berhasil di-load")
    
    def predict(self, data, model_name='random_forest'):
        """
        Melakukan prediksi pada data baru
        
        Args:
            data (pd.DataFrame or str): Data untuk prediksi atau path ke file
            model_name (str): Nama model yang digunakan
            
        Returns:
            np.array: Array prediksi AQI
        """
        # Load model jika belum di-load
        if self.model is None or self.model_name != model_name:
            self.load_model(model_name)
        
        # Load data jika berupa path
        if isinstance(data, str):
            data = pd.read_csv(data)
        
        # Prepare data
        X = self._prepare_prediction_data(data)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        logger.info(f"Prediksi selesai untuk {len(predictions)} sampel")
        return predictions
    
    def predict_single(self, features_dict, model_name='random_forest'):
        """
        Prediksi untuk satu sampel data
        
        Args:
            features_dict (dict): Dictionary berisi nilai fitur
            model_name (str): Nama model yang digunakan
            
        Returns:
            float: Prediksi AQI
        """
        # Load model jika belum di-load
        if self.model is None or self.model_name != model_name:
            self.load_model(model_name)
        
        # Convert dictionary to DataFrame
        input_data = pd.DataFrame([features_dict])
        
        # Prepare data
        X = self._prepare_prediction_data(input_data)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        logger.info(f"Prediksi single: {prediction}")
        return prediction
    
    def _prepare_prediction_data(self, data):
        """
        Prepare data untuk prediksi
        """
        # Implementasi sesuai kebutuhan
        # Misalnya: scaling, feature engineering, dll.
        # Untuk sekarang kita asumsikan data sudah siap
        return data