import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        pass

    def preprocess(self, data):
        """
        Melakukan preprocessing dasar pada data.

        Args:
            data (pd.DataFrame): Data mentah.

        Returns:
            pd.DataFrame: Data yang sudah dipreprocess.
        """
        logger.info("Memulai preprocessing data...")
        
        # Cek jika data kosong
        if data.empty:
            logger.warning("Data kosong diterima untuk preprocessing.")
            return data
        
        # 1. Handling missing values
        logger.info("Menangani missing values...")
        data = self._handle_missing_values(data)
        
        # 2. Konversi tipe data
        logger.info("Mengonversi tipe data...")
        data = self._convert_data_types(data)
        
        # 3. Feature engineering dasar
        logger.info("Membuat fitur dasar...")
        data = self._create_basic_features(data)
        
        logger.info("Preprocessing selesai.")
        return data

    def _handle_missing_values(self, data):
        # Isi dengan median untuk numerik, modus untuk kategorik
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                if data[col].isnull().sum() > 0:
                    median_val = data[col].median()
                    data[col].fillna(median_val, inplace=True)
            else:
                if data[col].isnull().sum() > 0:
                    mode_val = data[col].mode()[0]
                    data[col].fillna(mode_val, inplace=True)
        return data

    def _convert_data_types(self, data):
        # Konversi kolom tanggal jika ada
        date_cols = [col for col in data.columns if 'date' in col.lower()]
        for col in date_cols:
            data[col] = pd.to_datetime(data[col], errors='coerce')
        
        # Konversi kolom kategorik
        cat_cols = data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            data[col] = data[col].astype('category')
        
        return data

    def _create_basic_features(self, data):
        # Ekstrak fitur waktu jika ada kolom tanggal
        date_cols = [col for col in data.columns if 'date' in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            data['year'] = data[date_col].dt.year
            data['month'] = data[date_col].dt.month
            data['day'] = data[date_col].dt.day
            data['hour'] = data[date_col].dt.hour
            data['day_of_week'] = data[date_col].dt.dayofweek
        
        return data