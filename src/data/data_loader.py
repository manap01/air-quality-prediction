# src/data/data_loader.py
import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        pass

    def load_data(self, file_path, **kwargs):
        """
        Memuat dataset dari file CSV.

        Args:
            file_path (str): Path ke file CSV.
            **kwargs: Keyword arguments tambahan untuk pd.read_csv.

        Returns:
            pd.DataFrame: Data yang dimuat.
        """
        logger.info(f"Memuat data dari: {file_path}")
        
        # Periksa apakah file ada
        if not os.path.exists(file_path):
            logger.error(f"File tidak ditemukan: {file_path}")
            raise FileNotFoundError(f"File tidak ditemukan: {file_path}")
        
        try:
            # Coba baca file dengan encoding berbeda jika perlu
            try:
                data = pd.read_csv(file_path, **kwargs)
            except UnicodeDecodeError:
                logger.warning("Mencoba encoding 'latin1'...")
                data = pd.read_csv(file_path, encoding='latin1', **kwargs)
            
            # Periksa apakah data kosong
            if data.empty:
                logger.warning("Data kosong, mencoba tanpa header...")
                data = pd.read_csv(file_path, header=None, **kwargs)
            
            logger.info(f"Data berhasil dimuat. Shape: {data.shape}")
            return data
        except pd.errors.EmptyDataError:
            logger.error("File kosong atau tidak berisi data kolom")
            raise
        except Exception as e:
            logger.error(f"Error saat memuat data: {e}")
            raise