# File: create_sample_datasets.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_air_quality_data(n_samples=900):
    """Membuat dataset kualitas udara sintetis"""
    np.random.seed(42)
    start_date = datetime(2020, 1, 1)
    
    data = {
        'date': [start_date + timedelta(hours=i) for i in range(n_samples)],
        'PM2.5': np.random.normal(35, 15, n_samples).clip(0),
        'PM10': np.random.normal(60, 20, n_samples).clip(0),
        'NO2': np.random.normal(25, 10, n_samples).clip(0),
        'SO2': np.random.normal(8, 3, n_samples).clip(0),
        'CO': np.random.normal(1.2, 0.5, n_samples).clip(0),
        'O3': np.random.normal(45, 15, n_samples).clip(0),
        'temperature': np.random.normal(25, 8, n_samples),
        'humidity': np.random.normal(60, 20, n_samples).clip(0, 100),
        'wind_speed': np.random.normal(10, 5, n_samples).clip(0),
        'pressure': np.random.normal(1013, 10, n_samples),
    }
    
    # Buat AQI berdasarkan polutan
    data['AQI'] = (
        data['PM2.5'] * 0.3 +
        data['PM10'] * 0.2 +
        data['NO2'] * 0.15 +
        data['SO2'] * 0.1 +
        data['CO'] * 0.1 +
        data['O3'] * 0.15 +
        np.random.normal(0, 10, n_samples)
    ).clip(0, 500)
    
    return pd.DataFrame(data)

def create_weather_data(n_samples=900):
    """Membuat dataset cuaca sintetis"""
    np.random.seed(42)
    start_date = datetime(2020, 1, 1)
    
    data = {
        'date': [start_date + timedelta(hours=i) for i in range(n_samples)],
        'temperature': np.random.normal(25, 8, n_samples),
        'humidity': np.random.normal(60, 20, n_samples).clip(0, 100),
        'wind_speed': np.random.normal(10, 5, n_samples).clip(0),
        'wind_direction': np.random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], n_samples),
        'pressure': np.random.normal(1013, 10, n_samples),
        'precipitation': np.random.exponential(0.5, n_samples).clip(0, 20),
        'visibility': np.random.normal(10, 3, n_samples).clip(1, 20),
        'cloud_cover': np.random.randint(0, 101, n_samples),
    }
    
    return pd.DataFrame(data)

def create_processed_data(n_samples=900):
    """Membuat dataset hasil pemrosesan sintetis"""
    np.random.seed(42)
    start_date = datetime(2020, 1, 1)
    
    data = {
        'date': [start_date + timedelta(hours=i) for i in range(n_samples)],
        'PM2.5': np.random.normal(35, 15, n_samples).clip(0),
        'PM10': np.random.normal(60, 20, n_samples).clip(0),
        'NO2': np.random.normal(25, 10, n_samples).clip(0),
        'SO2': np.random.normal(8, 3, n_samples).clip(0),
        'CO': np.random.normal(1.2, 0.5, n_samples).clip(0),
        'O3': np.random.normal(45, 15, n_samples).clip(0),
        'temperature': np.random.normal(25, 8, n_samples),
        'humidity': np.random.normal(60, 20, n_samples).clip(0, 100),
        'wind_speed': np.random.normal(10, 5, n_samples).clip(0),
        'pressure': np.random.normal(1013, 10, n_samples),
        'year': [2020] * n_samples,
        'month': [(i % 12) + 1 for i in range(n_samples)],
        'day': [((i // 24) % 28) + 1 for i in range(n_samples)],
        'hour': [i % 24 for i in range(n_samples)],
        'day_of_week': [(i // 24) % 7 for i in range(n_samples)],
        'AQI': np.random.normal(120, 40, n_samples).clip(0, 500),
    }
    
    return pd.DataFrame(data)

# Buat direktori jika belum ada
import os
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/external', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# Buat dan simpan dataset
print("Membuat dataset sintetis...")
air_quality_raw = create_air_quality_data(900)
weather_data = create_weather_data(900)
processed_data = create_processed_data(900)

air_quality_raw.to_csv('data/raw/air_quality_raw.csv', index=False)
weather_data.to_csv('data/external/weather_data.csv', index=False)
processed_data.to_csv('data/processed/air_quality_processed.csv', index=False)

print("✅ Dataset berhasil dibuat:")
print(f" - data/raw/air_quality_raw.csv ({len(air_quality_raw)} baris)")
print(f" - data/external/weather_data.csv ({len(weather_data)} baris)")
print(f" - data/processed/air_quality_processed.csv ({len(processed_data)} baris)")