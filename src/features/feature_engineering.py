import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Class untuk feature engineering pada data kualitas udara
    """
    
    def __init__(self):
        self.feature_selector = None
        self.polynomial_features = None
        self.feature_importance = {}
    
    def create_features(self, data, target_column='AQI'):
        """
        Membuat fitur-fitur baru untuk meningkatkan performa model
        
        Args:
            data (pd.DataFrame): Data yang sudah dipreprocessing
            target_column (str): Nama kolom target
            
        Returns:
            pd.DataFrame: Data dengan fitur tambahan
        """
        logger.info("Memulai feature engineering...")
        
        # Copy data
        engineered_data = data.copy()
        
        # 1. Create interaction features
        engineered_data = self._create_interaction_features(engineered_data)
        
        # 2. Create ratio features
        engineered_data = self._create_ratio_features(engineered_data)
        
        # 3. Create aggregate features
        engineered_data = self._create_aggregate_features(engineered_data)
        
        # 4. Create domain-specific features
        engineered_data = self._create_domain_features(engineered_data)
        
        # 5. Create lag features (if datetime is available)
        engineered_data = self._create_lag_features(engineered_data, target_column)
        
        # 6. Create statistical features
        engineered_data = self._create_statistical_features(engineered_data)
        
        logger.info(f"Feature engineering selesai. Shape: {engineered_data.shape}")
        return engineered_data
    
    def _create_interaction_features(self, data):
        """
        Membuat fitur interaksi antara polutan dan kondisi meteorologi
        """
        logger.info("Membuat interaction features...")
        
        # Pollutant interactions
        pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        meteorology_cols = ['temperature', 'humidity', 'wind_speed', 'pressure']
        
        existing_pollutants = [col for col in pollutant_cols if col in data.columns]
        existing_meteorology = [col for col in meteorology_cols if col in data.columns]
        
        # Temperature interactions with pollutants
        if 'temperature' in data.columns:
            for pollutant in existing_pollutants:
                data[f'{pollutant}_temp_interaction'] = data[pollutant] * data['temperature']
        
        # Humidity interactions with pollutants
        if 'humidity' in data.columns:
            for pollutant in existing_pollutants:
                data[f'{pollutant}_humidity_interaction'] = data[pollutant] * data['humidity']
        
        # Wind speed interactions (inversely related to pollutant concentration)
        if 'wind_speed' in data.columns:
            for pollutant in existing_pollutants:
                # Hindari pembagian dengan nol
                wind_speed_adj = data['wind_speed'].replace(0, 0.1)
                data[f'{pollutant}_wind_ratio'] = data[pollutant] / wind_speed_adj
        
        # PM2.5 and PM10 interaction
        if 'PM2.5' in data.columns and 'PM10' in data.columns:
            data['PM_interaction'] = data['PM2.5'] * data['PM10']
        
        return data
    
    def _create_ratio_features(self, data):
        """
        Membuat fitur rasio antar polutan
        """
        logger.info("Membuat ratio features...")
        
        # PM2.5 to PM10 ratio
        if 'PM2.5' in data.columns and 'PM10' in data.columns:
            # Hindari pembagian dengan nol
            pm10_adj = data['PM10'].replace(0, 0.1)
            data['PM2.5_PM10_ratio'] = data['PM2.5'] / pm10_adj
        
        # NO2 to SO2 ratio
        if 'NO2' in data.columns and 'SO2' in data.columns:
            # Hindari pembagian dengan nol
            so2_adj = data['SO2'].replace(0, 0.1)
            data['NO2_SO2_ratio'] = data['NO2'] / so2_adj
        
        # Temperature to humidity ratio
        if 'temperature' in data.columns and 'humidity' in data.columns:
            # Hindari pembagian dengan nol
            humidity_adj = data['humidity'].replace(0, 0.1)
            data['temp_humidity_ratio'] = data['temperature'] / humidity_adj
        
        # Pressure to wind speed ratio
        if 'pressure' in data.columns and 'wind_speed' in data.columns:
            # Hindari pembagian dengan nol
            wind_speed_adj = data['wind_speed'].replace(0, 0.1)
            data['pressure_wind_ratio'] = data['pressure'] / wind_speed_adj
        
        return data
    
    def _create_aggregate_features(self, data):
        """
        Membuat fitur agregat dari polutan
        """
        logger.info("Membuat aggregate features...")
        
        pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        existing_pollutants = [col for col in pollutant_cols if col in data.columns]
        
        if len(existing_pollutants) >= 2:
            # Total pollutant load
            data['total_pollutants'] = data[existing_pollutants].sum(axis=1)
            
            # Average pollutant concentration
            data['avg_pollutants'] = data[existing_pollutants].mean(axis=1)
            
            # Maximum pollutant concentration
            data['max_pollutants'] = data[existing_pollutants].max(axis=1)
            
            # Standard deviation of pollutants
            data['std_pollutants'] = data[existing_pollutants].std(axis=1)
            
            # Pollutant diversity index
            data['pollutant_diversity'] = data[existing_pollutants].apply(
                lambda x: len(x[x > x.median()]), axis=1
            )
        
        return data
    
    def _create_domain_features(self, data):
        """
        Membuat fitur spesifik domain kualitas udara
        """
        logger.info("Membuat domain-specific features...")
        
        # Air Quality Index categories for different pollutants
        if 'PM2.5' in data.columns:
            # Pastikan tidak ada nilai NaN sebelum membuat kategori
            data['PM2.5'] = data['PM2.5'].fillna(0)
            data['PM2.5_category'] = pd.cut(
                data['PM2.5'], 
                bins=[0, 12, 35.4, 55.4, 150.4, np.inf], 
                labels=[0, 1, 2, 3, 4]
            )
            # Konversi ke integer dengan penanganan NaN
            data['PM2.5_category'] = data['PM2.5_category'].cat.codes.replace({-1: 0})
        
        if 'PM10' in data.columns:
            # Pastikan tidak ada nilai NaN sebelum membuat kategori
            data['PM10'] = data['PM10'].fillna(0)
            data['PM10_category'] = pd.cut(
                data['PM10'], 
                bins=[0, 54, 154, 254, 354, np.inf], 
                labels=[0, 1, 2, 3, 4]
            )
            # Konversi ke integer dengan penanganan NaN
            data['PM10_category'] = data['PM10_category'].cat.codes.replace({-1: 0})
        
        # Weather comfort index
        if 'temperature' in data.columns and 'humidity' in data.columns:
            data['comfort_index'] = (
                np.abs(data['temperature'] - 22) + 
                np.abs(data['humidity'] - 50)
            ) / 2
        
        # Atmospheric stability indicator
        if 'temperature' in data.columns and 'wind_speed' in data.columns:
            # Hindari pembagian dengan nol
            wind_speed_adj = data['wind_speed'].replace(0, 0.1)
            data['atmospheric_stability'] = data['temperature'] / wind_speed_adj
        
        # Seasonal pollution pattern
        if 'month' in data.columns:
            # Winter months typically have higher pollution
            data['winter_season'] = data['month'].isin([12, 1, 2]).astype(int)
            data['summer_season'] = data['month'].isin([6, 7, 8]).astype(int)
        
        # Rush hour indicator
        if 'hour' in data.columns:
            data['rush_hour'] = data['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
            data['night_time'] = data['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
        
        return data
    
    def _create_lag_features(self, data, target_column):
        """
        Membuat lag features jika ada urutan temporal
        """
        logger.info("Membuat lag features...")
        
        # Sort by datetime if available
        datetime_cols = [col for col in data.columns if 'date' in col.lower()]
        
        if datetime_cols and len(datetime_cols) > 0:
            date_col = datetime_cols[0]
            
            # Pastikan kolom tanggal dalam format datetime
            if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
                try:
                    data[date_col] = pd.to_datetime(data[date_col])
                    logger.info(f"Kolom {date_col} dikonversi ke datetime")
                except Exception as e:
                    logger.warning(f"Gagal mengonversi kolom {date_col} ke datetime: {e}")
                    return data
                    
            data = data.sort_values(date_col)
            
            # Create lag features for key pollutants and target
            lag_columns = ['PM2.5', 'PM10', 'NO2', target_column]
            existing_lag_cols = [col for col in lag_columns if col in data.columns]
            
            for col in existing_lag_cols:
                # 1-hour lag
                data[f'{col}_lag1'] = data[col].shift(1)
                
                # 24-hour lag (daily pattern)
                data[f'{col}_lag24'] = data[col].shift(24)
                
                # 7-day lag (weekly pattern)
                data[f'{col}_lag168'] = data[col].shift(168)
                
                # Rolling averages
                data[f'{col}_rolling_mean_3'] = data[col].rolling(window=3, min_periods=1).mean()
                data[f'{col}_rolling_mean_24'] = data[col].rolling(window=24, min_periods=1).mean()
                
                # Rolling standard deviation
                data[f'{col}_rolling_std_24'] = data[col].rolling(window=24, min_periods=1).std()
            
            # Fill NaN values created by lag features
            data = data.fillna(method='bfill').fillna(method='ffill')
        
        return data
    
    def _create_statistical_features(self, data):
        """
        Membuat fitur statistik
        """
        logger.info("Membuat statistical features...")
        
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        existing_pollutants = [col for col in pollutant_cols if col in numerical_cols]
        
        if len(existing_pollutants) >= 2:
            # Skewness and kurtosis for pollutant distribution
            for col in existing_pollutants[:3]:  # Limit to avoid too many features
                # Pastikan tidak ada nilai negatif
                data[col] = data[col].clip(lower=0.1)
                
                # Transformasi
                data[f'{col}_squared'] = data[col] ** 2
                data[f'{col}_log'] = np.log1p(data[col])
                data[f'{col}_sqrt'] = np.sqrt(data[col])
        
        # Create interaction terms for top features
        if 'PM2.5' in data.columns and 'temperature' in data.columns:
            data['PM2.5_temp_squared'] = (data['PM2.5'] * data['temperature']) ** 2
        
        return data
    
    def select_features(self, data, target_column='AQI', method='correlation', k=20):
        """
        Seleksi fitur terbaik
        
        Args:
            data (pd.DataFrame): Data dengan semua fitur
            target_column (str): Nama kolom target
            method (str): Metode seleksi ('correlation', 'univariate', 'rfe')
            k (int): Jumlah fitur yang dipilih
        """
        logger.info(f"Melakukan feature selection dengan metode: {method}")
        
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        if method == 'correlation':
            # Select features based on correlation with target
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            selected_features = correlations.head(k).index.tolist()
            
        elif method == 'univariate':
            # Select features using univariate statistical tests
            selector = SelectKBest(score_func=f_regression, k=k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif method == 'rfe':
            # Recursive Feature Elimination
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            selector = RFE(estimator=estimator, n_features_to_select=k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # Store feature importance
        if method == 'correlation':
            self.feature_importance = correlations.head(k).to_dict()
        
        logger.info(f"Selected {len(selected_features)} features")
        
        # Return data with selected features plus target
        selected_data = data[selected_features + [target_column]]
        return selected_data
    
    def create_polynomial_features(self, data, target_column='AQI', degree=2, 
                                 interaction_only=True, include_bias=False):
        """
        Membuat polynomial features
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Target column name
            degree (int): Degree of polynomial features
            interaction_only (bool): Only interaction features
            include_bias (bool): Include bias column
        """
        logger.info(f"Membuat polynomial features degree {degree}...")
        
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Limit features to avoid memory issues
        if len(X.columns) > 10:
            # Select top 10 features by correlation
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            top_features = correlations.head(10).index.tolist()
            X = X[top_features]
        
        # Create polynomial features
        self.polynomial_features = PolynomialFeatures(
            degree=degree, 
            interaction_only=interaction_only, 
            include_bias=include_bias
        )
        
        X_poly = self.polynomial_features.fit_transform(X)
        
        # Create feature names
        feature_names = self.polynomial_features.get_feature_names_out(X.columns)
        
        # Create DataFrame with polynomial features
        poly_df = pd.DataFrame(X_poly, columns=feature_names, index=data.index)
        
        # Add target column back
        poly_df[target_column] = y
        
        logger.info(f"Polynomial features created. New shape: {poly_df.shape}")
        return poly_df
    
    def get_feature_importance_report(self, data, target_column='AQI'):
        """
        Generate feature importance report
        """
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Calculate correlations
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        
        # Use Random Forest for feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        rf_importance = pd.Series(rf.feature_importances_, index=X.columns)
        rf_importance = rf_importance.sort_values(ascending=False)
        
        report = {
            'correlation_ranking': correlations.to_dict(),
            'random_forest_importance': rf_importance.to_dict(),
            'total_features': len(X.columns),
            'feature_types': {
                'original': len([col for col in X.columns if not any(
                    suffix in col for suffix in ['ratio', 'interaction', 'lag', 'rolling', 'category']
                )]),
                'engineered': len([col for col in X.columns if any(
                    suffix in col for suffix in ['ratio', 'interaction', 'lag', 'rolling', 'category']
                )])
            }
        }
        
        return report