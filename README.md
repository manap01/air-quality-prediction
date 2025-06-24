# Air Quality Prediction Project

![Air Quality Visualization](reports/figures/random_forest_actual_vs_predicted.png)

## Overview
This project predicts Air Quality Index (AQI) using machine learning models. The pipeline includes data loading, preprocessing, feature engineering, model training, and visualization. The system supports multiple models including Random Forest, XGBoost, and Linear Regression.

## Features
- 🌀 End-to-end pipeline for air quality prediction
- 📊 Comprehensive data preprocessing and feature engineering
- 🤖 Multiple ML model support (Random Forest, XGBoost, Linear Regression)
- 📈 Performance metrics and visualizations
- 💾 Model persistence and prediction capabilities

## Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/air-quality-prediction.git
cd air-quality-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
```
air-quality-prediction/
├── data/
│   ├── raw/                  # Raw data files
│   └── processed/            # Processed data files
├── models/
│   ├── trained_models/       # Saved ML models
│   └── model_artifacts/      # Model artifacts
├── notebooks/                # Jupyter notebooks
├── reports/                  # Output reports
│   └── figures/              # Visualization images
├── src/                      # Source code
│   ├── data/                 # Data handling
│   ├── features/             # Feature engineering
│   ├── models/               # ML models
│   └── visualization/        # Visualization tools
├── main.py                   # Main pipeline script
├── create_sample_datasets.py # Data generation script
└── requirements.txt          # Dependencies
```

## Usage
### Training Mode
Train a model with default settings:
```bash
python main.py --mode train
```

Train a specific model:
```bash
python main.py --mode train --model_name xgboost
```

### Prediction Mode
Make predictions using a trained model:
```bash
python main.py --mode predict --data_path data/raw/new_data.csv
```

### Supported Models
- `random_forest` (default)
- `xgboost`
- `linear_regression`

## Results
After training, the pipeline will generate:
- Performance metrics (MSE, RMSE, MAE, R²)
- Model file (saved in `models/trained_models/`)
- Visualizations (saved in `reports/figures/`)

### Sample Visualizations
1. **Feature Importance**  
   ![Feature Importance](reports/figures/random_forest_feature_importance.png)

2. **Actual vs Predicted Values**  
   ![Actual vs Predicted](reports/figures/random_forest_actual_vs_predicted.png)

3. **Residual Analysis**  
   ![Residual Analysis](reports/figures/random_forest_residuals.png)

## Generating Sample Data
To create sample datasets for testing:
```bash
python create_sample_datasets.py
```

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
Hanif - [@yourtwitter](https://twitter.com/yourtwitter) - your-email@example.com

Project Link: [https://github.com/your-username/air-quality-prediction](https://github.com/your-username/air-quality-prediction)