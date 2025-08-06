# Asset Lifecycle Prediction System

This project implements a predictive maintenance system for assets, analyzing their condition sequences to predict time-to-failure and failure probabilities. The system processes historical condition data, splits it into discrete lifecycles, and builds machine learning models to forecast when assets might reach a "Ruim" (poor) state.

## Project Structure

- **`preprocess.py`**: Handles data loading, lifecycle segmentation, and feature engineering.
  - `create_dataframe`: Loads raw data from CSV into a pandas DataFrame.
  - `separate_discrete_lifecycles`: Splits asset condition sequences into lifecycles based on condition improvements.
  - `prepare_duration_features`: Generates features for duration prediction, including years to failure.
  
- **`models.py`**: Contains machine learning models for prediction.
  - `build_duration_regression`: Trains a Random Forest Regressor to predict years until failure.
  - `generate_prediction_output`: Produces predictions for next state and failure probabilities, saving results to CSV.

- **`plot.py`**: Generates visualizations for model performance and insights.
  - `plot_failure_prediction_analysis`: Creates plots showing prediction accuracy, residuals, failure timelines, and model performance over time.

- **`data/`**: Directory for input data (e.g., `mockup_vector_state.csv`).
- **`outputs/`**: Directory for generated outputs (e.g., `asset_predictions.csv`, `failure_prediction_analysis.png`).

## Requirements

- Python 3.8+
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

Install dependencies using:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage

1. **Prepare Data**: Ensure your input data is in `data/mockup_vector_state.csv` with columns `Serial` and `vetor_condicoes` (comma-separated condition sequences: "Bom", "Regular", "Ruim").

2. **Run the Pipeline**:
   ```bash
   python preprocess.py
   python models.py
   python plot.py
   ```
   Alternatively, create a main script to run all steps sequentially:
   ```python
   from preprocess import create_dataframe, separate_discrete_lifecycles, prepare_duration_features
   from models import build_duration_regression, generate_prediction_output
   from plot import plot_failure_prediction_analysis

   df = create_dataframe('../data/mockup_vector_state.csv')
   lifecycle_df = separate_discrete_lifecycles(df)
   duration_df = prepare_duration_features(lifecycle_df)
   model, features = build_duration_regression(duration_df)
   output_df = generate_prediction_output(lifecycle_df, duration_df)
   ```

3. **Outputs**:
   - `outputs/asset_predictions.csv`: Contains predictions for each asset (current state, predicted next state, and failure probabilities for 1 and 2 years).
   - `outputs/failure_prediction_analysis.png`: Visualizations of model performance.

## Key Features

- **Lifecycle Analysis**: Segments asset condition sequences into discrete lifecycles based on condition improvements.
- **Duration Prediction**: Uses Random Forest Regression to predict years until an asset reaches "Ruim".
- **Failure Probability**: Estimates the likelihood of failure within 1 and 2 years using Random Forest Classification.
- **Visualizations**: Provides plots for prediction accuracy, residuals, failure timelines by state, and model performance over time.

## Notes

- The system assumes the input CSV has a specific format. Adjust `create_dataframe` in `preprocess.py` if your data format differs.
- The model is tuned for assets with states "Bom", "Regular", and "Ruim". Modify `state_rank` in `preprocess.py` for different state hierarchies.
- Outputs are saved to the `outputs/` directory, which is created automatically if it doesn't exist.

## Future Improvements

- Add cross-validation for more robust model evaluation.
- Incorporate additional features (e.g., asset type, environmental factors).
- Optimize model hyperparameters using grid search.
- Add support for real-time predictions via an API.