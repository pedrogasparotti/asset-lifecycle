from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np

from plot import plot_failure_prediction_analysis

def build_duration_regression(duration_df):
    """
    Build regression model to predict years until Ruim state
    Focus on samples that actually reach Ruim
    """
    # Filter to samples that will reach Ruim
    failure_samples = duration_df[duration_df['will_reach_ruim'] == 1].copy()
    
    # Features for prediction
    feature_cols = ['years_elapsed', 'bom_ratio', 'regular_ratio', 'degradation_rate']
    
    # Add current state as dummy variables
    state_dummies = pd.get_dummies(failure_samples['current_state'], prefix='state')
    features_df = pd.concat([failure_samples[feature_cols], state_dummies], axis=1)
    
    X = features_df
    y = failure_samples['years_to_ruim']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Random Forest (handles non-linear patterns better than linear regression)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("=== REGRESSION MODEL RESULTS ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Train MAE: {train_mae:.2f} years")
    print(f"Test MAE: {test_mae:.2f} years")
    print(f"Train R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")
    
    print(f"\nFeature Importance:")
    print(feature_importance.head(8))
    
    print(f"\nPrediction vs Reality (test set):")
    comparison = pd.DataFrame({
        'actual': y_test.values,
        'predicted': y_pred_test,
        'error': np.abs(y_test.values - y_pred_test)
    }).round(2)
    print(comparison.head(10))
    
    # Prediction accuracy by time horizon
    print(f"\nAccuracy by prediction horizon:")
    for horizon in [1, 2, 3]:
        mask = y_test <= horizon
        if mask.sum() > 0:
            horizon_mae = mean_absolute_error(y_test[mask], y_pred_test[mask])
            print(f"≤{horizon} years: MAE = {horizon_mae:.2f} years ({mask.sum()} samples)")

    plot_failure_prediction_analysis(model, X_test, y_test, y_pred_test, duration_df)
    
    
    return model, X.columns

def generate_prediction_output(lifecycle_df, duration_df):
    """
    Generate CSV with next state predictions and failure probabilities
    """
    # Get latest state for each original asset
    latest_states = []
    
    for serial in lifecycle_df['original_serial'].unique():
        asset_lifecycles = lifecycle_df[lifecycle_df['original_serial'] == serial]
        
        # Get the most recent lifecycle (last one created)
        latest_lifecycle = asset_lifecycles.iloc[-1]
        latest_sequence = latest_lifecycle['lifecycle_sequence'].split()
        current_state = latest_sequence[-1]
        
        # Build features for prediction (simulating current position)
        years_elapsed = len(latest_sequence)
        bom_count = latest_sequence.count('Bom')
        regular_count = latest_sequence.count('Regular')
        
        # Degradation rate
        changes = 0
        for i in range(len(latest_sequence)-1):
            if latest_sequence[i] != latest_sequence[i+1]:
                changes += 1
        degradation_rate = changes / years_elapsed if years_elapsed > 0 else 0
        
        latest_states.append({
            'Serial': serial,
            'current_state': current_state,
            'years_elapsed': years_elapsed,
            'bom_ratio': bom_count / years_elapsed,
            'regular_ratio': regular_count / years_elapsed,
            'degradation_rate': degradation_rate
        })
    
    current_df = pd.DataFrame(latest_states)
    
    # Predict next state (simple transition logic based on historical patterns)
    def predict_next_state(row):
        if row['current_state'] == 'Ruim':
            return 'Ruim'  # Already at worst state
        elif row['current_state'] == 'Regular':
            # High degradation rate = likely to go to Ruim
            if row['degradation_rate'] > 0.5:
                return 'Ruim'
            else:
                return 'Regular'
        else:  # Bom
            # Based on degradation pattern
            if row['degradation_rate'] > 0.3:
                return 'Regular'
            else:
                return 'Bom'
    
    current_df['predicted_next_state'] = current_df.apply(predict_next_state, axis=1)
    
    # Train failure probability models for year 1 and year 2
    # Use duration data to create binary classifiers
    failure_samples = duration_df[duration_df['will_reach_ruim'] == 1].copy()
    
    # Features for probability prediction
    prob_features = ['years_elapsed', 'bom_ratio', 'regular_ratio', 'degradation_rate']
    
    # Add state dummies
    state_dummies_train = pd.get_dummies(failure_samples['current_state'], prefix='state')
    X_prob_train = pd.concat([failure_samples[prob_features], state_dummies_train], axis=1)
    
    # Binary targets for year 1 and year 2
    y_year1 = (failure_samples['years_to_ruim'] <= 1).astype(int)
    y_year2 = (failure_samples['years_to_ruim'] <= 2).astype(int)
    
    # Train probability models
    prob_model_year1 = RandomForestClassifier(n_estimators=100, random_state=42)
    prob_model_year2 = RandomForestClassifier(n_estimators=100, random_state=42)
    
    prob_model_year1.fit(X_prob_train, y_year1)
    prob_model_year2.fit(X_prob_train, y_year2)
    
    # Prepare current data for probability prediction
    current_state_dummies = pd.get_dummies(current_df['current_state'], prefix='state')
    
    # Ensure all state columns exist
    for col in state_dummies_train.columns:
        if col not in current_state_dummies.columns:
            current_state_dummies[col] = 0
    
    # Reorder columns to match training data
    current_state_dummies = current_state_dummies.reindex(columns=state_dummies_train.columns, fill_value=0)
    
    X_current = pd.concat([current_df[prob_features], current_state_dummies], axis=1)
    
    # Get probabilities
    prob_year1 = prob_model_year1.predict_proba(X_current)[:, 1]  # Probability of class 1 (failure)
    prob_year2 = prob_model_year2.predict_proba(X_current)[:, 1]
    
    # Create final output
    output_df = pd.DataFrame({
        'Serial': current_df['Serial'],
        'current_state': current_df['current_state'],
        'predicted_next_state': current_df['predicted_next_state'],
        'probability_of_ruim_year1': np.round(prob_year1, 3),
        'probability_of_ruim_year2': np.round(prob_year2, 3)
    })
    
    # Save to outputs directory
    os.makedirs('../outputs', exist_ok=True)
    output_path = '../outputs/asset_predictions.csv'
    output_df.to_csv(output_path, index=False)
    
    print(f"=== PREDICTION OUTPUT GENERATED ===")
    print(f"File saved: {output_path}")
    print(f"Assets processed: {len(output_df)}")
    print(f"High risk assets (>50% chance year 1): {(output_df['probability_of_ruim_year1'] > 0.5).sum()}")
    print(f"Medium risk assets (>30% chance year 2): {(output_df['probability_of_ruim_year2'] > 0.3).sum()}")
    
    print(f"\nSample predictions:")
    print(output_df.head(10))
    
    return output_df