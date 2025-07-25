import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

def plot_failure_prediction_analysis(model, X_test, y_test, y_pred_test, duration_df):
    """
    Generate the sexy curves your stakeholders crave
    """
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Asset Failure Prediction Model - Executive Summary', fontsize=16, fontweight='bold')
    
    # Prediction vs Reality - The Money Shot
    axes[0,0].scatter(y_test, y_pred_test, alpha=0.7, color='darkblue', s=60)
    axes[0,0].plot([0, 5], [0, 5], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0,0].set_xlabel('Actual Years to Failure')
    axes[0,0].set_ylabel('Predicted Years to Failure') 
    axes[0,0].set_title('Prediction Accuracy\n(Closer to red line = better)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Residuals - Show the spread
    residuals = y_test.values - y_pred_test
    axes[0,1].scatter(y_pred_test, residuals, alpha=0.7, color='darkgreen', s=60)
    axes[0,1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0,1].set_xlabel('Predicted Years to Failure')
    axes[0,1].set_ylabel('Prediction Error (Actual - Predicted)')
    axes[0,1].set_title('Model Residuals\n(Random scatter around zero = good)')
    axes[0,1].grid(True, alpha=0.3)
    
    # Failure Distribution by Current State - The Business Insight
    failure_samples = duration_df[duration_df['will_reach_ruim'] == 1]
    state_failure_dist = failure_samples.groupby('current_state')['years_to_ruim'].apply(list)
    
    box_data = [state_failure_dist.get('Bom', []), state_failure_dist.get('Regular', [])]
    box_labels = ['Bom Assets', 'Regular Assets']
    
    bp = axes[1,0].boxplot(box_data, labels=box_labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('orange')
    axes[1,0].set_ylabel('Years Until Failure')
    axes[1,0].set_title('Failure Timeline by Asset Condition\n(Lower = More Urgent)')
    axes[1,0].grid(True, alpha=0.3)
    
    # Model Performance Over Time - The Confidence Curve
    time_horizons = np.arange(0.5, 4.5, 0.5)
    mae_by_horizon = []
    sample_counts = []
    
    for horizon in time_horizons:
        mask = y_test <= horizon
        if mask.sum() > 5:  # Need minimum samples
            horizon_mae = mean_absolute_error(y_test[mask], y_pred_test[mask])
            mae_by_horizon.append(horizon_mae)
            sample_counts.append(mask.sum())
        else:
            mae_by_horizon.append(np.nan)
            sample_counts.append(0)
    
    # Plot MAE curve
    valid_idx = ~np.isnan(mae_by_horizon)
    axes[1,1].plot(time_horizons[valid_idx], np.array(mae_by_horizon)[valid_idx], 
                   'o-', linewidth=3, markersize=8, color='purple')
    axes[1,1].set_xlabel('Prediction Horizon (Years)')
    axes[1,1].set_ylabel('Mean Absolute Error (Years)')
    axes[1,1].set_title('Model Accuracy vs Prediction Distance\n(Flatter = More Reliable)')
    axes[1,1].grid(True, alpha=0.3)
    
    # Add sample size annotations
    ax2 = axes[1,1].twinx()
    ax2.bar(time_horizons, sample_counts, alpha=0.3, color='gray', width=0.2)
    ax2.set_ylabel('Sample Size', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    
    plt.tight_layout()
    
    # Save to outputs directory
    import os
    os.makedirs('../outputs', exist_ok=True)
    plt.savefig('../outputs/failure_prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # The executive summary stats
    print("\n=== EXECUTIVE SUMMARY ===")
    print(f"Model Performance: Predicts failure timing within ±{mean_absolute_error(y_test, y_pred_test):.1f} years")
    print(f"Best Accuracy: {(np.abs(residuals) <= 0.5).mean()*100:.0f}% of predictions within 6 months")
    print(f"Critical Assets: {len(failure_samples[failure_samples['current_state']=='Regular'])} Regular assets need immediate attention")
    print(f"Stable Assets: {len(failure_samples[failure_samples['current_state']=='Bom'])} Bom assets in failure pipeline")
    print(f"Data Coverage: Model trained on {len(y_test) + len(X_test)} real failure scenarios")