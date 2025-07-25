import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from preprocess import create_dataframe, separate_discrete_lifecycles, prepare_duration_features

from models import build_duration_regression, generate_prediction_output

if __name__ == '__main__':

    df = create_dataframe("mockup_vector_state.csv")

    lifecycle_df = separate_discrete_lifecycles(df)

    duration_df = prepare_duration_features(lifecycle_df)

    build_duration_regression(duration_df)

    generate_prediction_output(lifecycle_df, duration_df)