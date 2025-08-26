import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from preprocess import create_dataframe, separate_discrete_lifecycles

from models import build_duration_regression, generate_prediction_output

if __name__ == '__main__':

    df = create_dataframe("Documents/github/asset-lifecycle/data/mockup_vector_state_test.csv")

    historical_lifecycles, current_lifecycles = separate_discrete_lifecycles(df)

    print(historical_lifecycles.describe())

    print(current_lifecycles.describe())