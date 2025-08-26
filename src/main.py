import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from preprocess import create_dataframe, separate_discrete_lifecycles

from models import build_duration_regression, generate_prediction_output

from pathlib import Path

from plot import simple_duration_histogram

if __name__ == '__main__':

    df = create_dataframe("Documents/github/asset-lifecycle/data/mockup_vector_state_test.csv")
    
    df_kartado = pd.read_csv(Path.joinpath(Path.home() , "Documents/github/asset-lifecycle/data/tabela_kartado.csv"))

    historical_lifecycles, current_lifecycles = separate_discrete_lifecycles(df)

    merged_with_km = historical_lifecycles.merge(df_kartado[['original_serial', 'km']],
                                                 on='original_serial',
                                                 how='left'
                                                 )

    print(merged_with_km.columns)

    simple_duration_histogram(merged_with_km)