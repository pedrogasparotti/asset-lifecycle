import numpy as np
import pandas as pd
from pathlib import Path

def create_dataframe(file_name):

    # define global path
    p = Path('.')

    file_arg = '**/*' + file_name

    # find path that contains the file
    path = list(p.glob(file_arg))[0]

    raw_data = pd.read_csv(path)

    raw_dataframe = pd.DataFrame(raw_data)

    print(raw_dataframe.head())

    print(f"created dataframe from file in {path}")

if __name__ == '__main__':

    create_dataframe("mockup_vector_state.csv")