import pandas as pd
import valohai
from valohai.inputs import get_input_path
from valohai.outputs import get_output_path
from utils import preprocess

data = {
    "data": "https://valohai-hospital-demo.s3-eu-west-1.amazonaws.com/training_v2.csv",
    "dictionary": "https://valohai-hospital-demo.s3-eu-west-1.amazonaws.com/dictionary.csv"
}

valohai.prepare(step="preprocess", parameters={}, inputs=data)

df_train = pd.read_csv(get_input_path("data/data.csv"))
df_dictionary = pd.read_csv(get_input_path("dictionary"))
df_preprocessed = preprocess(df_train, df_dictionary)
df_preprocessed.to_csv(get_output_path("data/data.csv"), index=False)
