from catboost import CatBoostClassifier
import pandas as pd
import valohai
from valohai.inputs import get_input_path
from valohai.paths import get_output_path
from utils import predict


def inference(model_path, data_path, results_path):
    m = CatBoostClassifier()
    m.load_model(model_path)
    df_data = pd.read_csv(data_path)
    results = predict(df_data, m)
    results[["encounter_id", "hospital_death"]].to_csv(results_path, index=False)


data = {
    "data": "https://valohai-hospital-demo.s3-eu-west-1.amazonaws.com/training_v2.csv",
    "model": "https://valohai-hospital-demo.s3-eu-west-1.amazonaws.com/model.cbm",
}

valohai.prepare(step="predict", parameters={}, inputs=data)
inference(
    model_path=get_input_path("model"),
    data_path=get_input_path("data"),
    results_path=get_output_path("results.csv")
)
