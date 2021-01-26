import valohai
from sklearn.model_selection import train_test_split
from valohai.inputs import get_input_path
import pandas as pd
from valohai.parameters import get_parameter
from valohai.paths import get_output_path

data = {
    "data": "https://valohai-hospital-demo.s3-eu-west-1.amazonaws.com/training_v2.csv",
}

params = {
    "testsize": 0.1,
}

valohai.prepare(step="split", parameters=params, inputs=data)

df = pd.read_csv(get_input_path("data"))
df_train, df_test = train_test_split(df, test_size=get_parameter("testsize"))
df_train.drop("hospital_death").to_csv(get_output_path("train/data.csv"), index=False)
df_test["encounter_id", "hospital_death"].to_csv(get_output_path("train/labels.csv"), index=False)
df_test.drop("hospital_death").to_csv(get_output_path("test/data.csv"), index=False)
df_test["encounter_id", "hospital_death"].to_csv(get_output_path("test/labels.csv"), index=False)
