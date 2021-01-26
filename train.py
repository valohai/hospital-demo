import pandas as pd
import valohai

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from catboost import Pool, CatBoostClassifier

from valohai.parameters import get_parameter
from valohai.inputs import get_input_path
from valohai.paths import get_output_path


def clip_column(df, key, lower, upper):
    df[key] = df[key].clip(lower=lower, upper=upper)


def normalize_column(df, key):
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    df[key] = scaler.fit_transform(df[[key]])


data = {
    "data": "https://valohai-hospital-demo.s3-eu-west-1.amazonaws.com/preprocessed.csv",
    "labels": "https://valohai-hospital-demo.s3-eu-west-1.amazonaws.com/labels.csv",
}

params = {
    "learning_rate": 0.001,
    "iterations": 10000,
    "early_stopping_rounds": 200,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_seed": 123,
}

valohai.prepare(step="train", parameters=params, inputs=data)

df_train = pd.read_csv(get_input_path("data"))
df_train["hospital_death"] = pd.read_csv(get_input_path("labels"))["hospital_death"]
df_train, df_train_val = train_test_split(df_train, test_size=0.1)

pd.set_option('display.max_rows', 100)
pd.set_option('display.min_rows', 100)

fields_list = list(df_train.columns)
fields_list.remove('hospital_death')
fields_list.remove('encounter_id')

y_train = df_train['hospital_death']
x_train = df_train[fields_list]
y_test = df_train_val['hospital_death']
x_test = df_train_val[fields_list]

# roc = RocCallback(training_data=(x_train, y_train), validation_data=(x_test, y_test))
# model = Sequential()
# model.add(Dense(get_parameter("layer_count")*2, input_shape=(len(fields_list),), activation='relu', name='fc1'))
# model.add(Dropout(get_parameter("dropout")))
# model.add(Dense(get_parameter("layer_count"), activation='relu', name='fc2'))
# model.add(Dropout(get_parameter("dropout")))
# model.add(Dense(2, activation='softmax', name='output'))
#
# optimizer = SGD(lr=get_parameter("learning_rate"), nesterov=True)
#
#
# model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# model.fit(
#     x_train, y_train, validation_data=(x_test, y_test),
#     verbose=0, batch_size=get_parameter("batch_size"), epochs=get_parameter("epochs"),
#     callbacks=[LogsCallback(), roc, LogsFlush()])

categorical_fields = ['apache_3j_diagnosis', 'gcs_verbal_apache', 'gcs_motor_apache', 'gcs_eyes_apache', 'ethnicity', 'gender', 'hospital_admit_source', 'icu_admit_source', 'icu_stay_type', 'icu_type', 'apache_3j_bodysystem', 'apache_2_bodysystem', 'apache_2_diagnosis']

train_pool = Pool(data=x_train, label=y_train, cat_features=categorical_fields)

params = {
    'task_type': 'GPU',
    'eval_metric': 'AUC',
    'gpu_ram_part': 0.95,
    'od_type': "Iter",
    'depth': get_parameter("depth"),
    'l2_leaf_reg': get_parameter("l2_leaf_reg"),
    'early_stopping_rounds': get_parameter("early_stopping_rounds"),
    'learning_rate': get_parameter("learning_rate"),
    'iterations': get_parameter("iterations"),
    'random_seed': get_parameter("seed"),
    'verbose': 100,
}

model = CatBoostClassifier(train_pool)
model.set_params(**params)
model.fit(train_pool, plot=False)

# grid = {
#     'learning_rate': [0.001, 0.003, 0.005],
#     'depth': [4, 6, 10],
#     'l2_leaf_reg': [1, 3, 5, 7, 9],
#     "iterations": [10000]
# }
#
# grid_search_result = model.grid_search(
#     grid,
#     train_pool,
#     plot=False,
#     refit=True,
#     verbose=100,
#     partition_random_seed=123
# )
# print(grid_search_result)

print(model.get_best_score()['learn'])
model.save_model(get_output_path("model/model.cbm"))
