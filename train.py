import pandas as pd
import valohai

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow_core.python.keras.layers.core import Dropout
from tensorflow_core.python.keras.layers.normalization_v2 import BatchNormalization
from catboost import Pool, cv, CatBoostClassifier, CatBoostRegressor

from valohai.parameters import get_parameter
from valohai.inputs import get_input_path
from valohai.paths import get_output_path
from valohai.logs import log_partial, flush_logs
from sklearn.metrics import roc_auc_score


class RocCallback(Callback):
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        if epoch % get_parameter("logfreq") == 0:
            y_pred_train = self.model.predict_proba(self.x)
            y_pred_train = y_pred_train[:, 1]
            roc_train = roc_auc_score(self.y, y_pred_train)
            y_pred_val = self.model.predict_proba(self.x_val)
            y_pred_val = y_pred_val[:, 1]
            roc_val = roc_auc_score(self.y_val, y_pred_val)
            log_partial('roc-auc_train', str(round(roc_train, 4)))
            log_partial('roc-auc_val', str(round(roc_val, 4)))
            return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def clip_column(df, key, lower, upper):
    df[key] = df[key].clip(lower=lower, upper=upper)


def normalize_column(df, key):
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    df[key] = scaler.fit_transform(df[[key]])


class LogsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % get_parameter("logfreq") == 0:
            log_partial("epoch", epoch)
            log_partial("val_loss", logs['val_loss'])
            log_partial("accuracy", float(logs['accuracy']))


class LogsFlush(Callback):
    def on_epoch_end(self, epoch, logs=None):
        flush_logs()


data = {
    "data": "https://valohai-hospital-demo.s3-eu-west-1.amazonaws.com/preprocessed.csv",
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

# df_train = pd.read_csv("training_v2_processed_cat.csv")
# df_test = pd.read_csv("unlabeled_processed_cat.csv")

df_train, df_train_val = train_test_split(df_train, test_size=get_parameter("testsize"))

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

# test1 = df_test[fields_list + ['hospital_death', 'encounter_id']].copy()
# probstest = model.predict_proba(df_test[fields_list])
# probstest = probstest[:, 1]
# test1["hospital_death"] = probstest
# test1[["encounter_id", "hospital_death"]].to_csv(get_output_path("result.csv"), index=False)
