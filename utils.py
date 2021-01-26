import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def clip_column(df, key, lower, upper):
    df[key] = df[key].clip(lower=lower, upper=upper)


def normalize_column(df, key):
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    df[key] = scaler.fit_transform(df[[key]])


def preprocess(df, df_dictionary):
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.min_rows', 50)

    all_fields = {}
    removed = []
    system_fields = ['hospital_death', 'encounter_id']
    categorical_fields = ['apache_3j_diagnosis', 'gcs_verbal_apache', 'gcs_motor_apache', 'gcs_eyes_apache',
                          'ethnicity', 'gender', 'hospital_admit_source', 'icu_admit_source', 'icu_stay_type',
                          'icu_type', 'apache_3j_bodysystem', 'apache_2_bodysystem', 'apache_2_diagnosis']
    categorical_floats = ['apache_3j_diagnosis', 'gcs_verbal_apache', 'gcs_motor_apache', 'gcs_eyes_apache',
                          'apache_2_diagnosis']
    numeric_fields = ['bmi']

    for column in df.columns:
        if column not in list(df_dictionary['Variable Name']):
            print(f'{column} unknown field from df_train')

    for index, row in df_dictionary[['Variable Name', 'Data Type']].iterrows():
        if row['Variable Name'] in df:
            if row['Variable Name'] not in system_fields:
                if row['Data Type'] in ["numeric", "binary"] or row['Variable Name'] in numeric_fields:
                    all_fields[row['Variable Name']] = {'type': 'normalized'}
                elif row['Variable Name'] in categorical_fields:
                    all_fields[row['Variable Name']] = {'type': 'categorical'}
                else:
                    print("Drop unused", row['Variable Name'])
                    removed.append(row['Variable Name'])
        else:
            for df in [df]:
                if row['Variable Name'] in df:
                    print("Drop missing", row['Variable Name'])
                    removed.append(row['Variable Name'])

    for df in [df]:
        df.drop(columns=removed, inplace=True)

    print(len(df.columns))

    all_fields_original = all_fields.copy()
    for df in [df]:
        clip_column(df, "apache_4a_icu_death_prob", 0.0, 1.0)
        clip_column(df, "apache_4a_hospital_death_prob", 0.0, 1.0)

    use_onehot = False

    for field, info in all_fields_original.items():
        if info['type'] == "normalized":
            print("preprocess " + field + " normalized")
            for df in [df]:
                df[field + '_na'] = df.apply(lambda r: 0.0 if pd.isnull(r[field]) else 1.0, axis=1)
                df[field].fillna(df[field].mean(), inplace=True)
                normalize_column(df, field)
                all_fields[field + '_na'] = {'type': 'na'}
        if info['type'] == "categorical":
            if use_onehot:
                print("preprocess " + field + " categorical")
                all_onehot = set()
                for df in [df]:
                    for unique_val in list(df[field].unique()):
                        if unique_val not in all_fields:
                            key = f'onehot_{field}_{str(unique_val)}'
                            all_fields[key] = {'type': 'onehot'}
                            all_onehot.add(key)
                for df in [df]:
                    dummies_df = pd.get_dummies(df[field], prefix='onehot_' + field, dummy_na=True, dtype=np.float)
                    for dummy_field in dummies_df.columns:
                        all_fields[dummy_field] = {'type': 'onehot'}
                    df.drop(columns=[field], inplace=True)
                    df[dummies_df.columns] = dummies_df
                    for missing_field in list(filter(lambda f: f not in dummies_df.columns, all_onehot)):
                        print("missing", missing_field)
                        df[missing_field] = 1.0
                print(df[all_onehot])
            else:
                print("preprocess " + field + " categorical2")
                for df in [df]:
                    df[field].fillna(0, inplace=True)
                    if field in categorical_floats:
                        df[field] = df[field].astype(int)
                    print(df[field])

    print(df.dtypes)
    print("All fields", all_fields)
    print("Total features", len(all_fields))
    return df


def predict(df, model):
    fields_list = list(df.columns)
    fields_list.remove('encounter_id')

    probstest = model.predict_proba(df[fields_list])
    probstest = probstest[:, 1]
    df["hospital_death"] = probstest
    return df[["encounter_id", "hospital_death"]]
