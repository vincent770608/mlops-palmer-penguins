import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from google.cloud import aiplatform
import time

# 定義特徵的配置
NUMERICAL_FEATURES = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
CATEGORICAL_FEATURES = ['island']
# Label 不需要放進 Input，但需要做 One-Hot
LABEL_COLUMN = 'species'

CONFIG = {
    "PARAMS": {
        "NN_LAYERS": [
            {"nodes": 32, "active_fun": "relu"},
            {"nodes": 32, "active_fun": "relu"},
            {"nodes": 3, "active_fun": "softmax"} # 輸出層
        ],
        "TRAIN_SPLIT_FRAC": 0.8,
        "LEARNING_RATE": 0.1,
        "EPOCHS": 10,
        "BATCH_SIZE": 32,
        "OPTIMIZER": "adam",
        "LOSS_FUNC": "categorical_crossentropy"
    },
    "DATA_PATH": "penguin_data/20251225_195439/data.csv",
    "BUCKET_NAME": "vincent-sandbox-mlops-palmer-penguins"
}


def df_to_dataset(dataframe, shuffle, batch_size):
    """
    將 Pandas DF 轉換為 tf.data.Dataset
    """
    df = dataframe.copy()
    labels = df.pop(LABEL_COLUMN)
    # Label 轉 One-Hot
    labels = pd.get_dummies(labels, prefix=LABEL_COLUMN)

    data_dict = {}
    for name, value in df.items():
        val = value.values
        if name in NUMERICAL_FEATURES:
            val = val.astype('float32')[:, np.newaxis]
        else:
            val = val[:, np.newaxis]
        data_dict[name] = val

    ds = tf.data.Dataset.from_tensor_slices((data_dict, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


def train_model(project_id, bucket_name, data_path, model_dir):
    # # --- 1. 從 BigQuery 讀取資料 ---
    # print("Loading data from BigQuery...")
    # client = bigquery.Client(project=project_id)
    # query = """
    #     SELECT species, island, culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g
    #     FROM `bigquery-public-data.ml_datasets.penguins`
    #     WHERE body_mass_g IS NOT NULL
    # """
    # df = client.query(query).to_dataframe()
    # df.dropna(inplace=True)

    # --- 1. 從 GCS 讀取資料 ---
    params = CONFIG["PARAMS"]
    print(f"Loading data from hardcoded path: {data_path}")
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
    except Exception as e:
        print(f"Error reading data: {e}")
        raise
    df.dropna(inplace=True)

    # 切分訓練與驗證集
    train_df = df.sample(frac=params["TRAIN_SPLIT_FRAC"], random_state=0)
    test_df = df.drop(train_df.index)

    # 轉為 tf.data.Dataset
    batch_size = params["BATCH_SIZE"]
    train_ds = df_to_dataset(train_df, shuffle=True, batch_size=batch_size)
    test_ds = df_to_dataset(test_df, shuffle=False, batch_size=batch_size)

    # --- 2. 建立模型 (包含前處理) ---
    all_inputs = {}
    encoded_features = []

    # A. 處理數值特徵
    for header in NUMERICAL_FEATURES:
        numeric_col = tf.keras.Input(shape=(1,), name=header, dtype="float32")
        normalization_layer = tf.keras.layers.Normalization()

        # 確保 adapt 用的資料也是 (N, 1) 且 float32
        adapt_data = train_df[header].values.astype('float32')[:, np.newaxis]
        normalization_layer.adapt(adapt_data)

        encoded_numeric_col = normalization_layer(numeric_col)
        all_inputs[header] = numeric_col
        encoded_features.append(encoded_numeric_col)

    # B. 處理類別特徵
    for header in CATEGORICAL_FEATURES:
        cat_col = tf.keras.Input(shape=(1,), name=header, dtype="string")
        lookup_layer = tf.keras.layers.StringLookup(output_mode="one_hot")

        # 確保 adapt 用的資料也是 (N, 1)
        adapt_data = train_df[header].values[:, np.newaxis]
        lookup_layer.adapt(adapt_data)

        encoded_cat_col = lookup_layer(cat_col)
        all_inputs[header] = cat_col
        encoded_features.append(encoded_cat_col)

    # --- 組合模型 (Functional API) ---
    all_features = tf.keras.layers.concatenate(encoded_features)

    x = tf.keras.layers.Dense(params["NN_LAYERS"][0]["nodes"], activation=params["NN_LAYERS"][0]["active_fun"])(all_features)
    x = tf.keras.layers.Dense(params["NN_LAYERS"][1]["nodes"], activation=params["NN_LAYERS"][1]["active_fun"])(x)
    output = tf.keras.layers.Dense(params["NN_LAYERS"][2]["nodes"], activation=params["NN_LAYERS"][2]["active_fun"])(x)

    # 建立模型，指定 Inputs (字典) 和 Outputs
    model = tf.keras.Model(inputs=all_inputs, outputs=output)

    # 設定 Optimizer
    opt = params["OPTIMIZER"]
    if "LEARNING_RATE" in params:
        if params["OPTIMIZER"] == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=params["LEARNING_RATE"])

    model.compile(optimizer=opt,
                  loss=params["LOSS_FUNC"],
                  metrics=['categorical_accuracy'])

    # --- 3. 訓練 ---
    model.fit(train_ds, epochs=params["EPOCHS"], validation_data=test_ds)

    # --- 4. 評估與紀錄 ---
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test Accuracy: {accuracy}")

    # Vertex AI Logging
    timestamp = int(time.time())
    run_id = f"penguin-run-{timestamp}"
    aiplatform.init(project=project_id, experiment='penguin-experiment', location='asia-east1',
                    staging_bucket=f'gs://{bucket_name.replace("gs://", "")}')
    aiplatform.start_run(run=run_id)

    params_to_log = {"DATA_PATH": data_path}
    for key, value in params.items():
        # 如果是 list 或 dict，就轉成字串
        if isinstance(value, (list, dict)):
            params_to_log[key] = str(value)
        else:
            params_to_log[key] = value

    aiplatform.log_params(params_to_log)
    aiplatform.log_metrics({"accuracy": accuracy, "loss": loss})
    aiplatform.end_run()

    # --- 5. 儲存模型 ---
    print(f"Saving model to {model_dir}")
    # export 會保存包含 StringLookup 和 Normalization 的完整模型
    try:
        model.export(model_dir)
    except AttributeError:
        # 如果環境不小心退回 Keras 2 (TF < 2.16)，export 不存在，改用 save
        tf.saved_model.save(model, model_dir)


if __name__ == '__main__':
    data_path = f"gs://{CONFIG["BUCKET_NAME"].replace("gs://", "")}/{CONFIG["DATA_PATH"]}"

    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=str, required=True)
    parser.add_argument('--model_dir', type=str, default=os.environ.get('AIP_MODEL_DIR'))
    parser.add_argument('--data_path', type=str, default=data_path)
    parser.add_argument('--bucket_name', type=str, default=CONFIG["BUCKET_NAME"])

    args = parser.parse_args()

    train_model(args.project_id, args.bucket_name, args.data_path, args.model_dir)