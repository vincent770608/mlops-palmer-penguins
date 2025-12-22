import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from google.cloud import bigquery
from google.cloud import aiplatform
import time

# 定義特徵的配置
NUMERICAL_FEATURES = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
CATEGORICAL_FEATURES = ['island']
# Label 不需要放進 Input，但需要做 One-Hot
LABEL_COLUMN = 'species'


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    """
    修正版：將 Pandas DF 轉換為 tf.data.Dataset
    1. 強制轉換數值為 float32
    2. 強制將 shape 從 (N,) 轉為 (N, 1) 以符合 Keras Input(shape=(1,))
    """
    df = dataframe.copy()
    labels = df.pop(LABEL_COLUMN)

    # Label One-Hot (保持不變)
    labels = pd.get_dummies(labels, prefix=LABEL_COLUMN)

    # --- 🔥 關鍵修正開始 🔥 ---
    data_dict = {}

    # 遍歷所有特徵欄位，手動調整形狀與型別
    for name, value in df.items():
        # 取出 numpy array
        val = value.values

        if name in NUMERICAL_FEATURES:
            # 數值特徵：轉 float32 並增加一個維度
            # 例如: [0.1, 0.5] -> [[0.1], [0.5]]
            val = val.astype('float32')[:, np.newaxis]
        else:
            # 字串特徵：雖然不用轉 float，但也要增加維度
            val = val[:, np.newaxis]

        data_dict[name] = val
    # --- 🔥 關鍵修正結束 🔥 ---

    # 這裡傳入處理好的 data_dict
    ds = tf.data.Dataset.from_tensor_slices((data_dict, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


def train_model(project_id, model_dir, bucket_name):
    # --- 1. 從 BigQuery 讀取資料 ---
    print("Loading data from BigQuery...")
    client = bigquery.Client(project=project_id)
    query = """
        SELECT species, island, culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g
        FROM `bigquery-public-data.ml_datasets.penguins`
        WHERE body_mass_g IS NOT NULL
    """
    df = client.query(query).to_dataframe()
    df.dropna(inplace=True)

    # 切分訓練與驗證集
    train_df = df.sample(frac=0.8, random_state=0)
    test_df = df.drop(train_df.index)

    # 轉為 tf.data.Dataset
    batch_size = 32
    train_ds = df_to_dataset(train_df, batch_size=batch_size)
    test_ds = df_to_dataset(test_df, shuffle=False, batch_size=batch_size)

    # --- 2. 建立模型 (包含前處理) ---
    all_inputs = {}
    encoded_features = []

    # A. 處理數值特徵
    for header in NUMERICAL_FEATURES:
        numeric_col = tf.keras.Input(shape=(1,), name=header, dtype="float32")
        normalization_layer = tf.keras.layers.Normalization()

        # 🔥 修改：確保 adapt 用的資料也是 (N, 1) 且 float32
        adapt_data = train_df[header].values.astype('float32')[:, np.newaxis]
        normalization_layer.adapt(adapt_data)

        encoded_numeric_col = normalization_layer(numeric_col)
        all_inputs[header] = numeric_col
        encoded_features.append(encoded_numeric_col)

    # B. 處理類別特徵
    for header in CATEGORICAL_FEATURES:
        cat_col = tf.keras.Input(shape=(1,), name=header, dtype="string")
        lookup_layer = tf.keras.layers.StringLookup(output_mode="one_hot")

        # 🔥 修改：確保 adapt 用的資料也是 (N, 1)
        adapt_data = train_df[header].values[:, np.newaxis]
        lookup_layer.adapt(adapt_data)

        encoded_cat_col = lookup_layer(cat_col)
        all_inputs[header] = cat_col
        encoded_features.append(encoded_cat_col)

    # --- 組合模型 (Functional API) ---
    all_features = tf.keras.layers.concatenate(encoded_features)

    x = tf.keras.layers.Dense(32, activation="relu")(all_features)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    output = tf.keras.layers.Dense(3, activation="softmax")(x)

    # 建立模型，指定 Inputs (字典) 和 Outputs
    model = tf.keras.Model(inputs=all_inputs, outputs=output)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # --- 3. 訓練 ---
    model.fit(train_ds, epochs=10, validation_data=test_ds)

    # --- 4. 評估與紀錄 ---
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test Accuracy: {accuracy}")

    # Vertex AI Logging (省略部分重複代碼...)
    timestamp = int(time.time())
    run_id = f"penguin-run-{timestamp}"
    aiplatform.init(project=project_id, experiment='penguin-experiment', location='asia-east1',
                    staging_bucket=f'gs://{bucket_name.replace("gs://", "")}')
    aiplatform.start_run(run=run_id)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=str, required=True)
    parser.add_argument('--model_dir', type=str, default=os.environ.get('AIP_MODEL_DIR'))
    parser.add_argument('--bucket_name', type=str)
    args = parser.parse_args()

    train_model(args.project_id, args.model_dir, args.bucket_name)