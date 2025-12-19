import os
import argparse
import pandas as pd
import tensorflow as tf
from google.cloud import bigquery
from google.cloud import aiplatform
import time


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

    # --- [修改重點 1] 使用 One-Hot Encoding ---
    # 對 Feature 'island' 做 One-Hot
    # 原本: island 一個欄位 (0, 1, 2)
    # 現在: 會變成三個欄位 (island_Biscoe, island_Dream, island_Torgersen)
    df = pd.get_dummies(df, columns=['island'], prefix='island')

    # 對 Target 'species' 做 One-Hot
    # 原本: species 一個欄位 (0, 1, 2)
    # 現在: 變成三個欄位 (species_Adelie Penguin (Pygoscelis adeliae), ...)
    target_col_prefix = 'species'
    df = pd.get_dummies(df, columns=['species'], prefix=target_col_prefix)

    # 分離特徵與標籤
    # 找出所有開頭是 'species_' 的欄位當作 Label
    label_cols = [c for c in df.columns if c.startswith(target_col_prefix)]

    train_dataset = df.sample(frac=0.8, random_state=0)
    test_dataset = df.drop(train_dataset.index)

    # 取出 Label (y) 和 Features (x)
    train_labels = train_dataset[label_cols]
    test_labels = test_dataset[label_cols]

    train_features = train_dataset.drop(columns=label_cols)
    test_features = test_dataset.drop(columns=label_cols)

    print(f"Training features shape: {train_features.shape}")
    print(f"Training labels shape: {train_labels.shape}")

    # --- 2. 建立並訓練模型 ---
    print("Training model...")
    model = tf.keras.Sequential([
        # input_shape 會自動抓取 Feature 數量 (現在變多了，因為 island 拆成了3個欄位)
        tf.keras.layers.Dense(10, activation='relu', input_shape=[len(train_features.keys())]),
        tf.keras.layers.Dense(10, activation='relu'),
        # Output 層維持 3 (因為有 3 種企鵝)
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    # --- [修改重點 2] Loss Function 變更 ---
    # 因為 Label 已經是 One-Hot 格式 (例如 [1, 0, 0])，所以用 categorical_crossentropy
    # 如果 Label 是整數 (例如 0)，才用 sparse_categorical_crossentropy
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_features, train_labels, epochs=10, verbose=1)

    # --- 3. 評估與紀錄 Metadata ---
    loss, accuracy = model.evaluate(test_features, test_labels, verbose=0)
    print(f"Test Accuracy: {accuracy}")

    # 1. 產生一個不會重複的 Run ID (加上時間戳記)
    timestamp = int(time.time())
    run_id = f"penguin-run-{timestamp}"

    aiplatform.init(project=project_id,
                    experiment='penguin-experiment',  # <--- 加上這行，指定實驗名稱
                    location='asia-east1',
                    staging_bucket=f'gs://{bucket_name}'  # (選填) 建議加上你的 bucket
                    )

    # [選填] 啟動一個 Run (回合)。
    # 如果不寫 start_run，直接 log_metrics，Vertex AI SDK 通常會自動幫你產生一個隨機名稱的 Run
    # 為了方便辨識，我們可以手動給一個前綴
    aiplatform.start_run(run=run_id)

    aiplatform.log_metrics({"accuracy": accuracy, "loss": loss})
    # 結束 Run
    aiplatform.end_run()

    # --- 4. 儲存模型 ---
    # 改用 tf.saved_model.save 來確保輸出格式是 SavedModel (資料夾)，
    # 這樣無論在本地(Keras 3)還是雲端(Keras 2)都兼容，且符合 Vertex AI 需求。
    try:
        model.export(model_dir)
    except AttributeError:
        # 如果環境不小心退回 Keras 2 (TF < 2.16)，export 不存在，改用 save
        tf.saved_model.save(model, model_dir)
    print(f"Model saved to {model_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=str, required=True)
    parser.add_argument('--model_dir', type=str, default=os.environ.get('AIP_MODEL_DIR'))
    parser.add_argument('--bucket_name', type=str, required=True)
    args = parser.parse_args()

    train_model(args.project_id, args.model_dir, args.bucket_name)