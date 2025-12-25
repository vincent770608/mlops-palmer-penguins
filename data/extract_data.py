import argparse
from google.cloud import bigquery
from google.cloud import aiplatform
import pandas as pd
import time


def extract_and_version_data(project_id, bucket_name, sql_file_path):
    # 1. 讀取 SQL 檔案 (Code Versioning)
    with open(sql_file_path, 'r') as file:
        query = file.read()
    print(f"Executing query from {sql_file_path}...")

    # 2. 執行 Query (Extraction)
    client = bigquery.Client(project=project_id)
    df = client.query(query).to_dataframe()

    # 3. 儲存到 GCS (Data Versioning)
    # 使用 timestamp 或 semantic version 作為資料夾名稱
    version = time.strftime("%Y%m%d_%H%M%S")
    gcs_path = f"gs://{bucket_name}/penguin_data/{version}/data.csv"

    df.to_csv(gcs_path, index=False)
    print(f"Data saved to {gcs_path}")

    # 4.紀錄 Metadata (Lineage)
    # 這樣你未來看到這個 CSV，就會知道它是用哪個 SQL 跑出來的
    aiplatform.init(project=project_id, location='asia-east1')

    # 建立一個 Dataset Artifact
    dataset_artifact = aiplatform.Artifact.create(
        schema_title="system.Dataset",
        display_name=f"penguin-dataset-{version}",
        uri=gcs_path,
        metadata={
            "sql_source": query,  # 直接把 SQL 內文存進去
            "sql_file_name": sql_file_path,
            "row_count": len(df),
            "generated_at": version
        }
    )

    print(f"Metadata recorded. Artifact Resource Name: {dataset_artifact.resource_name}")
    return gcs_path


if __name__ == "__main__":
    # 你可以手動執行這個腳本：
    # python src/extract_data.py --sql_file sql/extract_penguins_v1.sql ...
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=str, required=True)
    parser.add_argument('--bucket_name', type=str, required=True)
    parser.add_argument('--sql_file', type=str, default='sql/extract_penguins_v1.sql')
    args = parser.parse_args()

    extract_and_version_data(args.project_id, args.bucket_name, args.sql_file)