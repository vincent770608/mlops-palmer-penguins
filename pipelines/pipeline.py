import argparse
import os
from kfp import dsl
from kfp import compiler
# 引入 Importer 與 Model 定義
from kfp.dsl import importer
from google_cloud_pipeline_components.types import artifact_types
# Google 官方組件庫
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp

# 如果沒讀到 (例如本地測試)，就用 placeholder
# 這樣可以確保路徑是靜態字串，避免 KFP 執行時解析錯誤
TRAINING_IMAGE_URI = os.environ.get("TRAINING_IMAGE_URI", "placeholder")
BUCKET_NAME = os.environ.get("BUCKET_NAME", "placeholder_bucket")
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_root"


@dsl.container_component
def custom_training_job(
    project_id: str,
    bucket_name: str,
    model_dir_str: str
):
    return dsl.ContainerSpec(
        image=TRAINING_IMAGE_URI,
        args=[
            '--project_id', project_id,
            '--bucket_name', bucket_name,
            # KFP 會自動生成一個路徑填入這裡，例如 gs://bucket/.../model_output
            '--model_dir', model_dir_str
        ]
    )


# 定義一個組件專門做資料快照
@dsl.component(packages_to_install=["google-cloud-bigquery", "pandas", "pyarrow"])
def get_data_snapshot(
        project_id: str,
        query: str,
        dataset_output: dsl.Output[dsl.Dataset]  # 這裡是關鍵，KFP 會自動追蹤這個 Dataset 物件
):
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    df = client.query(query).to_dataframe()

    # 版控:將當下的資料存成檔案
    df.to_csv(dataset_output.path, index=False)

    # 紀錄 Metadata：例如你是撈哪個 Table、當下的時間點
    dataset_output.metadata["source_query"] = query
    dataset_output.metadata["row_count"] = len(df)

# Step 1.5: 新增一個「配置組件」 (關鍵救星！)
# 它的工作很簡單：拿到模型 -> 加上啟動參數 Metadata -> 輸出模型
@dsl.component(base_image="python:3.12",
               packages_to_install=["google-cloud-pipeline-components==2.22.0"])
def configure_serving_metadata(
        model_uri: str,
        ready_model: dsl.Output[artifact_types.UnmanagedContainerModel],
        serving_image_uri: str
):
    # 1. 直接複製路徑 (指向同一個 GCS 位置，不需要移動檔案)
    ready_model.uri = model_uri

    # 2. 注入 Serving 設定 (這就是我們之前想塞給 Importer 的東西)
    # Vertex AI 會讀取這個 metadata 來知道如何啟動容器
    ready_model.metadata = {
        "containerSpec": {
            "imageUri": serving_image_uri,
            "command": ["/usr/bin/tensorflow_model_server"],
            "args": [
                "--port=8500",
                "--rest_api_port=8080",
                "--model_name=default",
                "--model_base_path=$(AIP_STORAGE_URI)"  # 這裡保持原樣，Vertex 會執行時替換
            ],
            "predictRoute": "/v1/models/default:predict",
            "healthRoute": "/v1/models/default"
        }
    }


@dsl.pipeline(name="penguin-training-pipeline")
def pipeline(
        project_id: str,
        run_id: str,
        serving_container_image_uri: str = "asia-east1-docker.pkg.dev/vincent-sandbox-470814/mlops-palmer-penguins/tf-serving:2.16"
):
    # 動態組建路徑
    # 這是 Importer 要看的地方 (父目錄)
    # 例如: gs://.../model_output/commit-a1b2c3d
    model_base_dir = f"{PIPELINE_ROOT}/model_output/{run_id}"

    # 這是 Train 要寫入的地方 (子目錄，永遠叫 1)
    # 例如: gs://.../model_output/commit-a1b2c3d/1
    model_version_dir = f"{model_base_dir}/1"

    # 使用傳入的 bucket_name 變數
    # pipeline_root = f"gs://{bucket_name}/pipeline_root"
    # model_dir = f"{pipeline_root}/model_output"
    location = "asia-east1"
    model_display_name = "penguin-model"

    # 步驟 1: 訓練
    # 實例化 component
    train_task = custom_training_job(
        project_id=project_id,
        bucket_name=BUCKET_NAME,
        model_dir_str=model_version_dir  # 告訴 train.py 寫到這裡
    )
    train_task.set_caching_options(False)

    # Step 1.5: 將路徑轉為 Artifact
    # 我們用這個小組件取代了 Importer
    configure_task = configure_serving_metadata(
        model_uri=model_base_dir,
        serving_image_uri=serving_container_image_uri
    )
    # 重要：因為沒有資料流依賴 (Data Dependency)，我們必須手動加順序
    # 確保訓練完之後，才去包裝模型
    configure_task.after(train_task)

    # Step 2: 上傳
    model_upload_op = ModelUploadOp(
        project=project_id,
        display_name=model_display_name,
        location=location,
        unmanaged_container_model=configure_task.outputs["ready_model"]
    ).after(configure_task)

    # 步驟 3: 建立 Endpoint
    endpoint_create_op = EndpointCreateOp(
        project=project_id,
        display_name="penguin-endpoint",
        location=location
    ).after(model_upload_op)

    # 步驟 4: 部署模型
    ModelDeployOp(
        model=model_upload_op.outputs["model"],
        endpoint=endpoint_create_op.outputs["endpoint"],
        dedicated_resources_machine_type="e2-medium",
        dedicated_resources_min_replica_count=0,
        dedicated_resources_max_replica_count=1
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=str, required=True)
    args = parser.parse_args()

    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="pipeline.json",
        pipeline_parameters={
            "project_id": args.project_id
        }
    )