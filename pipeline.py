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

# 預先組好路徑
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_root"
MODEL_DIR = f"{PIPELINE_ROOT}/model_output"


@dsl.container_component
def custom_training_job(
    project_id: str,
    model_dir: str,
    bucket_name: str
):
    return dsl.ContainerSpec(
        image=TRAINING_IMAGE_URI,
        args=[
            '--project_id', project_id,
            '--model_dir', model_dir,
            '--bucket_name', bucket_name
        ]
    )


@dsl.pipeline(name="penguin-training-pipeline")
def pipeline(
        project_id: str,
        serving_container_image_uri: str = "asia-east1-docker.pkg.dev/vincent-sandbox-470814/mlops-palmer-penguins/tf-serving:2.16"
):
    # 使用傳入的 bucket_name 變數
    # pipeline_root = f"gs://{bucket_name}/pipeline_root"
    # model_dir = f"{pipeline_root}/model_output"
    location = "asia-east1"
    # 步驟 1: 訓練
    # 實例化 component
    train_task = custom_training_job(
        project_id=project_id,
        model_dir=MODEL_DIR,
        bucket_name=BUCKET_NAME
    )
    # 這裡覆寫 image，使用 CI/CD 傳進來的 image_uri
    # train_task.image = image_uri
    # 關閉快取 (Demo用)
    train_task.set_caching_options(False)

    # Step 1.5: 將路徑轉為 Artifact (關鍵修正)
    import_unmanaged_model_task = importer(
        artifact_uri=MODEL_DIR,
        artifact_class=artifact_types.UnmanagedContainerModel,
        reimport=False,
        metadata={
            "containerSpec": {
                "imageUri": serving_container_image_uri
            }
        }
    ).after(train_task)

    # 步驟 2: 上傳模型
    model_upload_op = ModelUploadOp(
        project=project_id,
        display_name="penguin-model",
        location=location,
        # 使用 importer 的輸出
        unmanaged_container_model=import_unmanaged_model_task.output,
    ).after(import_unmanaged_model_task)

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
        dedicated_resources_machine_type="e2-standard-4",
        dedicated_resources_min_replica_count=1,
        dedicated_resources_max_replica_count=1,
        service_account="mlops-144@vincent-sandbox-470814.iam.gserviceaccount.com"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=str, required=True)
    args = parser.parse_args()

    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="pipeline.json",
        pipeline_parameters={
            "project_id": args.project_id,
        }
    )