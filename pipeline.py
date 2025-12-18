import os
import argparse
from google.cloud import aiplatform
from kfp import dsl
from kfp.v2 import compiler


@dsl.component(base_image="python:3.12")
def custom_training_job(
    project_id: str,
    model_dir: str,
):
    # 這裡只是 Placeholder，實際執行邏輯在 Docker Image 裡
    pass


@dsl.pipeline(name="penguin-training-pipeline")
def pipeline(
        project_id: str,
        bucket_name: str,  # 新增這個參數
        image_uri: str,
        serving_container_image_uri: str = "asia-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest"
):
    # 使用傳入的 bucket_name 變數
    pipeline_root = f"gs://{bucket_name}/pipeline_root"

    # 步驟 1: 訓練
    train_op = dsl.ContainerOp(
        name="train-model",
        image=image_uri, # 這會由 CI/CD 傳入最新的 Image TAG
        arguments=[
            "--project_id", project_id,
            "--model_dir", f"{pipeline_root}/model_output",
        ],
    ).set_caching_options(False) # 為了 Demo 方便，關閉快取

    # 步驟 2: 上傳模型 (記得這裡也要改成 pipeline_root)
    model_upload_op = aiplatform.ModelUploadOp(
        project=project_id,
        display_name="penguin-model",
        artifact_uri=f"{pipeline_root}/model_output",
        serving_container_image_uri=serving_container_image_uri,
    ).after(train_op)

    # 步驟 3: 自動部署到 Endpoint (關鍵需求 4)
    endpoint_create_op = aiplatform.EndpointCreateOp(
        project=project_id,
        display_name="penguin-endpoint",
    ).after(model_upload_op)

    aiplatform.ModelDeployOp(
        model=model_upload_op.outputs["model"],
        endpoint=endpoint_create_op.outputs["endpoint"],
        dedicated_resources_machine_type="n1-standard-2",
        dedicated_resources_min_replica_count=1,
        dedicated_resources_max_replica_count=1,
    )


if __name__ == "__main__":
    # 使用 argparse 讓 Python 腳本可以從外部接收參數
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=str, required=True)
    parser.add_argument('--bucket_name', type=str, required=True)
    args = parser.parse_args()

    # 編譯時將參數寫入 json 預設值
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="pipeline.json",
        pipeline_parameters={
            "project_id": args.project_id,
            "bucket_name": args.bucket_name
        }
    )