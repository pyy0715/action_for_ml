import os
import numpy as np
from dotenv import load_dotenv
from typing import List

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository

load_dotenv()

mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

client = MlflowClient(tracking_uri=mlflow_tracking_uri)
print("MLflow Version:", mlflow.__version__)
print("MLflow Tracking URI:", mlflow.get_tracking_uri())


def search_model(model_name: str):
    filter_string = "name='{}'".format(model_name)
    results = client.search_model_versions(filter_string)

    for res in results:
        print(
            "name={}; run_id={}; version={}; current_stage={}".format(
                res.name, res.run_id, res.version, res.current_stage
            )
        )


def load_model(model_name: str, model_version: int):
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    return model


def transition_model(model_name: str, model_version: int, stage: str):
    valid_stages = client.get_model_version_stages(model_name, model_version)
    if stage not in valid_stages:
        raise RuntimeError(f"Stage:{stage} not in valid stages")
    client.transition_model_version_stage(
        name=model_name, version=model_version, stage=stage
    )


# TODO: fix model uri
def download_model(model_name, deploy_version, download_path):
    filter_string = "name='{}'".format(model_name)
    results = client.search_model_versions(filter_string)
    for res in results:
        if res.current_stage == deploy_version:
            print(
                "Download name={}; run_id={}; version={}; current_stage={}".format(
                    res.name, res.run_id, res.version, res.current_stage
                )
            )
    model_uri = client.get_model_version_download_uri(res.name, res.version)
    # run_id, path = get_run_id_and_model_relative_path(model_uri)
    print(f"Model Download URI: {model_uri}")
    client.download_artifacts(res.run_id, model_uri, dst_path=download_path)


def delete_model(model_name: str, stage: List[str] = None):
    versions = client.get_latest_versions(model_name, stage)
    print(f"Deleting {len(versions)} latest versions for model '{model_name}'")
    for v in versions:
        print(
            f"  latest version={v.version} status={v.status} stage={v.current_stage} run_id={v.run_id}"
        )
        client.transition_model_version_stage(model_name, v.version, "Archived")
        client.delete_model_version(model_name, v.version)
