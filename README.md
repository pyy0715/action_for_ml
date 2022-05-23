[![GitHub Super-Linter](https://github.com/pyy0715/action_for_ml/workflows/Lint%20Code%20Base/badge.svg)](https://github.com/marketplace/actions/super-linter)

# Mlflow

![image](https://www.mlflow.org/docs/latest/_images/scenario_4.png)

To record all runs’ MLflow entities, the MLflow client interacts with the tracking server via a series of REST requests:

- Part 1a and b:

    - The MLflow client creates an instance of a RestStore and sends REST API requests to log MLflow entities

    - The Tracking Server creates an instance of an SQLAlchemyStore and connects to the remote host to insert MLflow entities in the database

For artifact logging, the MLflow client interacts with the remote Tracking Server and artifact storage host:

- Part 2a, b, and c:

    - The MLflow client uses RestStore to send a REST request to fetch the artifact store URI location from the Tracking Server

    - The Tracking Server responds with an artifact store URI location (an S3 storage URI in this case)

    - The MLflow client creates an instance of an S3ArtifactRepository, connects to the remote AWS host using the boto client libraries, and uploads the artifacts to the S3 bucket URI location

# Bentoml