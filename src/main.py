import mlflow
import argparse
import os
def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:9283"))
    with mlflow.start_run() as run:
        mlflow.run(".", "stage_01", use_conda = False)
        mlflow.run(".", "stage_02", use_conda = False)
        mlflow.run(".", "stage_03", use_conda = False)
        mlflow.run(".", "stage_04", use_conda = False)


if __name__ == "__main__":
    main()