#!/usr/bin/env python
"""
Test regression model W&B artifact
"""
import argparse
import logging
import wandb
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="test_regression_model")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################
    logger.info("Downloading artifacts")
    model_local_path = run.use_artifact(args.mlflow_model).download()
    test_local_path = run.use_artifact(args.test_dataset).file()

    X_test = pd.read_csv(test_local_path)
    y_test = X_test.pop("price")

    logger.info("Loading model and performing inference on test set")
    sk_pipe = mlflow.sklearn.load_model(model_local_path)
    y_pred = sk_pipe.predict(X_test)

    logger.info("Scoring")
    r_squared = sk_pipe.score(X_test, y_test)

    mae = mean_absolute_error(y_test, y_pred)

    logger.info(f"Score: {r_squared}")
    logger.info(f"MAE: {mae}")

    run.summary["r2"] = r_squared
    run.summary["mae"] = mae


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test regression model")


    parser.add_argument(
        "--mlflow_model", 
        type=str,
        help="An MLflow serialized model",
        required=True
    )

    parser.add_argument(
        "--test_dataset", 
        type=str,
        help="The test artifact",
        required=True
    )


    args = parser.parse_args()

    go(args)
