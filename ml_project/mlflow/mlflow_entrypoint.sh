#!/bin/sh

mlflow server --backend-store-uri ./logs/ --host 0.0.0.0 --port 5001