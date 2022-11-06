FROM python:3.9
RUN pip3 install mlflow==1.22.0
RUN pip3 install protobuf==3.20.*
COPY mlflow_entrypoint.sh mlflow_entrypoint.sh

CMD ["mlflow", "server", "--backend-store-uri",  "sqlite:///mlflow.db", "--default-artifact-root", "/logs/", "--host", "0.0.0.0", "--port", "5001"]