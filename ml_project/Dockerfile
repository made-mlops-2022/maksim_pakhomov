FROM python:3.9

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY dist/ml-0.1.0.tar.gz /ml-0.1.0.tar.gz
RUN pip install /ml-0.1.0.tar.gz

COPY configs/ /configs
WORKDIR .
