FROM continuumio/miniconda3

RUN pip install mlflow boto3 pymysql

COPY wait-for-it.sh wait-for-it.sh 
RUN chmod +x wait-for-it.sh