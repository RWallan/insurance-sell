FROM python:3.13-slim

RUN apt-get update && apt-get install -y curl
RUN pip install mlflow[extras] psycopg2-binary boto3 cryptography pymysql

EXPOSE 5000
