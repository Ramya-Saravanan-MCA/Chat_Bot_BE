import boto3
import json
import os
from botocore.exceptions import ClientError

def get_secrets():
    secret_name = "dev_droog_mvp1"
    region_name = "ap-south-1"
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        secret = get_secret_value_response['SecretString']
        return json.loads(secret)
    except ClientError:
        print("Warning: Could not load secrets from AWS, falling back to environment/defaults.")
        return {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
            "GROQ_API_KEY": os.getenv("GROQ_API_KEY", ""),
            "LANCEDB_PATH": os.getenv("LANCEDB_PATH", "s3://droogbucket/lancedb"),
            "CHATDB_PATH": os.getenv("CHATDB_PATH", "s3://droogbucket/lancedb/chatdb"),
            "DATA_DIR": os.getenv("DATA_DIR", "s3://droogbucket/data"),
            "AWS_REGION": os.getenv("AWS_REGION", "ap-south-1"),
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", ""),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
        }

def get_secret_value(key: str) -> str:
    secrets = get_secrets()
    return secrets.get(key, "")

def load_secrets_into_env():
    secrets = get_secrets()
    for key, value in secrets.items():
        if value is not None:
            os.environ[key] = str(value)