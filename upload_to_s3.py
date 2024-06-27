import boto3
import os

def upload_to_s3(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket"""
    # Read AWS credentials from environment variables
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION')

    # Debug prints
    print(f"AWS_ACCESS_KEY_ID: {aws_access_key_id}")
    print(f"AWS_SECRET_ACCESS_KEY: {aws_secret_access_key}")
    print(f"AWS_REGION: {aws_region}")
    print(f"Bucket: {bucket}")
    print(f"File name: {file_name}")
    print(f"Object name: {object_name or file_name}")

    s3_client = boto3.client(
        's3',
        region_name=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    
    try:
        response = s3_client.upload_file(file_name, bucket, object_name or file_name)
    except Exception as e:
        print(f"Error uploading {file_name}: {e}")
        return False
    return True

if __name__ == "__main__":
    processed_data_file = 'artifacts/data_ingestion/card_transdata.csv'
    model_file = 'artifacts/training/model.pkl'
    bucket_name = os.getenv('AWS_S3_BUCKET_NAME')  # Read bucket name from environment variable
    
    # Debug prints
    print(f"Bucket Name: {bucket_name}")

    # Upload files
    upload_to_s3(processed_data_file, bucket_name, os.getenv('AWS_S3_OBJECT_NAME_PROCESSED', 'processed_data.csv'))
    upload_to_s3(model_file, bucket_name, os.getenv('AWS_S3_OBJECT_NAME_MODEL', 'model.pkl'))
