import boto3
import os

def upload_to_s3(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket"""
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name or file_name)
    except Exception as e:
        print(f"Error uploading {file_name}: {e}")
        return False
    return True

if __name__ == "__main__":
    processed_data_file = 'artifacts/data_ingestion/card_transdata.csv'
    model_file = 'artifacts/training/model.pkl'
    bucket_name = 'mlops-dev-s3 '
    
    # Upload files
    upload_to_s3(processed_data_file, bucket_name, 'processed_data.csv')
    upload_to_s3(model_file, bucket_name, 'model.pkl')
