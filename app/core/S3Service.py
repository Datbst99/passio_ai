import os
from pathlib import Path

import boto3

from config.core import Core


class S3Service:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=Core.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Core.AWS_SECRET_ACCESS_KEY,
            region_name=Core.AWS_DEFAULT_REGION,
        )
        self.bucket_name = Core.AWS_BUCKET
        self.public_url = Core.AWS_URL + "/audio_ai"

    def upload(self, file_path, file_name = None):
        try:
            if file_name is None:
                file_name = os.path.basename(file_path)
            self.s3_client.upload_file(file_path, self.bucket_name, file_name)
            file_url = f"{self.public_url}/{file_name}"
            delete_file = Path(file_path)
            if delete_file.exists():
                delete_file.unlink()

            return file_url
        except Exception as e:
            print(e)
            return False
