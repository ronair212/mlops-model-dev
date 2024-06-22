import os
import urllib.request as request
import zipfile
from XGBClassifier import logger
from XGBClassifier.utils.common import get_size
from XGBClassifier.entity.config_entity import DataIngestionConfig
from pathlib import Path
import gdown

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            
            file_id = self.config.source_URL.split("/")[-2]
            prefix = "https://drive.google.com/uc?/export=download&id="
            gdown.download(prefix+file_id, self.config.local_data_file)
            
            filename = prefix+file_id
            
            logger.info(f" file from url {filename} downloaded and saved at path {self.config.local_data_file}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")  


    
    def extract_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory if it is a zip file.
        If not a zip file, prints the file type and file name.
        Function returns None
        """
        file_path = self.config.local_data_file
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        
        if zipfile.is_zipfile(file_path):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            logger.info(f" Extracted zip file: {file_path}")
        else:
            file_extension = os.path.splitext(file_path)[1]
            logger.info(f" File '{file_path}' is of type '{file_extension}'")

