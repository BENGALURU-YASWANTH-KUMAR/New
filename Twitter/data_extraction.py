import os
import pandas as pd

def download_and_extract_data():
    # Ensure the Kaggle API is configured
    kaggle_config_dir = os.path.expanduser('~/.kaggle')
    os.makedirs(kaggle_config_dir, exist_ok=True)
    
    # Copy the kaggle.json file to the Kaggle config directory
    !cp kaggle.json ~/.kaggle/kaggle.json
    !chmod 600 ~/.kaggle/kaggle.json
    
    # Download the dataset using the Kaggle API
    !kaggle datasets download -d kazanova/sentiment140
    
    # Unzip the downloaded dataset
    !unzip -o sentiment140.zip
    
    # Load the dataset into a pandas DataFrame
    column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
    df = pd.read_csv('training.1600000.processed.noemoticon.csv', names=column_names, encoding='latin-1')
    return df

if __name__ == "__main__":
    df = download_and_extract_data()
    print(df.head())
