import requests
import os
import shutil
from zipfile import ZipFile
from kaggle.api.kaggle_api_extended import KaggleApi


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    if not os.path.isdir('data'):
        os.mkdir('data')
    destination = os.path.join('data', destination)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def get_dataset_from_kaggle(comp_name):
    api = KaggleApi()
    api.authenticate()

    api.competition_download_files(comp_name)

    filename = comp_name + ".zip"
    if not os.path.isdir('data'):
        os.mkdir("data")
    arch_path = os.path.join("data", filename)

    shutil.move(filename, arch_path)

    file = ZipFile(arch_path)
    file.extractall("data")

    return 'Complete'