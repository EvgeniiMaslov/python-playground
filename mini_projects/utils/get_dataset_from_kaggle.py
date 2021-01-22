from argparse import ArgumentParser
import os
import shutil
from zipfile import ZipFile
from kaggle.api.kaggle_api_extended import KaggleApi

parser = ArgumentParser()
parser.add_argument('comp_name', type=str, help="Competition name")
args = parser.parse_args()

api = KaggleApi()
api.authenticate()

api.competition_download_files(args.comp_name)

filename = args.comp_name + ".zip"
os.mkdir("data")
arch_path = os.path.join("data", filename)

shutil.move(filename, arch_path)

file = ZipFile(arch_path)
file.extractall("data")