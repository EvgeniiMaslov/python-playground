import requests
import shutil
import tarfile
import os

url = "http://opihi.cs.uvic.ca/sound/genres.tar.gz"
r = requests.get(url, allow_redirects=True)


path = "data"
os.mkdir(path)
filename = "genres.tar.gz"
new_path = os.path.join(path, filename)

shutil.move(filename, new_path)
file = tarfile.open(new_path)
file.extractall(path)