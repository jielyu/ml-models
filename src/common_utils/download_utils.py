# encoding: utf-8

import os
from pathlib import Path
import requests
import shutil
import zipfile


def __process_output_path__(output_path, zip_file):
    ml4a_downloads = "./downloads"
    if zip_file:
        folder, filename = output_path, "temp.zip"
    else:
        folder, filename = os.path.split(output_path)
    output_folder = os.path.join(ml4a_downloads, folder)
    output_filename = os.path.join(output_folder, filename)
    output_exists = os.path.exists(output_folder if zip_file else output_filename)
    return output_folder, output_filename, output_exists


def unzip(zip_file, output_folder, erase_zipfile=True):
    with zipfile.ZipFile(zip_file, "r") as zipref:
        zipref.extractall(output_folder)
        zipref.close()
    if erase_zipfile:
        os.remove(zip_file)


def download_data_file(url, output_path, zip_file=False, overwrite=False):
    output_folder, output_filename, output_exists = __process_output_path__(
        output_path, zip_file
    )
    if not output_exists or overwrite:
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        print("Downloading %s to %s" % (url, output_folder))
        with requests.get(url, stream=True) as r:
            with open(output_filename, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        if zip_file:
            unzip(output_filename, output_folder)
    output = output_folder if zip_file else output_filename
    return output


def main():
    url = "https://storage.googleapis.com/glow-demo/z_manipulate.npy"
    output_path = "./glow/z_manipulate.npy"
    p = download_data_file(url, output_path, zip_file=False)
    print(p)


if __name__ == "__main__":
    main()
