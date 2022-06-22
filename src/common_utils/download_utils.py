# encoding: utf-8

import os
from pathlib import Path
import requests
import shutil
import zipfile
import gdown
import urllib


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


def download_text_file(url, output_path, overwrite=False):
    output_folder, output_filename, output_exists = __process_output_path__(
        output_path, False
    )
    if not output_exists or overwrite:
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        filedata = urllib.request.urlopen(url)
        with open(output_filename, "wb") as f:
            f.write(filedata.read())
    return output_filename


def download_from_gdrive(gdrive_fileid, output_path, zip_file=False, overwrite=False):
    output_folder, output_filename, output_exists = __process_output_path__(
        output_path, zip_file
    )
    if not output_exists or overwrite:
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        gdrive_url = "https://drive.google.com/uc?id=%s" % gdrive_fileid
        gdown.download(gdrive_url, output_filename, quiet=False)
        if zip_file:
            unzip(output_filename, output_folder)
    output = output_folder if zip_file else output_filename
    return output


def test_download_data_file():
    url = "https://storage.googleapis.com/glow-demo/z_manipulate.npy"
    output_path = "./glow/z_manipulate.npy"
    p = download_data_file(url, output_path, zip_file=False)
    print(p)


def test_download_basnet_model():
    gdrive_field = "1s52ek_4YTDRt_EOkx1FS53u-vJa0c4nu"
    output_path = "basnet/basnet.pth"
    p = download_from_gdrive(gdrive_field, output_path, zip_file=False)
    print(p)


def test_download_starnight_picture():
    gdrive_field = "1LLV7lyuCPOCDa5c7CkWjSfeckXBou1W7"
    output_path = "ml4a_basnet/starnight.jpg"
    p = download_from_gdrive(gdrive_field, output_path, zip_file=False)
    print(p)


def test_download_cartoonization_model():
    model_subfolder = "ml4a_cartoonization/White-box-Cartoonization/"

    download_data_file(
        "https://raw.githubusercontent.com/SystemErrorWang/White-box-Cartoonization/master/test_code/saved_models/model-33999.data-00000-of-00001",
        os.path.join(model_subfolder, "model-33999.data-00000-of-00001"),
    )
    download_data_file(
        "https://raw.githubusercontent.com/SystemErrorWang/White-box-Cartoonization/master/test_code/saved_models/model-33999.index",
        os.path.join(model_subfolder, "model-33999.index"),
    )
    download_text_file(
        "https://raw.githubusercontent.com/SystemErrorWang/White-box-Cartoonization/master/test_code/saved_models/checkpoint",
        os.path.join(model_subfolder, "checkpoint"),
    )


def test_download_photo_sketch_model():
    model_subfolder = "ml4a_photosketch/pretrained"
    download_from_gdrive(
        "1TQf-LyS8rRDDapdcTnEgWzYJllPgiXdj",
        model_subfolder,
        zip_file=True,
    )


def main():
    # test_download_data_file()
    # test_download_cartoonization_model()
    test_download_photo_sketch_model()


if __name__ == "__main__":
    main()
