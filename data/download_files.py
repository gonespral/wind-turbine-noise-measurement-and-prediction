# Downloads data from surfdrive and processed it in downloads/ directory

import os
import tqdm
import requests

urls = ["https://surfdrive.surf.nl/files/index.php/s/G9Xkz0wtZyhG6XZ/download?path=%2F&files=DU96-W-180.cor",
        "https://surfdrive.surf.nl/files/index.php/s/G9Xkz0wtZyhG6XZ/download?path=%2F&files=U08_Background.mat",
        "https://surfdrive.surf.nl/files/index.php/s/G9Xkz0wtZyhG6XZ/download?path=%2F&files=U08_Wind%20turbine.mat",
        "https://surfdrive.surf.nl/files/index.php/s/G9Xkz0wtZyhG6XZ/download?path=%2F&files=U09_Background.mat",
        "https://surfdrive.surf.nl/files/index.php/s/G9Xkz0wtZyhG6XZ/download?path=%2F&files=U09_Wind%20turbine.mat",
        "https://surfdrive.surf.nl/files/index.php/s/G9Xkz0wtZyhG6XZ/download?path=%2F&files=U10_Background.mat",
        "https://surfdrive.surf.nl/files/index.php/s/G9Xkz0wtZyhG6XZ/download?path=%2F&files=U10_Wind%20turbine.mat",
        "https://surfdrive.surf.nl/files/index.php/s/G9Xkz0wtZyhG6XZ/download?path=%2F&files=U11_Background.mat",
        "https://surfdrive.surf.nl/files/index.php/s/G9Xkz0wtZyhG6XZ/download?path=%2F&files=U11_Wind%20turbine.mat",
        "https://surfdrive.surf.nl/files/index.php/s/G9Xkz0wtZyhG6XZ/download?path=%2F&files=U12_Background.mat",
        "https://surfdrive.surf.nl/files/index.php/s/G9Xkz0wtZyhG6XZ/download?path=%2F&files=U12_Wind%20turbine.mat"]


def download_file(url, filename):
    r = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        for chunk in tqdm.tqdm(r.iter_content(chunk_size=1024), total=int(int(r.headers['Content-Length']) / 1024)):
            if chunk:
                f.write(chunk)
                f.flush()


def download_data():
    if not os.path.exists("downloads"):
        os.makedirs("downloads")
    for i, url in enumerate(urls):
        filename = url.split("=")[-1]
        # Check if file already exists
        if os.path.exists(os.path.join("downloads", filename)):
            print(f"File {filename} already exists, skipping")
            continue
        else:
            print(f"Downloading file {i} of {len(urls)}: {filename}")
            download_file(url, os.path.join("downloads", filename))
    print("Download complete")


if __name__ == "__main__":
    try:
        download_data()
    except KeyboardInterrupt:
        print("Download interrupted. Current file is likely incomplete. Delete it and re-run to continue.")
