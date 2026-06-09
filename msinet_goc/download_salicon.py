"""Tải SALICON dataset từ Google Drive."""
import os, zipfile
import gdown

def download_salicon(data_path):
    default_path   = os.path.join(data_path, "salicon")
    stimuli_train  = os.path.join(default_path, "stimuli",  "train")
    saliency_train = os.path.join(default_path, "saliency", "train")

    if os.path.exists(stimuli_train) and len(os.listdir(stimuli_train)) > 0:
        print("✅ SALICON đã tồn tại.")
        return default_path

    for d in [stimuli_train,
              os.path.join(default_path, "stimuli",  "val"),
              saliency_train,
              os.path.join(default_path, "saliency", "val"),
              os.path.join(default_path, "fixations")]:
        os.makedirs(d, exist_ok=True)

    tmp  = os.path.join(data_path, "tmp.zip")
    ids  = [
        "1g8j-hTT-51IG1UFwP0xTGhLdgIUCW5e5",  # stimuli
        "1P-jeZXCsjoKO79OhFUgnj6FGcyvmLDPj",  # fixations
        "1PnO7szbdub1559LfjYHMy65EDC4VhJC8",  # saliency maps
    ]
    paths = [
        os.path.join(default_path, "stimuli"),
        os.path.join(default_path, "fixations"),
        os.path.join(default_path, "saliency"),
    ]
    for i, fid in enumerate(ids):
        print(f"  Tải file {i+1}/3 ...", end="", flush=True)
        gdown.download(f"https://drive.google.com/uc?id={fid}&export=download", tmp, quiet=True)
        with zipfile.ZipFile(tmp, "r") as z:
            for f in z.namelist():
                if "test" not in f:
                    z.extract(f, paths[i])
        os.remove(tmp)
        print(" done!")

    images_dir = os.path.join(default_path, "stimuli", "images")
    if os.path.exists(images_dir):
        import shutil
        for sub in os.listdir(images_dir):
            shutil.move(os.path.join(images_dir, sub), os.path.join(default_path, "stimuli", sub))
        os.rmdir(images_dir)

    print("✅ SALICON tải xong!")
    return default_path
