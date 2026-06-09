"""
Tải MIT1003 dataset.
Cấu trúc sau khi giải nén:
  data/mit1003/stimuli/   — 1003 ảnh .jpeg
  data/mit1003/saliency/  — 1003 saliency maps .jpg  (gaussian-blurred)
  data/mit1003/fixations/ — 1003 fixation maps .jpg  (binary)
"""
import os, zipfile, urllib.request

MIT1003_STIMULI_URL  = "https://people.csail.mit.edu/tjudd/WherePeopleLook/ALLSTIMULI.zip"
MIT1003_FIXMAP_URL   = "https://people.csail.mit.edu/tjudd/WherePeopleLook/ALLFIXATIONMAPS.zip"
# Saliency (gaussian) maps — same server
MIT1003_SALMAP_URL   = "https://people.csail.mit.edu/tjudd/WherePeopleLook/ALLFIXATIONMAPS.zip"


def download_mit1003(data_path):
    root       = os.path.join(data_path, "mit1003")
    stim_dir   = os.path.join(root, "stimuli")
    sal_dir    = os.path.join(root, "saliency")
    fix_dir    = os.path.join(root, "fixations")

    n_imgs = sum(1 for _,_,fs in os.walk(stim_dir)
                for f in fs if f.lower().endswith((".jpg",".jpeg",".png")))
    if n_imgs >= 1000:
        print("✅ MIT1003 đã tồn tại, bỏ qua.")
        return root

    for d in [stim_dir, sal_dir, fix_dir]:
        os.makedirs(d, exist_ok=True)

    tmp = os.path.join(data_path, "tmp_mit.zip")

    def _dl(url, dest_dir, skip_suffix=None):
        print(f"  Tải {url.split('/')[-1]} ...", end="", flush=True)
        urllib.request.urlretrieve(url, tmp)
        with zipfile.ZipFile(tmp, "r") as z:
            for name in z.namelist():
                base = os.path.basename(name)
                if not base:
                    continue
                if skip_suffix and not base.lower().endswith(skip_suffix):
                    continue
                with z.open(name) as src, open(os.path.join(dest_dir, base), "wb") as dst:
                    dst.write(src.read())
        os.remove(tmp)
        print(" done!")

    _dl(MIT1003_STIMULI_URL, stim_dir, skip_suffix=(".jpeg", ".jpg", ".png"))
    _dl(MIT1003_FIXMAP_URL,  sal_dir,  skip_suffix=(".jpg",  ".jpeg", ".png"))
    print("✅ MIT1003 tải xong!")
    return root
