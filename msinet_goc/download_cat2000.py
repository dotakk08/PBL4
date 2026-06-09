"""
Tải CAT2000 dataset.
Cấu trúc sau khi giải nén:
  data/cat2000/stimuli/   — 2000 ảnh (20 danh mục × 100 ảnh)
  data/cat2000/saliency/  — 2000 saliency maps  (gaussian-blurred)
  data/cat2000/fixations/ — 2000 fixation maps  (binary)
"""
import os, zipfile, time
import requests

CAT2000_URL = "http://saliency.mit.edu/trainSet.zip"


def download_cat2000(data_path):
    root     = os.path.join(data_path, "cat2000")
    stim_dir = os.path.join(root, "stimuli")

    if os.path.exists(stim_dir) and len(os.listdir(stim_dir)) >= 20:
        print("✅ CAT2000 đã tồn tại, bỏ qua.")
        return root

    os.makedirs(data_path, exist_ok=True)
    tmp = os.path.join(data_path, "tmp_cat.zip")

    # --- download với requests + retry + resume ---------------------------
    MAX_RETRIES = 5
    CHUNK       = 1024 * 1024   # 1 MB

    for attempt in range(1, MAX_RETRIES + 1):
        downloaded = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        headers    = {"Range": f"bytes={downloaded}-"} if downloaded else {}
        try:
            print(f"  Tải CAT2000 (~1 GB) — lần {attempt} ", end="", flush=True)
            resp = requests.get(CAT2000_URL, headers=headers,
                                stream=True, timeout=(30, 120))
            # 206 = resume OK, 200 = server không hỗ trợ resume (restart)
            if resp.status_code == 200 and downloaded:
                downloaded = 0   # server gửi lại từ đầu
            if resp.status_code not in (200, 206):
                raise RuntimeError(f"HTTP {resp.status_code}")
            mode = "ab" if downloaded else "wb"
            with open(tmp, mode) as f:
                for chunk in resp.iter_content(CHUNK):
                    if chunk:
                        f.write(chunk)
                        print(".", end="", flush=True)
            print(" done!")
            break   # thành công
        except Exception as e:
            print(f" lỗi: {e}")
            if attempt < MAX_RETRIES:
                wait = 5 * attempt
                print(f"  Thử lại sau {wait}s...", flush=True)
                time.sleep(wait)
            else:
                raise RuntimeError("Không tải được CAT2000 sau nhiều lần thử.") from e

    # --- giải nén ---------------------------------------------------------
    print("  Đang giải nén...", end="", flush=True)
    with zipfile.ZipFile(tmp, "r") as z:
        for name in z.namelist():
            if not ("Output" in name or "allFixData" in name):
                z.extract(name, data_path)

    # --- rename thư mục ---------------------------------------------------
    train_path = os.path.join(data_path, "trainSet")
    if os.path.exists(train_path) and not os.path.exists(root):
        os.rename(train_path, root)

    for old, new in [("Stimuli",      "stimuli"),
                     ("FIXATIONLOCS", "fixations"),
                     ("FIXATIONMAPS", "saliency")]:
        src = os.path.join(root, old)
        dst = os.path.join(root, new)
        if os.path.exists(src) and not os.path.exists(dst):
            os.rename(src, dst)

    os.remove(tmp)
    print(" done!")
    print("✅ CAT2000 tải xong!")
    return root
