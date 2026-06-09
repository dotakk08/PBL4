# MSINet — Visual Saliency Prediction Benchmarking

Dự án triển khai, cải tiến và đánh giá hiệu năng của mô hình **MSINet (Multi-Scale Input Network)** cho bài toán dự đoán độ nổi bật của hình ảnh (Saliency Prediction). Dự án thực hiện các thử nghiệm so sánh giữa kiến trúc gốc và 2 phương án cải tiến trên 3 bộ dữ liệu: **SALICON**, **MIT1003**, và **CAT2000**.

---

## 📁 Cấu trúc thư mục

```
├── msinet_goc/
├── msinet_caitien_encoder_efficientnetb0/
└── msinet_caitien_decoder_fpn/
```

Mỗi thư mục có cấu trúc giống nhau:

```
msinet_<tên>/
├── results/
│   ├── history/          # File CSV log & biểu đồ Loss
│   └── images/           # Ảnh Saliency Map dự đoán
├── config.py
├── data_loader.py
├── download_cat2000.py
├── download_mit1003.py
├── download_salicon.py
├── loss.py
├── metrics.py
├── model.py
├── msinet_<tên>.ipynb
└── msinet_<tên>_(clear_output).ipynb
```

---

## 🧠 Kiến trúc mô hình

### Bản gốc (`msinet_goc`)

- **Framework:** TensorFlow 2.12 / Keras
- **Encoder:** VGG16-style CNN (conv1 → conv5, 13 conv layers), load pretrained **VGG16-Hybrid weights** (ImageNet + Places)
- **Multi-Scale Input (MSI):** Ảnh được xử lý ở 3 scale song song (1×, 1/2×, 1/4×); feature maps từ 3 nhánh được upsample về cùng kích thước rồi concatenate → 1536 channels
- **Decoder:** UpSampling2D 16× + Conv 1×1 + Sigmoid + Resize về kích thước đầu vào
- **Loss:** KL Divergence
- **Pipeline data:** Resize giữ aspect ratio + symmetric padding (126 cho RGB, 0 cho saliency) + tạo 3 scale input
- **Training:** Train trên SALICON → Fine-tune MIT1003 → Fine-tune CAT2000

### Bản cải tiến 1 (`msinet_caitien_encoder_efficientnetb0`)

Thay encoder VGG16 bằng **EfficientNet-B0** để tăng hiệu quả tham số.

### Bản cải tiến 2 (`msinet_caitien_decoder_fpn`)

Tích hợp **FPN (Feature Pyramid Network)** vào decoder để khai thác đặc trưng đa tỉ lệ tốt hơn.

---

## 🛠️ Mô tả các module

- **config.py** — Siêu tham số huấn luyện: learning rate (`1e-5`), batch size (`4`), số epochs (`10`), kích thước ảnh theo từng dataset.
- **model.py** — Định nghĩa kiến trúc mạng và hàm `load_vgg16_hybrid_weights()` để nạp pretrained weights vào cả 3 nhánh encoder.
- **loss.py** — Hàm mất mát KL Divergence (normalize cả `y_true` lẫn `y_pred` trước khi tính).
- **metrics.py** — Tính các chỉ số đánh giá chuẩn: **KLD**, **CC**, **SIM**, **NSS**, **AUC-Judd**.
- **data_loader.py** — `tf.data` pipeline: resize giữ aspect ratio (area khi shrink, bicubic khi enlarge), symmetric padding, normalize, tạo 3 scale input cho MSI.
- **download_salicon.py** — Tải SALICON từ Google Drive qua `gdown` (3 file: stimuli, fixations, saliency maps).
- **download_mit1003.py** — Tải MIT1003 từ `people.csail.mit.edu` qua `urllib` (1003 ảnh + fixation maps).
- **download_cat2000.py** — Tải CAT2000 từ `saliency.mit.edu` (~1 GB) qua `requests` với cơ chế retry và resume.

---

## 🚀 Hướng dẫn chạy trên Kaggle Notebook

**Môi trường khuyến nghị:** Kaggle Notebook với GPU P100.

### Bước 1 — Upload notebook

Upload file **`msinet_<tên>_(clear_output).ipynb`** của model cần chạy lên Kaggle.

> Dùng bản `clear_output` để tránh file nặng do output cũ. **Không cần upload các file `.py` riêng** — notebook tự sinh ra toàn bộ module bằng `%%writefile` khi chạy.

### Bước 2 — Bật GPU

Vào **Settings → Accelerator → GPU P100** (hoặc T4 nếu P100 không khả dụng).

### Bước 3 — Run All

Nhấn **Run All**. Notebook sẽ tự động thực hiện toàn bộ pipeline theo thứ tự:

| Bước | Nội dung |
|------|----------|
| 1 | Cài đặt thư viện: `tensorflow==2.12.0`, `gdown`, `h5py`, `scipy`, `matplotlib`, `requests` |
| 2 | Tạo thư mục làm việc tại `/kaggle/working/` |
| 3 | Ghi các file module `.py` vào `/kaggle/working/` |
| 4 | Tải 3 bộ dữ liệu: SALICON (10k train / 5k val), MIT1003 (1003 ảnh), CAT2000 (2000 ảnh / 20 danh mục) |
| 5 | Khởi tạo model MSINet + nạp VGG16-Hybrid pretrained weights |
| 6 | Train trên **SALICON** (10 epochs, batch 4, lr 1e-5) |
| 7 | Fine-tune trên **MIT1003** |
| 8 | Fine-tune trên **CAT2000** |
| 9–15 | Đánh giá metrics, visualize saliency maps, lưu model |
| 16 | Đóng gói toàn bộ output thành `msinet_full_<timestamp>.zip` |

---

## 📈 Kết quả đầu ra

Sau khi chạy xong, toàn bộ kết quả được nén vào file `msinet_full_<timestamp>.zip` tại `/kaggle/working/`. Cấu trúc bên trong:

```
results/
├── ckpts/best/
│   ├── msinet_salicon_best.weights.h5
│   ├── msinet_mit1003_best.weights.h5
│   └── msinet_cat2000_best.weights.h5
├── history/
│   ├── training_log.csv
│   ├── salicon_train_loss.png
│   ├── mit1003_finetune_loss.png
│   ├── cat2000_finetune_loss.png
│   └── loss_curve.png
├── images/
│   └── msinet_predictions.png
├── viz_SALICON_val.png
├── viz_MIT1003_val.png
├── viz_CAT2000_val.png
├── baseline_final.keras
└── baseline_weights.weights.h5
weights/
├── vgg16_hybrid.ckpt.data-00000-of-00001
└── vgg16_hybrid.ckpt.index
config.py / loss.py / model.py / data_loader.py / metrics.py / download_*.py
```

- **ckpts/best/** — Trọng số tốt nhất (best checkpoint) của mỗi lần training.
- **history/** — File CSV log lịch sử training và biểu đồ Loss theo epochs để theo dõi hội tụ.
- **images/ & viz_*.png** — Saliency Map dự đoán xếp cạnh Ground Truth để đánh giá định tính trực quan.

---

## 📊 Metrics đánh giá

| Metric | Ý nghĩa | Tốt hơn khi |
|--------|---------|-------------|
| KLD | KL Divergence | Thấp hơn |
| CC | Pearson Correlation Coefficient | Cao hơn |
| SIM | Similarity (histogram intersection) | Cao hơn |
| NSS | Normalized Scanpath Saliency | Cao hơn |
| AUC-Judd | Area Under ROC Curve | Cao hơn |

---

## ⚙️ Siêu tham số mặc định

| Tham số | Giá trị |
|---------|---------|
| Learning rate | `1e-5` |
| Batch size | `4` |
| Epochs | `10` |
| Optimizer | Adam |
| Input size (SALICON) | 240 × 320 |
| Input size (MIT1003) | 360 × 360 |
| Input size (CAT2000) | 216 × 384 |
