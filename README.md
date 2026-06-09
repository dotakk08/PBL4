# MSINet Saliency Prediction Benchmarking

Dự án này tập trung vào việc triển khai, cải tiến và đánh giá hiệu năng của mô hình **MSINet (Multi-Scale Information Network)** cho bài toán dự đoán độ nổi bật của hình ảnh (Saliency Prediction). Dự án thực hiện các thử nghiệm so sánh giữa kiến trúc gốc và 2 phương án cải tiến trên 3 bộ dữ liệu: **SALICON**, **MIT1003**, và **CAT2000**.
```
├── 📁 msinet_caitien_decoder_fpn
│   ├── 📁 results
│   │   ├── 📁 history
│   │   │   ├── 📄 cat2000_log.csv
│   │   │   ├── 📄 mit1003_log.csv
│   │   │   ├── 📄 salicon_log.csv
│   │   │   ├── 🖼️ v2_all_loss_curves.png
│   │   │   └── 🖼️ v2_salicon_loss.png
│   │   └── 📁 images
│   │       ├── 🖼️ v2_pred_cat2000_val.png
│   │       ├── 🖼️ v2_pred_mit1003_val.png
│   │       └── 🖼️ v2_pred_salicon_val.png
│   ├── 🐍 config.py
│   ├── 🐍 data_loader.py
│   ├── 🐍 download_cat2000.py
│   ├── 🐍 download_mit1003.py
│   ├── 🐍 download_salicon.py
│   ├── 🐍 loss.py
│   ├── 🐍 metrics.py
│   ├── 🐍 model.py
│   ├── 📄 msinet_caitien_decoder_fpn.ipynb
│   └── 📄 msinet_caitien_decoder_fpn_(clear_output).ipynb
├── 📁 msinet_caitien_encoder_efficientnetb0
│   ├── 📁 results
│   │   ├── 📁 history
│   │   │   ├── 🖼️ cat2000_finetune_loss.png
│   │   │   ├── 🖼️ cat2000_finetune_loss_inline.png
│   │   │   ├── 🖼️ loss_curve.png
│   │   │   ├── 🖼️ mit1003_finetune_loss.png
│   │   │   ├── 🖼️ mit1003_finetune_loss_inline.png
│   │   │   ├── 🖼️ salicon_train_loss.png
│   │   │   └── 📄 training_log.csv
│   │   ├── 📁 images
│   │   │   └── 🖼️ msinet_predictions.png
│   │   ├── 🖼️ viz_CAT2000_val.png
│   │   ├── 🖼️ viz_MIT1003_val.png
│   │   └── 🖼️ viz_SALICON_val.png
│   ├── 🐍 config.py
│   ├── 🐍 data_loader.py
│   ├── 🐍 download_cat2000.py
│   ├── 🐍 download_mit1003.py
│   ├── 🐍 download_salicon.py
│   ├── 🐍 loss.py
│   ├── 🐍 metrics.py
│   ├── 🐍 model.py
│   ├── 📄 msinet_caitien_encoder_efficientnetb0.ipynb
│   └── 📄 msinet_caitien_encoder_efficientnetb0_(clear_output).ipynb
├── 📁 msinet_goc
│   ├── 📁 results
│   │   ├── 📁 history
│   │   │   ├── 🖼️ cat2000_finetune_loss.png
│   │   │   ├── 🖼️ cat2000_finetune_loss_inline.png
│   │   │   ├── 🖼️ loss_curve.png
│   │   │   ├── 🖼️ mit1003_finetune_loss.png
│   │   │   ├── 🖼️ mit1003_finetune_loss_inline.png
│   │   │   ├── 🖼️ salicon_train_loss.png
│   │   │   └── 📄 training_log.csv
│   │   ├── 📁 images
│   │   │   └── 🖼️ msinet_predictions.png
│   │   ├── 🖼️ viz_CAT2000_val.png
│   │   ├── 🖼️ viz_MIT1003_val.png
│   │   └── 🖼️ viz_SALICON_val.png
│   ├── 🐍 config.py
│   ├── 🐍 data_loader.py
│   ├── 🐍 download_cat2000.py
│   ├── 🐍 download_mit1003.py
│   ├── 🐍 download_salicon.py
│   ├── 🐍 loss.py
│   ├── 🐍 metrics.py
│   ├── 🐍 model.py
│   ├── 📄 msinet_goc.ipynb
│   └── 📄 msinet_goc_(clear_output).ipynb
└── ⚙️ .gitignore
```

---
## 🛠️ Tổng quan về cấu trúc thư mục
Mỗi thư mục mô hình chứa một bộ mã nguồn PyTorch hoàn chỉnh, độc lập:

* **config.py:** Quản lý các siêu tham số huấn luyện (Learning rate, Batch size, Epochs, kích thước ảnh, đường dẫn thư mục...).

* **data_loader.py:** Đọc dữ liệu, thực hiện tiền xử lý ảnh (Resize, Normalize) và đóng gói thành DataLoader của PyTorch.

* **download_*.py:** Các script tự động hóa tác vụ tải và giải nén các bộ dữ liệu từ nguồn lưu trữ trực tuyến.

* **model.py:** Định nghĩa cấu trúc mạng nơ-ron (Bản gốc, bản thay đổi Encoder, bản tích hợp FPN).

* **loss.py:** Định nghĩa hàm mất mát kết hợp tối ưu cho bài toán Saliency (ví dụ: tổ hợp của KL-Divergence, CC, và NSS).

* **metrics.py:** Công cụ đo lường và tính toán các chỉ số đánh giá chuẩn: AUC-Judd, SIM, CC, NSS.

* ***.ipynb:** File Notebook tích hợp quy trình chạy huấn luyện, log kết quả và trực quan hoá đồ thị. (Bản _clear_output đã được xóa sạch kết quả chạy để tối ưu dung lượng Git).

## 🚀 Hướng dẫn sử dụng chi tiết (Usage)
Dự án hỗ trợ linh hoạt trên cả 3 môi trường huấn luyện phổ biến. Hãy chọn một trong các cách dưới đây phù hợp với tài nguyên phần cứng của bạn:

### 💡 Bước chuẩn bị chung cho Cloud (Colab / Kaggle)
Nếu chạy trên Cloud, trước tiên hãy tải toàn bộ mã nguồn của dự án này lên Google Drive (đối với Colab) hoặc nén thành file .zip rồi upload lên phần Datasets (đối với Kaggle).

**Môi trường 1:** Google Colab (Khuyến nghị sử dụng GPU T4/A100)
Mở Google Colab, chọn Upload và tải file Notebook lên (Ví dụ: msinet_caitien_decoder_fpn.ipynb).

- Kết nối với Google Drive để truy cập các file code bổ trợ bằng cách thêm một ô code mới ở đầu Notebook:
```python
# Đây là code Python
from google.colab import drive
drive.mount('/content/drive')
```

- Di chuyển thư mục làm việc của Colab đến vị trí chứa code dự án của bạn trên Drive:

```Bash
# Đây là code Bash
%cd /content/drive/MyDrive/PBL4/msinet_caitien_decoder_fpn
```
- Chạy các file script để tải tự động dữ liệu trực tiếp về môi trường Cloud:

```Bash
# Đây là code Bash
!python download_salicon.py
!python download_mit1003.py
!python download_cat2000.py
```
- Tiến hành chạy tuần tự các ô code còn lại trong Notebook để huấn luyện mô hình.

**Môi trường 2:** Kaggle Notebook (Khuyến nghị sử dụng GPU P100/T4 x2)
Tạo một New Notebook trên Kaggle, kích chọn mục Accelerator là GPU T4 x2 hoặc GPU P100 ở bảng điều khiển bên phải.

- Upload folder source code của bạn lên Kaggle dưới dạng Dataset, sau đó kiểm tra đường dẫn thư mục code (thường nằm trong /kaggle/input/).

- Sao chép thư mục code sang thư mục có quyền ghi (/kaggle/working/) để có thể chạy script và lưu kết quả:

```Bash
# Đây là code Bash
!cp -r /kaggle/input/ten-dataset-cua-ban /kaggle/working/msinet_project
%cd /kaggle/working/msinet_project/msinet_caitien_decoder_fpn
```
- Thực thi các script tải dữ liệu:

```Bash
# Đây là code Bash
!python download_salicon.py
!python download_mit1003.py
!python download_cat2000.py
```
- Mở/Copy nội dung code từ file Notebook gốc vào Kaggle Notebook và tiến hành Run All.

**Môi trường 3:** Jupyter Notebook (Chạy Local trên máy tính cá nhân)
Cài đặt thư viện: Mở Terminal / Command Prompt và cài đặt các thư viện cần thiết:

```Bash
# Đây là code Bash
pip install torch torchvision numpy pandas matplotlib opencv-python pillow tqdm jupyter
```
- Tải dữ liệu: Di chuyển vào thư mục của mô hình bạn muốn thực nghiệm và chạy 3 file script để tải tự động:
```Bash
# Đây là code Bash
cd msinet_caitien_decoder_fpn
python download_salicon.py
python download_mit1003.py
python download_cat2000.py
```
- Khởi động và Chạy: Mở giao diện Jupyter Notebook từ Terminal:

```Bash
# Đây là code Bash
jupyter notebook
```
- Trình duyệt tự động mở ra, hãy chọn file Notebook tương ứng (Ví dụ: msinet_caitien_decoder_fpn.ipynb) và nhấn Run All để bắt đầu quá trình huấn luyện.
## 📈 Kết quả đầu ra (Results)
- Sau khi quá trình chạy hoàn tất, toàn bộ kết quả thực nghiệm sẽ tự động kết xuất ra thư mục results/ tương ứng trong môi trường bạn đang chạy:

- Log và Đồ thị (results/history/): Lưu file lịch sử huấn luyện dạng .csv và biểu đồ Loss theo chu kỳ Epochs (ví dụ: v2_all_loss_curves.png, loss_curve.png) để theo dõi tốc độ hội tụ.

- Hình ảnh trực quan (results/images/): Chứa ảnh Saliency Map do mô hình dự đoán được xếp cạnh Ground Truth (ảnh thực tế) để đánh giá định tính trực quan bằng mắt (ví dụ: v2_pred_salicon_val.png, msinet_predictions.png).
