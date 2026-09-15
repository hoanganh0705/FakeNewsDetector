# FakeNewsDetector — Phát hiện tin giả tiếng Việt sử dụng học máy và học sâu

> Đồ án ngành của sinh viên **Nguyễn Hoàng Anh** (MSSV: 2351010009) — Trường Đại học Mở TP. Hồ Chí Minh.
> Ngành: **Trí tuệ nhân tạo** — GVHD: **TS. Lê Quang Minh**.

Đồ án xây dựng và so sánh **bốn mô hình phát hiện tin giả** trên văn bản tiếng Việt:

| # | Mô hình | Nhóm |
|---|---------|------|
| 1 | Logistic Regression (LR) | Học máy truyền thống (TF-IDF) |
| 2 | Support Vector Machine (SVM) | Học máy truyền thống (TF-IDF) |
| 3 | BiLSTM | Học sâu (FastText embedding) |
| 4 | PhoBERT | Transformer tiền huấn luyện (BPE) |

Ngoài việc so sánh hiệu năng (Accuracy, F1-macro, ROC-AUC), đồ án còn thực hiện các phân tích mở rộng:
- Kiểm định McNemar + Holm–Bonferroni
- Khoảng tin cậy Bootstrap 95 %
- 5-fold Cross-Validation
- Ablation study trên pipeline TF-IDF + LR
- Phân tích hiệu chuẩn xác suất (ECE / MCE / Brier) + post-hoc calibration
- Token-level explainability (SHAP, Integrated Gradients, Attention Rollout)
- Knowledge Distillation (BiLSTM teacher → Student BiLSTM)

## 📁 Cấu trúc dự án

```
FakeNewsDetector/
├── paper/                          # Thư mục chính của đồ án
│   ├── main.tex                    # File chính LaTeX
│   ├── preamble.sty                # Cấu hình gói lệnh + custom commands
│   ├── data_commands.sty           # Tập trung mọi số liệu (\newcommand)
│   ├── refs.bib                    # Tài liệu tham khảo (BibTeX)
│   ├── build.sh                    # Script build PDF (pdflatex + bibtex)
│   │
│   ├── front/                      # Phần mở đầu
│   │   ├── 0.1-BiaNgoai.tex        # Bìa ngoài (HCMOU)
│   │   ├── 0.2-BiaTrong.tex        # Bìa trong (MSSV + GVHD)
│   │   ├── MucLuc_DanhMuc.tex      # Mục lục + 3 danh mục
│   │   └── pages/
│   │       ├── LoiCamOn.tex        # Lời cảm ơn
│   │       ├── GVHDNhanXet.tex     # Nhận xét của GVHD
│   │       ├── TomTat.tex          # Tóm tắt
│   │       ├── Abbreviations.tex   # File trống (nội dung đặt trong MucLuc)
│   │       └── PhuLuc.tex          # Phụ lục A–D
│   │
│   ├── chapters/                   # 5 chương nội dung + kết luận
│   │   ├── Chuong01_TongQuan.tex   # Chương 1: Tổng quan
│   │   ├── Chuong02_CoSoLyThuyet.tex  # Chương 2: Cơ sở lý thuyết
│   │   ├── Chuong03_DuLieu.tex     # Chương 3: Dữ liệu & tiền xử lý
│   │   ├── Chuong04_PhuongPhap.tex # Chương 4: Phương pháp
│   │   ├── Chuong05_KetQua.tex     # Chương 5: Kết quả thực nghiệm
│   │   └── Chuong99_KetLuan.tex    # Kết luận + TLTK + Phụ lục
│   │
│   ├── tables/                     # 9 bảng LaTeX được include
│   │   ├── table1_dataset.tex
│   │   ├── table2_results.tex
│   │   ├── table3_perclass.tex
│   │   ├── table4_hyperparams.tex
│   │   ├── table5_complexity.tex
│   │   ├── table_calibration.tex
│   │   ├── table_post_hoc_calibration.tex
│   │   ├── table_distillation.tex
│   │   ├── table_hard_cases_annotated.tex
│   │   └── table_attribution_faithfulness.tex
│   │
│   ├── figures/                    # Thư mục hình ảnh (figures/*.png|jpg)
│   ├── src/                        # Mã nguồn Python huấn luyện & đánh giá
│   ├── data/                       # Dữ liệu thô + đã xử lý
│   └── main.pdf                    # ← Kết quả build (sinh ra từ build.sh)
│
└── readme.md                       # File này
```

## 🛠️ Yêu cầu môi trường

- **TeX Live 2023+** (pdflatex + bibtex)
- **Java JRE ≥ 11** (cho `py_vncorenlp`)
- **Python ≥ 3.11** (chạy src/ — không bắt buộc để build PDF)

## 🚀 Cách build PDF

```bash
cd paper
./build.sh              # Build ra main.pdf
./build.sh clean        # Xóa file trung gian
./build.sh fullclean    # Xóa cả main.pdf
```

Build script thực hiện 4 bước chuẩn:
1. `pdflatex` lần 1 — sinh `.aux` cho bibtex
2. `bibtex main` — đọc `refs.bib` và `main.aux`, sinh `main.bbl`
3. `pdflatex` lần 2 — ổn định TOC + danh mục
4. `pdflatex` lần 3 — ổn định citation + cross-reference

Kết quả: **`paper/main.pdf`** (~5 MB, ~100 trang).

## 📊 Số liệu chính (trên tập test)

| Mô hình | Accuracy | F1-macro | ROC-AUC |
|---------|----------|----------|---------|
| Logistic Regression | 83,48 % | 83,29 % | 0,9188 |
| SVM | 84,34 % | 84,10 % | 0,9190 |
| BiLSTM | 82,52 % | 82,32 % | 0,9048 |
| **PhoBERT** | **90,07 %** | **89,88 %** | **0,9495** |

PhoBERT vượt trội với **+5,78 đến +7,56 điểm F1** so với các mô hình còn lại; sự khác biệt có ý nghĩa thống kê theo McNemar + Holm–Bonferroni (`p < 0,001`).

## 📝 Quy ước & cấu trúc LaTeX

### Mọi số liệu được đặt tại `data_commands.sty`

Mọi con số trong đồ án (kích thước tập dữ liệu, accuracy, F1, kích thước vocabulary,…) đều được định nghĩa dưới dạng `\newcommand{\tên_biến}{giá trị}` trong file `data_commands.sty`. Tác giả chỉ cần sửa MỘT chỗ → toàn bộ đồng bộ.

Ví dụ:

```latex
% Trong data_commands.sty
\newcommand{\phobertfone}{89,88}

% Trong Chuong05_KetQua.tex
PhoBERT đạt F1-macro cao nhất (\phobertfone\%).
```

### Custom commands tiện ích (trong `preamble.sty`)

| Lệnh | Mục đích |
|------|---------|
| `\phobert` | In chuỗi "PhoBERT" (có xử lý khoảng trắng) |
| `\bilstm` | In chuỗi "BiLSTM" |
| `\studentname` | Họ tên sinh viên |
| `\thesistitle` | Tên đồ án |
| `\studentid` | MSSV |
| `\major` | Ngành học |
| `\supervisor` | GVHD |
| `\currentyear` | Năm hiện tại |
| `\imagesource{url}` | In footnote "nguồn ảnh: <url>" ở chân trang |
| `\smarturl{url}` | Hiển thị URL gọn, có hyperlink |

## 🐛 Các lỗi LaTeX đã sửa (2026-09)

Trong quá trình compile, các lỗi sau đã được phát hiện và sửa:

1. **Chuong02_CoSoLyThuyet.tex** — Công thức `P(y=1|x) = ...` thiếu `\[` mở đầu.
   - Trước: `P(y=1 ... \]`
   - Sau: `\[ P(y=1 ... \]`
2. **Chuong02_CoSoLyThuyet.tex** — `\imagesource` có cú pháp Markdown `[url](url)`.
   - Trước: `\imagesource{[https://...](https://...)}`
   - Sau: `\imagesource{https://...}`

## ⚠️ Lưu ý về figures/

Thư mục `paper/figures/` được khai báo trong `preamble.sty` qua lệnh:
```latex
\graphicspath{{./figures/}{../figures/}{./}{../}}
```

Các hình ảnh minh họa (`textpreprocessing.jpg`, `lstm.jpeg`, `fig1_model_comparison.png`, …) cần được đặt vào đây. Nếu thiếu, pdflatex vẫn build thành công nhưng **chỉ có caption, không có hình ảnh**.

## 🔧 Tái lập thực nghiệm

Xem chi tiết trong `paper/front/pages/PhuLuc.tex` mục D.