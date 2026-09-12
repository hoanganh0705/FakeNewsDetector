# Bài Báo Cáo — Phát Hiện Tin Giả Tiếng Việt

Bài báo cáo đồ án ngành đã được **tích hợp với khung template "DoAnNganh"** của Trường Đại học Mở TP.HCM.

> ✅ **Refactored (2026-09)**: file `main.tex` đã được tách thành 9 file nhỏ để dễ quản lý.
> Trước: `main.tex` 2601 dòng (khó kiểm soát). Sau: `main.tex` ~53 dòng, mỗi chương là 1 file riêng.
> ✅ **Bug-fixed (2026-09-08)**: sửa lỗi "Missing \begin{document}" do quên `\begin{document}` trong main.tex, lỗi duplicate `hyperref` colorlinks, lỗi duplicate page labels bằng cách dùng La Mã cho front matter. Hiện build ra PDF sạch 0 errors / 0 warnings.
> ✅ **Bug-fixed (2026-09-09)**: sửa lỗi `fig_token_attribution_lr_svm.png` bị trống — `lr_kernel_shap` dùng `predict_proba` + `link="logit"` khiến SHAP attributions bị nén về ~1e-4 (do `p·(1-p)` scaling với probability cực trị). Đổi sang `decision_function` (raw log-odds), bỏ `link`. Tăng `n_samples` mặc định 100 → 200 cho ổn định hơn trên TF-IDF sparse.
> ✅ **Bug-fixed (2026-09-09)**: sửa lỗi `fig_token_attribution_phobert.png`, `fig_method_agreement.png`, `fig_cross_model_agreement.png` bị placeholder — (1) `agreement_matrix` crash khi các phương pháp có độ dài vector khác nhau → thêm NaN mask cho các cặp không so sánh được; (2) `_align_to_words` split n-gram thành từng từ riêng (ví dụ "việt_nam" → "việt", "nam") nhưng word list chứa "việt_nam" → không khớp → scores = 0; đổi thành split trên "_" để giữ nguyên từ ghép; (3) `cross_model_agreement` yêu cầu đủ 4 models → đổi thành chấp nhận subset 2–4 models; (4) `_resolve_vocab` không xử lý `joblib.load`-style dict → thêm trường hợp giải nén dict từ `EmbeddingFeatureExtractor.save()`.

## Cấu Trúc Thư Mục (MỚI — sau khi refactor)

```
paper/
├── main.tex                        # File chính — ~53 dòng, chỉ chứa \input
├── preamble.sty                    # Gói lệnh + thông tin đồ án + custom commands
├── data_commands.sty               # Toàn bộ \newcommand{...} số liệu
├── build.sh                        # Script build PDF (4 bước chuẩn)
├── refs.bib                        # Tài liệu tham khảo (BibTeX)
├── main.pdf                        # Output: bài báo PDF (113 trang)
│
├── front/                          # PHẦN MỞ ĐẦU (front matter)
│   ├── 0.1-BiaNgoai.tex            # Bìa ngoài (HCMOU)
│   ├── 0.2-BiaTrong.tex            # Bìa trong (MSSV + GVHD)
│   ├── MucLuc_DanhMuc.tex          # Mục lục + 3 danh mục (viết tắt, hình, bảng)
│   └── pages/
│       ├── TomTat.tex              # Tóm tắt đồ án ngành
│       ├── Abbreviations.tex       # Danh mục viết tắt (file rỗng — danh sách ở MucLuc_DanhMuc)
│       ├── LoiCamOn.tex            # Lời cảm ơn
│       ├── GVHDNhanXet.tex         # Nhận xét GVHD
│       └── PhuLuc.tex              # Phụ lục
│
├── chapters/                       # PHẦN THÂN BÀI (5 chương + kết luận)
│   ├── Chuong01_TongQuan.tex       # Tổng quan (37 dòng)
│   ├── Chuong02_CoSoLyThuyet.tex   # Cơ sở lý thuyết (~470 dòng)
│   ├── Chuong03_DuLieu.tex         # Bộ dữ liệu (~210 dòng)
│   ├── Chuong04_PhuongPhap.tex     # Phương pháp (~260 dòng)
│   ├── Chuong05_KetQua.tex         # Kết quả thực nghiệm (~880 dòng)
│   └── Chuong99_KetLuan.tex        # Kết luận + TLTK + Phụ lục
│
├── src/                            # Logo & hình bìa
│   └── hcmcou.png                  # Logo Trường ĐH Mở TP.HCM
│
├── figures/                        # Hình ảnh minh họa trong bài
├── tables/                         # Bảng số liệu (\input{tables/...})
└── verify_link.py                  # Script kiểm tra link "nguồn ảnh" có click được không
```

## Cách dùng `\imagesource` (ghi nguồn ảnh vào footnote)

Mỗi hình trong bài có một footnote "nguồn ảnh" chứa URL nguồn. Cú pháp:

```latex
\begin{figure}[H]
  \centering
  \includegraphics[width=0.95\textwidth]{hinh1.png}
  \caption{Minh họa XYZ \protect\imagesourcemark}
  \label{fig:xyz}
\end{figure}
\imagesource{https://example.com/hinh1.png}
```

- `\imagesourcemark` được đặt **trong `\caption`** (sau `\protect` để hoạt động trong bookmarks).
- `\imagesource{URL}` được đặt **sau `\end{figure}`**.
- Hệ thống tự động tăng số thứ tự: Hình 1.1 → footnote 1, Hình 1.2 → footnote 2, v.v.
- Số trong caption và số trong footnote luôn khớp nhau.

## Cách Build PDF

```bash
cd paper
./build.sh              # Build: chạy 4 bước (pdflatex → bibtex → pdflatex → pdflatex)
./build.sh clean        # Dọn dẹp các file trung gian (.aux, .toc, .log, .fls, ...)
./build.sh fullclean    # Clean + xóa luôn main.pdf
```

Hoặc build thủ công:

```bash
cd paper
pdflatex main.tex       # Lần 1: sinh .aux cho bibtex
bibtex main             # Sinh main.bbl từ refs.bib
pdflatex main.tex       # Lần 2: cập nhật TOC + citation
pdflatex main.tex       # Lần 3: ổn định cross-references
```

## Cấu Trúc Tài Liệu (theo mẫu template)

### Phần mở đầu (đánh số La Mã i, ii, iii, ...)

| Trang | Nội dung | File nguồn |
|-------|----------|------------|
| 1     | **Bìa ngoài** — HCMOU, tên SV, tên đồ án, ngành, năm | `front/0.1-BiaNgoai.tex` |
| 2     | **Bìa trong** — MSSV, GVHD | `front/0.2-BiaTrong.tex` |
| 3     | **Lời cảm ơn** | `front/pages/LoiCamOn.tex` |
| 4     | **Nhận xét GVHD** | `front/pages/GVHDNhanXet.tex` |
| 5     | **Tóm tắt đồ án ngành** | `front/pages/TomTat.tex` |
| 6-7   | **Mục lục** + **Danh mục viết tắt** | `front/MucLuc_DanhMuc.tex` |
| 8-9   | **Danh mục hình vẽ** + **Danh mục bảng** | `front/MucLuc_DanhMuc.tex` |

### Phần thân bài (đánh số Ả-rập 1, 2, 3, ...)

| Chương | Nội dung | File |
|--------|----------|------|
| (Mở đầu) | Tổng quan về đề tài | `chapters/Chuong01_TongQuan.tex` |
| 1       | Cơ sở lý thuyết — NLP tiếng Việt, ML, Deep Learning, Transformer, PhoBERT | `chapters/Chuong02_CoSoLyThuyet.tex` |
| 2       | Bộ dữ liệu và tiền xử lý | `chapters/Chuong03_DuLieu.tex` |
| 3       | Xác định tin giả (Phương pháp đề xuất) | `chapters/Chuong04_PhuongPhap.tex` |
| 4       | Kết quả thực nghiệm | `chapters/Chuong05_KetQua.tex` |
| (Kết luận) | Kết luận + Tài liệu tham khảo + Phụ lục | `chapters/Chuong99_KetLuan.tex` |

## Thông Tin Cần Cập Nhật Trước Khi Nộp

Trong `preamble.sty`, **mục Thông tin đồ án** (cuối file):

```latex
\newcommand{\studentname}{NGUYỄN HOÀNG ANH}
\newcommand{\thesistitle}{PHÁT HIỆN TIN GIẢ TIẾNG VIỆT SỬ DỤNG HỌC MÁY VÀ HỌC SÂU}
\newcommand{\studentid}{2351010009}              % ← Thay bằng MSSV thật
\newcommand{\major}{TRÍ TUỆ NHÂN TẠO}            % ← Kiểm tra tên ngành chính xác
\newcommand{\supervisor}{TS. LÊ QUANG MINH}      % ← Kiểm tra tên GVHD
\newcommand{\currentyear}{2026}                  % ← Năm bảo vệ
```

> **Lưu ý:** Các biến này được dùng bởi cả `0.1-BiaNgoai.tex` (trang bìa) và `pages/PhuLuc.tex`.

## Cập Nhật Số Liệu Thực Nghiệm

Tất cả số liệu (F1, AUC, ECE, ...) đã được tập trung trong `data_commands.sty`. 
Sửa một chỗ → toàn bộ bài đồng bộ theo. Ví dụ:

```latex
% Trong data_commands.sty
\newcommand{\phobertfone}{89,88}    % ← Sửa tại đây
```

## Về Hình Ảnh Trong Bài (figures/)

Toàn bộ hình ảnh được lưu trong `paper/figures/`. Có **2 loại**:

### 1. Hình thực nghiệm (tự tạo bằng Python/matplotlib)

Đây là những hình được sinh ra từ chính code thực nghiệm trong project (thư mục `experiments/`):

| Hình | Mục đích | Cách tạo lại |
|------|----------|---------------|
| `fig1_model_comparison` | So sánh Accuracy/F1/AUC của 4 mô hình | `experiments/<model>/evaluate.py` |
| `fig5_per_class` | Precision/Recall/F1 theo lớp | `experiments/aggregate.py` |
| `fig6_paradigm_comparison` | So sánh 2 paradigm | tự sinh từ notebooks |
| `fig2_confusion_matrices` | Ma trận nhầm lẫn 4 mô hình | `experiments/<model>/evaluate.py` |
| `fig3_roc_curves`, `fig4_pr_curves` | Đường cong ROC và PR | tự sinh |
| `fig0a_overall_distribution`, `fig0b_split_distribution` | Phân phối dữ liệu | `experiments/data_prep.py` |
| `fig_calibration_curves`, `fig_calibration_overlay` | Hiệu chuẩn | tự sinh |
| `fig_reliability_diagrams_before_after` | Reliability diagrams | tự sinh |
| `fig_token_attribution_lr_svm`, `fig_token_attribution_phobert` | SHAP attribution | tự sinh |
| `fig_method_agreement`, `fig_cross_model_agreement` | Jaccard heatmap | tự sinh |
| `fig_fp_fn_comparison`, `fig_text_length_analysis` | Phân tích lỗi | tự sinh |
| `fig_hard_case_distribution` | Phân bố hard cases | tự sinh |
| `feature_importance` | Top 15 đặc trưng LR | tự sinh |
| `error_taxonomy` | Phân loại lỗi | tự sinh |

> **Tóm lại:** Bạn **không cần tự tạo** các hình này vì chúng đã được sinh ra từ code thực nghiệm. Khi cập nhật số liệu, chỉ cần chạy lại các script tương ứng trong `experiments/` để regenerate.

### 2. Hình minh họa lý thuyết (lấy từ nguồn bên ngoài có ghi nguồn)

Đây là những hình minh họa khái niệm lý thuyết, **lấy từ các nguồn public** và đã có footnote "nguồn ảnh" ghi rõ URL:

| Hình | Nguồn | Ghi chú |
|------|-------|---------|
| `textpreprocessing.jpg/png` | LinkedIn blog, Couchbase blog | Minh họa TF-IDF, tiền xử lý |
| `1.1.3.png` | maelfabien.github.io | Minh họa bài toán phân loại văn bản |
| `1.3.2.2.png` | statusneo.com | Minh họa thuật toán SVM |
| `tfidf_example.png` | dataaspirant.com | Ví dụ tính TF-IDF |
| `1.4.2.png` | machinelearningmastery | Minh họa Word Embedding |
| `annlayers.png` | Medium blog | Minh họa ANN layers |
| `lstm.jpeg` | colah.github.io | Minh họa LSTM |
| `bilstm.png` | geeksforgeeks.org | Minh họa BiLSTM |
| `encoder.jpg` | d2l.ai | Minh họa Transformer encoder |

> **Bạn không cần tự tạo/tìm các hình này.** Chúng đã được tải về và lưu trong `figures/`. Nếu một ngày nào đó nguồn gốc bị die, có thể tìm lại bằng cách Google "TF-IDF illustration", "BiLSTM architecture diagram", v.v. — nhưng **không cần làm gì bây giờ**.

### Nếu sau này muốn thay hình minh họa

Bạn có 2 lựa chọn:

**Lựa chọn A: Tự vẽ bằng draw.io / PowerPoint / TikZ**
- Đặt file mới vào `paper/figures/` (cùng tên hoặc tên mới).
- Cập nhật `\includegraphics{...}` trong file chương tương ứng.
- Nếu muốn ghi nguồn: thêm `\imagesource{URL}` sau `\end{figure}` (xem hướng dẫn ở phần trên).

**Lựa chọn B: Lấy từ nguồn public**
- Tìm hình minh họa trên Google Scholar, Wikipedia, blog uy tín.
- Đảm bảo hình có giấy phép sử dụng được (CC-BY, public domain, hoặc của tác giả bài báo được cite).
- Lưu vào `figures/`, cập nhật caption, ghi URL vào `\imagesource{URL}`.

## Cấu Trúc Preamble (preamble.sty)

Preamble được tổ chức thành 5 phần rõ ràng:

1. **Font & Encoding** — tự động chọn pdfTeX (vntex) hoặc XeTeX/LuaTeX (fontspec)
2. **Mathematics, Graphics, Algorithm** — các gói toán, hình vẽ, pseudocode
3. **Page Layout & Formatting** — geometry, setspace, fncychap (style "Chương X")
4. **Typography, Tables, Figures** — microtype, booktabs, multirow, subcaption, ...
5. **Hyperlinks, Bibliography, Utilities** — hyperref, cleveref, cite, xspace, crefname

## Lợi Ích Của Cấu Trúc Mới

| Trước (1 file) | Sau (nhiều file) |
|----------------|------------------|
| `main.tex` 2601 dòng | `main.tex` ~53 dòng |
| Muốn sửa Chương 3 → scroll 1200 dòng | Mở thẳng `Chuong03_DuLieu.tex` |
| Lỗi dòng 1500 → tìm trong 2600 dòng | Lỗi ngay trong file đang đọc |
| Khó review/sửa song song | Mỗi người 1 chương, không đụng |
| Preamble trộn với nội dung | Phân tách rõ `.sty` vs `.tex` |

## Lưu Ý Khi Build

- **Trình biên dịch**: `pdflatex` (khuyến nghị TeX Live 2023 trở lên)
- **Bibliography**: `bibtex` + style `ieeetr`
- **Font tiếng Việt**: hỗ trợ cả `pdfTeX` (vntex) lẫn `XeTeX/LuaTeX` (fontspec + polyglossia) — chọn tự động qua `\ifPDFTeX`
- **TOC**: dùng `tocloft` để có dấu chấm lửng + numbered "Chương X" (theo template)
- **Hyperref**: hyperlink màu (xanh dương cho nội bộ, tím cho citation, đỏ cho URL) — đã tắt màu cho phần mục lục để hiển thị như chữ thường

## Câu Hỏi Thường Gặp

**Q: Tôi muốn sửa nội dung Chương 4?**
A: Mở file `chapters/Chuong04_PhuongPhap.tex`, sửa trực tiếp. Không cần đụng vào các file khác.

**Q: Tôi muốn thay đổi tên trên bìa?**
A: Sửa các macro `\studentname`, `\thesistitle`, ... trong `preamble.sty`. Các thay đổi sẽ tự động cập nhật trang bìa ngoài và trong.

**Q: Tôi sửa số liệu thực nghiệm nhưng PDF không đổi?**
A: Số liệu được định nghĩa qua macro trong `data_commands.sty`. Sau khi sửa, chạy lại `./build.sh` để rebuild.

**Q: File build tạo ra rất nhiều .aux, .log, .fls,...?**
A: Chạy `./build.sh clean` để dọn dẹp tất cả file trung gian ở root + `front/` + `chapters/`.

**Q: Tôi muốn thêm chương mới (ví dụ Chương 6 — Tổng quan)?**
A: Tạo file mới `chapters/Chuong06_TongQuanMoi.tex`, rồi thêm `\input{chapters/Chuong06_TongQuanMoi}` vào `main.tex`.

**Q: Tôi muốn restore lại main.tex cũ (2601 dòng)?**
A: File backup ở `main.tex.bak_pre_template` (chỉ có nếu chưa bị `./build.sh clean` xóa).
