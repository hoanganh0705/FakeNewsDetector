#!/usr/bin/env bash
# ── Build script cho paper/main.tex ─────────────────────────
# Chạy 4 bước chuẩn của pdflatex + bibtex để PDF ra đúng citation,
# mục lục và danh sách bảng/hình.
#
# Cách dùng:
#     cd paper
#     ./build.sh            # build ra main.pdf
#     ./build.sh clean      # xóa hết file trung gian
#     ./build.sh fullclean  # clean + xóa luôn PDF
# ────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")"

# Danh sách file trung gian cần xóa khi clean.
# Lưu ý: sau khi refactor (2026-09), một số file .aux nằm trong
# front/ và chapters/ nên phải xóa recursive.
AUX_PATTERNS=(
  "main.aux" "main.toc" "main.lof" "main.lot" "main.out"
  "main.bbl" "main.blg" "main.fls" "main.fdb_latexmk" "main.synctex.gz"
  # File cũ (trước refactor)
  "_BCTK.aux" "_BCTK.bbl" "_BCTK.fls" "_BCTK.fdb_latexmk" "_BCTK.lof"
  "_BCTK.log" "_BCTK.lot" "_BCTK.pdf" "_BCTK.synctex.gz" "_BCTK.toc"
  "_.aux" "_.bbl" "_.blg" "_.fdb_latexmk" "_.fls" "_.lof" "_.lot"
  "_.pdf" "_.synctex.gz" "_.toc"
  "_NoiDung.aux" "_NoiDung.fdb_latexmk" "_NoiDung.fls" "_NoiDung.log"
)

case "${1:-build}" in
  clean)
    # Xóa file trung gian ở root
    rm -f "${AUX_PATTERNS[@]}"
    # Xóa file trung gian ở thư mục con (recursive)
    find front chapters -type f \( \
        -name "*.aux" -o -name "*.log" -o -name "*.fls" -o \
        -name "*.fdb_latexmk" -o -name "*.toc" -o -name "*.out" -o \
        -name "*.synctex.gz" \
      \) -delete 2>/dev/null || true
    # Xóa file backup main.tex.bak_pre_template nếu có
    rm -f main.tex.bak_pre_template
    echo "✓ Đã xóa file trung gian."
    ;;
  fullclean)
    ./build.sh clean
    rm -f main.pdf
    echo "✓ Đã xóa cả main.pdf."
    ;;
  build|"")
    # Bước 1: pdflatex lần đầu — sinh .aux cho bibtex
    pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
    # Bước 2: bibtex — đọc refs.bib và main.aux, sinh main.bbl
    bibtex main >/dev/null
    # Bước 3 & 4: pdflatex thêm 2 lần để TOC + citation ổn định
    pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
    pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
    echo "✓ Đã build xong: paper/main.pdf"
    ;;
  *)
    echo "Usage: $0 [build|clean|fullclean]"
    exit 1
    ;;
esac
