# LaTeX Paper Source (`paper/`) — Springer 1-Column Overleaf Package

Thư mục này chứa toàn bộ mã nguồn bài báo khoa học định dạng **Springer Nature 1 cột (`sn-jnl`, numbered `sn-mathphys-num`)** viết bằng tiếng Anh dựa trên toàn bộ số liệu thực nghiệm Phase 1–6.

## Cấu trúc thư mục
- `main.tex`: Toàn văn bản thảo (`sn-jnl` 1 cột).
- `main.pdf`: Bản PDF đã biên dịch hoàn chỉnh (18 trang, không lỗi LaTeX/BibTeX) để review nhanh.
- `references.bib`: Toàn bộ tài liệu tham khảo BibTeX đã xác thực.
- `sn-jnl.cls` & `sn-mathphys-num.bst`: Bộ class và BibTeX style chính thức của Springer Nature (LPPL 1.3c).
- `figures/`: 8 hình ảnh độ phân giải cao (kiến trúc pipeline, biểu đồ tối ưu RPi 4/3/2, Accuracy-Latency, Robustness Degradation, 3 Radar Levels 1–3, lát cắt Source B).

## Cách mở và biên dịch trên Overleaf hoặc Docker TeX Live
Toàn bộ template Springer Nature đã được đóng gói sẵn trong thư mục `paper/`:
1. **Trên Overleaf:** Nén thư mục `paper/` (hoặc chọn `main.tex`, `references.bib`, `sn-jnl.cls`, `sn-mathphys-num.bst`, và `figures/`) upload trực tiếp lên Overleaf → **Upload Project**, chọn Compiler = **`pdfLaTeX`** và bấm **Recompile**.
2. **Tại local qua Docker (`texlive/texlive:latest-small`):**
   ```powershell
   docker run --rm -v "${PWD}/paper:/workdir" -w /workdir texlive/texlive:latest-small bash -c "pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex"
   ```
   *(Yêu cầu các gói TeX: `ncctools`, `sttools`, `threeparttable`, `crop`, `appendix`, `float`, `jknapltx`, `wrapfig`, `multirow`.)*

## Lưu ý điền thông tin tác giả
Trong `main.tex`, phần `\author` và `\affil` đang để placeholder cho 6 thành viên nhóm nghiên cứu — hãy thay bằng họ tên đầy đủ, email và tên trường/khoa trước khi xuất PDF nộp bài.
