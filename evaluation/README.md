# Chuẩn bị dữ liệu đánh giá FAS

## WMCA RGB Grandtest

Xin quyền tại [Zenodo WMCA](https://zenodo.org/records/4580313) và giải nén bản preprocessed RGB. Giữ nguyên tên HDF5 và các thư mục ngày.

```bash
python prepare_wmca.py --metadata-only
python -m pip install h5py
python prepare_wmca.py --rgb-root /path/to/preprocessed-face-station_RGB
python benchmark.py run --manifest data/wmca/manifest.csv --input-mode cropped --output runs/wmca
```

Script tải metadata chính thức và xác minh SHA-256. Train bona fide dùng làm reference cho DMFC (124 clip), Dev dùng để chọn ngưỡng (557 clip) và Eval dùng để chấm (557 clip). Eval có 137 mặt nạ mềm và 49 mặt nạ cứng. Xem APCER riêng cho hai loại trong `video_metrics.csv`. Đã kiểm tra metadata chính thức và HDF5 giả lập. Cần file RGB thật để kiểm tra layout và chạy benchmark thực tế.

## Nguồn khác

Tạo inventory theo `inventory.example.csv`, giữ ID người, phiên, video và mặt nạ.

```bash
python prepare_eval.py --inventory /path/to/inventory.csv --manifest data/public/manifest.csv --frames-per-video 8
```

Với test cross-dataset, dùng `--holdout-source` và lấy reference/calibration từ nguồn khác. Kiểm tra lịch sử train và chọn checkpoint theo từng nguồn bằng `checkpoints.example.csv`.
