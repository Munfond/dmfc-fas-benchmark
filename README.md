# Benchmark face anti-spoofing RGB

Benchmark inference từ checkpoint pretrained cho DMFC-FAS, MobileNetV3 và MiniFASNetV2. Có adapter cho GAC-FAS, MVP-FAS và FLIP-MCL. Mọi model dùng cùng manifest, với score càng cao càng nghiêng về spoof. Kết quả có cả theo ảnh và theo clip, phục vụ so sánh ảnh in, replay và mặt nạ 3D.

## 1. Chọn dữ liệu

| Nguồn | Vai trò | Dữ liệu | Lệnh setup |
|---|---|---|---|
| Minivision demo | Kiểm tra cài đặt và inference | 3 ảnh từ repo tác giả | `python setup_data.py smoke` tự tải ảnh |
| WMCA RGB Grandtest | Benchmark chính cho mặt nạ 3D | RGB, bona fide, ảnh in, replay, mặt nạ cứng/mềm, mặt nạ giấy, đầu mannequin và kính ngụy trang | `python setup_data.py wmca --root /path/to/preprocessed-face-station_RGB` sau khi tải bản RGB |
| Replay-Attack Grandtest | Benchmark bổ sung cho ảnh in và màn hình | Video webcam, bona fide, ảnh in, replay điện thoại/tablet | `python setup_data.py replay-attack --root /path/to/replayattack` sau khi tải video |
| Dữ liệu riêng, ví dụ ZED Mini | Kiểm tra khả năng chuyển sang camera robot | Ảnh hoặc video có nhãn và ID nhóm | `prepare_eval.py` với inventory của bạn |

**WMCA và Replay-Attack là bộ dữ liệu nghiên cứu có metadata công khai và file giới hạn truy cập.** Xin quyền qua [WMCA](https://zenodo.org/records/4580313) và [Replay-Attack](https://zenodo.org/records/4593128). Sau khi được duyệt, tải archive chính thức và giải nén vào máy chạy benchmark. Code và tài liệu nằm trong repo, còn ảnh/video giữ trong máy bạn. Xem [quy ước thư mục, split và kiểm tra lỗi](evaluation/README.md).

## 2. Cài môi trường và chạy thử

Cần Python 3.10+. Ví dụ dùng CPU trên Linux x86-64.

```bash
git clone https://github.com/Munfond/dmfc-fas-benchmark.git
cd dmfc-fas-benchmark
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python benchmark.py fetch
python setup_data.py smoke
python benchmark.py run --manifest data/smoke/manifest.csv \
  --input-mode cropped --output runs/smoke
```

`fetch` tải trọng số và code kiến trúc MiniFASNet đã ghim phiên bản, có xác minh SHA-256. Smoke có một ảnh live để tạo reference DMFC và hai ảnh spoof để kiểm tra luồng chấm điểm. Các chỉ số cần ngưỡng sẽ để trống vì smoke thiếu calibration. Trên Jetson, dùng bản PyTorch và torchvision tương thích JetPack. Mặc định inference chạy CPU. Thêm `--device cuda` sau `run` để chạy các model PyTorch trên GPU, còn MN3 và YuNet chạy CPU.

## 3. Chuẩn bị benchmark và chạy inference

### WMCA RGB cho 3D mask

```bash
python setup_data.py wmca --metadata-only
python -m pip install h5py
python setup_data.py wmca --root /path/to/preprocessed-face-station_RGB --frames-per-clip 8
python benchmark.py run --manifest data/wmca/manifest.csv \
  --input-mode cropped --output runs/wmca
```

Metadata chính thức được tải tự động và kiểm tra SHA-256. Script giữ split Grandtest, khớp clip theo thư mục ngày và tên HDF5, rồi trích tối đa 8 frame mỗi clip. Train bona fide cung cấp 124 clip reference. Dev có 557 clip và Eval có 557 clip, gồm 115 bona fide và 442 spoof. Eval có 137 clip mặt nạ mềm và 49 clip mặt nạ cứng. Xem [protocol của Idiap](https://www.idiap.ch/en/dataset/wide-multi-channel-presentation-attack-wmca).

### Replay-Attack cho print và replay

Giải nén bản chính thức. `--root` trỏ tới thư mục chứa trực tiếp `train/`, `devel/` và `test/`. Tải [YuNet ONNX](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) bản `face_detection_yunet_2023mar.onnx` cho ảnh nguyên khung.

```bash
python setup_data.py replay-attack --root /path/to/replayattack --frames-per-clip 8
python benchmark.py run --manifest data/replay-attack/manifest.csv \
  --input-mode raw --detector-weights /path/to/face_detection_yunet_2023mar.onnx \
  --output runs/replay-attack
```

Script kiểm tra split và số clip Grandtest trước khi trích frame. Train dùng 60 bona fide làm reference. Devel có 360 clip và Test có 480 clip. Lưu ý checkpoint GAC trong [sổ theo dõi](evaluation/checkpoints.example.csv) đã dùng Replay-Attack khi train.

### Dữ liệu tự thu hoặc tổng hợp

Sao chép [inventory mẫu](evaluation/inventory.example.csv), điền đường dẫn ảnh/video cùng nguồn, label, loại tấn công và ID người, phiên, clip hoặc mặt nạ. Khai báo split chính thức nếu có.

```bash
python prepare_eval.py --inventory /path/to/inventory.csv \
  --manifest data/custom/manifest.csv --frames-per-video 8
```

Dùng `--holdout-source TEN_DATASET` để dành toàn bộ một nguồn cho test khi reference/calibration đến từ nguồn khác. Nếu ảnh đã crop, chọn `--input-mode cropped`. Nếu là video hay ảnh nguyên khung, dùng cấu hình YuNet như ví dụ Replay-Attack. Raw mode căn landmark cho DMFC, crop bbox cho MN3 và mở rộng bbox 2.7 cho MiniFASNetV2. Crop sẵn quá sát có thể làm mất ngữ cảnh MiniFASNet cần. Tiền xử lý có thể khác code tác giả.

## 4. Split, ngưỡng và cách đọc kết quả

| Split | Cách dùng |
|---|---|
| `reference` | Ảnh bona fide tạo vector tham chiếu DMFC |
| `calibration` | Dùng score bona fide để chọn ngưỡng riêng cho từng model và từng đơn vị ảnh/clip |
| `test` | Chấm kết quả bằng ngưỡng đã chốt. Cần nhãn để tính APCER/BPCER |

Mục tiêu mặc định là BPCER calibration 5%, chỉnh bằng `--bpcer` sau `run`. Score clip là median của frame chấm được. **APCER** là tỷ lệ mẫu spoof bị nhận nhầm thành live. **BPCER** là tỷ lệ mẫu live bị nhận nhầm thành spoof. Đọc `coverage` cùng các tỷ lệ lỗi vì clip không có frame chấm được nằm ngoài mẫu số tính lỗi.

| File | Nội dung |
|---|---|
| `video_metrics.csv` | BPCER và APCER theo nguồn và loại tấn công ở cấp clip. Tìm `3d_mask_flexible` và `3d_mask_rigid` khi chạy WMCA. |
| `video_predictions.csv` | Score, dự đoán và số frame chấm được mỗi clip |
| `metrics.csv`, `predictions.csv` | Kết quả ở cấp ảnh/frame |
| `calibration_metrics.csv` | Chỉ số trên tập calibration |
| `run.json` | Cấu hình, checksum, ngưỡng và reference dùng trong lần chạy |
| `dmfc_reference.npy` | Vector tham chiếu khi chạy DMFC |

Để so sánh checkpoint, giữ cùng manifest, detector, chế độ crop và số frame. Kiểm tra tập train, nguồn dùng để chọn checkpoint, và trùng người/clip/mặt nạ. CelebA-Spoof đã dùng để train MN3. Nguồn train MiniFASNetV2 chưa xác minh. WMCA từng là target công bố của checkpoint MVP được gợi ý. Một split chuẩn giúp tái lập phép đo, còn trạng thái độc lập phụ thuộc từng checkpoint. Xem [sổ theo dõi checkpoint](evaluation/checkpoints.example.csv).

## 5. Baseline mặc định

| CLI | Model | Score |
|---|---|---|
| `dmfc` | DMFC-FAS ResNet34, epoch 150, seed 42 | Khoảng cách từ embedding đến reference bona fide. Dùng `--no-tta` sau `run` để tắt ảnh lật. |
| `mn3` | Open Model Zoo anti-spoof-mn3 | Xác suất spoof từ ONNX, ảnh RGB 128 × 128 |
| `mini` | MiniFASNetV2 scale 2.7 | Tổng xác suất hai lớp spoof, ảnh BGR 80 × 80 |

Để chạy riêng một model, đặt `--models` trước `run`, ví dụ `python benchmark.py --models mn3 run ...`.

## 6. Model nghiên cứu tùy chọn

Ba baseline cũ vẫn là mặc định. Dùng các tham số `--models` lặp lại để chọn thêm model. Các adapter dùng crop bbox chung của YuNet ở raw mode, hoặc dùng crop bạn cung cấp. Vì vậy kết quả trên tập của bạn có thể khác số công bố dùng preprocessing của tác giả.

| CLI | Model | Trọng số và mã nguồn |
|---|---|---|
| `gac` | GAC-FAS, CVPR 2024, ResNet18 RGB | Lấy checkpoint theo protocol từ mục pretrained weights trong [repo tác giả](https://github.com/leminhbinh0209/CVPR24-FAS#31-running). |
| `mvp` | MVP-FAS, ICCV 2025, CLIP ViT-B/16 + slot attention | Lấy `MVP_FAS_P2_W.pth` từ [Protocol 2](https://drive.google.com/drive/folders/15Sa3P-AaHGmedEtyEs9Tv-aVQsxnQ2dK) cho cấu hình tác giả train SURF/CeFA và test WMCA. [Code tác giả](https://github.com/Elune001/MVP-FAS). |
| `flip-mcl` | FLIP-MCL, ICCV 2023, CLIP image–text | Chọn checkpoint đã xác minh là bản MCL từ [Model Zoo](https://github.com/koushiksrivats/FLIP/blob/main/docs/model_zoo.md) hoặc từ lần train chính thức. Adapter kiểm tra có state của nhánh `image_mlp` để tránh nạp nhầm FLIP-V/IT. |

`fetch` tự tải ba baseline mặc định. Với các checkpoint nghiên cứu, dùng link tác giả ở bảng trên và lưu file local. Nên chạy model CLIP offline trên máy benchmark có đủ bộ nhớ, hoặc chạy từng model với cùng manifest vào thư mục kết quả riêng. PyTorch 2.6+ được khuyến nghị cho các adapter nghiên cứu vì chúng dùng trình nạp `weights_only`.

```bash
python -m pip install -r requirements-research.txt
git clone https://github.com/Elune001/MVP-FAS.git vendor/mvp
git -C vendor/mvp checkout fad83b94a326cec1925b73b43fb27708f1bdcf23
mkdir -p weights
curl -fL https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt \
  -o weights/ViT-B-16.pt

python benchmark.py --models dmfc --models mn3 --models mini --models mvp run \
  --mvp-checkpoint weights/MVP_FAS_P2_W.pth \
  --mvp-repo vendor/mvp --clip-weights weights/ViT-B-16.pt \
  --manifest /path/to/manifest.csv --input-mode cropped --output runs/with_mvp

python benchmark.py --models gac run --gac-checkpoint weights/gac.pth \
  --manifest /path/to/manifest.csv --input-mode cropped --output runs/gac

git clone https://github.com/koushiksrivats/FLIP.git vendor/flip
git -C vendor/flip checkout 4f95def259e135a0cbaff1d770f559ca739c4c9f

python benchmark.py --models flip-mcl run --flip-repo vendor/flip \
  --flip-mcl-checkpoint weights/flip-mcl.pth \
  --manifest /path/to/manifest.csv --input-mode cropped --output runs/flip
```

Giữ checkout upstream sạch. CLI lưu checksum của các checkpoint nghiên cứu vào `run.json`. Đọc [sổ theo dõi nguồn train](evaluation/checkpoints.example.csv) trước khi diễn giải kết quả cross-dataset.

## Nguồn

- [DMFC-FAS chính thức](https://github.com/wangjl1993/DMFC-FAS/tree/ce33f053c9b9c9ca2d9daac9f9e35086a7a1198e)
- [Đặc tả anti-spoof-mn3](https://docs.openvino.ai/2023.3/omz_models_model_anti_spoof_mn3.html)
- [Minivision Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/tree/b6d5f04ad78778917853b25c778acef6d5626d15)
