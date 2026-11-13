# Benchmark DMFC-FAS trên tập ảnh gộp

CLI chạy ba model RGB trên cùng manifest và xuất `spoof_score` theo chiều **điểm càng cao, ảnh càng giống spoof**. Các model dùng checkpoint đã huấn luyện. DMFC-FAS cần ảnh mặt thật tham chiếu để tính điểm cosine.

| Tên CLI | Model | Cách lấy điểm |
|---|---|---|
| `dmfc` | DMFC-FAS ResNet34 epoch 150, seed 42 | `1 - clip(cosine, 0, 1)` theo code tác giả. Ảnh gốc và ảnh lật ngang được gộp ở mức embedding khi bật TTA. |
| `mn3` | MobileNetV3 anti-spoof-mn3 của Open Model Zoo, train trên CelebA-Spoof | Xác suất class spoof từ ONNX, input 128 × 128 RGB. |
| `mini` | MiniFASNetV2 80 × 80, checkpoint scale 2.7 của Minivision | Tổng xác suất hai lớp spoof, tương đương `1 - P(live)`. Checkpoint V2 nhận BGR dạng float trong khoảng 0–255 theo code Minivision. |

## 1. Cài môi trường và tải model

Python 3.10 trở lên. Trên Linux x86-64, có thể bắt đầu với bản PyTorch CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python benchmark.py fetch
```

Trên Jetson, dùng cặp PyTorch/torchvision phù hợp với JetPack và môi trường của máy. Lệnh cài PyTorch phía trên dành cho Linux x86-64. CLI mặc định chạy CPU. `--device cuda` áp dụng cho DMFC-FAS và MiniFASNetV2, còn MN3 dùng ONNX Runtime CPU. Lệnh `fetch` tải trọng số và mã kiến trúc Minivision đã ghim phiên bản, rồi xác minh SHA-256. Chỉ bước này cần mạng nếu đã có các file đầu vào và dependencies ở máy.

## 2. Chuẩn bị manifest

Điền đường dẫn tương đối từ vị trí CSV hoặc đường dẫn tuyệt đối. Dùng `reference` cho ảnh mặt thật dùng tạo vector DMFC, `calibration` cho ảnh chọn ngưỡng và `test` cho ảnh đánh giá.

```csv
image_path,source_dataset,split,label,attack_type,video_id
images/ref1.jpg,source_a,reference,live,,ref_01
images/cal1.jpg,source_a,calibration,live,,cal_01
images/test1.jpg,source_a,test,live,,test_01
images/mask1.jpg,source_b,test,spoof,3d_mask,test_02
images/replay1.jpg,source_b,test,spoof,replay,test_03
```

Ảnh test có thể để trống nhãn. Các cột `subject_id`, `session_id`, `mask_id` và `video_id` là tùy chọn. Dùng ID video ổn định và giữ mọi frame từ cùng clip trong cùng split. Nếu dataset nguồn có ID video trùng giữa các thư mục, dùng đường dẫn video tương đối làm ID. Script chặn trùng đường dẫn ảnh và chặn một ID video xuất hiện ở nhiều split trong cùng dataset. Kiểm tra thêm ảnh gần trùng, trùng người hoặc trùng mặt nạ giữa các nguồn trước khi đánh giá.

## 3. Chạy so sánh

Với dữ liệu đã crop mặt, từ thư mục chứa `benchmark.py` chạy:

```bash
python benchmark.py run \
  --manifest /path/to/manifest.csv \
  --input-mode cropped \
  --output runs/public_mix \
  --bpcer 0.05
```

Ở chế độ này cả ba model lấy ảnh crop đã có làm đầu vào. MiniFASNetV2 nhận toàn bộ crop và không thể khôi phục vùng ngữ cảnh ngoài crop. Nếu ảnh đã bị crop sát mặt, lưu ý điều này khi đọc kết quả của MiniFASNetV2.

Với ảnh nguyên khung, lấy [YuNet ONNX](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) bản `face_detection_yunet_2023mar.onnx`, rồi chạy:

```bash
python benchmark.py run \
  --manifest /path/to/manifest.csv \
  --input-mode raw \
  --detector-weights /path/to/face_detection_yunet_2023mar.onnx \
  --output runs/public_mix_raw
```

Raw mode dùng chung detector cho cả ba model, căn landmark cho DMFC-FAS, crop bbox cho MN3 và mở rộng bbox theo scale 2.7 cho MiniFASNetV2. Đây là cấu hình benchmark dùng YuNet. Repo DMFC-FAS gốc dùng InsightFace để tạo crop. Với ảnh raw, ảnh không thấy mặt hoặc có nhiều mặt vẫn xuất dòng trạng thái. Có thể chạy riêng một model bằng `python benchmark.py --models dmfc run --manifest ... --input-mode cropped --output ...`. Dùng `--no-tta` sau `run` để tắt lượt ảnh lật của DMFC.

## 4. Đọc kết quả

| File | Nội dung |
|---|---|
| `predictions.csv` | Mỗi cặp ảnh và model có score, trạng thái, ngưỡng, dự đoán và thời gian xử lý. Có cả ảnh calibration và test. |
| `metrics.csv` | Chỉ tính trên test có nhãn, gồm BPCER, APCER gộp và APCER theo loại tấn công. Cột nguồn trống chỉ kết quả gộp qua các nguồn. `rate` và `coverage` là tỷ lệ 0–1. |
| `run.json` | Checksum asset và manifest, thiết bị, ngưỡng, số mẫu calibration và danh sách ảnh tham chiếu đã dùng. |
| `dmfc_reference.npy` | Vector tham chiếu đã dùng khi chạy DMFC-FAS. |

- Ngưỡng riêng cho mỗi model được chọn từ **cùng tập calibration live** để BPCER trên calibration không vượt mục tiêu. Khi chưa có ảnh calibration live chấm được, CLI vẫn xuất score và để trống dự đoán cùng metric phụ thuộc ngưỡng.
- So **APCER của 3D mask** và BPCER thực tế trên test. `scored` là mẫu số dùng tính tỷ lệ lỗi. `total` gồm cả ảnh không chấm được, vì vậy xem thêm `coverage`. Mỗi hàng là một ảnh. Khi có nhiều frame trên mỗi clip, kết quả theo ảnh có thể chịu ảnh hưởng của số frame.
- `preprocess_ms` là thời gian chung để đọc ảnh và chuẩn bị crop. `model_ms` gồm resize/chuẩn hóa riêng và chấm điểm. Ảnh đầu có thể bao gồm thời gian khởi động. Đo cùng thiết bị và cấu hình nếu cần so độ trễ.
- Nếu tập test có CelebA-Spoof, ghi rõ đây là dataset đã dùng để train MN3. Khi đánh giá khả năng chuyển dataset, xem riêng các nguồn độc lập với tập train và giữ reference/calibration ngoài nguồn test theo protocol đã chọn.

## Nguồn

- [DMFC-FAS chính thức](https://github.com/wangjl1993/DMFC-FAS/tree/ce33f053c9b9c9ca2d9daac9f9e35086a7a1198e)
- [Đặc tả anti-spoof-mn3](https://docs.openvino.ai/2023.3/omz_models_model_anti_spoof_mn3.html)
- [Minivision Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/tree/b6d5f04ad78778917853b25c778acef6d5626d15)

## Chuẩn bị dữ liệu và chấm theo clip

Dùng `python prepare_wmca.py --metadata-only` để tải và kiểm tra metadata Grandtest chính thức. Sau khi Idiap cấp quyền tải WMCA RGB, chạy `prepare_wmca.py --rgb-root /path/to/preprocessed-face-station_RGB` để tự tạo manifest. Xem [hướng dẫn dữ liệu đánh giá](evaluation/README.md) để chạy WMCA hoặc nhập dữ liệu từ nguồn khác. `prepare_eval.py` nhận ảnh hoặc video, chia theo nhóm, lấy frame và tạo báo cáo số mẫu. Có thể dùng `--holdout-source WMCA` để giữ nguyên nguồn đó làm test. Ba ảnh demo chính thức của Minivision có thể tải qua `python prepare_smoke.py` để kiểm tra CLI chạy được.

Ngoài kết quả theo ảnh, CLI xuất `video_predictions.csv`, `video_metrics.csv` và `calibration_metrics.csv`. Điểm video là median của các frame chấm được. Ngưỡng ảnh và ngưỡng video được chọn riêng từ calibration live theo mục tiêu BPCER. Khi thiếu `video_id`, mỗi ảnh là một lần trình diện riêng trong báo cáo video.

## Model nghiên cứu tùy chọn

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
