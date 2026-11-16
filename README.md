# Benchmark face anti-spoofing RGB

Benchmark inference từ checkpoint pretrained cho DMFC-FAS, MobileNetV3 và MiniFASNetV2. Có adapter cho GAC-FAS, MVP-FAS và FLIP-MCL. Mọi model dùng cùng manifest, với score càng cao càng nghiêng về spoof. Kết quả có cả theo ảnh và theo clip, phục vụ so sánh ảnh in, replay và mặt nạ 3D.

## 1. Chọn dữ liệu

| Nguồn | Vai trò | Dữ liệu | Lệnh setup |
|---|---|---|---|
| Minivision demo | Kiểm tra cài đặt và inference | 3 ảnh từ repo tác giả | `python setup_data.py smoke` tự tải ảnh |
| CelebA-Spoof | Benchmark ảnh tải từ nguồn chính thức | Print, Replay, Paper Cut, 3D mask chưa phân vật liệu | `python celeba_spoof.py download` rồi `extract` và `prepare` |
| UniAttackData+ physical | Kiểm tra đủ 6 nhóm theo challenge | Print, Replay, Cutouts, Transparent, Plaster, Resin | `python setup_data.py uniattack-plus --root /path/to/UniAttackDataPlus` sau khi nhận dữ liệu và nhãn test |
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

### CelebA-Spoof tải trực tiếp

Bản ECCV 2020 có metadata và split train/test chính thức. Tác giả công bố dữ liệu cho nghiên cứu phi thương mại. Link Drive hiện có 74 phần, tổng khoảng 77 GB trước khi giải nén. Chuẩn bị thêm dung lượng cho dữ liệu đã giải nén và cài 7-Zip (`7z` hoặc `7zz` trong PATH). Xem [trang phát hành](https://github.com/ZhangYuanhan-AI/CelebA-Spoof#dataset-downloads) và [điều khoản](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebA_Spoof.html).

```bash
python -m pip install gdown==6.5.1
python celeba_spoof.py download
python celeba_spoof.py extract
python celeba_spoof.py prepare --root data/celeba-spoof-raw/CelebA_Spoof --metadata-only
python celeba_spoof.py prepare --root data/celeba-spoof-raw/CelebA_Spoof
python benchmark.py run --manifest data/celeba-spoof/manifest.csv \
  --input-mode raw --detector-weights /path/to/face_detection_yunet_2023mar.onnx \
  --output runs/celeba-spoof
```

`--root` trỏ đến thư mục chứa `Data/train/`, `Data/test/` và `metas/intra_test/`. Nếu đã tải các phần ZIP từ trang tác giả, dùng `extract --parts /path/to/parts`. Sau khi một lượt tải bị gián đoạn, chạy lại `download` để tiếp tục. Nếu giải nén lỗi giữa chừng, dùng thư mục output trống cho lượt kế tiếp.

Script giữ nguyên test theo JSON chính thức, loại ID người xuất hiện trong test khỏi các ứng viên reference/calibration, và lấy một ảnh bona fide cho mỗi người được chọn từ train. Metadata đã kiểm tra có 3 ID xuất hiện ở cả hai file nhãn chính thức. Script dùng nhãn JSON thay cho tên thư mục để xác định live/spoof. Mặc định có 50 người làm reference và 200 người khác làm calibration, có thể đổi bằng `--reference-subjects` và `--calibration-subjects`. Đọc `metrics.csv` ở cấp ảnh. MN3 đã train trên CelebA-Spoof, nên báo kết quả của checkpoint này là in-dataset. Dùng cùng detector để so sánh các model và báo kèm coverage.

Nhóm Paper Cut được xuất là `cutout` theo taxonomy của CelebA-Spoof. Nhãn 3D Mask được xuất là `3d_mask_unspecified` vì metadata không phân biệt Transparent, Plaster và Resin. Để kiểm tra manifest bất kỳ theo tiêu chí sáu nhóm UniAttackData+, chạy:

```bash
python setup_data.py check-physical --manifest /path/to/manifest.csv --min-test-per-category 1
```

Lệnh báo số mẫu và trả mã lỗi nếu thiếu bona fide hoặc nhóm vật lý. Nó đếm mỗi `video_id` một lần trong từng nguồn, và đếm theo ảnh nếu thiếu ID video. Mức 1 chỉ kiểm tra nhóm có mặt. Để đánh giá, chốt số mẫu tối thiểu trước và đối chiếu độ đa dạng theo người, phiên quay và vật mẫu. CelebA-Spoof sẽ báo thiếu ba nhóm vật liệu 3D khi chạy kiểm tra này.

### UniAttackData+ cho đủ sáu nhóm physical

[Xin bộ dữ liệu và nhãn subtype từ ban tổ chức](https://sites.google.com/view/face-anti-spoofing-challenge/dataset-download/uniattackdata-iccv2025). Dùng thư mục có `Data-train`, `Data-val`, `Data-test` và ba file `Protocol-{train,val,test}.txt`. Mỗi dòng cần đường dẫn tương đối và mã nhãn. Nếu file test của challenge chỉ có đường dẫn, cung cấp file test có nhãn do tác giả cấp bằng `--test-protocol /path/to/labeled-test.txt`. Xem [quy ước chuẩn bị dữ liệu của DMFC-FAS](https://github.com/wangjl1993/DMFC-FAS/tree/master/prepare_data).

```bash
python setup_data.py uniattack-plus --root /path/to/UniAttackDataPlus --metadata-only
python setup_data.py uniattack-plus --root /path/to/UniAttackDataPlus
```

Script giữ split nguồn, lấy bona fide từ Train làm reference, dùng Val cho calibration và bỏ các mẫu digital. Test phải có bona fide và đủ sáu nhãn spoof. Có thể tăng `--min-test-per-category` để yêu cầu nhiều mẫu hơn mỗi nhóm. Đếm theo dòng protocol. Nếu bản phát hành gồm ảnh rời mà thiếu ID video/người, số dòng không thể dùng làm số video/người độc lập.

| Mã UniAttackData+ | Nhãn xuất |
|---|---|
| `1_0_0` | `print` |
| `1_0_1` | `replay` |
| `1_0_2` | `cutout` |
| `1_1_0` | `3d_mask_transparent` |
| `1_1_1` | `3d_mask_plaster` |
| `1_1_2` | `3d_mask_resin` |

Sau khi setup, dùng `data/uniattack-plus/manifest.csv` với `benchmark.py run`. Chọn `--input-mode raw` và YuNet nếu đầu vào là ảnh nguyên khung, hoặc `--input-mode cropped` nếu đã crop mặt. Với DMFC-FAS, báo đây là đánh giá trong dataset đã dùng để huấn luyện. CASIA-SURF, CeFA và HiFiMask có đóng góp mẫu vào UniAttackData+, nên cần kiểm tra trùng mẫu nếu muốn dùng chúng làm test ngoài tập cho checkpoint này. Xem [báo cáo challenge](https://openaccess.thecvf.com/content/ICCV2025W/FAS2025/html/Ma_UniAttackData_Unified_Physical-Digital_Attack_Detection_Challenge_ICCVW_2025_paper.html).

### Tạo tập test cân bằng tối đa 5 GB

Chuẩn bị một manifest ứng viên có bona fide và đủ sáu nhãn trong bảng UniAttackData+ ở trên. Với dữ liệu tự thu hoặc bộ được cấp quyền, dùng [inventory mẫu](evaluation/inventory.example.csv) và `prepare_eval.py` để nhập nhãn đã xác minh. Điền `subject_id` cho mỗi dòng và `mask_id` cho mỗi mẫu mặt nạ 3D. Dùng cùng một `mask_id` cho mọi lần xuất hiện của cùng vật mẫu. Dùng `video_id` nếu nhiều frame đến từ cùng clip và điền `session_id` nếu metadata có phiên quay. Với nguồn đã crop, dùng cùng chế độ đầu vào cho lần benchmark.

Importer `uniattack-plus` đọc nhãn chính thức. Khi file protocol thiếu ID người, clip hoặc vật mẫu, bổ sung từ metadata xác thực trước khi xuất subset. Nếu chỉ có tên file ẩn danh, bạn có thể chạy chấm theo nhóm trên split chính thức, còn bước xuất subset cân bằng cần ID người và vật mẫu để kiểm tra đa dạng.

```bash
python prepare_physical_subset.py --manifest data/custom/manifest.csv \
  --output data/physical-test --per-class 100 --min-subjects 20 --min-masks 5 --seed 42

python setup_data.py check-physical --manifest data/physical-test/manifest.csv \
  --min-test-per-category 100
```

- Mặc định lấy **100 lượt trình diện mỗi nhóm**, gồm 100 bona fide và 600 spoof, với ít nhất 20 ID người mỗi nhóm và 5 vật mẫu cho từng nhóm mặt nạ 3D. Một ảnh là một lượt khi thiếu `video_id`. Với dữ liệu trích từ video, luôn giữ ID clip để đếm đúng. Các ngưỡng mặc định là cấu hình đề xuất cho lần so sánh đầu, có thể đổi theo dữ liệu hoặc sai số mong muốn.
- Script giữ nguyên split, ưu tiên các nguồn, ID người và vật mẫu ít được đại diện trong mỗi nhóm, đồng thời kiểm tra ID bị chia qua nhiều split trong từng nguồn và file trùng byte. Nếu biết một người hoặc vật mẫu trùng giữa các nguồn, ghi `group_id` chung ở inventory để `prepare_eval.py` kiểm tra trước. Giữ số frame mỗi clip nhất quán khi chuẩn bị manifest.
- Giới hạn **5.000.000.000 byte** áp dụng cho thư mục xuất, gồm dữ liệu test, manifest và `distribution.json`. Reference/calibration vẫn được đọc từ đường dẫn gốc. Nếu vượt ngân sách, giảm số frame/clip hoặc giảm `--per-class` theo cùng một mức cho mọi nhóm. Đối chiếu số clip, ảnh, người, phiên và vật mẫu theo nhóm trong báo cáo trước khi chạy.
- Đây là phân phối **cân bằng để so model**, với tỷ lệ live/spoof 1:6. Báo APCER riêng từng nhóm và BPCER. Tỷ lệ tấn công khi triển khai cần một phép đo riêng. Trùng dữ liệu với checkpoint vẫn cần kiểm tra theo nguồn train và nguồn chọn trọng số.

Chạy `benchmark.py run` với `--manifest data/physical-test/manifest.csv` và chế độ crop tương ứng. Với dữ liệu có nhiều frame mỗi clip, đọc `video_metrics.csv` và coverage. Giữ manifest cố định cho mọi model.

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
