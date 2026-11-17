# Checklist kiểm tra inference FAS

Mục tiêu là xác định vì sao GAC-FAS, MiniFASNetV2 và MN3 cho kết quả thấp trên tập public gộp nhiều nguồn. Checklist bám theo repo [Munfond/dmfc-fas-benchmark](https://github.com/Munfond/dmfc-fas-benchmark). Đường dẫn dự kiến trong repo là `evaluation/INFERENCE_AUDIT.md`.

## 1. Chốt cấu hình trước khi chạy lại

- [ ] Ghi commit, lệnh chạy, checkpoint và SHA-256, manifest và SHA-256, protocol GAC, detector, backend, thiết bị và đơn vị đánh giá (ảnh hoặc clip). Đối chiếu với `run.json`.
- [ ] Dùng `raw` cho ảnh nguyên khung có detector YuNet và `cropped` cho ảnh đã crop. Nếu dữ liệu chứa cả hai kiểu, chạy riêng theo kiểu đầu vào. Kiểm tra trực quan một số crop live và spoof của từng nguồn, gồm biên crop, padding, orientation và trường hợp không tìm được mặt.
- [ ] Đối chiếu nhãn và nhóm tấn công với metadata gốc. Giữ frame cùng clip và bản sao do resize hoặc encode lại trong cùng split. Kiểm tra ID người, phiên quay và vật mẫu theo protocol đã chốt.
- [ ] Ghi overlap theo cặp checkpoint–dataset. MN3 đã train trên CelebA-Spoof. GAC-FAS gốc dùng nguồn theo từng config và paper kiểm tra 3D mask ở thiết lập unseen attack. Nguồn train của checkpoint MiniFASNetV2 cần xác minh. [Sổ checkpoint](https://github.com/Munfond/dmfc-fas-benchmark/blob/main/evaluation/checkpoints.example.csv) · [Paper GAC-FAS](https://openaccess.thecvf.com/content/CVPR2024/papers/Le_Gradient_Alignment_for_Cross-Domain_Face_Anti-Spoofing_CVPR_2024_paper.pdf)

## 2. Đối chiếu từng adapter với inference gốc

Lấy một nhóm ảnh có live, print, replay và 3D mask nếu có. Đầu tiên, chạy **cùng checkpoint và cùng crop** qua adapter và code tác giả. So tensor đầu vào, logits hoặc xác suất, và điểm theo từng ảnh. Sau đó, chạy lại từ ảnh nguyên khung để kiểm tra mức chênh lệch do detector hoặc crop.

| Model | Các điểm cần kiểm |
|---|---|
| **GAC-FAS** | RGB 256×256, chuẩn hóa ImageNet. Head một logit như `ICM2O` trả logit live `z` theo upstream, adapter dùng `sigmoid(-z)` làm điểm spoof. Head hai logit như `OMI2C` lấy softmax cột 0 làm điểm spoof. So đúng config và tách khác biệt giữa crop YuNet của benchmark với crop MTCNN trong paper. [Adapter](https://github.com/Munfond/dmfc-fas-benchmark/blob/main/research_models.py#L89) · [Upstream](https://github.com/BOVIFOCR/gacfas/blob/main/helpers/loops.py#L182) |
| **MiniFASNetV2** | Checkpoint `2.7_80x80_MiniFASNetV2`, đầu vào BGR 80×80 dạng float theo thang byte, điểm spoof bằng `1 - p(class 1)`. Raw mode mở rộng bbox theo scale 2.7. Dùng riêng checkpoint V2 khi so điểm vì demo upstream mặc định có thể gộp nhiều checkpoint và lấy argmax. [Adapter](https://github.com/Munfond/dmfc-fas-benchmark/blob/main/benchmark.py#L278) · [Demo upstream](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/test.py#L34) |
| **MN3** | ONNX gốc nhận RGB 128×128, chuẩn hóa bằng mean và scale theo đơn vị pixel, trả hai xác suất `[live, spoof]`. Lấy cột 1. Kiểm tra không đảo kênh, chia 255, chuẩn hóa hoặc áp softmax thêm. Với bản OpenVINO đã chuyển đổi, đối chiếu hợp đồng đầu vào BGR của bản đó. [Adapter](https://github.com/Munfond/dmfc-fas-benchmark/blob/main/benchmark.py#L260) · [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/anti-spoof-mn3/README.md) |

- [ ] Cả ba adapter dùng **điểm cao = spoof**. Khi tính ROC-AUC từ `spoof_score`, đặt **spoof là lớp dương**.
- [ ] Ghi điểm upstream, điểm adapter và độ lệch theo ảnh. Nếu điểm lệch rõ trên cùng crop hoặc thứ tự điểm bị đảo, khoanh vùng lỗi trước khi chỉnh ngưỡng.

## 3. Tái chạy benchmark

~~~bash
python benchmark.py --models mini --models mn3 fetch

python benchmark.py --models gac --models mini --models mn3 run \
  --gac-checkpoint weights/gac.pth \
  --manifest /path/to/manifest.csv \
  --input-mode raw \
  --detector-weights /path/to/face_detection_yunet_2023mar.onnx \
  --bpcer 0.05 \
  --output runs/inference-audit
~~~

Nếu mọi ảnh đã crop, dùng `--input-mode cropped` và bỏ `--detector-weights`. Mức 5% là mục tiêu BPCER mặc định trên calibration. Khi tái lập một lần chạy cũ, dùng lại cấu hình của lần đó trước.

- [ ] Từ `predictions.csv` hoặc `video_predictions.csv`, tính ROC-AUC theo từng nguồn có đủ live và spoof. So với AUC gộp. Nếu dùng video, giữ cùng đơn vị clip khi so model. Repo lấy median của các frame chấm được.
- [ ] Từ `metrics.csv` hoặc `video_metrics.csv`, báo BPCER và APCER theo nguồn và theo loại print, replay, cutout, transparent, plaster, resin nếu có. Báo kèm số lỗi, số mẫu chấm được và coverage. Giữ mask chưa rõ vật liệu thành nhóm riêng.
- [ ] Ghi ngưỡng riêng của mỗi model và mỗi đơn vị ảnh/clip, số mẫu live dùng calibration và BPCER quan sát trên test. Giữ ngưỡng cố định khi mở test.

## 4. Đọc kết quả

| Quan sát | Việc cần kiểm tiếp |
|---|---|
| Điểm lệch upstream trên cùng crop | So từng bước tiền xử lý, checkpoint và mapping lớp. |
| AUC theo nguồn ổn nhưng lỗi tăng ở ngưỡng đã chọn | So phân phối điểm theo nguồn và độ lệch giữa calibration với test. |
| Print/replay ổn nhưng 3D mask tệ | So APCER theo vật liệu tại cùng ngưỡng của từng model. Đối chiếu dữ liệu train của đúng checkpoint. |
| Tỷ lệ lỗi thấp nhưng coverage thấp | Kiểm tra detector/crop và tỷ lệ ảnh hoặc clip không chấm được. |

**Đầu ra để review** gồm `run.json`, bảng so điểm adapter với upstream, và bảng chỉ số theo model × nguồn × loại tấn công. Khi chia sẻ công khai, dùng kết quả tổng hợp hoặc ID đã ẩn danh.
