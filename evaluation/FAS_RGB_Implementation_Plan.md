# Kế hoạch triển khai FAS trên ảnh RGB

## 1. Mục tiêu và thứ tự triển khai

Đầu vào là một ảnh RGB. Đầu ra là điểm spoof và nhãn live/spoof sau hiệu chỉnh ngưỡng. Chạy local, batch 1, không yêu cầu depth, video hoặc API LLM. Chốt trần **1.000.000.000 byte cho tổng dung lượng trọng số của mỗi cấu hình triển khai**, bao gồm mọi backbone, head và adapter cần để chạy. Đo bộ nhớ đỉnh riêng vì nó còn phụ thuộc phần cứng, runtime và preprocessing.

| Thứ tự | Model | Trọng số và điều kiện để chạy |
|---|---|---|
| 1 | **MVP-FAS, ICCV 2025** | Repo benchmark đã có adapter `mvp`. Tác giả công bố checkpoint FAS Protocol 2. Chọn `MVP_FAS_P2_W.pth` cho phép thử SURF + CeFA → WMCA. CLIP đầy đủ khoảng 600 MB FP32, cộng head và slot attention. Kiểm tra dung lượng tải về và dung lượng bản xuất riêng. [Checkpoint](https://github.com/Elune001/MVP-FAS#checkpoints) |
| 1 | **FLIP-MCL, ICCV 2023** | Repo đã có adapter `flip-mcl`. Model Zoo có mục WMCA 0-shot. Kiểm tra file tải được đúng biến thể MCL và head ViT-B/16 trước khi chạy. Sau khi cache text embeddings và chỉ giữ visual encoder, trọng số khoảng 172 MB FP16. [Model Zoo](https://github.com/koushiksrivats/FLIP/blob/main/docs/model_zoo.md) |
| 2 | **DINOv2 with Registers cho FAS** | Meta công bố backbone. Chưa xác minh được checkpoint FAS của paper CVPRW 2026. Bắt đầu bằng ViT-S/14 + classifier FAS, khoảng 42 MB backbone FP16. Thử ViT-B/14 khoảng 173 MB FP16 nếu cần đối chứng với backbone của paper. [Meta DINOv2](https://github.com/facebookresearch/dinov2) · [Trang dự án FAS](https://gsisaoki.github.io/FAS-VFMbenchmark-CVPRW2026/) |
| 3 | **FoundPAD ViT-B/16, WACV Workshop 2025** | Chạy vision encoder + LoRA + head local. Ước tính khoảng 173 MB trọng số FP16. Tác giả công bố nơi xin quyền tải checkpoint. Nếu được cấp, đối chiếu preprocessing và protocol của đúng checkpoint rồi mới tích hợp. [Repo và checkpoint](https://github.com/gurayozgur/FoundPAD) |

Giữ **GAC-FAS, MiniFASNetV2 và MN3** làm baseline. GD-FAS và CP-JEPA có thể được cân nhắc sau khi có checkpoint phù hợp hoặc khi mục tiêu chuyển sang tái lập paper.

## 2. Chuẩn bị benchmark chung

1. Đóng băng một manifest RGB hiện tại. Giữ `image_path,source_dataset,split,label,attack_type,subject_id,session_id,video_id,mask_id` theo quy ước repo. Đối chiếu nhãn gốc, ảnh trùng, camera, loại tấn công và số lượng theo nguồn. Nếu có nhiều ảnh cùng clip thì giữ clip cùng split.
2. Tách **train / validation / calibration / test** cho model tự huấn luyện. Validation dùng chọn checkpoint. Calibration dùng đặt ngưỡng. Test giữ riêng đến khi chốt model và ngưỡng. Nếu dữ liệu ít, có thể dùng chung validation với calibration và ghi rõ. Khi đánh giá chuyển domain, giữ riêng ít nhất một nguồn theo điều kiện dữ liệu thực tế cho phép. Ghi overlap train của checkpoint public, kể cả nguồn khác tên nhưng có ảnh trùng.
3. Dùng cùng crop RGB nguồn khi so kiến trúc. Mỗi adapter resize và normalize theo checkpoint tương ứng. Dùng `raw` khi ảnh nguyên khung cần YuNet và `cropped` khi đã crop. Nếu có hai kiểu đầu vào, chạy theo từng nhóm. Giữ dòng lỗi và báo coverage theo nguồn và loại tấn công. Với ảnh trích từ clip, có thể xuất thêm kết quả clip, nhưng chọn một đơn vị chính trước khi so model.

## 3. Chạy MVP-FAS và FLIP-MCL trên repo hiện có

Cài PyTorch và torchvision phù hợp môi trường, sau đó chạy `python -m pip install -r requirements-research.txt`. Theo README, nên dùng PyTorch 2.6+ cho loader của các adapter nghiên cứu. Chuẩn bị checkpoint đúng biến thể từ link ở trên và ghi SHA-256. Các lệnh dưới dùng ảnh đã crop và CPU để đối chiếu. Đổi `--device cuda` khi đo trên GPU.

~~~bash
git clone https://github.com/Elune001/MVP-FAS.git vendor/mvp
git -C vendor/mvp checkout fad83b94a326cec1925b73b43fb27708f1bdcf23

git clone https://github.com/koushiksrivats/FLIP.git vendor/flip
git -C vendor/flip checkout 4f95def259e135a0cbaff1d770f559ca739c4c9f

mkdir -p weights
curl -fL https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt \
  -o weights/ViT-B-16.pt

python benchmark.py --models mvp run \
  --mvp-checkpoint weights/MVP_FAS_P2_W.pth \
  --mvp-repo vendor/mvp --clip-weights weights/ViT-B-16.pt \
  --manifest data/public/manifest.csv --input-mode cropped \
  --bpcer 0.05 --device cpu --output runs/mvp-fp32

python benchmark.py --models flip-mcl run \
  --flip-mcl-checkpoint weights/flip-mcl.pth --flip-repo vendor/flip \
  --manifest data/public/manifest.csv --input-mode cropped \
  --bpcer 0.05 --device cpu --output runs/flip-mcl-fp32
~~~

1. **Kiểm tra tương đương upstream trước khi chạy full.** Dùng cùng checkpoint và vài crop có cả live/spoof. So tensor, logits và chiều score. Repo quy ước `spoof_score` càng cao càng nghiêng spoof. Nếu lệch, kiểm tra resize, RGB/BGR, mean/std, thứ tự prompt, thứ tự lớp và cách tổng hợp text embedding. Adapter hiện dùng ImageNet normalization chung cho nhóm nghiên cứu. Xác nhận giá trị này với test loader của từng model trước khi diễn giải số; lưu các lần chạy dùng cấu hình khác nhau thành các thư mục riêng.
2. **MVP-FAS.** Dùng checkpoint FAS để nạp đúng mô hình, còn CLIP gốc là đầu vào để khởi tạo kiến trúc của adapter. Khi chấm ảnh dùng CLS, patch embeddings, text embeddings đã tính sẵn, slot attention và classifier. Kiểm tra đầu ra trước và sau khi tối ưu trên cùng crop. [Adapter](https://github.com/Munfond/dmfc-fas-benchmark/blob/main/research_models.py#L122)
3. **FLIP-MCL.** Xác nhận checkpoint có trọng số `image_mlp` như adapter yêu cầu và patch size 16. So với forward chính thức của MCL, gồm cách gom các prompt và chuẩn hóa trước khi tính similarity. Nếu link Model Zoo trả về checkpoint khác biến thể hoặc checkpoint không nạp được, ghi lại trạng thái và tiếp tục với MVP-FAS. [Adapter](https://github.com/Munfond/dmfc-fas-benchmark/blob/main/research_models.py#L157)
4. **Giảm trọng số đang nạp.** Cả hai adapter hiện cache text embeddings nhưng vẫn giữ đối tượng CLIP đầy đủ ở FP32. Sau khi có mốc đúng, triển khai một wrapper inference chỉ giữ visual encoder, cached text embeddings, logit scale hoặc các module FAS cần cho forward. Chuyển FP16 trên thiết bị hỗ trợ. So điểm FP32/FP16 và quyết định quanh ngưỡng, rồi đo lại peak memory và latency trước khi dùng bản tối ưu để chốt kết quả.

## 4. Triển khai DINOv2 và FoundPAD

### DINOv2 with Registers

- **Bản thử đầu.** Dùng `dinov2_vits14_reg`, tách đặc trưng CLS từ ảnh mặt RGB. Bắt đầu với 224×224 và normalization theo backbone. Freeze encoder và train linear head hai lớp theo quy ước `[spoof,live]` bằng cross-entropy. Có thể cache CLS khi train head để tiết kiệm GPU. Lưu checkpoint head cùng định danh và checksum backbone, class order, preprocessing và split.
- **Nếu head tuyến tính không cải thiện trên validation.** Thử fine-tune các block cuối hoặc dùng LoRA, rồi thử ViT-B/14 nếu chi phí cho phép. Dùng cùng split để so biến thể. Nếu mục tiêu là tái lập paper, dùng **ViT-B/14**, fine-tune toàn backbone và bổ sung giám sát cấp patch, patch-wise augmentation và attention-weighted patch loss. Paper dùng benchmark MICO. Vì vậy, báo bản Small + linear head là baseline riêng. [Paper CVPRW 2026](https://openaccess.thecvf.com/content/CVPR2026W/BIOM2026/html/Feng_Benchmarking_Vision_Foundation_Models_for_Domain-Generalizable_Face_Anti-Spoofing_CVPRW_2026_paper.html)
- **Tích hợp.** Thêm adapter DINOv2 vào `research_models.py` và đăng ký model mới ở `benchmark.py`. Dùng loader và scorer hiện có khi phù hợp. Xuất `spoof_score` đúng quy ước. Ghi cấu hình cho cả hai artifact backbone/head để kiểm tra trần 1 GB trên tổng trọng số dùng thực tế.

### FoundPAD

Xin checkpoint **ViT-B/16** theo form của tác giả. Khi có file, xác minh base CLIP, rank và cách nạp LoRA, head và thứ tự nhãn từ config/checkpoint. Đối chiếu logits với code tác giả trước khi đăng ký adapter. Nếu kế hoạch gồm huấn luyện lại, dùng cùng split với DINOv2 và ghi rõ đó là checkpoint tự train. Nếu chưa được cấp file và chưa có kế hoạch train, ưu tiên hoàn thành MVP-FAS, FLIP-MCL và DINOv2 trước.

## 5. Báo cáo và tiêu chí hoàn thành

| Kiểm tra | Đầu ra cần có |
|---|---|
| Tái lập | `run.json`, checkpoint và SHA-256, commit repo, detector, kích thước ảnh, precision, device và manifest hash. Ghi một phép so adapter–upstream trên cùng crop. |
| Chất lượng | `predictions.csv`, `metrics.csv`, ROC-AUC gộp và theo từng nguồn có đủ hai lớp. So APCER riêng theo print, replay, cutout, transparent, plaster và resin nếu metadata có. Báo BPCER thực tế, coverage và số mẫu cho từng nhóm. |
| Ngưỡng | Dùng cùng mục tiêu BPCER trên calibration, ví dụ 5% theo mặc định hiện có, nhưng chọn ngưỡng riêng cho từng model. Cố định ngưỡng cho test và công bố BPCER quan sát trên test. |
| Chi phí | Dung lượng tổng trọng số triển khai dưới 1 GB. Báo batch 1, latency p50/p95 sau warm-up, peak RAM/VRAM và phần cứng. Đo riêng forward và end-to-end gồm decode/crop/preprocess. Khi đo CUDA, đồng bộ ở biên đo để lấy đúng thời gian thực thi. |

**Thứ tự chạy đề xuất** là xác minh MVP-FAS và FLIP-MCL trên vài crop, benchmark cả hai cùng CNN baseline, rồi mới xây baseline DINOv2 Small. Chọn model theo APCER ở các nhóm tấn công bạn quan tâm, BPCER quan sát, coverage và chi phí thực tế. Nếu số clip ít, báo cả số lỗi và mẫu số để thấy độ chắc chắn của chênh lệch.
