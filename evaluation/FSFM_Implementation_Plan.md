# Plan triển khai thử FSFM cho RGB Face Anti-Spoofing

Ngày: 19/11/2026. Phạm vi: kế hoạch triển khai, chưa phải bộ script đã chạy kiểm chứng.

## 1. Mục tiêu và phạm vi

Thử FSFM CVPR 2025 ViT-B/16 bằng checkpoint đã fine-tune cho FAS, trên cùng bộ dữ liệu RGB và cùng danh sách test đã dùng cho các baseline. Chạy local, không gọi API LLM/VLM, không pretrain lại. Đánh giá checkpoint gốc trước; fine-tune là thí nghiệm riêng sau đó.

Giữ giới hạn tập test cuối tối đa 5 GB. Dữ liệu calibration, checkpoint và cache crop được tính riêng, không âm thầm cộng vào tập test. Không tải lại dataset nếu đã có dữ liệu hợp lệ.

Không trộn FSFM CVPR25 nhánh main với FS-VFM extension hay detector unified 4-class: chúng có checkpoint, task head và nguồn train khác nhau.

## 2. Tài nguyên chính thức và checkpoint

- Code: https://github.com/wolo-wolo/FSFM-CVPR25
- Checkpoint FAS: https://huggingface.co/Wolowolo/fsfm-3c/tree/main/finetuned_models/MCIO_protocol
- Checkpoint khởi đầu đã thấy công khai: `finetuned_models/MCIO_protocol/OCI2M/msu_vit_0_shot_checkpoint_run_2.pth.tar`, khoảng 345 MB.
- Các protocol: OCI2M, OMI2C, OCM2I, ICM2O. M=MSU-MFSD, C=CASIA-MFSD, I=Replay-Attack, O=OULU-NPU.

Mỗi thí nghiệm dùng một checkpoint; không cần tải toàn bộ model hub. Ghi repository commit, hub revision, tên file, SHA256, nguồn pretrain/fine-tune, normalization và threshold vào model card nội bộ. Checkpoint fine-tune khác checkpoint backbone pretrained; không dùng nhầm trọng số deepfake/DiFF.

Tải qua `huggingface_hub.hf_hub_download` với repo_id `Wolowolo/fsfm-3c` và filename nêu trên. Nếu link không còn hoạt động, dừng và cập nhật tài nguyên; không thay bằng checkpoint khác mà không ghi lại.

## 3. Kiểm tra độ độc lập của dữ liệu trước khi benchmark

Tạo bảng checkpoint × dataset test với ba trạng thái:

- Held-out: nguồn test không tham gia FAS fine-tune và chưa bị dùng để chọn model/ngưỡng trong pipeline của mình.
- Seen-domain: nguồn dữ liệu đã tham gia fine-tune; tách thêm có/không trùng sample nếu xác minh được.
- Unknown: không đủ thông tin về nguồn train hoặc provenance.

OCI2M không đồng nghĩa checkpoint chưa thấy mọi dataset trong bộ tổng hợp. Theo thiết lập paper còn có CelebA-Spoof làm dữ liệu bổ sung; cần kiểm tra script/log của đúng checkpoint. Pretrain VGGFace2 và chọn checkpoint bằng target benchmark cũng phải được ghi nhận. Không tuyên bố hoàn toàn unseen chỉ dựa vào tên protocol.

Nếu dữ liệu gồm nhiều nguồn, báo cáo riêng các nhóm trên. Không chọn checkpoint cho từng ảnh theo nhãn hoặc theo kết quả test. Nếu benchmark nhiều checkpoint, chọn danh sách trước và công bố tất cả; model triển khai được chọn bằng calibration/validation độc lập.

## 4. Môi trường và kiểm tra tương thích

Khởi đầu trên Linux + NVIDIA GPU nếu có. CPU vẫn cần được hỗ trợ bằng `--device cpu`, dùng FP32 và batch nhỏ; chưa cam kết latency. Với Jetson, dùng bản PyTorch/CUDA tương ứng JetPack, không cài wheel CUDA desktop trực tiếp.

Môi trường tham chiếu upstream: Python 3.9.21, PyTorch 1.13.1/CUDA 11.7, torchvision 0.14.1; requirements ghi timm 0.4.5. Code cũng có comment đề cập các phiên bản timm khác, nên phải kiểm tra shape và forward trước khi khóa dependency. Không nâng timm tùy tiện vì có thể đổi `forward_features`.

Tách environment FSFM khỏi các model đang chạy. Sau khi smoke test pass, xuất lock file và ghi GPU/driver/CUDA. GPU phục vụ tăng tốc, không phải yêu cầu logic của inference.

## 5. Bộ file cần triển khai

| File dự kiến | Nhiệm vụ |
|---|---|
| `configs/fsfm.yaml` | Checkpoint, preprocessing, device, batch, threshold, class mapping |
| `scripts/download_fsfm.py` | Tải đúng checkpoint/revision và tính checksum |
| `scripts/inspect_fsfm_checkpoint.py` | Liệt kê key/shape, metadata, kiểm tra head |
| `src/models/fsfm_adapter.py` | Dựng đúng kiến trúc và nạp trọng số FAS |
| `src/data/manifest.py` | Đọc CSV, kiểm tra nhãn, provenance, nhóm video |
| `src/data/face_preprocess.py` | Face crop có cache và metadata |
| `scripts/infer_fsfm.py` | Chạy score ảnh, không cần label |
| `scripts/calibrate_fsfm.py` | Chọn ngưỡng chỉ từ calibration |
| `scripts/evaluate_fsfm.py` | Tính metric tổng và theo nhóm |
| `scripts/profile_fsfm.py` | Đo latency/RAM/VRAM trên thiết bị thực |
| `docs/FSFM_SETUP.md` | Hướng dẫn setup và xử lý lỗi |

Đây là interface cần viết, chưa phải các file upstream sẵn có. Có thể đặt adapter vào cấu trúc repository hiện tại thay vì tạo pipeline thứ hai.

## 6. Load model đúng với checkpoint

Đối chiếu `fsfm-3c/finuetune/cross_domain_FAS/fas.py` và `fsfm-3c/models_vit.py`:

- Dùng ViT-B/16, cấu hình `global_pool=True` như upstream.
- Giữ feature embedder Linear(768,512), ReLU, Dropout và classifier Linear(512,2).
- Giữ normalization trong embedder/classifier và `norm_flag=True`; không thay bằng head Linear(768,2) của timm thông thường.
- Checkpoint fine-tune được upstream lưu dưới key `state_dict`; đọc metadata riêng. Chỉ bỏ prefix `module.` nếu thực sự có.
- Tạo cấu trúc model không tự tải ImageNet/pretrain khác, rồi nạp toàn bộ checkpoint FAS. Không truyền checkpoint FAS vào nhánh `--pt_model` dành cho pretrain rồi vô tình khởi tạo lại head.
- Yêu cầu strict load khớp toàn bộ model; mọi missing/unexpected key đều phải giải thích, không bỏ qua bằng `strict=False` cho xong.
- Chạy `model.eval()` và `torch.inference_mode()`.

Trong code hiện tại, tensor logits có thể chứa trục token: evaluator upstream lấy `softmax(cls_out, dim=-1)[:,0,1]`. Kiểm tra đúng shape thực tế và giữ cách lấy score này; không lấy trung bình token hay tự đổi pooling. Nhãn upstream: **0=spoof, 1=live**. Score chuẩn hóa nội bộ là `score_live`; `score_spoof=1-score_live`. Đây là score softmax, chưa mặc nhiên là xác suất được calibration tốt.

## 7. Manifest và tiền xử lý

CSV tối thiểu:

```csv
image_path,label,dataset,subject_id,presentation_id,attack_type,split
images/live_a.jpg,1,source_a,s001,p001,live,calibration
images/mask_a.jpg,0,source_b,s002,p002,resin,test
```

Quy ước attack: live, print, replay, cutout, transparent_mask, plaster_mask, resin_mask, other, unknown. Không ép nhãn mask không rõ vật liệu sang một nhóm cụ thể. Đọc nhãn từ protocol/annotation gốc, không đoán qua tên file.

Tách calibration/test theo subject và presentation/video; loại trùng byte và kiểm tra near-duplicate giữa hai split. Khi không xác minh được identity xuyên dataset, ghi hạn chế đó. Frame từ cùng một video không được rải vào cả hai tập.

Tiền xử lý inference đối chiếu upstream: OpenCV BGR→RGB, chuyển ảnh uint8, PIL resize 224×224, ToTensor, Normalize. Khớp interpolation với môi trường đã khóa, không thêm augmentation test ngẫu nhiên.

Normalization upstream chọn thống kê `pretrain_ds_mean_std.txt` nếu cấu hình lúc train dùng file đó; nếu không thì ImageNet mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]. Phải truy nguyên đúng script/log checkpoint, không chọn mean/std bằng cách xem model nào đạt test cao hơn. Nếu thiếu metadata, ghi giả định và tách thí nghiệm.

Loader FAS upstream không tự detect/crop mặt: nó nhận ảnh đã chuẩn bị. Với ảnh đã face-crop đúng chuẩn, tránh crop lần hai. Với full frame, tái sử dụng face detector hiện tại và đối chiếu crop của protocol gốc. Face parsing/masking là thành phần pretrain, không tự động thêm vào inference.

Ảnh không có mặt, lỗi đọc ảnh, nhiều mặt: trả `status` rõ ràng. Nếu nhiều mặt, áp dụng chính sách chọn mặt đã định trước hoặc xuất từng mặt. Không tự gán lỗi detect thành live/spoof; báo cáo coverage và số lỗi trên toàn tập.

## 8. Chạy inference và calibration

Luồng: manifest/ảnh → face crop nếu cần → RGB/resize/normalize → ViT + embedder + classifier → score_live → threshold → live/spoof.

Đầu ra `predictions.csv`: sample_id, image_path, presentation_id, dataset, attack_type, score_live, prediction, status, checkpoint_sha256, preprocessing_id, inference_ms. Lưu score đủ độ chính xác để đánh giá lại mà không chạy model.

Ba cách ngưỡng:

1. 0.5: baseline tham chiếu, không mặc định tối ưu.
2. Ngưỡng calibration-EER: dùng để báo cáo HTER cân bằng.
3. Ngưỡng theo APCER mục tiêu (ví dụ 1%) trên calibration, sau đó đo BPCER trên test. Nếu số presentation attack không đủ để chứng minh mức 1%, công bố số mẫu và độ bất định; không coi zero observed errors là zero true risk.

Khóa `threshold.json` trước test. Ngưỡng có sẵn trong checkpoint chỉ là tham chiếu của nguồn cũ, không phải ngưỡng triển khai đã được xác thực trên miền mới. Evaluator upstream tính ngưỡng EER từ tập đang đánh giá; không dùng cách đó làm số liệu triển khai chính trên test của bạn.

## 9. Đánh giá công bằng

- Giữ nguyên manifest, nhãn và số presentation giữa FSFM và baseline. Dùng preprocessing đặc thù đúng cho từng model, lưu cấu hình để so sánh được.
- Mỗi model được calibration theo cùng quy tắc trên cùng tập calibration, rồi giữ nguyên ngưỡng khi test.
- Báo cáo ROC-AUC, EER mô tả đường ROC, accuracy/balanced accuracy, APCER, BPCER, ACER/HTER tại ngưỡng đã khóa. Tách rõ EER tính từ test chỉ để mô tả, không dùng nó chọn ngưỡng triển khai.
- Với quy ước score_live≥threshold là live: APCER là tỷ lệ attack bị nhận live; BPCER là tỷ lệ live bị nhận spoof.
- Báo cáo per-dataset, per-attack và ba nhóm provenance; số presentation phải đi kèm metric. Đặc biệt tách print/replay/cutout/transparent/plaster/resin.
- Với video, lấy trung bình score frame theo presentation, rồi đánh giá presentation-level làm kết quả chính; image/frame-level là phụ. Không dùng nhiều frame để biến một mask/video thành nhiều mẫu độc lập.
- Báo cáo micro và macro theo nhóm, nhóm không có mẫu ghi N/A. Bootstrap khoảng tin cậy theo subject/presentation khi đủ dữ liệu.
- Tạo histogram score live/spoof, ROC và gallery false accept/false reject để phân tích lỗi.

Kết quả MCIO trong paper không chứng minh sẵn khả năng phân biệt mọi vật liệu 3D mask. Chỉ kết luận về các category thực sự có mẫu và provenance kiểm chứng được.

## 10. Thứ tự thực hiện và nghiệm thu

| Giai đoạn | Công việc | Điều kiện hoàn thành |
|---|---|---|
| A | Khóa checkpoint, code, nguồn dữ liệu | Có model card và bảng seen/held-out/unknown |
| B | Dựng adapter và preprocessing | Load không thiếu key; score hữu hạn, đúng chiều live/spoof |
| C | Smoke test 20–50 ảnh | Batch/single cho score gần nhau; chạy lặp lại ổn định; lỗi input có status |
| D | Đối chiếu upstream | Score adapter khớp evaluator trên cùng tensor/model FP32, ghi sai số; mục tiêu ban đầu max abs diff≤1e-5 trên cùng thiết bị |
| E | Calibration và chạy test cố định | Có threshold.json, predictions.csv, metrics và breakdown |
| F | Profile | Đo p50/p95 latency, throughput, RAM/VRAM và coverage trên máy đích |

Chỉ sau khi FP32 đúng mới thử FP16, ONNX/TensorRT. So sánh score và metric với FP32 trước khi chấp nhận. Đo cả latency model-only và end-to-end có face detection, warm-up và CUDA synchronization khi cần. 345 MB trọng số không tương đương 345 MB VRAM.

Không đặt mục tiêu phải vượt 90–95% accuracy trước khi chạy. FSFM đáng giữ nếu cải thiện các attack quan trọng tại cùng BPCER/APCER mục tiêu và đáp ứng ngân sách latency thực tế.

## 11. Fine-tune tùy chọn sau baseline

Nếu checkpoint gốc yếu trên mask hoặc một nguồn mới, tạo thí nghiệm fine-tune riêng: head-only trước, rồi unfreeze block cuối hoặc LoRA. Chỉ dùng train/validation độc lập; giữ final test nguyên vẹn. Ghi các mẫu/attack/nguồn đã tham gia fine-tune và tính cả backbone khi báo cáo dung lượng LoRA.

## 12. Tài liệu đã đối chiếu

- Repository/README: https://github.com/wolo-wolo/FSFM-CVPR25
- Kiến trúc: https://github.com/wolo-wolo/FSFM-CVPR25/blob/main/fsfm-3c/finuetune/cross_domain_FAS/fas.py
- Loader: https://github.com/wolo-wolo/FSFM-CVPR25/blob/main/fsfm-3c/finuetune/cross_domain_FAS/utils/dataset.py
- Evaluator: https://github.com/wolo-wolo/FSFM-CVPR25/blob/main/fsfm-3c/finuetune/cross_domain_FAS/utils/evaluate.py
- Lưu checkpoint: https://github.com/wolo-wolo/FSFM-CVPR25/blob/main/fsfm-3c/finuetune/cross_domain_FAS/utils/utils.py
- FAS checkpoint: https://huggingface.co/Wolowolo/fsfm-3c/tree/main/finetuned_models/MCIO_protocol

Trạng thái: đã đọc các file code nêu trên để lập kế hoạch; chưa tải/load checkpoint hoặc chạy inference trên dữ liệu của bạn. Metadata normalization của đúng checkpoint cần xác minh khi triển khai.
