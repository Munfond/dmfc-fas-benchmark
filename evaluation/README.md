# Chuẩn bị dữ liệu đánh giá FAS

## Dữ liệu để chạy

| Nguồn | Dùng để | Cách lấy |
|---|---|---|
| Minivision demo | Smoke test thật/giả, kiểm tra loader và các model chạy được | `python prepare_smoke.py` tải 3 ảnh ví dụ từ repo tác giả. |
| WMCA RGB | Test public trên mặt thật, ảnh in, replay và mặt nạ | Lấy bản RGB và metadata theo điều kiện truy cập của [Idiap](https://www.idiap.ch/en/dataset/wide-multi-channel-presentation-attack-wmca). |
| Replay-Attack | Benchmark phụ cho ảnh in và replay | Xin quyền tại [Zenodo](https://zenodo.org/records/4593128) rồi dùng `setup_data.py replay-attack`. |
| ZED Mini mới thu | Blind test cho robot | Chốt checkpoint và quy trình chấm trước khi thu hoặc mở tập test. Giữ riêng người, phiên và mẫu mặt nạ. |

Ảnh demo có thể trùng nguồn train. Dùng chúng để kiểm tra kỹ thuật chạy và để trống calibration. Dùng dataset có giấy phép phù hợp cho benchmark thật. Ảnh, video và checkpoint không được đóng gói trong repo.

## WMCA RGB Grandtest

1. Mở [record WMCA trên Zenodo](https://zenodo.org/records/4580313) và gửi **Request access**. Idiap cấp quyền cho nghiên cứu phi thương mại theo EULA. Form yêu cầu thông tin người ký có vị trí thường trực tại đơn vị và email của đơn vị. Sau khi được duyệt, tải và giải nén **bản preprocessed RGB**. Giữ nguyên các thư mục ngày và tên HDF5. Trỏ script vào thư mục RGB, ví dụ `preprocessed-face-station_RGB`.
2. Từ thư mục repo, kiểm tra protocol và tạo manifest. Bước đầu chỉ tải metadata nhỏ `bob.db.batl` từ PyPI và kiểm tra SHA-256. Bước thứ hai đọc HDF5 đã tải từ Zenodo, khớp theo thư mục ngày và ID clip, rồi lấy tối đa 8 frame trải đều trên mỗi clip.

```bash
python prepare_wmca.py --metadata-only
python -m pip install h5py
python prepare_wmca.py --rgb-root /path/to/preprocessed-face-station_RGB \
  --manifest data/wmca/manifest.csv --frames-per-clip 8
python benchmark.py run --manifest data/wmca/manifest.csv \
  --input-mode cropped --output runs/wmca
```

Nếu HDF5 lưu RGB dưới một dataset 4 chiều thay cho các nhóm `Frame_N/array`, chỉ định đúng key bằng `--dataset-key`. Dùng `--color-order bgr` nếu bản đã tải lưu BGR. Nếu dùng bản ảnh nguyên khung, chạy inference ở `--input-mode raw` với YuNet. Tham khảo [hướng dẫn của Idiap](https://www.idiap.ch/en/dataset/wide-multi-channel-presentation-attack-wmca) cho các bản phát hành.

| Split Grandtest | Dùng trong pipeline | Số clip |
|---|---|---:|
| Train | Reference DMFC, chỉ dùng bona fide | 124 |
| Dev | Calibration | 557 |
| Eval | Test | 557 |

Metadata khớp số lượng Grandtest của Idiap sau khi loại wig/makeup. Eval có 115 bona fide và 442 spoof, gồm 137 mặt nạ mềm và 49 mặt nạ cứng. Đọc hai hàng `3d_mask_flexible` và `3d_mask_rigid` trong `video_metrics.csv`. Script kiểm tra `client_id` và PAI tách giữa các split theo Grandtest. Người thực hiện trình diện có thể lặp giữa các split theo metadata gốc. Nếu dùng WMCA làm test cross-dataset, cần reference và calibration từ nguồn khác, và cần kiểm tra lịch sử từng checkpoint. Cấu hình trên dùng train/dev của WMCA để tạo reference và chọn ngưỡng.

Đã kiểm tra metadata chính thức và đường xuất manifest trên HDF5 giả lập. Cần quyền Zenodo để xác nhận layout HDF5 của bản RGB thực tế và chạy benchmark trên ảnh thật.

## Replay-Attack Grandtest

Tải archive chính thức sau khi Idiap duyệt quyền. Trỏ `--root` vào thư mục có dạng sau. Các thư mục enrollment dành cho nhận diện nằm ngoài quy trình PAD này.

```text
replayattack/
  train/
    real/
    attack/
  devel/
    real/
    attack/
  test/
    real/
    attack/
```

```bash
python setup_data.py replay-attack --root /path/to/replayattack --frames-per-clip 8
```

Script quét video trong các thư mục trên, kể cả các thư mục con như `fixed` và `hand`. Nó lấy ID client từ tên file chính thức, gộp `mobile` và `highdef` thành `replay`, kiểm tra client tách split và đối chiếu số clip (60/300 Train, 60/300 Devel, 80/400 Test theo live/spoof). Kết quả nằm ở `data/replay-attack/manifest.csv` và `manifest.summary.json`. Dùng YuNet ở chế độ raw để chấm video nguyên khung. Đã kiểm tra importer với cấu trúc và video giả lập. Cần archive được cấp quyền để xác nhận trên video thật.

## Nhập dữ liệu từ các nguồn khác

1. Mở `inventory.example.csv`. Thay mỗi đường dẫn ví dụ bằng một ảnh hoặc một video RGB đã có. Với dataset RGB được phát hành theo dạng HDF5 hoặc archive, dùng công cụ của tác giả để xuất RGB và giữ ID bản ghi gốc. Giữ label và loại tấn công theo metadata chính thức. Dùng `3d_mask`, `print`, `replay`, `cutout`, hoặc nhãn gốc cụ thể hơn nếu cần. Ví dụ, báo riêng mannequin head nếu dataset tách lớp đó khỏi mặt nạ.
2. Nếu đã có official split, điền cột `split` tường minh. Với WMCA grandtest, ví dụ có thể lấy một nhóm bona fide từ `train` làm reference, `dev` làm calibration và `eval` làm test. Nếu tự chia, để trống để script chia theo nhóm. Dùng `subject_id` và `mask_id` ổn định trong từng nguồn. `session_id` phải là ID phiên duy nhất trong nguồn, hoặc ghép người với phiên nếu dataset lặp số phiên theo người. `group_id` dùng để nối cùng người, vật thể hoặc nội dung được replay qua các dataset khi biết chúng trùng nhau. Nếu một nhóm được chọn làm reference có cả live và spoof, các file spoof trong nhóm đó được loại khỏi lần chạy để tránh đưa chúng sang calibration hoặc test.
3. Chọn một nguồn giữ riêng cho test. Các nhóm được nối qua người, phiên, video, mask hoặc `group_id` không bị tách qua nhiều split. Script từ chối file có cùng SHA-256. Kiểm tra thêm frame gần trùng do resize hoặc encode lại. Với split tường minh xung đột, giữ nguyên group ở một split rồi chạy lại.

```bash
python prepare_eval.py \
  --inventory /path/to/inventory.csv \
  --manifest data/public/manifest.csv \
  --holdout-source WMCA \
  --frames-per-video 8

python benchmark.py run \
  --manifest data/public/manifest.csv \
  --input-mode raw \
  --detector-weights /path/to/face_detection_yunet_2023mar.onnx \
  --output runs/public
```

Dùng `--input-mode cropped` nếu ảnh đã crop. Khi xuất ảnh từ video, giữ cùng `video_id` cho các frame của một lần trình diện. Script lấy frame ở giữa từng khoảng thời gian theo chỉ số frame và ghi thống kê split vào `manifest.summary.json`. Nếu bản chia tự động thiếu nhóm live cho reference hoặc calibration, dùng thêm dữ liệu hoặc chỉ định split trong inventory. Lệnh trên dùng chế độ giữ riêng toàn bộ nguồn WMCA, nên reference và calibration phải đến từ nguồn khác. Nếu dùng grandtest của chính WMCA như ví dụ ở bước 2, bỏ `--holdout-source WMCA`.

## So sánh checkpoint

- Dùng `checkpoints.example.csv` làm mẫu sổ theo dõi. Xác minh **đúng file**, protocol, tập train và cách tác giả chọn checkpoint. Giữ tập dùng để chọn threshold và tập reference ở ngoài nguồn blind test.
- Ghi trạng thái overlap theo cặp **checkpoint–dataset**. Giữ một nguồn khỏi bước chuẩn bị manifest không chứng minh checkpoint đã phát hành chưa từng dùng nguồn đó. Nếu model dùng dữ liệu hoặc kết quả nguồn đó để chọn checkpoint, ghi rõ khi báo kết quả. Với MiniFASNetV2, nguồn train chưa xác minh.
- Đọc **APCER từng loại**, **BPCER**, số mẫu và coverage ở `video_metrics.csv` nếu đơn vị ra quyết định là một clip. Score clip lấy median từ các frame chấm được. `video_predictions.csv` ghi cả số frame chấm được. Không gộp nhiều clip chỉ vì cùng người hoặc cùng mặt nạ.
- Giữ checkpoint, threshold, TTA, detector và cách lấy frame cố định khi mở blind test. Báo test coverage cạnh tỷ lệ lỗi, vì ảnh hoặc clip detector bỏ sót không có score.
