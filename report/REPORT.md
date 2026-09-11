# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026 

**Runtime Colab:** GPU T4

**Python / PyTorch / Ultralytics:** 3.13.15 / 2.11.0+cu128 / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** 

> Khi nộp, đặt trực tiếp `REPORT.md` và `day1_lab_outputs/` trong thư mục `report/` của repository. Không đưa họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân của người nộp vào báo cáo hay output.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `day1_lab_outputs/classification_predictions.json`, sample `traffic`.

- **Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):** `(468, "cab", 1, 0.510915, "ImageNet-1K")`, tương đương score 51,0915%.
- **Record này mô tả toàn ảnh như thế nào?** Checkpoint xếp `cab` là nhãn có score cao nhất cho toàn bộ ảnh `traffic` kích thước 640 × 428. Đây là prediction cấp ảnh: nó không định vị, không đếm và không gán nhãn riêng cho từng xe hoặc người trong cảnh.
- **Ai định nghĩa class list mà checkpoint có thể dự đoán?** Taxonomy của tập huấn luyện ImageNet-1K và quá trình huấn luyện checkpoint định nghĩa danh sách lớp; mapping giữa ID và tên lớp được mang theo checkpoint. Model không tự tạo lớp mới lúc inference và annotator không được tự ý đổi danh sách này.
- **Vì sao cần giữ cả ID, tên lớp và tên taxonomy?** `class_id` ổn định cho xử lý máy nhưng chỉ có nghĩa trong đúng namespace; `class_name` giúp con người đọc và QC; `taxonomy_name` xác định bộ nhãn mà ID/tên thuộc về, tránh nhầm các ID giống nhau giữa các taxonomy và giúp tái lập train/export/QC.
- **Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?** Phải quy định rõ bài toán single-label hay multi-label. Nếu single-label, cần có tiêu chí chọn chủ thể chính theo mục tiêu tác vụ, độ nổi bật hoặc diện tích; đồng thời quy định thứ tự ưu tiên và cách gắn cờ/escalate khi các chủ thể ngang nhau hoặc ảnh mơ hồ.
- **Vì sao model score không phải ground truth?** Score chỉ là độ tin cậy do model tính ra và có thể sai hoặc chưa được hiệu chỉnh, nhất là khi ảnh khác miền dữ liệu huấn luyện. Ground truth phải là nhãn tham chiếu được tạo và kiểm tra theo guideline; score cao không chứng minh nhãn đúng.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `day1_lab_outputs/detection_predictions.json` và `day1_lab_outputs/visuals/detection_predictions.png`, sample `kitchen`.

- **Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):** `("person", 0.912625, [385.33, 69.24, 498.92, 348.92], 113.58, 279.68)`; đơn vị tọa độ và kích thước là pixel.
- **Diễn giải vị trí box bằng lời:** Với gốc tọa độ ở góc trên trái, `x` tăng sang phải và `y` tăng xuống dưới, box đi từ góc trên-trái `(385.33, 69.24)` đến góc dưới-phải `(498.92, 348.92)`. Box bao quanh người đứng ở nửa phải ảnh 640 × 427, chiếm khoảng 17,75% chiều rộng và 65,50% chiều cao ảnh.
- **So sánh số prediction ở hai threshold:** Theo quy ước giữ record có `score >= threshold`, ngưỡng 0,35 có **11 box** (`5 bowl`, `2 cup`, `2 oven`, `2 person`). Khi lọc chính các record này ở ngưỡng 0,70 còn **3 box** (`2 bowl`, `1 person`). Con số ở ngưỡng 0,70 là kết quả lọc hậu kỳ từ JSON được sinh ở ngưỡng 0,35, không phải một lần inference thứ hai.
- **Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?** Ngưỡng 0,35 giữ thêm 8 ứng viên nên có khả năng tăng độ bao phủ/recall, nhưng reviewer phải xem 11 thay vì 3 box (khoảng 3,67 lần). Ngưỡng 0,70 giảm 72,73% số box cần xem nhưng có thể bỏ sót object đúng có score thấp; khi chưa có ground truth, không thể kết luận các box bị lọc đều là lỗi.
- **Đề xuất một quy tắc box chặt:** Vẽ hình chữ nhật song song với trục nhỏ nhất bao hết các pixel nhìn thấy của đúng một instance; không thêm padding, bóng, nền hoặc vật thể bên cạnh. Giữ box trong biên ảnh và dùng thống nhất quy ước visible/modal, không tự suy đoán phần bị che.
- **Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?** Guideline cần quy định tỷ lệ nhìn thấy tối thiểu và mức nhận dạng tối thiểu để annotate hay bỏ qua; box của object cắt mép dừng tại mép ảnh và nên có cờ `truncated`, object bị vật khác che nên có cờ `occluded`. Escalate khi không chắc lớp, không chắc các phần rời có thuộc cùng instance, hoặc mức che khuất nằm sát tiêu chí tối thiểu.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `day1_lab_outputs/segmentation_predictions.json` và `day1_lab_outputs/visuals/segmentation_prediction.png`, sample `kitchen`.

- **Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):** `instance_id = "kitchen-001"`, `class_name = "person"`, `score = 0.899318`, `polygon_point_count = 348`; sáu điểm đầu là `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0], [441.0, 73.0]]`. Mảng thực tế có đúng 348 cặp điểm, đơn vị pixel trên ảnh 640 × 427.
- **Polygon bổ sung chi tiết gì so với box?** Box chỉ là hình chữ nhật bao ngoài nên chứa nhiều pixel nền. Polygon bám theo đường bao để xác định pixel thuộc instance, thể hiện hình dạng, diện tích, chỗ lõm và mức chồng lấn chính xác hơn; ví dụ mask người đi theo tóc, thân, tay và chân thay vì tô toàn bộ box.
- **`instance_id` dùng để làm gì và không phải loại ID nào?** Đây là khóa phân biệt từng prediction trong một sample, dùng để nối class, score, box, polygon và kết quả QC của cùng instance. Nó không phải `class_id`, không phải ID taxonomy/ảnh COCO, không phải danh tính của người/vật ngoài đời và không phải tracking ID bền vững qua nhiều ảnh hoặc video.
- **Đề xuất một quy tắc biên mask:** Dùng quy ước visible/modal: chỉ tô pixel nhìn thấy của đúng instance; đặt biên sát ranh giới ở độ phân giải gốc; loại nền, bóng đổ, vật che và khoảng rỗng nhìn xuyên; không suy diễn phần bị che hoặc ngoài khung. Vật khác nhau, kể cả cùng lớp hoặc đang tiếp xúc, vẫn có mask và `instance_id` riêng khi còn đủ dấu hiệu phân cách.
- **Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?** Guideline phải chốt visible/modal hay amodal, cách đặt biên trong dải chuyển tiếp mờ, cách biểu diễn nhiều mảnh rời, lỗ và phần chạm mép. Với visible/modal, không nối mask xuyên qua vật che. Gắn cờ và escalate khi không xác định được pixel thuộc instance nào, không biết hai vùng là một hay hai instance, hoặc schema chỉ hỗ trợ một polygon trong khi vùng nhìn thấy cần multipolygon/RLE.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một record cho mỗi ảnh: class ID/tên thuộc taxonomy; single-label hoặc danh sách multi-label theo guideline | Ảnh `traffic` có nhiều xe và người nhưng model phải chọn nhãn cấp ảnh; top-1 `cab` chỉ đạt 0.510915, nên chủ thể chính và chế độ single/multi-label dễ mơ hồ | Đọc toàn ảnh, áp dụng đúng taxonomy và quy tắc chọn nhãn chính/đa nhãn; gắn cờ khi không đủ căn cứ | Kiểm tra đúng taxonomy, đúng chế độ nhãn, tính nhất quán giữa ảnh tương tự và các ca bị gắn cờ |
| Phát hiện vật thể | Một record cho mỗi object: class và box `xyxy` (pixel hoặc normalized phải được khai báo) | Có object cắt mép trái, nhiều vật gần/chồng nhau và các nhãn `bowl`/`cup` score thấp; threshold làm thay đổi mạnh số ứng viên | Tìm đủ object hợp lệ, gán lớp, vẽ box chặt theo phần nhìn thấy và gắn cờ `occluded`/`truncated` | Kiểm tra bỏ sót, dư/duplicate, sai lớp, box quá rộng/chật, quy tắc mép ảnh và ca escalation; không coi score là ground truth |
| Instance segmentation | Một mask/polygon hoặc RLE cho mỗi instance, kèm class và `instance_id` | Biên người–nền, các vật tiếp xúc trên bàn, object nhỏ, phần bị che và mask chạm mép làm ranh giới khó xác định | Tách từng instance, vẽ biên theo quy ước visible/modal, loại nền/vật che và gắn cờ vùng mơ hồ | Kiểm tra độ bám biên, pixel rò/thiếu, lỗ, chồng lấn, tách instance, tính đầy đủ và sự nhất quán với guideline |

Prediction chỉ là đầu ra để QC hoặc gợi ý gán nhãn; nó trở thành ground truth chỉ sau khi được con người kiểm tra và sửa theo guideline.

## 5. An toàn dữ liệu

- **Một quy tắc bảo vệ dữ liệu:** Chỉ dùng dữ liệu nằm trong phạm vi bài học và kênh lưu trữ được phê duyệt; không đưa PII, dữ liệu nhạy cảm hoặc ảnh riêng tư vào repository và không sao chép/chia sẻ dữ liệu ra ngoài phạm vi được phép.
- **Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:** giảng viên/trợ giảng hoặc đầu mối quản lý dữ liệu của môn học qua kênh được phê duyệt; không tiếp tục xử lý hay tải tệp đó lên repository.

## 6. Danh sách bằng chứng

- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS` — chưa thể xác nhận vì gói hiện tại không có notebook hoặc log của ô validation.
- [x] Không có họ tên, MSSV, email, số điện thoại hoặc dữ liệu nhạy cảm của người nộp trong báo cáo/output. Tên tác giả ảnh công khai trong `IMAGE_ATTRIBUTION.md` được giữ lại để tuân thủ yêu cầu attribution của giấy phép.

**Kiểm tra artifact cục bộ:** `PASS` — cả ba JSON parse hợp lệ; có lần lượt 15 record classification, 53 record detection và 49 record segmentation; mọi `polygon_point_count` khớp số cặp điểm thực tế. Kết quả này không thay thế trạng thái ô validation cuối notebook.
