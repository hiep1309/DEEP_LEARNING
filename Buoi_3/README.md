Bài tập 1:
- Công nghệ sử dụng
Python
Pandas: đọc file CSV, xử lý và phân tích dữ liệu
NumPy: chuyển dữ liệu sang mảng số phục vụ học máy / học sâu

- Cách hoạt động
Đọc file howlongwelive.csv vào DataFrame bằng Pandas
Khám phá dữ liệu ban đầu:
Xem 2 dòng đầu và 2 dòng cuối
Kiểm tra kích thước và tên các cột
Thống kê mô tả bằng .describe()
Tiền xử lý dữ liệu:
Xóa các cột có nhiều giá trị thiếu hoặc dư thừa (Hepatitis B, Population)
Mã hóa cột Status về dạng số (0/1)
Đổi tên cột cho đúng ý nghĩa
Chuẩn bị dữ liệu cho mô hình:
Tách đặc trưng X (tất cả cột trừ Life Expectancy)
Tách nhãn y (Life Expectancy)
Chuyển cả X và y sang mảng NumPy

- Kết quả
Dữ liệu được làm sạch và chuẩn hóa
Các cột dư thừa và nhiều NaN đã bị loại bỏ
Dữ liệu phân loại (Status) đã được mã hóa số
Thu được:
X: mảng NumPy chứa các đặc trưng đầu vào
y: mảng NumPy chứa nhãn tuổi thọ
Dữ liệu sẵn sàng để dùng cho Machine Learning / Deep Learning

Bài Tập 2:
- Công nghệ sử dụng
Python
Pandas: kiểm tra dữ liệu thiếu, groupby, tạo và gộp DataFrame
NumPy: tạo dữ liệu ngẫu nhiên

- Cách hoạt động
Kiểm tra dữ liệu thiếu: Dùng Pandas để đếm số lượng giá trị NaN trên từng cột
Xử lý dữ liệu thiếu: Thay thế tất cả giá trị NaN bằng giá trị trung bình (mean) của từng cột số
Phân tích theo quốc gia: Groupby theo Country
Tính tuổi thọ trung bình để tìm quốc gia có tuổi thọ thấp nhất và cao nhất
Phân tích theo mức độ phát triển:
Groupby theo Status (Developed / Developing)
So sánh tuổi thọ trung bình giữa hai nhóm
Tạo dữ liệu bổ sung:
Tạo DataFrame mới gồm ID (giống Country) và Noise_level (giá trị ngẫu nhiên)
Kết hợp dữ liệu:
Gộp (merge) DataFrame mới với DataFrame ban đầu dựa trên cột ID

- Kết quả
Xác định được số lượng dữ liệu thiếu trên từng cột
Dữ liệu được làm đầy NaN bằng giá trị trung bình, sẵn sàng cho phân tích
Tìm ra:
Quốc gia có tuổi thọ trung bình thấp nhất
Quốc gia có tuổi thọ trung bình cao nhất
Nhận thấy:
Các quốc gia Developed có tuổi thọ trung bình cao hơn rõ rệt so với Developing
DataFrame mới được tạo và gộp thành công với dữ liệu gốc, mở rộng thêm đặc trưng