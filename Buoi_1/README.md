#BTVN 1 
Công nghệ sử dụng:

PyTorch

Cách hoạt động:

Khai báo biến x có requires_grad=True.

PyTorch xây dựng đồ thị tính toán khi tính y.

Gọi y.backward() để tự động tính đạo hàm của y theo x.

#BTVN 2 
Công nghệ sử dụng:

PyTorch 

Cách hoạt động:

Khởi tạo x và cho phép tính gradient.

Mỗi vòng lặp: tính hàm y, dùng backward() để lấy đạo hàm.

Cập nhật x theo Gradient Descent và reset gradient.

Kết quả:

Giá trị x thay đổi sau mỗi vòng lặp.

x dần tiến tới điểm làm cho hàm số giảm nhỏ hơn.

#BTVN 3
Công nghệ sử dụng:

PyTorch (Tensor, Autograd, MSE Loss).

Cách hoạt động:

Tạo dữ liệu giả theo công thức y = 3x + 5 + noise.

Dự đoán y_pred = w*x + b.

Tính sai số bằng MSE, dùng backward() để tính gradient.

Cập nhật w, b bằng Gradient Descent và reset gradient.

Kết quả:

In ra giá trị loss ban đầu.

Tham số w và b được cập nhật, tiến gần hơn tới giá trị đúng 

#BTVN 4

#BTVN 5 
Công nghệ sử dụng:

PyTorch (Tensor).

Cách hoạt động:

Tạo tensor với các hàm: empty, zeros, ones, rand, arange.

Dùng view và view_as để thay đổi hình dạng tensor.

Kết quả:

In ra các tensor với giá trị khác nhau.

Thể hiện cách khởi tạo và reshape tensor trong PyTorch.