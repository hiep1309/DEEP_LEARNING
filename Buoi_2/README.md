#BTVN 1
Công nghệ sử dụng

Python: ngôn ngữ lập trình chính

NumPy:

Tạo và xử lý ma trận bàn cờ 3x3

Kiểm tra thắng nhanh bằng các phép toán trên mảng

Cách hoạt động

Bàn cờ được tạo dưới dạng ma trận 3x3, giá trị 99 là ô trống

Người chơi:

X = 1

O = 0

Hai người chơi nhập tọa độ (dòng, cột) luân phiên

Chương trình:

Kiểm tra ô đã đánh hay chưa

Kiểm tra thắng theo hàng, cột, đường chéo

Nếu có 3 ô liên tiếp giống nhau → kết thúc game

Kết quả

In bàn cờ sau mỗi lượt chơi

Thông báo:

X thắng hoặc O thắng

Trò chơi kết thúc ngay khi có người thắng


#BTVN 2
Công nghệ sử dụng

Python

NumPy: thư viện xử lý mảng và ma trận

Cách hoạt động

Chuyển list 2 chiều thành mảng NumPy

Truy xuất phần tử bằng index, slicing và indexing nâng cao

Lấy:

1 hàng

Nhiều phần tử không cùng vị trí

Một cột theo thứ tự đảo ngược

Kết quả 

list2 → ma trận 3×3

list2[1] → hàng thứ 2: [4 5 6]

list2[[0,1],1] → [2 5]

list2[[0,1],[2,0]] → [3 4]

list2[::-1,2] → [9 6 3]


#BTVN 3
Công nghệ sử dụng

Python

NumPy: xử lý mảng số và lọc dữ liệu

Cách hoạt động

Chuyển list Python thành mảng NumPy

Cách 1: dùng vòng lặp for và toán tử % để kiểm tra số chẵn

Cách 2: dùng Boolean indexing của NumPy để lọc số chẵn nhanh hơn

Kết quả

Mảng ban đầu: [1 2 3 4 5 6 7 8 9 10]

Kết quả in ra (cách 1): 2 4 6 8 10

Kết quả lọc (cách 2): [2 4 6 8 10]

#BTVN 4
Công nghệ sử dụng

Python

NumPy: tạo dữ liệu ngẫu nhiên, xử lý mảng, chia tập dữ liệu

Cách hoạt động

Tạo bộ dữ liệu ngẫu nhiên gồm 150 mẫu, 5 cột

Tách dữ liệu:

X: 4 cột đầu 

y: cột cuối 

Chia dữ liệu theo tỷ lệ:

70% train

30% test

Chia X_train thành 10 phần nhỏ bằng array_split 
Kết quả 

X_train: 105 mẫu × 4 đặc trưng

X_test: 45 mẫu × 4 đặc trưng

y_train: 105 nhãn

y_test: 45 nhãn

X_train_splits: 10 tập con gần bằng nhau