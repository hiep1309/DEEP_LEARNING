import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ===== Model giống notebook =====
class RNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, output_size=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# ===== Load model =====
model = RNN()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# ===== UI =====
st.title("📈 Dự đoán chuỗi thời gian bằng RNN")

st.write("👉 Nhập 20 dòng, mỗi dòng 3 số (cách nhau dấu phẩy)")
st.write("Ví dụ: 1,2,3")

inputs = []

for i in range(20):
    row = st.text_input(f"Dòng {i+1}", key=i)
    inputs.append(row)

if st.button("Dự đoán"):
    try:
        data = []

        for row in inputs:
            nums = [float(x) for x in row.split(",")]
            if len(nums) != 3:
                st.error("Mỗi dòng phải có 3 số!")
                st.stop()
            data.append(nums)

        data = np.array(data)

        # scale (fit lại tạm thời)
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        # reshape đúng (1, 20, 3)
        tensor = torch.tensor(data_scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred = model(tensor)

            # tạo mảng giả để inverse
            temp = np.zeros((1, 3))
            temp[0, -1] = pred.item()

            # inverse scale
            pred_real = scaler.inverse_transform(temp)[0, -1]

        st.success(f"🔮 Dự đoán (thực): {pred_real:.2f}")

        # vẽ chart
        st.line_chart(data)

    except:
        st.error("❌ Lỗi dữ liệu nhập!")