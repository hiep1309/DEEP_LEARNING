import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ================= MODEL =================
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)
        return self.fc(hidden[-1])

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = LSTMModel()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ================= LOAD SCALER =================
data_min, data_max = np.load("scaler.npy")

# ================= FUNCTION =================
def predict_future(model, data, time_step=5, future_steps=50):
    data = data.copy().tolist()

    for _ in range(future_steps):
        x_input = np.array(data[-time_step:])
        x_input = (x_input - data_min) / (data_max - data_min)

        x_input = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        with torch.no_grad():
            y_pred = model(x_input).item()

        # scale ngược lại
        y_pred = y_pred * (data_max - data_min) + data_min
        data.append(y_pred)

    return data

# ================= UI =================
st.title("📈 LSTM Time Series Prediction")

st.write("Demo dự đoán chuỗi sin + noise")

# slider
future_steps = st.slider("Số bước dự đoán", 10, 200, 50)

# tạo data giống lúc train
t = np.arange(0, 100, 0.1)
data = np.sin(t)
noise = np.random.normal(0, 0.1, len(data))
data = data + noise

# normalize
data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))

# predict
predicted = predict_future(model, data_norm, future_steps=future_steps)

# ================= PLOT =================
fig, ax = plt.subplots(figsize=(10,5))

ax.plot(range(len(data)), data, label="Dữ liệu thật")
ax.plot(range(len(predicted)), predicted, label="Dự đoán", linestyle='dashed')

ax.legend()
ax.set_title("Kết quả dự đoán")

st.pyplot(fig)