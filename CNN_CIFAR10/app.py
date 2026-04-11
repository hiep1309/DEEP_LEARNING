import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# ================= MODEL =================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256*4*4,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,10)
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ================= CLASS =================
classes = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),
                         (0.2023,0.1994,0.2010))
])

# ================= UI =================
st.set_page_config(page_title="CIFAR10 Demo", layout="centered")

st.title("🚀 CIFAR-10 Image Classification")
st.write("Upload image để dự đoán 1 trong 10 class")

uploaded_file = st.file_uploader("📷 Upload image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    label = classes[predicted.item()]

    # format đẹp hơn
    label_display = label.capitalize()

    st.success(f"Prediction: {label_display}")
    st.write(f"Confidence: {confidence.item():.2f}")

    # ===== TOP 5 =====
    st.subheader("Top 5 Predictions:")
    top5_prob, top5_idx = torch.topk(probs, 5)

    for i in range(5):
        st.write(f"{classes[top5_idx[0][i]]}: {top5_prob[0][i].item():.2f}")