import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import pickle

class ChestXrayCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(ChestXrayCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model architecture
    model = ChestXrayCNN(num_classes=2)
    model.to(device)

    # Load weights from state_dict file
    model.load_state_dict(torch.load('chest_xray_CNN_from_skratch_weights.pth', map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # for grayscale images
    ])

    image = Image.open(image_path).convert('L')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)
        pred_class = torch.argmax(probs, dim=1).item()

    with open('Data/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    predicted_label = le.inverse_transform([pred_class])[0]

    return predicted_label, probs[0][pred_class]

if __name__ == "__main__":
    image_path = r"Data\All_Data\NORMAL\IM-0070-0001.jpeg"
    predicted_label, confidence = predict(image_path)
    print(f"Predicted label: {predicted_label}, Confidence: {confidence:.4f}")
