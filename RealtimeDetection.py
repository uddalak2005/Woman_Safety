import torch
from ultralytics import YOLO
import cv2
import numpy as np

# Load trained YOLO model
yolo_model = YOLO("runs/classify/train4/weights/best.pt")  # Adjust path if needed

# Load trained LSTM model
class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_model = LSTMClassifier(input_dim=2, hidden_dim=128, num_layers=3, output_dim=2).to(device)
lstm_model.load_state_dict(torch.load("lstm_fight_detection.pth", map_location=device))
lstm_model.eval()

# Real-time video processing
def process_webcam(frame_interval=5, seq_length=10):
    cap = cv2.VideoCapture(0)  # Open webcam (use RTSP link for CCTV)
    
    if not cap.isOpened():
        print("❌ Error: Cannot access webcam.")
        return

    frame_count = 0
    features = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % frame_interval == 0:
            results = yolo_model(frame)  
            fight_prob = results[0].probs.data.cpu().numpy()[0]  # Extract Fight probability
            features.append([fight_prob, 1 - fight_prob])  # Fight, Non-Fight

            if len(features) >= seq_length:
                seq_features = np.array(features[-seq_length:])  # Take last `seq_length` features
                seq_features = torch.tensor(seq_features, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    lstm_output = lstm_model(seq_features)
                    predicted_class = torch.argmax(lstm_output, dim=1).cpu().numpy()[0]

                label = "Fight" if predicted_class == 1 else "Non-Fight"
                color = (0, 0, 255) if predicted_class == 1 else (0, 255, 0)

                # Display prediction on frame
                cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Show video feed
        cv2.imshow("Real-Time Fight Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run real-time detection
process_webcam()
