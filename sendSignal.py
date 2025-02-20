import torch
from ultralytics import YOLO
import cv2
import numpy as np
import socket
import threading
import time
import json
import geocoder

# Server Configuration
SECURITY_IP = "127.0.0.1"
ALERT_PORT = 9999

# Load trained YOLO model
yolo_model = YOLO("runs/classify/train4/weights/best.pt")

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

def get_location():
    """Get location using browser's Geolocation API"""
    try:
        g = geocoder.ip('me')
        if g.ok:
            return f"{g.latlng[0]}, {g.latlng[1]}"  # Return latitude, longitude
        return "Location unavailable"
    except Exception as e:
        print(f"Error getting location: {e}")
        return "Location unavailable"

def get_timestamp():
    """Get current timestamp"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def send_alert():
    """Send alert with essential incident details"""
    try:
        print("Attempting to send alert...")
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Create alert message
        alert_data = {
            "type": "VIOLENCE DETECTED",
            "location": get_location(),
            "timestamp": get_timestamp(),
            "device_id": "CAM_MODULE_001"
        }
        
        # Convert to JSON and send
        message = json.dumps(alert_data).encode('utf-8')
        client_socket.sendto(message, (SECURITY_IP, ALERT_PORT))
        print("Detailed alert sent to security system!")
    except Exception as e:
        print(f"Failed to send alert: {e}")
    finally:
        client_socket.close()

def process_webcam(frame_interval=5, seq_length=10):
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        return

    frame_count = 0
    features = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame.")
            break

        frame_count += 1

        if frame_count % frame_interval == 0:
            results = yolo_model(frame)
            if results and results[0].probs is not None:
                fight_prob = results[0].probs.data.cpu().numpy()[0]
                features.append([fight_prob, 1 - fight_prob])

                if len(features) >= seq_length:
                    seq_features = np.array(features[-seq_length:])
                    seq_features = torch.tensor(seq_features, dtype=torch.float32).unsqueeze(0).to(device)

                    with torch.no_grad():
                        lstm_output = lstm_model(seq_features)
                        predicted_class = torch.argmax(lstm_output, dim=1).cpu().numpy()[0]

                    label = "Fight" if predicted_class == 1 else "Non-Fight"
                    color = (0, 0, 255) if predicted_class == 1 else (0, 255, 0)

                    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                    if predicted_class == 1:
                        threading.Thread(target=send_alert).start()

        cv2.imshow("Real-Time Fight Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

process_webcam()