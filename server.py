import socket
import cv2
import numpy as np
import threading
import pygame  

# Server Configuration
ALERT_PORT = 9999
VIDEO_PORT = 5000
SERVER_IP = "127.0.0.1"

# Initialize UDP sockets
alert_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
alert_socket.bind((SERVER_IP, ALERT_PORT))

video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
video_socket.bind((SERVER_IP, VIDEO_PORT))

# Initialize pygame for siren sound
pygame.mixer.init()
pygame.mixer.music.load("siren.wav")  

# Global flag to control siren playback
siren_active = False

def play_siren():
    """Plays the siren sound in a loop until alert stops."""
    global siren_active
    pygame.mixer.music.play(-1)  # Play in infinite loop

    while siren_active:
        pass  # Keep looping while alert is active

    pygame.mixer.music.stop()  # Stop siren when alert stops

def receive_alert():
    """Listens for alert messages from the CCTV system."""
    global siren_active
    print("🔴 Waiting for alerts...")

    while True:
        data, addr = alert_socket.recvfrom(1024)
        message = data.decode()

        if message == "VIOLENCE DETECTED":
            if not siren_active:
                print("🚨 VIOLENCE DETECTED! TRIGGERING SIREN & VIDEO STREAM! 🚨")
                siren_active = True
                threading.Thread(target=play_siren).start()
                threading.Thread(target=receive_video).start()

        elif message == "NO VIOLENCE":
            print("✅ Alert resolved! Stopping siren.")
            siren_active = False  # This stops the siren loop

def receive_video():
    """Receives and displays video stream from the CCTV system."""
    print("📡 Receiving video stream...")

    cv2.namedWindow("LIVE CCTV FEED", cv2.WINDOW_NORMAL)

    while True:
        data, _ = video_socket.recvfrom(65536)
        frame = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        if frame is not None:
            cv2.imshow("LIVE CCTV FEED", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Start listening for alerts
receive_alert()
