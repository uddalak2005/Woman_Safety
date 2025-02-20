import socket
import threading
import pygame
import time
import json

# Server Configuration
ALERT_PORT = 9999
SERVER_IP = "127.0.0.1"

# Initialize UDP socket for alerts only
alert_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
alert_socket.bind((SERVER_IP, ALERT_PORT))

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
        time.sleep(0.1)  # Sleep to prevent busy waiting
    pygame.mixer.music.stop()  # Stop siren when alert stops

def receive_alert():
    """Listens for alert messages from the CCTV system."""
    global siren_active
    print("Waiting for alerts...")
    while True:
        data, addr = alert_socket.recvfrom(1024)
        try:
            alert = json.loads(data.decode())
            if alert["type"] == "VIOLENCE DETECTED":
                if not siren_active:
                    print(f"VIOLENCE DETECTED! TRIGGERING SIREN! 🚨")
                    print(f"Location: {alert['location']}")
                    print(f"Timestamp: {alert['timestamp']}")
                    print(f"Device ID: {alert['device_id']}")
                    siren_active = True
                    threading.Thread(target=play_siren).start()
            elif alert["type"] == "NO VIOLENCE":
                print("✅ Alert resolved! Stopping siren.")
                siren_active = False
        except json.JSONDecodeError:
            print("Received invalid alert format")
        except KeyError as e:
            print(f"Missing key in alert data: {e}")

def user_input_listener():
    """Listens for user input to stop the siren."""
    global siren_active
    while True:
        user_input = input("Type 'stop' to silence the siren: ")
        if user_input.lower() == 'stop':
            if siren_active:
                print("User command received. Stopping siren.")
                siren_active = False
            else:
                print("Siren is already off.")

# Start listening for alerts in a separate thread
threading.Thread(target=receive_alert, daemon=True).start()

# Start listening for user input
user_input_listener()