import socket

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(("127.0.0.1", 9999))
client_socket.sendall(b"TEST ALERT")
client_socket.close()

print("🚨 Test Alert Sent Successfully!")
