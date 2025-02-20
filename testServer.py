import socket

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("127.0.0.1", 9999))
server_socket.listen(1)

print("Server is listening on port 9999...")

while True:
    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")
    data = conn.recv(1024)
    if not data:
        break
    print("Received:", data.decode())
    conn.close()
