from http.server import HTTPServer,BaseHTTPRequestHandler

PORT=8080
HOST="localhost"
FORMAT="utf-8"

ADDR=(HOST,PORT)
class Serv(BaseHTTPRequestHandler):
   def do_GET(self):
       if self.path=='/':
           self.path='/presentation/index.html'
       try:
           file_to_open = open(self.path[1:]).read()
           self.send_response(200)
           print(f'the path is {self.path}')
       except :
           file_to_open ="File not found"
           self.send_response(404)
       self.end_headers()
       self.wfile(bytes(file_to_open,FORMAT))

print("[STARTING] server is starting.... ")
httpd= HTTPServer(ADDR,Serv)
httpd.serve_forever()

# import socket
# import threading

# HEADER=64
# POST=5050
# SERVER= socket.gethostbyname(socket.gethostname())
# ADDR=(SERVER,POST)
# FORMAT='utf-8'
# DISCONNECT_MESSAGE="!DISCONNECT"

# server=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# server.bind(ADDR)

# def send(message,client):
#     msg=message.encode(FORMAT)
#     msg_len=len(msg)
#     send_len=str(msg_len).encode(FORMAT)
#     send_len +=b' '*(HEADER - len(send_len))
#     client.send(send_len)
#     client.send(msg)

# def handle_client(conn,addr,clients):
#     print(f"[NEW CONNECTION] {addr} is connected")
#     connected=True
#     while connected:
#         msg_length = conn.recv(HEADER).decode(FORMAT)
#         if msg_length:
#             msg_length=int(msg_length)
#             msg = conn.recv(msg_length).decode(FORMAT)
#             if msg == DISCONNECT_MESSAGE:
#                 connected =False
#             print(f"[{addr}] {msg}")
#         for client in clients:
#             if client!=conn:
#                 send(msg,client)
#     conn.close()

# def start():
#     server.listen()
#     print(f"[LISTENING] Server is listening on {SERVER}")
#     clients = []
#     while True:
#         conn , addr = server.accept()
#         clients.append(conn)
#         thread=threading.Thread(target= handle_client,args= (conn,addr,clients))
#         thread.start()
#         print(f"[ACTIVE CONNECTIONS] {threading.active_count() -1}")

# if __name__ == '__main__':
#     print("[STARTING] server is starting.... ")
#     start()