from http.server import HTTPServer,BaseHTTPRequestHandler

PORT=8080
HOST="localhost"
FORMAT="utf-8"

ADDR=(HOST,PORT)
class Serv(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path=='/':
            self.path='/index.html'
        try:
            file_to_open = open(self.path[1:]).read()
            self.send_response(200)
        except :
            file_to_open ="File not found"
            self.send_response(404)
        self.end_headers()
        self.wfile(bytes(file_to_open,FORMAT))

httpd= HTTPServer(ADDR,Serv)
httpd.serve_forever()
