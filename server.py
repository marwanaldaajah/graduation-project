import http.server
import socketserver

class RequesHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        print(f'get fun: {self.path}')
        if self.path == '/':
            self.path='templates/search.html'
            return http.server.SimpleHTTPRequestHandler.do_GET(self)
        elif self.path == '/resolution':
            self.path='templates/resolution.html'
            return http.server.SimpleHTTPRequestHandler.do_GET(self)
        else:
            self.send_error(404, 'File Not Found: %s' % self.path)
    
    def do_POST(self):
        print(f'post fun: {self.path}')

HOST='localhost'
PORT=8000
ADDR=(HOST,PORT)
httpd = socketserver.TCPServer((ADDR),RequesHandler)
print(f"[STARTING] server is starting on PORT: {PORT}")
httpd.serve_forever()

