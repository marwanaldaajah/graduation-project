from http.server import HTTPServer, BaseHTTPRequestHandler

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/search':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('search.html', 'rb') as f:
                self.wfile.write(f.read())
        elif self.path == '/resolution':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('resolution.html', 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><body>404 Not Found</body></html>')
            
    def do_POST(self):
        if self.path == '/search':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('search.html', 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><body>404 Not Found</body></html>')

httpd = HTTPServer(('localhost', 8000), RequestHandler)
httpd.serve_forever()

