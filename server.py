import http.server
import socketserver
import os

PORT = 8000

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Get the file path and extension
        file_path = self.translate_path(self.path)
        _, ext = os.path.splitext(file_path)

        # Serve .js files with application/javascript MIME type
        if ext.lower() == '.js':
            try:
                with open(file_path, 'rb') as f:
                    self.send_response(200)
                    self.send_header('Content-type', 'application/javascript')
                    self.end_headers()
                    self.wfile.write(f.read())
            except IOError:
                self.send_error(404, 'File Not Found: %s' % self.path)
            return

        # Handle other files with default behavior
        super().do_GET()

# Set up the server
with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
    print(f"Serving at http://localhost:{PORT}")
    httpd.serve_forever()