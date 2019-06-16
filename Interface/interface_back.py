import numpy as np
from normalization_utils import sentence_to_features
from naiveBayes import predict
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import cgi

values = np.load('probabilities.npy')
feature_names = [line.rstrip('\n') for line in open('best_words_1000.txt', 'r')]

class Server(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def do_HEAD(self):
        self._set_headers()

    def do_GET(self):
        if self.path!='/':
            self.send_response(400)
            self.end_headers()
            return
        f = open('interface.html')
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()
        self.wfile.write(f.read().encode())
        f.close()

    # POST echoes the message adding a JSON field
    def do_POST(self):
        ctype, pdict = cgi.parse_header(self.headers.get('content-type'))

        # refuse to receive non-json content
        if ctype != 'application/json':
            self.send_response(400)
            self.end_headers()
            return

        # read the message and convert it into a python dictionary
        length = int(self.headers.get('content-length'))
        message = json.loads(self.rfile.read(length))

        news = message['news'].replace('\n', ' ')

        converted_news = sentence_to_features(news, feature_names)

        prediction_array = predict(converted_news.toarray(), values[2], values[3], values[0], values[1])

        prediction = 'true' if prediction_array[0] == 0 else 'fake'

        return_message = {'prediction': prediction}

        # send the message back
        self._set_headers()
        self.wfile.write(json.dumps(return_message).encode())

def run(server_class=HTTPServer, handler_class=Server, port=8008):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)

    print('Starting httpd on port %d...' % port)
    httpd.serve_forever()

run()
