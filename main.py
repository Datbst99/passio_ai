import os
import argparse
from app import create_app

app = create_app()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000, help="Port running")
    opt = parser.parse_args()
    app.run(debug=False, port=opt.port)


