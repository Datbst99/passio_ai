import os

from app import create_app
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, port=5500)


