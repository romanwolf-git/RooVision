import os
from application import create_app

app = create_app()

if __name__ == "__main__":
    server_port = os.environ.get('PORT', '8080')
    app.run(port=server_port, host='0.0.0.0')
