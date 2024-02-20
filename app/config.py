"""Flask configuration."""
from pathlib import Path
from os import environ, path
from dotenv import load_dotenv
import redis

# specificy `.env` file containing key/value config values
BASE_DIR = path.abspath(path.dirname(__file__))
env_file = path.join(BASE_DIR, ".env")
load_dotenv(env_file)


class Config:
    """Base config."""
    # General Config
    ENVIRONMENT = environ.get("ENVIRONMENT")

    # Flask Config
    FLASK_APP = "wsgi.py"
    FLASK_DEBUG = environ.get("FLASK_DEBUG")
    SECRET_KEY = environ.get("SECRET_KEY")

    # Flask-Session
    REDIS_URI = environ.get("REDIS_URI")
    SESSION_TYPE = "redis"
    SESSION_REDIS = redis.from_url(REDIS_URI)
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True

    # Flask-Assets
    STATIC_FOLDER = 'static'
    TEMPLATES_FOLDER = 'templates'
    UPLOADED_PHOTOS_DEST = 'application/image_upload/static/img'
    OUTPUT_PHOTOS_DEST = 'application/image_inference/static/output'
    MODEL_DEST = 'application/image_inference/static/models'

    # Flask-SQLAlchemy
    SQLALCHEMY_DATABASE_URI = environ.get("SQLALCHEMY_DATABASE_URI")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False

    # Roboflow
    PRIVATE_API_ROBOFLOW = environ.get("PRIVATE_API_ROBOFLOW")