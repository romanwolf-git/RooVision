"""Flask configuration."""
from pathlib import Path
from os import environ, path
from dotenv import load_dotenv
from google.cloud import secretmanager
import redis

# specificy `.env` file containing key/value config values
BASE_DIR = path.abspath(path.dirname(__file__))
env_file = path.join(BASE_DIR, ".env")
load_dotenv(env_file)


def get_secret(secret_version_id):
    """
    Retrieve a secret from Google secret manager.
    :arg: secret_id (str): The ID of the secret to retrieve.
    :return:
        str: The secret value.
    """
    # Create the Secret Manager client.
    client = secretmanager.SecretManagerServiceClient()
    # Build the resource name of the secret.
    response = client.access_secret_version(name=secret_version_id)
    # Get the secret payload.
    payload = response.payload.data.decode("UTF-8")
    return payload


class Config:
    """Base config."""
    # General Config
    ENVIRONMENT = environ.get("ENVIRONMENT")

    # Flask Config
    FLASK_APP = "wsgi.py"
    FLASK_DEBUG = 0
    SECRET_KEY = get_secret("projects/818170133534/secrets/SECRET_KEY_FLASK/versions/latest")

    # Flask-Session
    REDIS_URI = get_secret("projects/818170133534/secrets/REDIS_URI/versions/latest")
    SESSION_TYPE = "redis"
    SESSION_REDIS = redis.from_url(REDIS_URI)
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True

    # Flask-Assets
    STATIC_FOLDER = 'static'
    TEMPLATES_FOLDER = 'templates'
    UPLOADED_PHOTOS_DEST = 'application/prediction/static/img_upload'
    TEST_PHOTOS_DEST = 'application/prediction/static/img_test'
    OUTPUT_PHOTOS_DEST = 'application/prediction/static/img_output'
    MODEL_DEST = 'application/prediction/static/models'

    # Roboflow
    PRIVATE_API_ROBOFLOW = get_secret("projects/818170133534/secrets/PRIVATE_API_ROBOFLOW/versions/latest")
