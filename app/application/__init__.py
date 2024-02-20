"""Initialize app."""
from flask import Flask
from flask_session import Session
from flask_uploads import configure_uploads

session = Session()


def create_app():
    """Construct the core flask_session_tutorial."""
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object('config.Config')

    # Initialize Plugins
    session.init_app(app)

    # sys.path.append(str(Path.cwd().parent))
    # from src.model_architecture.get_model import get_roboflow_model
    # model = get_roboflow_model()

    # inject new variables/functions into all templates
    # Store something in session
    with app.app_context():
        from .home import routes as home_routes
        from .image_upload import routes as img_routes
        from .image_inference import routes as img_inf_routes
        from .video_feed import routes as vid_routes
        from .image_upload.routes import photos

        configure_uploads(app, photos)

        # register blueprints
        app.register_blueprint(home_routes.home_blueprint)
        app.register_blueprint(img_routes.image_upload_blueprint, url_prefix='/image_upload')
        app.register_blueprint(img_inf_routes.image_inference_blueprint, url_prefix='/image_inf')
        app.register_blueprint(vid_routes.video_feed_blueprint)

    return app
