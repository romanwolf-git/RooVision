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

    # inject new variables/functions into all templates
    # Store something in session
    with app.app_context():
        from .index import routes as idx_routes
        from .prediction import routes as pred_routes
        from .data import routes as data_routes
        from .model import routes as model_routes
        from .results import routes as results_routes

        from .prediction.routes import photos

        configure_uploads(app, photos)

        # register blueprints
        app.register_blueprint(idx_routes.index_blueprint)
        app.register_blueprint(pred_routes.prediction_blueprint)
        app.register_blueprint(data_routes.data_blueprint)
        app.register_blueprint(model_routes.model_blueprint)
        app.register_blueprint(results_routes.results_blueprint)

    return app
