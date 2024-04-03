"""
routes for the model blueprint
"""

from flask import (
    Blueprint,
    render_template,
)

# Blueprint Configuration
model_blueprint = Blueprint(
    'model_bp', __name__,
    template_folder='templates',
    static_folder='static',
    static_url_path='/model/static'
)


@model_blueprint.route('/model')
def model():
    return render_template('model.html')
