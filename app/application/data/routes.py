"""
prediction routes for the application
"""
from flask import (
    Blueprint,
    render_template,
)

# Blueprint Configuration
data_blueprint = Blueprint(
    'data_bp', __name__,
    template_folder='templates',
    static_folder='static',
    static_url_path='/data/static'
)


@data_blueprint.route('/data')
def data():
    return render_template('data.html')