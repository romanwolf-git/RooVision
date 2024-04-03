"""
routes for the results blueprint
"""

from flask import (
    Blueprint,
    render_template,
)

# Blueprint Configuration
results_blueprint = Blueprint(
    'results_bp', __name__,
    template_folder='templates',
    static_folder='static',
    static_url_path='/results/static'
)


@results_blueprint.route('/results')
def average_precision():
    return render_template('results.html')
