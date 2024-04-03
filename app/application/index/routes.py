"""
index routes
"""
from ..prediction.routes import UploadForm

from flask import (
    Blueprint,
    render_template,
)

# Blueprint Configuration
index_blueprint = Blueprint(
    'index_bp', __name__,
    template_folder='templates',
    static_folder='static',
    static_url_path='/index/static'
)


@index_blueprint.route('/')
@index_blueprint.route('/index')
def index():
    form = UploadForm()
    return render_template(template_name_or_list='index.html', form=form)
