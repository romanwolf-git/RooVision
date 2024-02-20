"""
home routes
"""
from ..image_upload.routes import UploadForm

from flask import (
    Blueprint,
    render_template,
)

# Blueprint Configuration
home_blueprint = Blueprint(
    'home_blueprint', __name__,
    template_folder='templates',
    static_folder='static',
    static_url_path='/home/static'
)


@home_blueprint.route('/')
@home_blueprint.route('/index')
def index():
    form = UploadForm()
    return render_template(template_name_or_list='index.html', form=form)


@home_blueprint.route('/data')
def data():
    form = UploadForm()
    return render_template(template_name_or_list='data.html', form=form)


@home_blueprint.route('/model')
def model():
    form = UploadForm()
    return render_template(template_name_or_list='model.html', form=form)


@home_blueprint.route('/average_precision')
def average_precision():
    return render_template(template_name_or_list='average_precision.html')
