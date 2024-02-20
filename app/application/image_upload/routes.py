import os
from flask import Blueprint, request, send_from_directory, session, flash, url_for, render_template
from flask import current_app as app
from flask_uploads import UploadSet, IMAGES
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField


# Blueprint Configuration
image_upload_blueprint = Blueprint(
    'image_upload_bp', __name__,
    template_folder='templates',
    static_folder='static',
    static_url_path='/image_upload/static'
)

# instantiate photos
photos = UploadSet('photos', IMAGES)


class UploadForm(FlaskForm):
    """Form class for uploading images."""
    photo = FileField(
        validators=[
            FileAllowed(photos, 'Only images are allowed'),
            FileRequired('File field should not be empty')
        ]
    )
    submit = SubmitField('Upload')


@image_upload_blueprint.route('/img/<path:file_name>')
def get_file(file_name):
    """Route to get uploaded files."""
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], file_name)


@image_upload_blueprint.route('/', methods=['GET', 'POST'])
def upload_image():
    """Route to handle image upload."""
    form = UploadForm()
    if form.validate_on_submit():
        # Save the uploaded photo and flash
        img_name = photos.save(form.photo.data)
        session['img_name'] = img_name
        flash('Photo saved.')
    else:
        img_name = None
    return render_template('index.html', form=form, img_name=img_name)
