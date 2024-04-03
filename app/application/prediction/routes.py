"""
prediction routes for the application
"""
import os
import subprocess
from pathlib import Path
import pickle

from flask import (
    Blueprint,

    flash,
    jsonify,
    render_template,
    request,
    send_from_directory,
    session,
    url_for
)
from flask import current_app as app
from flask_uploads import UploadSet, IMAGES
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

# Blueprint Configuration
prediction_blueprint = Blueprint(
    'pred_bp', __name__,
    template_folder='templates',
    static_folder='static',
    static_url_path='/prediction/static'
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


@prediction_blueprint.route('/prediction')
def prediction():
    form = UploadForm()
    session['confidence'] = 50
    session['overlap'] = 30
    return render_template(template_name_or_list='prediction.html',
                           confidence=session.get('confidence'),
                           overlap=session.get('overlap'),
                           form=form)


@prediction_blueprint.route('/img_upload/<path:file_name>')
def get_file(file_name):
    """Route to get uploaded files."""
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], file_name)


@prediction_blueprint.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    """Route to handle image upload."""

    # instantiate upload form and load model
    form = UploadForm()

    if form.validate_on_submit():
        # Save the uploaded photo and flash
        img_name = photos.save(form.photo.data)

        # save name and type in session
        session['img_name'] = img_name
        session['img_type'] = 'img_upload'
        session['roo_heading'] = None

        # inform user
        flash('Image saved.')
    else:
        img_name = None
        flash('Image not validated or saved.')

    return render_template(template_name_or_list='prediction.html',
                           confidence=session.get('confidence'),
                           overlap=session.get('overlap'),
                           form=form,
                           img_type=session.get('img_type'),
                           img_name=img_name,
                           roo_heading=session.get('roo_heading'))


@prediction_blueprint.route('/display_image', methods=['POST'])
def display_image():
    # instantiate upload form and load model
    form = UploadForm()

    # get selection from dropdown and compose image name
    roo_value = request.form.get('testImagesSelect')
    img_name = roo_value + '.jpg'

    # save name and type in session
    session['img_name'] = img_name
    session['img_type'] = 'img_test'

    # create roo heading
    roo_dict = {'bridled_nail_tail_wallaby': 'Bridled nail-tail wallaby',
                'brush_tailed_rock_wallaby': 'Brush-tailed rock-wallaby',
                'eastern_grey_kangaroo': 'Eastern grey kangaroo',
                'red_kangaroo': 'Red kangaroo',
                'red_necked_wallaby': 'Red-necked wallaby',
                'swamp_wallaby': 'Swamp wallaby',
                'western_grey_kangaroo': 'Western grey kangaroo'}
    roo_heading = roo_dict[roo_value]
    session['roo_heading'] = roo_heading

    return render_template(template_name_or_list='prediction.html',
                           confidence=session.get('confidence'),
                           overlap=session.get('overlap'),
                           form=form,
                           img_type=session.get('img_type'),
                           img_name=img_name,
                           roo_heading=roo_heading)


@prediction_blueprint.route('/get_slider_value', methods=['POST'])
def get_slider_value():
    data = request.get_json()
    slider_value = int(data['slider_value'])

    if data['elementId'] == 'confidenceInput':
        session['confidence'] = slider_value
    if data['elementId'] == 'overlapInput':
        session['overlap'] = slider_value

    return jsonify(slider_value=slider_value)


@prediction_blueprint.route('/detect_object', methods=['POST'])
def detect_object():
    """Route to handle roo detection of an image using a YOLOv9 model"""

    # get session variables
    confidence = session.get('confidence')
    overlap = session.get('overlap')
    img_name_in = session.get('img_name')
    img_type = session.get('img_type')

    # compose image paths
    if img_type == 'img_test':
        img_path_in = os.path.join(os.getcwd(), app.config['TEST_PHOTOS_DEST'], img_name_in)
    elif img_type == 'img_upload':
        img_path_in = os.path.join(os.getcwd(), app.config['UPLOADED_PHOTOS_DEST'], img_name_in)
    else:
        # if detect_object was called before (multiple clicks on detect button)
        img_path_in = os.path.join(os.getcwd(), app.config['OUTPUT_PHOTOS_DEST'], img_name_in)

    img_out_dir = os.path.join(os.getcwd(), app.config['OUTPUT_PHOTOS_DEST'])

    # keep track of detection status
    session['img_type'] = 'img_detect'

    # get project directory and set paths for the detection script and the weights
    project_dir = Path.cwd().parent
    detect_path = project_dir / 'src/yolov9/detect.py'
    weights_path = project_dir / 'src/yolov9/runs/train/18-03-2024_64Batch_300Epochs/weights/best_striped.pt'

    # infer the image using the YOLOv9 model
    cmd = (f"python {detect_path} --weights {weights_path} --source {img_path_in} --img 640"
           f" --conf {confidence/100} --iou {overlap/100} --line-thickness 2 --img_output {img_out_dir}")

    # Execute the command
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    # Capture standard output and standard error
    stdout, stderr = process.communicate()

    # Print standard output
    print("Output:")
    print(stdout.decode("utf-8"))

    # Print standard error
    print("Errors:")
    print(stderr.decode("utf-8"))

    # inform user
    flash('Inference completed.')

    form = UploadForm()
    return render_template(template_name_or_list='prediction.html',
                           confidence=confidence,
                           overlap=overlap,
                           form=form,
                           img_type=session.get('img_type'),
                           img_name=session.get('img_name'),
                           roo_heading=session.get('roo_heading'))
