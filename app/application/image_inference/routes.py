"""
home routes
"""
import pickle
import os
from flask import (
    Blueprint,

    flash,
    jsonify,
    render_template,
    request,
    session,
    url_for
)
from flask import current_app as app
from ..image_upload.routes import UploadForm

# Blueprint Configuration
image_inference_blueprint = Blueprint(
    'image_inference_bp', __name__,
    template_folder='templates',
    static_folder='static',
    static_url_path='/image_inference/static'
)


@image_inference_blueprint.route('/get_confidence', methods=['POST'])
def get_confidence():
    data = request.get_json()
    confidence = int(data['confidence'])
    session['confidence'] = confidence
    return jsonify(confidence=confidence)


@image_inference_blueprint.route('/detect_object', methods=['POST'])
def detect_object():
    img_name_in = session.get('img_name')
    img_name_out = '_inf.'.join(img_name_in.rsplit('.', 1))
    img_path_in = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], img_name_in)
    img_path_out = os.path.join(app.config['OUTPUT_PHOTOS_DEST'], img_name_out)
    confidence = session.get('confidence')
    iou_thresh = 30

    # open model from pickle file
    model_path = os.path.join(app.config['MODEL_DEST'], 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # predict the model, functionality from roboflow package
    model.predict(
        img_path_in,
        confidence=confidence,
        overlap=iou_thresh
    ).save(img_path_out)

    flash('Inference completed.')

    # Render the template with the form and file URL
    form = UploadForm()
    return render_template('index.html', form=form, img_name_out=img_name_out)
