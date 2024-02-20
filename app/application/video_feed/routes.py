import cv2
from flask import Blueprint, Response

# Blueprint Configuration
video_feed_blueprint = Blueprint(
    'video_feed_bp', __name__,
    template_folder='templates',
    static_folder='static'
)


def gen_frames():
    """Generate video frames from the video_feed.
    This function turns on the video_feed, reads frames in a loop, encodes them as JPEG,
    and yields the frames for streaming. The loop breaks if there is an issue
    reading frames or if the 'q' key is pressed.
    Yields:
        bytes: JPEG-encoded video frames as bytes.
    """
    camera = cv2.VideoCapture(0)
    while True:
        # read the video_feed frame
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@video_feed_blueprint.route('/video_feed', methods=['GET', 'POST'])
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')