let cameraOn = false;

function toggleCameraState() {
    const camera_button = document.getElementById('camera_button');
    const video_display = document.getElementById('video_display');

    // Toggle camera state
    cameraOn = !cameraOn;

    // Update UI
    if (cameraOn) {
        startCamera(video_display, camera_button);
    } else {
        stopCamera(video_display, camera_button);
    }
}

function startCamera(video_display, camera_button) {
    // Start camera
    console.log(videoFeedUrl)
    video_display.src = videoFeedUrl;
    camera_button.innerText = 'Camera off';
}

function stopCamera(video_display, camera_button) {
    // Stop camera
    video_display.src = "";
    camera_button.innerText = 'Camera on';
}

window.onload = function() {
    let slider = document.getElementById('rangeInput');
    let output = document.getElementById('demo');
    output.innerHTML = slider.value;

    slider.oninput = function() {
        output.innerHTML = this.value;
        sendConfidence(this.value);
    };

    slider.dispatchEvent(new Event('input'));
};

function sendConfidence(confidence) {
    $.ajax({
        url: '/get_confidence',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ 'confidence': confidence }),
        success: function(response) {
            console.log("confidence:", response.confidence);
        },
        error: function(error) {
            console.log("Error:", error);
        }
    });
}
