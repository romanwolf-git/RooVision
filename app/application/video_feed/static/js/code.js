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