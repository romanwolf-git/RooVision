// Dropdown selection
function selectInput() {
  var selectElement = document.getElementById("imageOption");
  var selectedOption = selectElement.value;
  console.log("Selected option: " + selectedOption);

  if (selectedOption === "upload_image") {
    document.getElementById("formImageUpload").style.display = "flex";
    document.getElementById("testImages").style.display = "none";
  }
  if (selectedOption === "use_test_image") {
    document.getElementById("formImageUpload").style.display = "none";
    document.getElementById("testImages").style.display = "flex";
  }
}

// Slider
window.onload = function() {
    initializeSlider('confidenceInput', 'confidenceValue');
    initializeSlider('overlapInput', 'overlapValue');
};

function initializeSlider(sliderId, outputId) {
    let slider = document.getElementById(sliderId);
    let output = document.getElementById(outputId);
    output.innerHTML = slider.value;

    slider.oninput = function() {
        output.innerHTML = this.value;
        sendSliderValue(this.id, this.value);
    };

    slider.dispatchEvent(new Event('input'));
}

function sendSliderValue(elementId, slider_value) {
    $.ajax({
        url: '/get_slider_value',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({'elementId': elementId, 'slider_value': slider_value }),
        success: function(response) {
            console.log("slider_value: ", response.slider_value);
        },
        error: function(error) {
            console.log("Error:", error);
        }
    });
}
