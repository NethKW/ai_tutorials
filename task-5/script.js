const video = document.getElementById("video");
const statusText = document.getElementById("status");

function startCamera() {
  statusText.innerText = "Requesting camera permission...";

  navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
      video.srcObject = stream;
      video.play();
      statusText.innerText = "Camera is ON!";
    })
    .catch(err => {
      console.error("Camera error:", err);
      statusText.innerText = "Camera permission denied or not working!";
      alert("Camera permission denied or not working!");
    });
}