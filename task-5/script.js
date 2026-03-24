const video = document.getElementById("video");
const statusText = document.getElementById("status");
const uploadInput = document.getElementById("upload");
const uploadCanvas = document.getElementById("uploadCanvas");

async function loadModel() {
  statusText.innerText = "Loading face detection model...";
  await faceapi.nets.tinyFaceDetector.loadFromUri("../models");
  statusText.innerText = "Model loaded successfully!";
}

window.onload = async () => {
  await loadModel();
};

// Start camera
async function startCamera() {
  const canvas = document.getElementById("canvas");

  if (!video || !canvas) return;

  statusText.innerText = "Requesting camera permission...";
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;

    // Wait for video to be ready
    video.onloadedmetadata = () => {
      video.play();
      statusText.innerText = "Camera is ON";

      // Get actual video dimensions
      const displaySize = {
        width: video.videoWidth,
        height: video.videoHeight,
      };

      // Set canvas size to match video
      canvas.width = displaySize.width;
      canvas.height = displaySize.height;

      // Match face-api dimensions
      faceapi.matchDimensions(canvas, displaySize);

      // Start detection loop
      setInterval(async () => {
        const detections = await faceapi.detectAllFaces(
          video,
          new faceapi.TinyFaceDetectorOptions(),
        );

        const resizedDetections = faceapi.resizeResults(
          detections,
          displaySize,
        );

        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        faceapi.draw.drawDetections(canvas, resizedDetections);
      }, 100);
    };
  } catch (err) {
    console.error("Camera error:", err);
    statusText.innerText = "Camera permission denied or not working!";
  }
}

//upload photo
uploadInput.addEventListener("change", async () => {
  const file = uploadInput.files[0];
  if (!file) return;

  const img = await faceapi.bufferToImage(file);
  const ctx = uploadCanvas.getContext("2d");

  // Clear previous
  ctx.clearRect(0, 0, uploadCanvas.width, uploadCanvas.height);

  // Draw image on canvas
  ctx.drawImage(img, 0, 0, uploadCanvas.width, uploadCanvas.height);

  // Detect faces
  const detections = await faceapi.detectAllFaces(
    img,
    new faceapi.TinyFaceDetectorOptions(),
  );

  const resizedDetections = faceapi.resizeResults(detections, {
    width: uploadCanvas.width,
    height: uploadCanvas.height,
  });

  faceapi.draw.drawDetections(uploadCanvas, resizedDetections);
});
