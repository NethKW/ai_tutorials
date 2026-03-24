const uploadInput = document.getElementById("upload");
const canvas = document.getElementById("canvas");
const statusText = document.getElementById("status");

async function loadModel() {
  statusText.innerText = "Loading model...";
  await faceapi.nets.tinyFaceDetector.loadFromUri("../models");
  await faceapi.nets.faceLandmark68Net.loadFromUri("../models");
  statusText.innerText = "Model loaded!";
}

window.onload = loadModel;

uploadInput.addEventListener("change", async () => {
  const file = uploadInput.files[0];
  if (!file) return;

  const img = await faceapi.bufferToImage(file);

  // Set canvas size to image
  const maxWidth = 600;
  const maxHeight = 500;

  let scale = Math.min(maxWidth / img.width, maxHeight / img.height);

  scale = Math.min(scale, 1);

  const newWidth = img.width * scale;
  const newHeight = img.height * scale;

  canvas.width = newWidth;
  canvas.height = newHeight;

  const ctx = canvas.getContext("2d");

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, newWidth, newHeight);

  // detection face and face lansmarks
  const detections = await faceapi
    .detectAllFaces(img, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks();

  // Resize results
  const resizedDetections = faceapi.resizeResults(detections, {
    width: canvas.width,
    height: canvas.height,
  });

  // displaying detection result
  faceapi.draw.drawDetections(canvas, resizedDetections);
  faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);

  statusText.innerText = ``;
});
