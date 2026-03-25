const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const statusText = document.getElementById("statusText");

async function loadModel() {
  statusText.innerText = "Loading model...";
  await faceapi.nets.ssdMobilenetv1.loadFromUri("../models");
  statusText.innerText = "Model loaded!";

  videoDetection();
}

window.onload = loadModel;

function videoDetection() {
  const ctx = canvas.getContext("2d");

  video.addEventListener("play", () => {
    const displaySize = {
      width: video.clientWidth,
      height: video.clientHeight,
    };
    canvas.width = displaySize.width;
    canvas.height = displaySize.height;

    faceapi.matchDimensions(canvas, displaySize);

    async function detectFrame() {
      const detections = await faceapi.detectAllFaces(
        video,
        new faceapi.SsdMobilenetv1Options(),
      );

      console.log(detections);

      const resizedDetections = faceapi.resizeResults(detections, displaySize);
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      faceapi.draw.drawDetections(canvas, resizedDetections);

      requestAnimationFrame(detectFrame);
    }
    detectFrame();
  });
}
