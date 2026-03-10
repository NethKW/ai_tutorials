import * as tf from "@tensorflow/tfjs";
import fs from "fs";
import jpeg from "jpeg-js";

const MODEL_PATH = "./task-3/output";

async function loadModel() {

  const json = JSON.parse(
    fs.readFileSync(`${MODEL_PATH}/model.json`)
  );

  const weightData = fs.readFileSync(`${MODEL_PATH}/weights.bin`);

  const model = await tf.loadLayersModel(
    tf.io.fromMemory({
      modelTopology: json.modelTopology,
      weightSpecs: json.weightSpecs,
      weightData: weightData
    })
  );

  console.log("Model loaded successfully!");

  return model;
}

function loadImage(filePath) {

  const jpegData = fs.readFileSync(filePath);
  const raw = jpeg.decode(jpegData, { useTArray: true });

  const buffer = new Uint8Array(raw.width * raw.height * 3);

  for (let i = 0, j = 0; i < raw.data.length; i += 4) {
    buffer[j++] = raw.data[i];
    buffer[j++] = raw.data[i + 1];
    buffer[j++] = raw.data[i + 2];
  }

  let tensor = tf.tensor3d(buffer, [raw.height, raw.width, 3]);

  tensor = tf.image.resizeNearestNeighbor(tensor, [128,128]);
  tensor = tensor.toFloat().div(255.0);

  return tensor.expandDims(0);
}

(async () => {

  const model = await loadModel();

  model.summary();

 const image = loadImage("./task-3/test3.jpg");

  const prediction = model.predict(image);

  prediction.print();

  const classes = ["cat","dog","elephant","parrot"];

  const result = prediction.argMax(-1).dataSync()[0];

  console.log("Prediction:", classes[result]);

})();