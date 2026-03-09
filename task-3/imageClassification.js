import * as tf from "@tensorflow/tfjs";
import fs from "fs";
import path from "path";
import jpeg from "jpeg-js";

const IMAGE_SIZE = 128;
const DATASET_PATH = "./task-3/dataset";
const SAVE_PATH = "task-3/output";

function loadImage(filePath) {
  const jpegData = fs.readFileSync(filePath);
  const rawImageData = jpeg.decode(jpegData, { useTArray: true });

  const { width, height, data } = rawImageData;

  // Convert RGBA -> RGB
  const buffer = new Uint8Array(width * height * 3);

  for (let i = 0, j = 0; i < data.length; i += 4) {
    buffer[j++] = data[i];
    buffer[j++] = data[i + 1];
    buffer[j++] = data[i + 2];
  }

  let tensor = tf.tensor3d(buffer, [height, width, 3]);

  tensor = tf.image.resizeNearestNeighbor(tensor, [IMAGE_SIZE, IMAGE_SIZE]); //resize image
  tensor = tensor.toFloat().div(255.0);

  return tensor;
}

function loadDataset() {
  const classNames = fs.readdirSync(DATASET_PATH);
  const images = [];
  const targets = [];

  classNames.forEach((label, index) => {
    const folderPath = path.join(DATASET_PATH, label);
    const files = fs.readdirSync(folderPath);

    files.forEach((file) => {
      const imagePath = path.join(folderPath, file);
      const imageTensor = loadImage(imagePath);

      images.push(imageTensor);
      targets.push(index);
    });
  });

  return {
    images: tf.stack(images),
    labels: tf.oneHot(tf.tensor1d(targets, "int32"), classNames.length),
    classNames,
  };
}

function createModel(numClasses) {
  const model = tf.sequential();

  model.add(
    tf.layers.conv2d({
      inputShape: [IMAGE_SIZE, IMAGE_SIZE, 3],
      filters: 16,
      kernelSize: 3,
      activation: "relu",
    }),
  );

  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  model.add(
    tf.layers.conv2d({
      filters: 32,
      kernelSize: 3,
      activation: "relu",
    }),
  );

  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  model.add(tf.layers.flatten());

  model.add(
    tf.layers.dense({
      units: 64,
      activation: "relu",
    }),
  );

  model.add(
    tf.layers.dense({
      units: numClasses,
      activation: "softmax",
    }),
  );

  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

async function saveModel(model) {
  await model.save(
    tf.io.withSaveHandler(async (artifacts) => {
      fs.writeFileSync(
        `${SAVE_PATH}/model.json`,
        JSON.stringify({
          modelTopology: artifacts.modelTopology,
          weightSpecs: artifacts.weightSpecs,
        }),
      );

      fs.writeFileSync(
        `${SAVE_PATH}/weights.bin`,
        Buffer.from(artifacts.weightData),
      );

      console.log("Model saved in output folder");

      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          modelTopologyType: "JSON",
        },
      };
    }),
  );
}

(async () => {
  console.log("Loading dataset...");
  const { images, labels, classNames } = loadDataset();

  console.log("Dataset size:", images.shape);
  console.log("Classes:", classNames);

  console.log("Creating model...");
  const model = createModel(classNames.length);

  console.log("Training...");
  await model.fit(images, labels, {
    epochs: 5,
    batchSize: 8,
    shuffle: true,
    callbacks: {
      onEpochBegin: (epoch) => {
        console.log(`Starting Epoch ${epoch + 1}`);
      },
      onEpochEnd: (epoch, logs) => {
        console.log(
          `Epoch ${epoch + 1} complete | loss=${logs.loss.toFixed(4)} | accuracy=${logs.acc || logs.accuracy}`,
        );
      },
    },
  });

  console.log("Training Complete!");

  if (!fs.existsSync(SAVE_PATH)) {
    fs.mkdirSync(SAVE_PATH, { recursive: true });
  }

  await saveModel(model);

  console.log("Model saved successfully!");
})();
