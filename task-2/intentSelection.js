import * as tf from "@tensorflow/tfjs";
import fs from "fs";

// Training data
const trainingData = [
  { text: "hi", intent: "greeting" },
  { text: "hello", intent: "greeting" },
  { text: "hey", intent: "greeting" },
  { text: "good morning", intent: "greeting" },

  { text: "burger", intent: "order_food" },
  { text: "pizza", intent: "order_food" },
  { text: "order coffee", intent: "order_food" },
  { text: "can i order fish bun", intent: "order_food" },

  { text: "bye", intent: "goodbye" },
  { text: "see you", intent: "goodbye" },
];

// Tokenizer
function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, "")
    .split(/\s+/)
    .filter(Boolean);
}

// Build vocabulary
const vocabulary = [];
trainingData.forEach((d) => {
  tokenize(d.text).forEach((word) => {
    if (!vocabulary.includes(word)) vocabulary.push(word);
  });
});

// Build intents
const intents = [];
trainingData.forEach((d) => {
  if (!intents.includes(d.intent)) intents.push(d.intent);
});

// Text → Bag of Words
function textToVector(text) {
  const tokens = tokenize(text);
  return vocabulary.map((word) => (tokens.includes(word) ? 1 : 0));
}

// Prepare tensors
const xs = tf.tensor2d(trainingData.map((d) => textToVector(d.text)));
const ys = tf.tensor2d(
  trainingData.map((d) =>
    intents.map((intent) => (intent === d.intent ? 1 : 0)),
  ),
);

// Build model
function createModel() {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [vocabulary.length],
      units: 16,
      activation: "relu",
    }),
  );
  model.add(tf.layers.dense({ units: intents.length, activation: "softmax" }));
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

let model = createModel();

// Train model
console.log("Training started...");
await model.fit(xs, ys, {
  epochs: 200,
  shuffle: true,
  verbose: 1,
  callbacks: {
    onEpochEnd: (epoch) =>
      console.log(`${(((epoch + 1) / 200) * 100).toFixed(2)}`),
  },
});
console.log("Training complete!");

// save model
async function saveModel(instance, path) {
  fs.mkdirSync(path, { recursive: true }); //create folder

  const handler = tf.io.withSaveHandler((artifacts) => artifacts);
  const artifacts = await instance.model.save(handler);

  const weightData = Buffer.from(artifacts.weightData);
  const outputData = JSON.stringify(instance.output);

  //model.json
  fs.writeFileSync(
    `${path}/model.json`,
    JSON.stringify({
      modelTopology: artifacts.modelTopology,
      format: artifacts.format,
      generatedBy: artifacts.generatedBy,
      convertedBy: artifacts.convertedBy,
      weightsManifest: [
        {
          paths: ["weights.bin"],
          weights: artifacts.weightSpecs,
        },
      ],
    }),
  );

  fs.writeFileSync(`${path}/weights.bin`, weightData); //weights.bin
  fs.writeFileSync(`${path}/output.json`, outputData); //output.json
}

// Save your model
await saveModel(
  { model, output: { vocabulary, intents, type: "text" } },
  "task-2/output",
);
console.log("Model saved to task-2/output!");

// Predict
function predict(text) {
  const vector = textToVector(text);
  const input = tf.tensor2d([vector]);
  const output = model.predict(input);
  const index = output.argMax(1).dataSync()[0];
  console.log(`"${text}" → ${intents[index]}`);
}

// Test predictions
predict("good evening");
predict("burger");
predict("i need pizza");
predict("bye guys");
