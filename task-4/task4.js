import * as tf from "@tensorflow/tfjs";
import * as use from "@tensorflow-models/universal-sentence-encoder";

// Training data
const trainingData = [
  { text: "hi", intent: "greeting" },
  { text: "hello", intent: "greeting" },
  { text: "good morning", intent: "greeting" },

  { text: "burger", intent: "order_food" },
  { text: "burger", intent: "order_food" },
  { text: "i want food", intent: "order_food" },
  { text: "order coffee", intent: "order_food" },

  { text: "bye", intent: "goodbye" },
  { text: "see you", intent: "goodbye" },
  { text: "catch you later", intent: "goodbye" },
];

const intents = [...new Set(trainingData.map(d => d.intent))];

async function main() {

  console.log("Loading USE model...");
  const useModel = await use.load();

  // Convert sentences → embeddings
  const sentences = trainingData.map(d => d.text);
  const embeddings = await useModel.embed(sentences); //convert into vector

  const xs = embeddings;

  const ys = tf.tensor2d(
    trainingData.map(d =>
      intents.map(intent => intent === d.intent ? 1 : 0)
    )
  );

  // Build classifier model
  const model = tf.sequential();

  model.add(tf.layers.dense({
    inputShape: [512], // USE embedding size
    units: 32,
    activation: "relu"
  }));

  model.add(tf.layers.dense({
    units: intents.length,
    activation: "softmax"
  }));

  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });

  console.log("Training...");
  await model.fit(xs, ys, {
    epochs: 200,
    shuffle: true
  });

  console.log("Training complete");

  async function predict(text) {

    const embedding = await useModel.embed([text]);//convert input text to embedding

    const prediction = model.predict(embedding);

    const index = prediction.argMax(-1).dataSync()[0];

    console.log(`"${text}" → ${intents[index]}`);
  }

  await predict("have a nice day");
  await predict("bring me some food");
  await predict("hey,how are you?");
  await predict("until next time");
  await predict("glad to see you");
  await predict("take care guys");

}

main();