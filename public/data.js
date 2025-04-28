// data.js
// Minimal TFJS-based MNIST data loader
// The user’s existing code with slight expansions for returning batch info.

import * as tf from '@tensorflow/tfjs';

const MNIST_IMAGES_SPRITE_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

const IMAGE_SIZE = 28 * 28;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

const TRAIN_TEST_SPLIT = 55000;

export class MnistData {
  constructor() {
    this.shuffledTrainIndex = 0;
    this.shuffledTestIndex = 0;
  }

  async load() {
    // (Identical to user’s code)
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const imgRequest = new Promise((resolve, reject) => {
      img.crossOrigin = '';
      img.onload = () => resolve();
      img.onerror = (err) => reject(err);
    });

    img.src = MNIST_IMAGES_SPRITE_PATH;
    const labelsRequest = fetch(MNIST_LABELS_PATH);

    await Promise.all([imgRequest, labelsRequest]);

    const labelsResponse = await labelsRequest;
    this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

    const {width, height} = img;
    const datasetBytesBuffer =
      new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);

    const chunkSize = 5000;
    canvas.width = width;
    canvas.height = chunkSize;

    for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
      const datasetBytesView = new Float32Array(
        datasetBytesBuffer,
        i * IMAGE_SIZE * chunkSize * 4,
        IMAGE_SIZE * chunkSize
      );
      ctx.drawImage(
        img,
        0, i * chunkSize, width, chunkSize,
        0, 0, width, chunkSize
      );
      const imageData = ctx.getImageData(0, 0, width, chunkSize);

      for (let j = 0; j < imageData.data.length / 4; j++) {
        // All channels are same grayscale
        datasetBytesView[j] = imageData.data[j * 4] / 255;
      }
    }
    this.datasetImages = new Float32Array(datasetBytesBuffer);

    // Training index order
    this.trainIndices = tf.util.createShuffledIndices(TRAIN_TEST_SPLIT);
    // Test index order
    this.testIndices = tf.util.createShuffledIndices(NUM_DATASET_ELEMENTS - TRAIN_TEST_SPLIT);

    this.trainImages = this.datasetImages.slice(0, IMAGE_SIZE * TRAIN_TEST_SPLIT);
    this.testImages = this.datasetImages.slice(IMAGE_SIZE * TRAIN_TEST_SPLIT);
    this.trainLabels = this.datasetLabels.slice(0, NUM_CLASSES * TRAIN_TEST_SPLIT);
    this.testLabels = this.datasetLabels.slice(NUM_CLASSES * TRAIN_TEST_SPLIT);
  }

  nextTrainBatch(batchSize) {
    return this.nextBatch(
      batchSize,
      [this.trainImages, this.trainLabels, this.trainIndices, this.shuffledTrainIndex],
      idx => { this.shuffledTrainIndex = idx; }
    );
  }

  nextTestBatch(batchSize) {
    return this.nextBatch(
      batchSize,
      [this.testImages, this.testLabels, this.testIndices, this.shuffledTestIndex],
      idx => { this.shuffledTestIndex = idx; }
    );
  }

  nextBatch(batchSize, dataset, updateIndexFunc) {
    const [images, labels, indices, idx] = dataset;
    let newIndex = idx;
    const xs = new Float32Array(batchSize * IMAGE_SIZE);
    const labs = new Uint8Array(batchSize * NUM_CLASSES);

    for (let i = 0; i < batchSize; i++) {
      const idxVal = indices[newIndex];
      const imageOffset = idxVal * IMAGE_SIZE;
      const labelOffset = idxVal * NUM_CLASSES;
      xs.set(images.slice(imageOffset, imageOffset + IMAGE_SIZE), i * IMAGE_SIZE);
      labs.set(labels.slice(labelOffset, labelOffset + NUM_CLASSES), i * NUM_CLASSES);

      newIndex = (newIndex + 1) % indices.length;
    }
    updateIndexFunc(newIndex);

    return {
      xs: tf.tensor2d(xs, [batchSize, IMAGE_SIZE]),
      labels: tf.tensor2d(labs, [batchSize, NUM_CLASSES])
    };
  }
}
