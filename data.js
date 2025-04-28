/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

const MNIST_IMAGES_SPRITE_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

/**
 * A class that fetches the sprited MNIST dataset and returns shuffled batches.
 *
 * NOTE: This will get much easier. For now, we do data fetching and
 * manipulation manually.
 */
export class MnistData {
  constructor() {
    this.shuffledTrainIndex = 0;
    this.shuffledTestIndex = 0;
  }

  async load() {
    // Make a request for the MNIST sprited image.
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const imgRequest = new Promise((resolve, reject) => {
      // Set the crossOrigin attribute to an empty string to allow loading images
      // from different domains (CORS - Cross-Origin Resource Sharing). 
      // 
      // What is crossOrigin?
      // The crossOrigin attribute is an HTML attribute that manages how the browser handles
      // requests for resources (like images) from different origins/domains.
      // - When set to '' (empty string) or 'anonymous', it makes a CORS request without sending user credentials
      // - When set to 'use-credentials', it makes a CORS request with credentials
      // 
      // Without this setting, the canvas would be "tainted" by cross-origin data,
      // preventing us from accessing pixel data with getImageData() due to security restrictions.
      // This is critical for our MNIST dataset processing where we need to extract
      // pixel values from the loaded sprite image.
      img.crossOrigin = ''; //necessarily set this to allow loading.
      img.onload = () => {
        const loadingIndicator = document.getElementById("loading-indicator");
        if (loadingIndicator) loadingIndicator.remove(); //remove loading screen
        img.width = img.naturalWidth; //so this is 784
        console.log("img.width: ",img.width);
        img.height = img.naturalHeight;

        // Create a buffer to store the pixel data of all MNIST images
        // The buffer size is 65000 * 784 * 4 = 203,840,000 bytes
        // This is because each image is 28x28 pixels, and each pixel has 4 values (RGBA)
        // So each image contributes 28 * 28 * 4 = 2560 bytes to the buffer
        // The buffer is a contiguous block of memory that stores all pixel data
        const datasetBytesBuffer =new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);

        const chunkSize = 5000;
        canvas.width = img.width; //784
        canvas.height = chunkSize;

        for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
          const datasetBytesView = new Float32Array(datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4, IMAGE_SIZE * chunkSize);
          // ctx.drawImage(image, sx, sy, sWidth, sHeight, dx, dy, dWidth, dHeight)
          // image: the source image
          // sx, sy: the source x,y coordinates to start copying from
          // sWidth, sHeight: the width and height of the source rectangle
          // dx, dy: the destination x,y coordinates to draw to
          // dWidth, dHeight: the width and height to draw the image in the destination
          ctx.drawImage(img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width, chunkSize);

          // ctx.getImageData(sx, sy, sw, sh, options) extracts pixel data from the canvas
          // - sx, sy: The x and y coordinates of the top-left corner to start extracting from
          // - sw, sh: The width and height of the rectangle to extract
          // - options: { willReadFrequently: true } is a performance hint that tells the browser
          //   we'll be reading from this canvas frequently, allowing it to optimize memory usage
          //   and potentially avoid creating a new copy of the data each time
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height, { willReadFrequently: true });

          // Loop through each pixel in the imageData
          // imageData.data is a flat array where each pixel uses 4 array elements (R,G,B,A)
          // So we divide the total length by 4 to get the number of pixels
          // j represents the index of each pixel in our destination array (datasetBytesView)
          for (let j = 0; j < imageData.data.length / 4; j++) {
            // All channels hold an equal value since the image is grayscale, so
            // just read the blue channel.
            //noramlize!!
            datasetBytesView[j] = imageData.data[j * 4] / 255;

            //so to my understanding imageData got a 28x width and basically 5000 height... confusing.
            //so wth is imageData.data.length?... well its divided by 4 so that means maybe we're navigating over a grayscale smth smth...hmm...
            //we're doing this for each 5,000 image chunk.. huh... we're adding it to canvas context, which I presume makes a new canvas in our 
            //page and creates it in html? Or is it a virtual holding space for the image data?
          }
        }
        this.datasetImages = new Float32Array(datasetBytesBuffer);

        resolve();
      };
      img.src = MNIST_IMAGES_SPRITE_PATH;
    });

    const labelsRequest = fetch(MNIST_LABELS_PATH);
    const [imgResponse, labelsResponse] = await Promise.all([imgRequest, labelsRequest]);

    this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

    // Create shuffled indices into the train/test set for when we select a
    // random dataset element for training / validation.
    this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
    console.log("trainIndices length via tf.util.createShuffledIndices: ",this.trainIndices.length);
    console.log("checking trainIndices via tf.util.createShuffledIndices: ",this.trainIndices);
    this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);

    // Slice the the images and labels into train and test sets.
    this.trainImages = this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.trainLabels = this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    this.testLabels = this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
  }

  nextTrainBatch(batchSize) {
    return this.nextBatch(batchSize, [this.trainImages, this.trainLabels], () => {
          this.shuffledTrainIndex = (this.shuffledTrainIndex + 1) % this.trainIndices.length;
          return this.trainIndices[this.shuffledTrainIndex];
        });
  }

  nextTestBatch(batchSize) {
    return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
      this.shuffledTestIndex = (this.shuffledTestIndex + 1) % this.testIndices.length;
      return this.testIndices[this.shuffledTestIndex];
    });
  }

  nextBatch(batchSize, data, index) {
    const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
    const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

    for (let i = 0; i < batchSize; i++) {
      const idx = index();

      const image = data[0].slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
      batchImagesArray.set(image, i * IMAGE_SIZE);

      const label = data[1].slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES);
      batchLabelsArray.set(label, i * NUM_CLASSES);
    }

    const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
    const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

    return {xs, labels};
  }
}