import { arrayBuffer } from "stream/consumers";
const BATCH_SIZE = 512;
const TRAIN_DATA_SIZE = 55000;
const TEST_DATA_SIZE = 10000;
const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;
let EPOCH_AMOUNT = 10
let LR = 0.01;
const itersTilFullTrainingSetUsed = TRAIN_DATA_SIZE / BATCH_SIZE;



const MNIST_IMAGES_SPRITE_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

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
        img.height = img.naturalHeight; //what is naturalHeight?

        // Create a buffer to store the pixel data of all MNIST images
        // The buffer size is 65000 * 784 * 4 = 203,840,000 bytes
        // This is because each image is 28x28 pixels, and each pixel has 4 values (RGBA)
        // So each image contributes 28 * 28 * 4 = 2560 bytes to the buffer
        // The buffer is a contiguous block of memory that stores all pixel data
        const datasetBytesBuffer = new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);

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
          // easier to use a canvas as a temporary holding ground for the image data.
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
            //noramlize to between 0-1!!
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




const { model } = require("mongoose");



const blocks = document.querySelectorAll('.block');
// let r = document.getElementById("source-area");
// r.addEventListener("dragStart",function(event){
//     event.preventDefault();
// });

    // 1. Make blocks draggable
blocks.forEach(function(box) {
    box.addEventListener('dragstart', function(event) {
    // Store which box is being dragged
    event.dataTransfer.setData('text/plain', box.id);
    });
});

//2. set def. behavior
document.getElementById("target").addEventListener("dragover", function(event) {
    event.preventDefault();
});


document.getElementById("target").addEventListener('drop', function(event){
    let boxId = event.dataTransfer.getData('text/plain');
    let origBox = document.getElementById(boxId);
    const newBox = origBox.cloneNode(true);
    newBox.id = boxId + '-clone';
    document.getElementById("target").appendChild(newBox);
    // Add trash icon to the newly dropped block
    addTrashIconToBlock(newBox);
});


function addTrashIconToBlock(block) {
    // Create a new div element that will be our trash icon
    const trashIcon = document.createElement('div');
    // This is not creating a new CSS class definition, but rather
    // assigning the element to an existing CSS class named 'trash-icon'
    // The actual CSS styles for this class would be defined elsewhere in a stylesheet
    // This line simply tags the element so it can be styled according to that class definition
    // If the 'trash-icon' class doesn't exist in any stylesheet, the element will still
    // have the class name but no special styling would be applied
    trashIcon.classList.add('trash-icon');

    // Set its content
    trashIcon.innerHTML = '🗑️';
    
    // Add click event listener to remove the block when trash icon is clicked
    trashIcon.addEventListener('click', function(event) {
        // Prevent the event from bubbling up to parent elements
        event.stopPropagation();
        
        // Remove the block containing this trash icon
        block.remove();
    });
    
    // Insert the trash icon as the first child of the block
    block.insertBefore(trashIcon, block.firstChild);

}

function buildModel() { //remove all ||s
    console.log('Building TensorFlow model...');

    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;

    let modelArr = [];
    let lossFunc;
    let optimizer;
    const targetArea = document.getElementById("target-area");
    const children = targetArea.children;

    for (let i = 0; i < children.length; i++) {
        const child = children[i];

        if (i === 0 && child.id === "conv2d") {
            modelArr.push(tf.layers.conv2d({
                inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
                kernelSize: child.kernelSize || 3,
                filters: child.filters || 16,
                strides: child.strides || 1,
                kernelInitializer: 'varianceScaling'
            }));
        } 
        else if (child.id === "conv2d") {
            modelArr.push(tf.layers.conv2d({
                kernelSize: child.kernelSize || 3,
                filters: child.filters || 16,
                strides: child.strides || 1,
                kernelInitializer: 'varianceScaling'
            }));
        } 
        else if (child.id === "dense") {
            modelArr.push(tf.layers.dense({
                inputShape: child.inputChannels ? [parseInt(child.inputChannels)] : undefined,
                units: child.outputChannels ? parseInt(child.outputChannels) : 10,
            }));
        } 
        else if (child.id === "relu") {
            modelArr.push(tf.layers.activation({activation: 'relu'}));
        } 
        else if (child.id === "sigmoid") {
            modelArr.push(tf.layers.activation({activation: 'sigmoid'}));
        } 
        else if (child.id === "tanh") {
            modelArr.push(tf.layers.activation({activation: 'tanh'}));
        } 
        else if (child.id === "maxpool") {
            modelArr.push(tf.layers.maxPooling2d({
                poolSize: child.poolSize || [2, 2],
                strides: child.poolSize || [2, 2]
            }));
        }
        else if (child.id === "loss") {
            const root = document.getElementById('loss');

            // ② its two <select> elements are the only ones inside that block
            const [optSel, lossSel] = root.getElementsByTagName('select');
          
            // ③ values you need
            const optimizerName = optSel.value;          // 'sgd' | 'adam' | 'rmsprop'
            const lossName      = lossSel.value;         // 'categoricalCrossentropy'
          
            // ④ map string → actual tf.Optimizer instance
            const optimizer = {
              sgd:     tf.train.sgd,
              adam:    tf.train.adam,
              rmsprop: tf.train.rmsprop
            }[optimizerName]();
        }
    }
    return [modelArr, lossName, optimizer];
}

function trainInBrowser(itersPerLog=1){
    // Create proper data structures for logging
    const trainingLogs = [];
    const activationsByLayer = [];
    const gradientsByLayer = [];
    
    // Build the model from the UI components
    let [tModel, lossFunc, optimizer] = buildModel();
    
    // Load and prepare data
    const data = new MnistData();
    data.load().then(() => {
        // Get training data
        const [trainXs, trainYs] = tf.tidy(() => {
            const d = data.nextTrainBatch(BATCH_SIZE);
            return [
                d.xs.reshape([BATCH_SIZE, 28, 28, 1]),
                d.labels
            ];
        });
        
        // Get validation data
        const [testXs, testYs] = tf.tidy(() => {
            const d = data.nextTestBatch(TEST_DATA_SIZE);
            return [
                d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
                d.labels
            ];
        });
        
        // Initialize visualization
        createWeightsViz(tModel);
        
        // Training loop
        let epoch = 0;
        const train = async () => {
            if (epoch >= EPOCH_AMOUNT) {
                console.log("Training complete!");
                return;
            }
            
            console.log(`Starting epoch ${epoch + 1}/${EPOCH_AMOUNT}`);
            
            for (let iter = 0; iter < itersTilFullTrainingSetUsed; iter++) {
                // Get batch
                const batchXs = trainXs.slice([iter * BATCH_SIZE, 0, 0, 0], [BATCH_SIZE, 28, 28, 1]);
                const batchYs = trainYs.slice([iter * BATCH_SIZE, 0], [BATCH_SIZE, NUM_CLASSES]);
                
                // Forward pass with activation tracking
                const activations = [];
                let currentLayer = batchXs;
                activations.push(currentLayer);
                
                // Track forward pass through each layer
                for (let i = 0; i < tModel.layers.length; i++) {
                    const layer = tModel.layers[i];
                    // Process each layer in the model
                    currentLayer = layer.apply(currentLayer);
                    activations.push(currentLayer);
                    console.log(`Layer ${i}: ${layer.name}, type: ${layer.getClassName()}`);
                }
                
                // Calculate loss
                const logits = activations[activations.length - 1];
                const loss = tf.tidy(() => {
                    return tf.losses.softmaxCrossEntropy(batchYs, logits).mean();
                });
                
                // Calculate accuracy
                const predictions = tf.argMax(logits, 1);
                const labels = tf.argMax(batchYs, 1);
                const accuracy = tf.tidy(() => {
                    return predictions.equal(labels).mean();
                });
                
                // Log metrics
                const lossValue = loss.dataSync()[0];
                const accuracyValue = accuracy.dataSync()[0];
                
                trainingLogs.push({
                    epoch,
                    iter,
                    loss: lossValue,
                    accuracy: accuracyValue
                });
                
                // Find worst performers (highest loss)
                const sampleLosses = tf.losses.softmaxCrossEntropy(batchYs, logits);
                const lossValues = sampleLosses.dataSync();
                
                // Get indices of 10 worst performers
                const worstIndices = Array.from(lossValues)
                    .map((loss, idx) => ({ loss, idx }))
                    .sort((a, b) => b.loss - a.loss)
                    .slice(0, 10)
                    .map(item => item.idx);
                
                // Store information about worst performers
                const worstPerformers = worstIndices.map(idx => ({
                    index: idx,
                    image: batchXs.slice([idx, 0, 0, 0], [1, 28, 28, 1]),
                    prediction: predictions.slice([idx], [1]).dataSync()[0],
                    actual: labels.slice([idx], [1]).dataSync()[0],
                    loss: lossValues[idx]
                }));
                
                // Update visualizations if needed
                if (iter % itersPerLog === 0) {
                    // Update weight visualizations
                    updateWeightViz(tModel);
                    
                    // Update activation visualizations for the first sample
                    for (let i = 0; i < activations.length; i++) {
                        updateActivationsViz(i, activations[i].slice([0, 0, 0, 0], [1, -1, -1, -1]));
                    }
                    
                    // Update visualization of worst performers
                    updateWorstPerformersViz(worstPerformers);
                    
                    // Update loss and accuracy graph
                    updateLossAccuracyGraph(trainingLogs);
                }
                
                // Clean up tensors
                loss.dispose();
                accuracy.dispose();
                predictions.dispose();
                labels.dispose();
                sampleLosses.dispose();
            }
            
            epoch++;
            setTimeout(train, 0); // Continue to next epoch with setTimeout to prevent UI blocking
        };
        
        // Start training
        train();
    });
}

// Function to add right-click context menu to images
function addContextMenuToImages() {
    // Find all canvas elements that display MNIST images
    const canvases = document.querySelectorAll('.performer canvas');
    
    canvases.forEach((canvas, idx) => {
        // Add context menu event listener
        canvas.addEventListener('contextmenu', function(event) {
            event.preventDefault();
            
            // Get the performer div parent to extract image data
            const performerDiv = this.closest('.performer');
            const index = performerDiv.dataset.index;
            
            // Extract image tensor
            const image = tf.tensor(performerDiv.dataset.imageData)
                .reshape([1, 28, 28, 1]);
            
            // Show similar images
            showSimilarImages(index, image);
            
            return false;
        });
    });
}

// Modified updateWorstPerformersViz to store image data and add context menu
function updateWorstPerformersViz(worstPerformers) {
    const container = document.getElementById('worstPerformers');
    if (!container) return;
    
    container.innerHTML = '';
    
    worstPerformers.forEach((performer, idx) => {
        const performerDiv = document.createElement('div');
        performerDiv.className = 'performer';
        performerDiv.dataset.index = performer.index;
        
        // Store image data as base64 for retrieval
        const imageData = performer.image.dataSync();
        performerDiv.dataset.imageData = JSON.stringify(Array.from(imageData));
        
        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        
        // Display the image
        tf.browser.toPixels(performer.image.reshape([28, 28, 1]), canvas);
        
        const infoDiv = document.createElement('div');
        infoDiv.innerHTML = `
            <p>Predicted: ${performer.prediction}</p>
            <p>Actual: ${performer.actual}</p>
            <p>Loss: ${performer.loss.toFixed(4)}</p>
        `;
        
        performerDiv.appendChild(canvas);
        performerDiv.appendChild(infoDiv);
        container.appendChild(performerDiv);
    });
    
    // Add context menu to images
    addContextMenuToImages();
}

// Function to update loss and accuracy graph
function updateLossAccuracyGraph(logs) {
    const container = document.getElementById('lossAccuracyGraph');
    if (!container) return;
    
    // Extract data for plotting
    const iterations = logs.map((log, idx) => idx);
    const lossValues = logs.map(log => log.loss);
    const accuracyValues = logs.map(log => log.accuracy);
    
    // Create data for tfjs-vis
    const lossData = {
        values: lossValues.map((y, x) => ({x, y})),
        series: 'Loss'
    };
    
    const accuracyData = {
        values: accuracyValues.map((y, x) => ({x, y})),
        series: 'Accuracy'
    };
    
    // Configure the chart
    const opts = {
        xLabel: 'Iteration',
        yLabel: 'Value',
        width: 600,
        height: 300,
        series: ['Loss', 'Accuracy']
    };
    
    // Render the chart
    tfvis.render.linechart(container, [lossData, accuracyData], opts);
}

function updateWeightViz(){
    
}
function updateGradsViz(layerIndex, gradientTensors) {
    const container = document.getElementById(`grads-${layerIndex}`);
    container.innerHTML = '';

    gradientTensors.forEach(tensor => {
        const canvas = document.createElement('canvas');
        tf.browser.toPixels(tensor.squeeze().clipByValue(0, 1), canvas);
        container.appendChild(canvas);
    });
}
function updateActivationsViz(layerIndex, activationTensor) {
    const container = document.getElementById(`activations-${layerIndex}`);
    container.innerHTML = '';

    if (activationTensor.shape.length > 2) {
        const slices = activationTensor.split(activationTensor.shape[-1], -1); // slice along channels
        slices.forEach(slice => {
            const canvas = document.createElement('canvas');
            tf.browser.toPixels(slice.squeeze().clipByValue(0, 1), canvas);
            container.appendChild(canvas);
        });
    } else {
        const canvas = document.createElement('canvas');
        tf.browser.toPixels(activationTensor.squeeze().clipByValue(0, 1), canvas);
        container.appendChild(canvas);
    }
}
function updatePredsViz(){}
function createWeightsViz(modelLayers) {
    const vizContainer = document.getElementById('modelLayerViz');
    vizContainer.innerHTML = ''; // clear previous if any

    for (let i = 0; i < modelLayers.length; i++) {
        const layerDiv = document.createElement('div');
        layerDiv.id = `layer-viz-${i}`;
        layerDiv.innerHTML = `<h4>Layer ${i}: ${modelLayers[i].getClassName()}</h4>
                              <div id="weights-${i}"></div>
                              <div id="activations-${i}"></div>
                              <div id="grads-${i}"></div>`;
        vizContainer.appendChild(layerDiv);
    }
}

// Function to perform PCA for dimensionality reduction on image embeddings
function performPCA(images, targetDimensions = 10) {
    return tf.tidy(() => {
        // Flatten images to vectors
        const flattenedImages = images.reshape([images.shape[0], -1]);
        
        // Center the data by subtracting the mean
        const mean = flattenedImages.mean(0);
        const centeredData = flattenedImages.sub(mean);
        
        // Compute covariance matrix
        const cov = tf.matMul(centeredData.transpose(), centeredData)
            .div(tf.scalar(images.shape[0] - 1));
        
        // Get eigenvectors using SVD
        const {u, s} = tf.linalg.svd(cov);
        
        // Take top components (eigenvectors)
        const principalComponents = u.slice([0, 0], [-1, targetDimensions]);
        
        // Project data onto principal components
        const projectedData = tf.matMul(centeredData, principalComponents);
        
        return {
            projectedData,           // Low-dimensional representation
            principalComponents,      // The PCA transformation matrix
            mean                      // Mean that was subtracted (needed for projecting new points)
        };
    });
}

// Function to find similar images using PCA embeddings
function findSimilarImages(sourceImage, allImages, pcaState, numToReturn = 10) {
    return tf.tidy(() => {
        // Flatten source image
        const flattenedSource = sourceImage.reshape([1, -1]);
        
        // Center using the same mean as PCA
        const centeredSource = flattenedSource.sub(pcaState.mean);
        
        // Project onto principal components
        const projectedSource = tf.matMul(centeredSource, pcaState.principalComponents);
        
        // Calculate distances to all other projected points
        const distances = tf.sum(
            tf.square(tf.sub(projectedSource, pcaState.projectedData)), 1
        );
        
        // Get indices of smallest distances
        const distanceValues = distances.dataSync();
        const indices = Array.from(distanceValues)
            .map((dist, idx) => ({ dist, idx }))
            .sort((a, b) => a.dist - b.dist)
            .slice(0, numToReturn)
            .map(item => item.idx);
        
        return indices;
    });
}

// Function to handle "See similar images" request
function showSimilarImages(sourceIndex, sourceImage) {
    // Check if we've already computed PCA
    if (!window.pcaState) {
        // Get subset of images for PCA (to keep computation manageable)
        const sampleSize = 5000;
        const sampleIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS)
            .slice(0, sampleSize);
            
        // Create tensor of sample images
        const sampleImages = tf.tidy(() => {
            const d = data.nextTrainBatch(sampleSize);
            return d.xs.reshape([sampleSize, 28, 28, 1]);
        });
        
        // Perform PCA
        window.pcaState = performPCA(sampleImages, 20); // Reduce to 20 dimensions
    }
    
    // Find similar images
    const similarIndices = findSimilarImages(sourceImage, null, window.pcaState);
    
    // Display similar images
    displaySimilarImages(sourceImage, similarIndices);
}

// Function to display similar images
function displaySimilarImages(sourceImage, similarIndices) {
    const container = document.getElementById('similarImages');
    if (!container) return;
    
    container.innerHTML = '<h3>Similar Images</h3>';
    
    // Display source image
    const sourceDiv = document.createElement('div');
    sourceDiv.className = 'source-image';
    sourceDiv.innerHTML = '<h4>Source Image</h4>';
    
    const sourceCanvas = document.createElement('canvas');
    sourceCanvas.width = 28;
    sourceCanvas.height = 28;
    tf.browser.toPixels(sourceImage.reshape([28, 28, 1]), sourceCanvas);
    
    sourceDiv.appendChild(sourceCanvas);
    container.appendChild(sourceDiv);
    
    // Create container for similar images
    const similarContainer = document.createElement('div');
    similarContainer.className = 'similar-images-container';
    container.appendChild(similarContainer);
    
    // Display each similar image
    similarIndices.forEach(async (idx) => {
        const imageDiv = document.createElement('div');
        imageDiv.className = 'similar-image';
        
        // Get the image data
        const d = data.nextTrainBatch(1, () => idx);
        const image = d.xs.reshape([28, 28, 1]);
        
        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        await tf.browser.toPixels(image, canvas);
        
        imageDiv.appendChild(canvas);
        similarContainer.appendChild(imageDiv);
        
        // Clean up tensors
        image.dispose();
    });
}

// Start training when button is clicked
document.getElementById('startTrainingButton').addEventListener('click', function() {
    trainInBrowser(5); // Update visualization every 5 iterations
});

