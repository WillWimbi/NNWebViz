const BATCH_SIZE = 512;
const TRAIN_DATA_SIZE = 55000;
const TEST_DATA_SIZE = 10000;
const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;
const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;
let EPOCH_AMOUNT = 10;
let LR = 0.01;
const itersTilFullTrainingSetUsed = TRAIN_DATA_SIZE / BATCH_SIZE;
let history = {};

document.addEventListener('DOMContentLoaded', async () => {
    console.log('Initializing MNIST data...');
    try {
        const data = new MnistData();
        await data.load();
        console.log('MNIST data loaded successfully');
        window.mnistData = data;
        const dataReadyEvent = new CustomEvent('mnistDataReady', { detail: { data } });
        document.dispatchEvent(dataReadyEvent);
    } catch (error) {
        console.error('Error loading MNIST data:', error);
        loadingIndicator.textContent = 'Error loading dataset. Please refresh the page.';
        loadingIndicator.style.backgroundColor = 'rgba(255, 0, 0, 0.7)';
    }
});

const MNIST_IMAGES_SPRITE_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

class MnistData {
    constructor() {
        this.shuffledTrainIndex = 0;
        this.shuffledTestIndex = 0;
    }

    async load() {
        const img = new Image();
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const imgRequest = new Promise((resolve, reject) => {
            img.crossOrigin = '';
            img.onload = () => {
                const loadingIndicator = document.getElementById("loading-indicator");
                if (loadingIndicator) loadingIndicator.remove();
                img.width = img.naturalWidth;
                console.log("img.width: ", img.width);
                img.height = img.naturalHeight;
                const datasetBytesBuffer = new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);
                const chunkSize = 5000;
                canvas.width = img.width;
                canvas.height = chunkSize;

                for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
                    const datasetBytesView = new Float32Array(
                        datasetBytesBuffer,
                        i * IMAGE_SIZE * chunkSize * 4,
                        IMAGE_SIZE * chunkSize
                    );
                    ctx.drawImage(
                        img,
                        0,
                        i * chunkSize,
                        img.width,
                        chunkSize,
                        0,
                        0,
                        img.width,
                        chunkSize
                    );
                    const imageData = ctx.getImageData(
                        0,
                        0,
                        canvas.width,
                        canvas.height,
                        { willReadFrequently: true }
                    );

                    for (let j = 0; j < imageData.data.length / 4; j++) {
                        datasetBytesView[j] = imageData.data[j * 4] / 255;
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
        this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
        console.log("trainIndices length via tf.util.createShuffledIndices: ", this.trainIndices.length);
        console.log("checking trainIndices via tf.util.createShuffledIndices: ", this.trainIndices);
        this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);
        this.trainImages = this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
        this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
        this.trainLabels = this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
        this.testLabels = this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    }

    nextTrainBatch(batchSize) {
        return this.nextBatch(
            batchSize,
            [this.trainImages, this.trainLabels],
            () => {
                this.shuffledTrainIndex = (this.shuffledTrainIndex + 1) % this.trainIndices.length;
                return this.trainIndices[this.shuffledTrainIndex];
            }
        );
    }

    nextTestBatch(batchSize) {
        return this.nextBatch(
            batchSize,
            [this.testImages, this.testLabels],
            () => {
                this.shuffledTestIndex = (this.shuffledTestIndex + 1) % this.testIndices.length;
                return this.testIndices[this.shuffledTestIndex];
            }
        );
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
        return { xs, labels };
    }
}

const blocks = document.querySelectorAll('.block');
blocks.forEach(function(box) {
    box.addEventListener('dragstart', function(event) {
        event.dataTransfer.setData('text/plain', box.id);
    });
});

document.getElementById("target").addEventListener("dragover", function(event) {
    event.preventDefault();
});

document.getElementById("target").addEventListener('drop', function(event) {
    let boxId = event.dataTransfer.getData('text/plain');
    let origBox = document.getElementById(boxId);
    const newBox = origBox.cloneNode(true);
    newBox.id = boxId + '-clone';
    document.getElementById("target").appendChild(newBox);
    addTrashIconToBlock(newBox);
});

function addTrashIconToBlock(block) {
    const trashIcon = document.createElement('div');
    trashIcon.classList.add('trash-icon');
    trashIcon.innerHTML = '🗑️';
    trashIcon.addEventListener('click', function(event) {
        event.stopPropagation();
        block.remove();
    });
    block.insertBefore(trashIcon, block.firstChild);
}

function buildModel() {
    console.log('Building TensorFlow model...');
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;
    let modelArr = [];
    let lossFunc;
    let optimizer;
    const targetArea = document.getElementById("target");
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
            modelArr.push(tf.layers.activation({ activation: 'relu' }));
        } 
        else if (child.id === "sigmoid") {
            modelArr.push(tf.layers.activation({ activation: 'sigmoid' }));
        } 
        else if (child.id === "tanh") {
            modelArr.push(tf.layers.activation({ activation: 'tanh' }));
        } 
        else if (child.id === "maxpool") {
            modelArr.push(tf.layers.maxPooling2d({
                poolSize: child.poolSize || [2, 2],
                strides: child.poolSize || [2, 2]
            }));
        }
        else if (child.id === "loss") {
            const root = document.getElementById('loss');
            const [optSel, lossSel] = root.getElementsByTagName('select');
            const optimizerName = optSel.value;
            // const lossName = lossSel.value;
            const optimizer = {
                sgd: tf.train.sgd,
                adam: tf.train.adam,
            }[optimizerName]();
        }
    }
    return [modelArr, optimizer];
}

function snapshot(t, meta = {}) {
    return {
      ...meta,                  // layer name, idx, etc.
      shape : t.shape.slice(),  // clone so it won’t mutate
      vals  : Array.from(t.dataSync())  // plain JS array for easy JSON/vis
    };
}
  async function trainOnlyLayers(data) {
  
      const layers = [
    
        tf.layers.conv2d({
          inputShape: [28, 28, 1],
          kernelSize: 5,
          filters: 8,
          strides: 1,
          activation: 'relu',
          kernelInitializer: 'varianceScaling'
        }),
      
        tf.layers.maxPooling2d({
          poolSize: [2, 2],
          strides: [2, 2]
        }),
      
        tf.layers.conv2d({
          kernelSize: 5,
          filters: 16,
          strides: 1,
          activation: 'relu',
          kernelInitializer: 'varianceScaling'
        }),
      
        tf.layers.maxPooling2d({
          poolSize: [2, 2],
          strides: [2, 2]
        }),
      
        tf.layers.flatten(),
      
        tf.layers.dense({
          units: 10,
          kernelInitializer: 'varianceScaling'
        })
      
      ];
      
      let dummy = tf.zeros([1, 28, 28, 1]);
      for (const layer of layers) {
          dummy = layer.apply(dummy);
      }
      dummy.dispose(); // clean up dummy tensor
      
      const optimizer = tf.train.adam(1e-3);
      const lossFn = (yTrue, yPred) =>
        tf.losses.softmaxCrossEntropy(yTrue, yPred).mean();
    
      const BATCH_SIZE = 512;
      const TRAIN_DATA_SIZE = 55000;
      const EPOCH_AMOUNT = 1;
      const itersTilFullTrainingSetUsed = Math.floor(TRAIN_DATA_SIZE / BATCH_SIZE);
    
      let iter = 0;
      let history = {};
      const activations = [];
      for (let epoch = 0; epoch < EPOCH_AMOUNT; ++epoch) {
    
        for (let outer = 0; outer < itersTilFullTrainingSetUsed; ++outer) {
          const trainBatch = data.nextTrainBatch(BATCH_SIZE);
          const testBatch = data.nextTestBatch(25); //compute validation loss later
          // const batch = window.mnistData.nextTrainBatch(BATCH_SIZE);
          const xsBatch = trainBatch.xs.reshape([BATCH_SIZE, 28, 28, 1]);
          const ysBatch = trainBatch.labels;
          const xsTestBatch = testBatch.xs.reshape([25, 28, 28, 1]);
          const ysTestBatch = testBatch.labels;
  
          const lossPerItem = new Float32Array(BATCH_SIZE);
          
          
    
          tf.tidy(() => {
    
            const trainableVars = [];
            for (const layer of layers) {
              for (const w of layer.trainableWeights) {
                trainableVars.push(w.val);
              }
            }
            //
            let act = xsBatch;
            
            //Forward pass, loss and gradient calculation.
            const {value, grads} = 
              tf.variableGrads(() => {
                
                for (const l of layers) {
                  act = l.apply(act);
                 
                    // console.log("rawattempt i =",outer," layername is: ",l.name,act.dataSync());
                    const vals = act.dataSync();  // call once
                    for (let r = 0; r < vals.length; r++) {
                      if (vals[r] < -10 || vals[r] > 10) {
                        // console.log("wow! its big/small!", vals[r]);
                      }
                  
                }
                // console.log('y1tofloat:',y1.toFloat().dataSync());
                const loss = lossFn(ysBatch, act);
                return loss;
              }, trainableVars);
  
  
            const gradMap = {};
            for (const v of trainableVars) gradMap[v.name] = grads[v.name];
            optimizer.applyGradients(gradMap);   // 
  
            
        if(outer===0){console.log("first iteration:");}
        lossPerItem[outer] = value.dataSync();
        console.log(`Epoch ${epoch} Iter ${iter} loss:`, lossPerItem[outer]);
  
  
          });
          console.log("n\n\n\n\n\n\n here is the test data: \n\n\n\n\n\n\n\n")
          const imgs  = xsTestBatch.reshape([25, 28, 28, 1]);
          
          //lets do a single forward pass on a single tensor
          for(let i=0;i<25;i++){
            
            const logits = tf.tidy(() => {
                let out = imgs.slice([i, 0, 0, 0], [1, 28, 28, 1]);
                if(i===0){
                  const {value,grads} = tf.variableGrads(() => {
                  for (const l of layers) {
                    out = l.apply(out)};
                    
                    activations.push(snapshot(out, {layer: l.name || l.getClassName()}));
                      
                    //record passes just for first one
                  return out;},trainableVars);
                  for (const v of trainableVars) gradMap[v.name] = grads[v.name];
                  //export gradMap in a similar way to snapshot
                  //then simply export weights as well
                }
                else
                {
                  for (const l of layers) {out = l.apply(out)};
                }
              });
            const preds = await logits.softmax().data(); //.argMax(-1)
            console.log("preds:",preds);
            console.log("actual label:",ysTestBatch.slice([i, 0], [1, 10]).argMax(-1).dataSync()[0]);
            console.log("actual backward pass:");
            // console.log(ys)
          }
          console.log("\n\n\n\n\n\n");
  
  
          // console.log("rawActivations just choosing 0th: ",activations[0].vals);
          // console.log("rawActivations just choosing 1th: ",activations[1].vals);
          // console.log("rawActivations just choosing 2th: ",activations[2].vals);
          // console.log("rawActivations just choosing 3th: ",activations[3].vals);
          history[iter] = {
            losses: Array.from(lossPerItem)
          };
   
    
          iter++;
    
          tf.dispose([xsBatch, ysBatch]);
          console.log("right before it!!",iter);if(iter===90){console.log("HEYAA\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");await showPredictions(data);  // call once training is done
            console.log('TRAINING DONE.');}
        }
        
      }
      console.log("testing a bunch!!!:::");
      //test on 30 outputs (designed with full batch in mind!!!!)
      // const newArray=[];
      // for(let g=0;g<25;g++){
      //   for (const l of layers) {
      //     act = l.apply(act);
      //   tf.losses.softmaxCrossEntropy(act);
  
      // }
      // tf.losses.softmaxCrossEntropy(act);
      /***** AFTER training *****/
      /***** AFTER training *****/
      /***** AFTER training *****/
}

function updateWeightViz() {
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
        const slices = activationTensor.split(activationTensor.shape[-1], -1);
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

function updatePredsViz() {
}

function createWeightsViz(modelLayers) {
    const vizContainer = document.getElementById('modelLayerViz');
    vizContainer.innerHTML = '';
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
