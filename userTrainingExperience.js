const BATCH_SIZE = 512;
const TRAIN_DATA_SIZE = 55000;
const TEST_DATA_SIZE = 10000;
const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;
const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;
let EPOCH_AMOUNT = 1;
let LR = 0.01;
const itersTilFullTrainingSetUsed = TRAIN_DATA_SIZE / BATCH_SIZE;
let history = {
    losses: [],
    vallosses: [],
    activations: {},
    gradients: {},
    weights: {}
    };

//run func
document.addEventListener('DOMContentLoaded', async () => {
    const ctx = document.getElementById('lossGraph').getContext('2d');

    const lossChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
        label: 'Loss',
        data: [],
        borderColor: 'blue',
        borderWidth: 2,
        fill: false
        }]
    },
    options: {
        animation: false,
        responsive: true,
        scales: {
        x: {
            title: { display: true, text: 'Batch' }
        },
        y: {
            title: { display: true, text: 'Loss' }
        }
        }
    }
    });
    
   
    const data = new MnistData();
    await data.load();

    document.getElementById("startTrainingButton").addEventListener("click", async () => {
        const [modelComponents, optimizer] = buildModel(); // requires DOM to exist
        trainOnlyLayers(data,modelComponents,optimizer); // needs model + data
    });
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
        if (i === 0 && child.dataset.type === "conv2d") {
            const inputs = child.querySelectorAll('input');
            const kernelSize = parseInt(inputs[0].value) || 3;
            const filters = parseInt(inputs[1].value) || 16;
            const strides = parseInt(inputs[2].value) || 1;

            modelArr.push({
                id: "conv2d",
                layer: tf.layers.conv2d({
                inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
                    kernelSize: kernelSize,
                    filters: filters,
                    strides: strides,
                    kernelInitializer: 'varianceScaling'
                }),
                info: {
                    kernelSize: kernelSize,
                    filters: filters,
                    strides: strides,
                kernelInitializer: 'varianceScaling'
                },
                currentTestActivations: [],
                currentTestGradients: [],
                currentTestWeights: []
            });
        } 
        else if (child.dataset.type === "conv2d") {
            const inputs = child.querySelectorAll('input');
            const kernelSize = parseInt(inputs[0].value) || 3;
            const filters = parseInt(inputs[1].value) || 16;
            const strides = parseInt(inputs[2].value) || 1;
            
            modelArr.push({
                id: "conv2d",
                layer: tf.layers.conv2d({
                    kernelSize: kernelSize,
                    filters: filters,
                    strides: strides,
                    kernelInitializer: 'varianceScaling'
                }),
                info: {
                    kernelSize: kernelSize,
                    filters: filters,
                    strides: strides,
                kernelInitializer: 'varianceScaling'
                },
                currentTestActivations: [],
                currentTestGradients: [],
                currentTestWeights: []
            });
        } 
        else if (child.dataset.type === "dense") {
            const inputs = child.querySelectorAll('input');
            const inputChannels = parseInt(inputs[0].value);
            const outputChannels = parseInt(inputs[1].value) || 10;
            
            modelArr.push({
                id: "dense",
                layer: tf.layers.dense({
                    inputShape: inputChannels ? [inputChannels] : undefined,
                    units: outputChannels
                }),
                info: {
                    inputShape: inputChannels ? [inputChannels] : undefined,
                    units: outputChannels
                },
                currentTestActivations: [],
                currentTestGradients: [],
                currentTestWeights: []
            });
        } 
        else if (child.dataset.type === "relu") {
            modelArr.push({
                id: "relu",
                layer: tf.layers.activation({ activation: 'relu' }),
                info: {
                    activation: 'relu'
                },
                currentTestActivations: [],
                currentTestGradients: [],
                currentTestWeights: []
            });
        } 
        else if (child.dataset.type === "sigmoid") {
            modelArr.push({
                id: "sigmoid",
                layer: tf.layers.activation({ activation: 'sigmoid' }),
                info: {
                    activation: 'sigmoid'
                },
                currentTestActivations: [],
                currentTestGradients: [],
                currentTestWeights: []
            });
        } 
        else if (child.dataset.type === "tanh") {
            modelArr.push({
                id: "tanh",
                layer: tf.layers.activation({ activation: 'tanh' }),
                info: {
                    activation: 'tanh'
                },
                currentTestActivations: [],
                currentTestGradients: [],
                currentTestWeights: []
            });
        } 
        else if (child.dataset.type === "maxpool") {
            const inputs = child.querySelectorAll('input');
            const poolSize = parseInt(inputs[0].value) || 2;
            const strides = parseInt(inputs[1].value) || 2;
            
            modelArr.push({
                id: "maxpool",
                layer: tf.layers.maxPooling2d({
                    poolSize: [poolSize, poolSize],
                    strides: [strides, strides]
                }),
                info: {
                    poolSize: [poolSize, poolSize],
                    strides: [strides, strides]
                },
                currentTestActivations: [],
                currentTestGradients: [],
                currentTestWeights: []
            });
        }
        else if (child.dataset.type === "flatten") {
            modelArr.push({
                id: "flatten",
                layer: tf.layers.flatten(),
                info: {
                    activation: 'flatten'
                },
                currentTestActivations: [],
                currentTestGradients: [],
                currentTestWeights: []
            });
            // console.log("flatten layer added to modelArr");
        }
        //TF.js doesn't offer standalone crossEntropyLayer, maybe we'll add softmax viz later.
        // else if(child.dataset.type === "softmax"){
        //     modelArr.push({
        //         id: "softmax",
        //         layer: tf.layers.activation({ activation: 'softmax' }),
        //         info: {
        //             activation: 'softmax'
        //         },
        //         currentTestActivations: [],
        //         currentTestGradients: [],
        //         currentTestWeights: []
        //     });
        // }
        else if (child.dataset.type === "loss") {
            const selects = child.querySelectorAll('select');
            const optimizerName = selects[0].value;
            
            optimizer = {
                sgd: tf.train.sgd,
                adam: tf.train.adam,
            }[optimizerName]();
        }
    }
    return [modelArr, optimizer];
}
//not finished yet.
function buildVizLayers(modelArr){
    let layersViz = document.getElementById('layersViz');
    let inputLayerImage = document.createElement('div');
    inputLayerImage.innerHTML = `
    <div class="layerBlock">
        <div class="layerMeta">Input Layer</div>
        <div class="canvasGrid" id="actGrid1">Gradients</div>
    </div>`;
    layersViz.appendChild(inputLayerImage);
    for(const layer of modelArr){
        const layerBlock = document.createElement('div');
        layerBlock.innerHTML =`
        <div class="layerBlock">
            <div class="layerMeta">${layer.id}</div>
            
            <div class="sectionRow">
                <div class="sectionLabel">Activations:</div>
                <div class="canvasGrid" id="actGrid1"></div>
            </div>

            <div class="sectionRow">
                <div class="sectionLabel">Gradients:</div>
                <div class="canvasGrid" id="gradGrid1"></div>
            </div>

            <div class="sectionRow">
                <div class="sectionLabel">Weights:</div>
                <div class="canvasGrid" id="weightGrid1"></div>
            </div>
        </div>`;
        layersViz.appendChild(layerBlock);
    }
}

function handleLayerVisualizationUpdates(activations,grads,weights){
    let layerViz = document.getElementById('layersViz');
    //should be layer containers
    for(const layer of layerViz.children){
        //go into viewing area grids:
        let activation = layer.querySelector('#actGrid1');
        let gradient = layer.querySelector('#gradGrid1');
        let weight = layer.querySelector('#weightGrid1');
            //should be canvas grids.
            //3 for loops, this one for weights.
            for(const weight of weights){
                let canvas = document.createElement('canvas');
                //.....

            }


    }
}


  async function trainOnlyLayers(data,modelComponents,optimizer) {

      let modelArr = modelComponents;
      //logic for selectig model from user input lego blocks.
      
      
      let dummy = tf.ones([1, 28, 28, 1]);
      for (const layer of modelArr) {
          try{
            dummy = layer.layer.apply(dummy);
            console.log("passed through layer: ", layer.id, " with shape: ", dummy.shape); // after apply

          }
          catch(e){
            console.log("ERROR IN MODEL CONFIGURATION!: ", e);
          }
      }
      let layers = modelArr.map(part => part.layer);
      for(const layer of layers){
        console.log("layer initial test!: ", layer.name, layer.shape);
      }
      dummy.dispose(); // clean up dummy tensor
    //   for (const [i, layer] of modelArr.entries()) {
    //     console.log(`Layer ${i} - Type: ${layer.layer?.getClassName?.() || 'Unknown'}`);
    //     console.log(`  Name: ${layer.layer?.name}`);
    //     console.log(`  Trainable: ${layer.layer?.trainable}`);
    //     console.log(`  Batch Input Shape: ${layer.layer?.batchInputShape}`);
    //     console.log(`  Output Shape: ${layer.layer?.outputShape}`);
    //     if (layer.layer?.trainableWeights?.length) {
    //       for (const w of layer.layer.trainableWeights) {
    //         console.log(`    Weight: ${w.name}, shape: ${w.shape}`);
    //       }
    //     }
    //   }
      const lossFn = (yTrue, yPred) =>
        tf.losses.softmaxCrossEntropy(yTrue, yPred).mean();
    
      const BATCH_SIZE = 512;
      const TRAIN_DATA_SIZE = 55000;
      const EPOCH_AMOUNT = 1;
      const itersTilFullTrainingSetUsed = Math.floor(TRAIN_DATA_SIZE / BATCH_SIZE);
    
      let iter = 0;
      
      const activations = [];
      for (let epoch = 0; epoch < EPOCH_AMOUNT; ++epoch) {
    
        for (let outer = 0; outer < itersTilFullTrainingSetUsed; ++outer) {
          const trainBatch = data.nextTrainBatch(BATCH_SIZE);
          const testBatch = data.nextTestBatch(50); //compute validation loss later
          // const batch = window.mnistData.nextTrainBatch(BATCH_SIZE);
          const xsBatch = trainBatch.xs.reshape([BATCH_SIZE, 28, 28, 1]);
          const ysBatch = trainBatch.labels;
          const xsTestBatch = testBatch.xs.reshape([50, 28, 28, 1]);
          const ysTestBatch = testBatch.labels;
  
          
          const trainableVars = [];
            for (const layer of layers) {
              for (const w of layer.trainableWeights) {
                trainableVars.push(w.val);
              }
            }
            if(iter===0){console.log("trainableVars: \n\n\n\n\n\n\n\n\n\n\n\n", trainableVars);}
    
          tf.tidy(() => {
    
            
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
                console.log("layer: ", l.name);
                console.log("act.shape: ", act.shape);
                // console.log('y1tofloat:',y1.toFloat().dataSync());

              }//end of const l for loop
              const loss = lossFn(ysBatch, act);
                           return loss;
            }, trainableVars);
  
  
            const gradMap = {};
            for (const v of trainableVars) gradMap[v.name] = grads[v.name];
            optimizer.applyGradients(gradMap);   // 
  
            
        if(outer===0){console.log("first iteration:");}
        let loss = value.dataSync()
        history.losses.push(loss);
        console.log(`Epoch ${epoch} Iter ${iter} loss:`, loss);
  
          });
          console.log("n\n\n\n\n\n\n here is the test data: \n\n\n\n\n\n\n\n")
          const imgs  = xsTestBatch.reshape([50, 28, 28, 1]);
          let temporaryLossArray = [];
          //lets do a single forward pass on a single tensor
          for(let i=0;i<50;i++){
            
            const [logits,grads] = (() => {
                let out = imgs.slice([i, 0, 0, 0], [1, 28, 28, 1]);
                
                  const {value,grads} = tf.variableGrads(() => {
                  for (const l of layers) {
                    out = l.apply(out);
                    if(i===0){
                        //log activations if first iteration of batch.
                        if (!history.activations[l.name]) {
                            history.activations[l.name] = [];
                        }
                        history.activations[l.name].push(out.dataSync());
                    }
                    }
                      tf.keep(out);
                    //record passes just for first one
                  return lossFn(ysTestBatch.slice([i, 0], [1, NUM_CLASSES]), out);},trainableVars);
                  
                  //export gradMap in a similar way to snapshot
                  //then simply export weights as well
                  
                console.log("out is unquestionably here: ", out);
                temporaryLossArray.push(value.dataSync());
                return [out,grads];
              })(); //});
              const gradMap = {};
                  for (const v of trainableVars) gradMap[v.name] = grads[v.name];
                  //update weights and gradients if first iteration of batch.
                  const layersLength = layers.length;
                  let iterator = 0;
                  if(i===0){
                    for(const layer of layers){
                        if (!history.gradients[layer.name]) {
                            history.gradients[layer.name] = [];
                        }
                        history.gradients[layer.name].push(gradMap);


                        if(layer.name.includes("conv2d")){
                            if (!history.weights[layer.name]) {
                                history.weights[layer.name] = [];
                            }
                            
                            history.weights[layer.name].push(trainableVars[iterator]);
                        }
                        else if(layer.name.includes("dense")){
                            if (!history.weights[layer.name]) {
                                history.weights[layer.name] = [];
                            }
                            history.weights[layer.name].push(trainableVars[iterator]);
                        }

                        iterator++;
                    }

                    for (const v of trainableVars)
                    for(const weightArr of trainableVars){
                        

                    }
                  }
            console.log("logits: ", logits);
            const preds = await logits.softmax().data(); //.argMax(-1)
            console.log("preds:",preds);
            console.log("actual label:",ysTestBatch.slice([i, 0], [1, NUM_CLASSES]).argMax(-1).dataSync()[0]);
            console.log("actual backward pass:");
            let averageValLoss = temporaryLossArray.reduce((acc, curr) => acc + curr/50, 0);
            history.vallosses.push(averageValLoss);
            // console.log(ys)
          }
          console.log("\n\n\n\n\n\n");
          console.log("history: ", history);
    
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
      async function showPredictions(data) {
  const BATCH = 25;
  const {xs}  = data.nextTestBatch(BATCH);
  const imgs  = xs.reshape([BATCH, 28, 28, 1]);

  /* ---------- forward pass ---------- */
  const logits = tf.tidy(() => {
    let out = imgs;
    for (const l of layers) out = l.apply(out);
    return out;
  });
  const preds = await logits.softmax().argMax(-1).data();
  logits.dispose();

  /* ---------- render ---------- */
  const grid = document.getElementById('demoGrid');
  grid.innerHTML = '';

  const imgTensors = tf.unstack(imgs);              // <-- create first
  console.log('imgTensors len =', imgTensors.length); // now it’s defined

  await Promise.all(
    imgTensors.map(async (t, i) => {
      const canvas = document.createElement('canvas');
      canvas.width = canvas.height = 28;
      await tf.browser.toPixels(t.squeeze(), canvas);
      t.dispose();

      const cell  = document.createElement('div');
      cell.className = 'demoCell';
      cell.appendChild(canvas);

      const label = document.createElement('div');
      label.textContent = `pred: ${preds[i]}`;
      cell.appendChild(label);

      grid.appendChild(cell);
    })
  );

  console.log('children in grid =', grid.childElementCount); // should print 25
  xs.dispose(); imgs.dispose();
  await tf.nextFrame();}
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
    const vizContainer = document.getElementById('netViz');
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

