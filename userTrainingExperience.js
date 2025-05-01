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

const builtPanels = new Set();

let seedCanvasArray = [];
for(let i = 0; i < 50; i++){
    seedCanvasArray.push(generateRandomSeed());
}

const itersTilFullTrainingSetUsed = TRAIN_DATA_SIZE / BATCH_SIZE;
let history = {
    losses: [],
    vallosses: [],
    activations: {},
    activationShapes: {},
    gradients: {},
    gradientShapes: {},
    weights: {},
    weightShapes: {},
    imagesPredsHoldingGround: [],
    userWantsToClassify: false,
    usersImg: null
    };

//run func
document.addEventListener('DOMContentLoaded', async () => {
    const ctx = document.getElementById('lossGraph').getContext('2d');
    let lossChart = new Chart(ctx, {
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
                title: { display: true, text: 'Loss' },
                min: 0,
                suggestedMax: 3.0,
                beginAtZero: true,
                adaption: {
                    maxOverflow: 0.2
                }
            }
            }
        }
        });
   
    const data = new MnistData();
    await data.load();

    document.getElementById("startTrainingButton").addEventListener("click", async () => {
        const [modelComponents, optimizer] = buildModel(); // requires DOM to exist
        // Pass the chart to the training function
        trainOnlyLayers(data, modelComponents, optimizer, lossChart); 
    });
});

// document.getElementById("userClassifyButton").addEventListener("click", () => {
//     history.userWantsToClassify = true;
//     history.usersImg = document.getElementById("userImage").src;
// });

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
// To generate a random number seed, you can use the following approach:
function generateRandomSeed() {
    // Use the current time to generate a seed
    return Math.floor(Math.random()*Math.random()*Math.random() * Date.now());
}



// Example usage:
const randomSeed = generateRandomSeed();
console.log("Generated Random Seed:", randomSeed);

// const blocks = document.querySelectorAll('.block');
// blocks.forEach(function(box) {
//     box.addEventListener('dragstart', function(event) {
//     event.dataTransfer.setData('text/plain', box.id);
//     });
// });

// document.getElementById("target").addEventListener("dragover", function(event) {
//     event.preventDefault();
// });

// document.getElementById("target").addEventListener('drop', function(event) {
//     let boxId = event.dataTransfer.getData('text/plain');
//     let origBox = document.getElementById(boxId);
//     const newBox = origBox.cloneNode(true);
//     newBox.id = boxId + '-clone';
//     document.getElementById("target").appendChild(newBox);
//     addTrashIconToBlock(newBox);
// });

// function addTrashIconToBlock(block) {
//     const trashIcon = document.createElement('div');
//     trashIcon.classList.add('trash-icon');
//     trashIcon.innerHTML = '🗑️';
//     trashIcon.addEventListener('click', function(event) {
//         event.stopPropagation();
//         block.remove();
//     });
//     block.insertBefore(trashIcon, block.firstChild);
// }

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

  
/* ───────── visual-helpers.js ───────── */
function addLayerPanel(id, parent){
    if (builtPanels.has(id)) return;    // do it only once
    const block = document.createElement('div');
    block.className = 'layerBlock';
    block.innerHTML = `
      <div class="sectionLabel">${id}</div>
      <div class="sectionLabel">Activations:</div><div class="canvasGrid" id="${id}-acts"></div>
      <div class="sectionLabel">Gradients:</div>   <div class="canvasGrid" id="${id}-grads"></div>
      <div class="sectionLabel">Weights:</div>     <div class="canvasGrid" id="${id}-wts"></div>`;
    parent.appendChild(block);
    builtPanels.add(id);
  }
  
  async function renderKind(flat, shape, grid){
    let t = tf.tensor(flat, shape).squeeze();      // remove batch
    let slices = [];
  
    /* ------------ rank-aware slicing ------------ */
    /* ------------ rank-aware slicing ------------ */
    if (t.rank === 4){                       // [H, W, Cin, Cout]
        const [H,W,Cin,Cout] = t.shape;
    
        for (let o = 0; o < Cout; o++){
        // Take the full [H,W,Cin] kernel for this output channel
        let k = t.slice([0,0,0,o], [H,W,Cin,1]).squeeze();  // [H,W,Cin]
    
        // Collapse Cin>1 by averaging → [H,W]
        if (Cin > 1) k = k.mean(2);                         // mean over Cin axis
    
        slices.push(k);     // we’ll get exactly Cout = 16 tiles, not 128
        }
    }
   else if (t.rank === 3){                       // [H,W,C]
      const [H,W,C] = t.shape;
      for (let c = 0; c < C; c++)
        slices.push(t.slice([0,0,c],[H,W,1]).squeeze());
    } else if (t.rank === 2){                       // [H,W]
      slices = [t];
      const m = t.shape[0] > t.shape[1] ? t.transpose() : t;   // [W,H]
      slices = [m];
    } else if (t.rank === 1){                       // [N]  (flatten)
      const N = t.shape[0], dim = Math.ceil(Math.sqrt(N));
      slices = [t.pad([[0, dim*dim - N]]).reshape([dim, dim])];
    }
  
    /* ------------ make / trim canvases ------------ */
    while (grid.children.length < slices.length)
      grid.appendChild(document.createElement('canvas'));
    while (grid.children.length > slices.length)
      grid.lastChild.remove();
  
    /* ------------ draw ------------ */
    for (let i = 0; i < slices.length; i++){
      const c = grid.children[i];
      const s = slices[i];
      if (c.width !== s.shape[1] || c.height !== s.shape[0]){
        c.width  = s.shape[1];
        c.height = s.shape[0];
      }
      const vis = s.sub(s.min()).div(s.max().sub(s.min()).add(1e-6));
      await tf.browser.toPixels(vis, c);
      s.dispose(); vis.dispose();
    }
    t.dispose();
  }
  
  

  
function getArrayShape(arr) {
    const shape = [];
    let current = arr;
    while (Array.isArray(current)) {
        shape.push(current.length);
        current = current[0];
    }
    return shape;
}

// drawConvMaps(tensor, canvas, b=0)


// if(layerName === "conv2d"){
//     newShape = shape.slice(1);
//     const [h,w,c] = newShape;
//     for(let i = 0){


//     }
// }




function handleLayerVisualizationUpdates(history){
    const layersViz = document.getElementById('layersViz');
  
    /* --------- create missing panels once --------- */
    for (const lname of Object.keys(history.activationShapes))
      addLayerPanel(lname, layersViz);
    console.log('Built panels for:', Array.from(builtPanels));
    console.log('Weight keys:', Object.keys(history.weights));
    console.log('Gradient keys:', Object.keys(history.gradients));
    
    /* --------- activations --------- */
    for (const [lname, act] of Object.entries(history.activations)){
      renderKind(act,
                 history.activationShapes[lname],
                 document.getElementById(`${lname}-acts`));
    }
  
    /* --------- gradients --------- */
    for (const lname of Object.keys(history.activationShapes)) {
      const gradGrid = document.getElementById(`${lname}-grads`);
      if (gradGrid) {
        // Find the section label (previous element with class "sectionLabel")
        const gradSectionLabel = gradGrid.previousElementSibling;
        
        // Check if any gradients belong to this layer
        const hasGradients = Object.keys(history.gradients).some(pname => pname.split('/')[0] === lname);
        
        // Show/hide gradients section based on content
        gradGrid.style.display = hasGradients ? '' : 'none';
        if (gradSectionLabel && gradSectionLabel.classList.contains('sectionLabel')) {
          gradSectionLabel.style.display = hasGradients ? '' : 'none';
        }
        
        if (hasGradients) {
          // Only render if we have gradients for this layer
          for (const [pname, g] of Object.entries(history.gradients)) {
            const base = pname.split('/')[0];
            if (base === lname) {
              renderKind(g, history.gradientShapes[pname], gradGrid);
            }
          }
        }
      }
    }
  
    /* --------- weights --------- */
    for (const lname of Object.keys(history.activationShapes)) {
      const weightGrid = document.getElementById(`${lname}-wts`);
      if (weightGrid) {
        // Find the section label (previous element with class "sectionLabel")
        const weightSectionLabel = weightGrid.previousElementSibling;
        
        // Check if any weights belong to this layer
        const hasWeights = Object.keys(history.weights).some(pname => pname.split('/')[0] === lname);
        
        // Show/hide weights section based on content
        weightGrid.style.display = hasWeights ? '' : 'none';
        if (weightSectionLabel && weightSectionLabel.classList.contains('sectionLabel')) {
          weightSectionLabel.style.display = hasWeights ? '' : 'none';
        }
        
        if (hasWeights) {
          // Only render if we have weights for this layer
          for (const [pname, w] of Object.entries(history.weights)) {
            const base = pname.split('/')[0];
            if (base === lname) {
              renderKind(w, history.weightShapes[pname], weightGrid);
            }
          }
        }
      }
    }
  }


  async function trainOnlyLayers(data,modelComponents,optimizer, lossChart) {

      let modelArr = modelComponents;
      //logic for selectig model from user input lego blocks.
      
      
      let dummy = tf.ones([1, 28, 28, 1]);
      for (const layer of modelArr) {
          try{
            dummy = layer.layer.apply(dummy);
            console.log("passed through layer: ", layer.id, " with shape: ", dummy.shape); // after apply
            //history.
          }
          catch(e){
            console.log("ERROR IN MODEL CONFIGURATION!: ", e);
          }
      }
      let layers = modelArr.map(part => part.layer);
      let trueLayerNames = [];
      for(const layer of layers){
        console.log("layer initial test!: ", layer.name, layer.batchInputShape);
        trueLayerNames.push(layer.name);
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
      const trainableVars = [];
      for (const layer of layers) {
        for (const w of layer.trainableWeights) {
          trainableVars.push(w.val);
        }
      }
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
            // console.log("ysBatch[0] dataSync():", ysBatch.slice([0, 0], [1, ysBatch.shape[1]]).dataSync());
            // console.log("ysBatch shape:", ysBatch.shape);

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
        
        // Update chart correctly
        lossChart.data.labels.push(iter);
        lossChart.data.datasets[0].data.push(loss[0]);
        const maxLoss = Math.max(...lossChart.data.datasets[0].data);

        // Set the y-axis range dynamically
        lossChart.options.scales.y.min = 0;
        lossChart.options.scales.y.max = maxLoss;

        // Now update the chart
        lossChart.update();
  
            });
            console.log("n\n\n\n\n\n\n here is the test data: \n\n\n\n\n\n\n\n")
            const imgs = xsTestBatch;
            let temporaryLossArray = [];
            //lets do a single forward pass on a single tensor
            for(let i=0;i<50;i++){
                
                const [loss,grads] = (() => {
                    let out = imgs.slice([i, 0, 0, 0], [1, 28, 28, 1]);
                    
                    const {value,grads} = tf.variableGrads(() => {
                    for (const l of layers) {
                        out = l.apply(out);
                        if(i===0){
                            //log activations if first iteration of batch.
                            if (!history.activations[l.name]) {
                                history.activations[l.name] = [];
                            }
                            history.activations[l.name]=out.dataSync();
                            history.activationShapes[l.name] = out.shape;
                        }
                        }
                        history.imagesPredsHoldingGround[i]=out.dataSync();
                        console.log("out.dataSync(): ", out.dataSync());
                        tf.keep(out);
                        //record passes just for first one
                    return lossFn(ysTestBatch.slice([i, 0], [1, NUM_CLASSES]), out);},trainableVars);
                    
                    //export gradMap in a similar way to snapshot
                    //then simply export weights as well
                    
                    //console.log("out is unquestionably here: ", out);
                    temporaryLossArray=value.dataSync();
                    return [out,grads];
                })(); //});
                //lets log a user digit classification:
                //if(userWantsToClassify){}.........
                //update weights and gradients if first iteration of batch.
                    const layersLength = layers.length;
                    let iterator = 0;
                    if(i===0){

                        for(const trw of trainableVars){

                            let currentName = trw.name;
                            let newName;
                            console.log("currentName: ", currentName, " and iterator value: ", iterator);
                            console.log("trainableVars[iterator].name: ", trainableVars[iterator].name);
                            if((currentName.includes("conv2d")&&!currentName.includes("bias"))){
                                newName = trw.name;
                                
                                if (!history.weights[newName]) {
                                    history.weights[newName] = [];
                                    history.weightShapes[newName] = trw.shape;
                                }
                                if(!history.gradients[newName]){
                                    history.gradients[newName] = [];
                                    console.log("logging  of grads[newName]: ", grads[trw.name]);
                                    history.gradientShapes[newName] = grads[trw.name].shape;
                                }
                                try{
                                    history.weights[newName]=(trw.dataSync());
                                    history.gradients[newName]=(grads[trw.name].dataSync());
                                    console.log("\n\n\n\ iterator success!: ", iterator, " and newName: ", newName);
                                    console.log("trw.dataSync(): ", trw.dataSync());
                                    console.log("trw.shape: ", trw.shape);
                                }
                                catch(e){
                                    console.log("error in weights: ", e);
                                }
                                iterator++;
                            }
                            else if((currentName.includes("dense")&&!currentName.includes("bias"))){
                                newName = trw.name;
                                if (!history.weights[newName]) {
                                    history.weights[newName] = [];
                                    history.weightShapes[newName] = trw.shape;
                                }
                                if(!history.gradients[newName]){
                                    history.gradients[newName] = [];
                                    history.gradientShapes[newName] = grads[trw.name].shape;
                                }
                                try{
                                    history.weights[newName]=(trw.dataSync());
                                    history.gradients[newName]=(grads[trw.name].dataSync());
                                    console.log("\n\n\n\ iterator success!: ", iterator, " and newName: ", newName);
                                    console.log("trw.dataSync(): ", trw.dataSync());
                                    console.log("trw.shape: ", trw.shape);
                                }
                                catch(e){
                                    console.log("error in weights: ", e);
                                }
                                iterator++;
                            }

                        }
                        //showPredictions(imgs, history.imagesPredsHoldingGround, ysTestBatch);
                    }
                // console.log("logits: ", logits);
                // const preds = await logits.softmax().data(); //.argMax(-1)
                // console.log("preds:",preds);
                // console.log("actual label:",ysTestBatch.slice([i, 0], [1, NUM_CLASSES]).argMax(-1).dataSync()[0]);
                // console.log("actual backward pass:");
                let averageValLoss = temporaryLossArray.reduce((acc, curr) => acc + curr/50, 0);
                history.vallosses.push(averageValLoss);
                if(i===0){
                    // console.log("ysTestbatch.arraySync(): ", ysTestBatch.arraySync());
                    // const ysTestBatchClone = ysTestBatch.arraySync();
                    await showPredictions(imgs, ysTestBatch);
                    //testing train
                    // await showPredictions(xsBatch.slice([0, 0, 0, 0], [50, 28, 28, 1]), ysBatch.slice([0, 0], [50, 10]));
                }
                // document.getElementById('outputText').textContent = JSON.stringify(history);
                
                // console.log(ys)
            }

          handleLayerVisualizationUpdates(history);
          console.log("\n\n\n\n\n\n");
          //console.log("history: ", history);
          iter++;
    
          tf.dispose([xsBatch, ysBatch]);
          console.log("right before it!!",iter);if(iter===90){console.log("HEYAA\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");  // call once training is done
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
    async function showPredictions(imgs, labels) {
        const preddies = history.imagesPredsHoldingGround;
        console.log("preddies[0]: ", preddies[0]);
        const imagesPredsTensor = tf.stack(preddies.map(pred => tf.tensor(pred, [1, NUM_CLASSES])));
        // imagesPredsHoldingGround = imagesPredsTensor;
        // const BATCH = 50;
        // const {xs}  = data.nextTestBatch(BATCH);
        // const imgs  = xs.reshape([BATCH, 28, 28, 1]);
        
        console.log("===== RUNNING SHOW PREDICTIONS =====");

        // /* ---------- forward pass ---------- */
        // const logits = tf.tidy(() => {
        //     let out = imgs;
        //     for (const l of layers) out = l.apply(out);
        //     return out;
        // });

        

        const preddios = await imagesPredsTensor.softmax().argMax(-1).data();
        console.log("truthios[0] before argmax: ", labels.slice([0, 0], [1, 10]).dataSync());
        const truthios = await labels.argMax(-1).data();
        console.log("truthios[0] after argmax: ", truthios[0]);
        for (let i = 0; i < 50; i++) {
            console.log(`Prediction for index ${i}: ${preddios[i]}`);
            console.log(`Truth for index ${i}: ${truthios[i]}`);
        }
        // console.log("truthios[0]: ", truthios[0]);
        // logits.dispose();

        /* ---------- render ---------- */
        const grid = document.getElementById('demoGrid');
        grid.innerHTML = ''; // Clear the grid

        const imgTensors = tf.unstack(imgs);              // <-- create first
        console.log('imgTensors len =', imgTensors.length); // now it's defined

        await Promise.all(
        imgTensors.map(async (t, i) => {
            let canvas;
            // grid.childElementCount>50?canvas=grid.getElementById(String(seedCanvasArray[i])):canvas = document.createElement('canvas');
            let cell;
            let predLabel;
            let trueLabel;
            if(grid.childElementCount<50){
                let seed = seedCanvasArray[i];
                canvas=document.createElement('canvas');
                canvas.id = seed;
                canvas.width = canvas.height = 28;
                await tf.browser.toPixels(t.squeeze(), canvas);
                t.dispose();
                cell  = document.createElement('div');
                cell.className = 'demoCell';
                predLabel = document.createElement('div');
                predLabel.id='predLabel';
                trueLabel = document.createElement('div');
                trueLabel.id='trueLabel';
                cell.appendChild(canvas);
                cell.appendChild(predLabel);
                cell.appendChild(trueLabel);

                grid.appendChild(cell);
                // console.log("canvas:all is GOING well", canvas);
            
            }
            else{
                canvas = document.getElementById(String(seedCanvasArray[i]));
                // console.log("canvas: ", canvas.id);
                canvas.width = canvas.height = 28;
                await tf.browser.toPixels(t.squeeze(), canvas);
                t.dispose();
                cell = canvas.parentElement;
                predLabel = cell.querySelector('#predLabel'); //predLabel should be only div in cell
                trueLabel = cell.querySelector('#trueLabel'); //trueLabel should be only div in cell
            }

            
            // cell.appendChild(canvas);

            // predLabel = document.createElement('div');
            predLabel.textContent = `pred: ${preddios[i]}`;
            // console.log("truthios[i]: ", truthios[i]);
            trueLabel.textContent = `true: ${truthios[i]}`;
        })
        );

        console.log('children in grid =', grid.childElementCount); // should print 50
        imagesPredsTensor.dispose(); 
        await tf.nextFrame();}
    }
  
// function updateWeightViz() {
// }

// function updateGradsViz(layerIndex, gradientTensors) {
//     const container = document.getElementById(`grads-${layerIndex}`);
//     container.innerHTML = '';
//     gradientTensors.forEach(tensor => {
//         const canvas = document.createElement('canvas');
//         tf.browser.toPixels(tensor.squeeze().clipByValue(0, 1), canvas);
//         container.appendChild(canvas);
//     });
// }

// function updateActivationsViz(layerIndex, activationTensor) {
//     const container = document.getElementById(`activations-${layerIndex}`);
//     container.innerHTML = '';
//     if (activationTensor.shape.length > 2) {
//         const slices = activationTensor.split(activationTensor.shape[-1], -1);
//         slices.forEach(slice => {
//             const canvas = document.createElement('canvas');
//             tf.browser.toPixels(slice.squeeze().clipByValue(0, 1), canvas);
//             container.appendChild(canvas);
//         });
//     } else {
//         const canvas = document.createElement('canvas');
//         tf.browser.toPixels(activationTensor.squeeze().clipByValue(0, 1), canvas);
//         container.appendChild(canvas);
//     }
// }

// function updatePredsViz() {
// }

// function createWeightsViz(modelLayers) {
//     const vizContainer = document.getElementById('netViz');
//     vizContainer.innerHTML = '';
//     for (let i = 0; i < modelLayers.length; i++) {
//         const layerDiv = document.createElement('div');
//         layerDiv.id = `layer-viz-${i}`;
//         layerDiv.innerHTML = `<h4>Layer ${i}: ${modelLayers[i].getClassName()}</h4>
//                               <div id="weights-${i}"></div>
//                               <div id="activations-${i}"></div>
//                               <div id="grads-${i}"></div>`;
//         vizContainer.appendChild(layerDiv);
//     }
// }

