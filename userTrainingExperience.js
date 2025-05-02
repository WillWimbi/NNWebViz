const BATCH_SIZE = 512;
const TRAIN_DATA_SIZE = 55000;
const TEST_DATA_SIZE = 10000;
const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;
const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;
let EPOCH_AMOUNT = 10000; //train forever  by default lol
let USER_CONST = 0.001;

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
    userWantsToClassify: false,
    usersImg: null,
    batchPreds: [],
    modelLayersCopy: []
    };

document.addEventListener('DOMContentLoaded', async () => {
    const ctx = document.getElementById('lossGraph').getContext('2d');
    let lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Training Loss',
                    data: [],
                    borderColor: 'blue',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 3,
                    pointBackgroundColor: 'blue',
                    fill: false,
                    zIndex: 10
                },
                {
                    label: 'Validation Loss',
                    data: [],
                    borderColor: 'red',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 3, 
                    pointBackgroundColor: 'red',
                    fill: false,
                    borderDash: [5, 5],
                    zIndex: 5
                }
            ]
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
        const [modelComponents, optimizer] = buildModel();
        const startBtn = document.getElementById("startTrainingButton");
        startBtn.style.transition = "opacity 5s";
        startBtn.style.opacity = "0";
        setTimeout(() => {
            if (startBtn && startBtn.parentNode) {
                startBtn.parentNode.removeChild(startBtn);
            }
        }, 5000);
        trainOnlyLayers(data, modelComponents, optimizer, lossChart); 
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
                img.height = img.naturalHeight;
        const datasetBytesBuffer = new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);
        const chunkSize = 5000;
                canvas.width = img.width;
        canvas.height = chunkSize;

        for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
                    const datasetBytesView = new Float32Array(datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4, IMAGE_SIZE * chunkSize); 
                    ctx.drawImage(img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width, chunkSize); 
                    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height, { willReadFrequently: true });

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
function generateRandomSeed() {
    // Use the current time to generate a seed
    return Math.floor(Math.random()*Math.random()*Math.random() * Date.now());
}

function buildModel() {
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
        }
        //TF.js doesn't offer standalone crossEntropyLayer, maybe we'll add softmax viz later.
        else if (child.dataset.type === "loss") {
            const selects = child.querySelectorAll('select');
            const optimizerName = selects[0].value;
            
            // Set learning rate for TensorFlow optimizer based on user input
            const lrInput = child.querySelector('input[placeholder="learning rate"]');
            const learningRate = lrInput ? parseFloat(lrInput.value) || 0.001 : 0.001;
            optimizer = {
                sgd: () => tf.train.sgd(learningRate),
                adam: () => tf.train.adam(learningRate),
            }[optimizerName]();
        }
    }
    return [modelArr, optimizer];
}

function addLayerPanel(id, parent){
    if (builtPanels.has(id)) return;
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
    let t = tf.tensor(flat, shape).squeeze();
    let slices = [];
  
    if (t.rank === 4){
        const [H,W,Cin,Cout] = t.shape;
    
        for (let o = 0; o < Cout; o++){
        let k = t.slice([0,0,0,o], [H,W,Cin,1]).squeeze();
    
        if (Cin > 1) k = k.mean(2);
    
        slices.push(k);
        }
    }
   else if (t.rank === 3){
      const [H,W,C] = t.shape;
      for (let c = 0; c < C; c++)
        slices.push(t.slice([0,0,c],[H,W,1]).squeeze());
    } else if (t.rank === 2){
      slices = [t];
      const m = t.shape[0] > t.shape[1] ? t.transpose() : t;
      slices = [m];
    } else if (t.rank === 1){
      const N = t.shape[0], dim = Math.ceil(Math.sqrt(N));
      slices = [t.pad([[0, dim*dim - N]]).reshape([dim, dim])];
    }
  
    while (grid.children.length < slices.length)
      grid.appendChild(document.createElement('canvas'));
    while (grid.children.length > slices.length)
      grid.lastChild.remove();
  
    for (let i = 0; i < slices.length; i++){
      const c = grid.children[i];
      const s = slices[i];
      if (c.width !== s.shape[1] || c.height !== s.shape[0]){
        c.width  = s.shape[1];
        c.height = s.shape[0];
        c.style.width  = `${c.width}px`;
        c.style.height = `${c.height}px`;
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

function handleLayerVisualizationUpdates(history){
    const layersViz = document.getElementById('layersViz');
  
    for (const lname of Object.keys(history.activationShapes))
      addLayerPanel(lname, layersViz);
    
    for (const [lname, act] of Object.entries(history.activations)){
      renderKind(act,
                 history.activationShapes[lname],
                 document.getElementById(`${lname}-acts`));
    }
  
    for (const lname of Object.keys(history.activationShapes)) {
      const gradGrid = document.getElementById(`${lname}-grads`);
      if (gradGrid) {
        // Find the section label (previous element with class "sectionLabel")
        const gradSectionLabel = gradGrid.previousElementSibling;
        
        const hasGradients = Object.keys(history.gradients).some(pname => pname.split('/')[0] === lname);
        gradGrid.style.display = hasGradients ? '' : 'none';
        if (gradSectionLabel && gradSectionLabel.classList.contains('sectionLabel')) {
          gradSectionLabel.style.display = hasGradients ? '' : 'none';
        }
        
        if (hasGradients) {
          for (const [pname, g] of Object.entries(history.gradients)) {
            const base = pname.split('/')[0];
            if (base === lname) {
              renderKind(g, history.gradientShapes[pname], gradGrid);
            }
          }
        }
      }
    }
  
    for (const lname of Object.keys(history.activationShapes)) {
      const weightGrid = document.getElementById(`${lname}-wts`);
      if (weightGrid) {
        // Find the section label (previous element with class "sectionLabel")
        const weightSectionLabel = weightGrid.previousElementSibling;      
        const hasWeights = Object.keys(history.weights).some(pname => pname.split('/')[0] === lname);
        weightGrid.style.display = hasWeights ? '' : 'none';
        if (weightSectionLabel && weightSectionLabel.classList.contains('sectionLabel')) {
          weightSectionLabel.style.display = hasWeights ? '' : 'none';
        }
        
        if (hasWeights) {
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
      optimizer.learningRate = USER_CONST;
      let modelArr = modelComponents;
      
      let dummy = tf.ones([1, 28, 28, 1]);
      for (const layer of modelArr) {
          try{
            dummy = layer.layer.apply(dummy);
          }
          catch(e){
          }
      }
      let layers = modelArr.map(part => part.layer);
      let trueLayerNames = [];
      for(const layer of layers){
        trueLayerNames.push(layer.name);
      }
      
      dummy.dispose();
      
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
          const testBatch = data.nextTestBatch(50);
          const xsBatch = trainBatch.xs.reshape([BATCH_SIZE, 28, 28, 1]);
          const ysBatch = trainBatch.labels;
          const xsTestBatch = testBatch.xs.reshape([50, 28, 28, 1]);
          const ysTestBatch = testBatch.labels;

          tf.tidy(() => {
            let act = xsBatch;
            
            const {value, grads} = 
              tf.variableGrads(() => {
                
                for (const l of layers) {
                  act = l.apply(act);
                 
                    const vals = act.dataSync();
                    for (let r = 0; r < vals.length; r++) {
                      if (vals[r] < -10 || vals[r] > 10) {
                      }
                }

              }
              const loss = lossFn(ysBatch, act);
                           return loss;
            }, trainableVars);
  
  
            const gradMap = {};
            for (const v of trainableVars) gradMap[v.name] = grads[v.name];
            optimizer.applyGradients(gradMap);
        const loss = value.dataSync()[0];        
        history.losses.push(loss);
            });
            const imgs = xsTestBatch;
            let temporaryLoss = 0;
            for(let i=0;i<50;i++){
                
                const [loss,grads] = (() => {
                    let out = imgs.slice([i, 0, 0, 0], [1, 28, 28, 1]);
                    const {value,grads} = tf.variableGrads(() => {
                    for (const l of layers) {
                        out = l.apply(out);
                        if(i===0){
                            if (!history.activations[l.name]) {
                                history.activations[l.name] = [];
                            }
                            history.activations[l.name]=out.dataSync();
                            history.activationShapes[l.name] = out.shape;
                        }
                        }
                        const safeLogits = Float32Array.from(out.dataSync());
                        history.batchPreds[i]  = safeLogits; 
                        tf.keep(out);
                    return lossFn(ysTestBatch.slice([i, 0], [1, NUM_CLASSES]), out);},trainableVars);
                    
                    temporaryLoss+=value.dataSync()/50;
                    return [out,grads];
                })();
                    const layersLength = layers.length;
                    let iterator = 0;
                    if(i===0){
                        for(const trw of trainableVars){
                            let currentName = trw.name;
                            let newName;
                            if((currentName.includes("conv2d")&&!currentName.includes("bias"))){
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
                                    iterator++;
                                }
                                catch(e){
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
                                    iterator++;
                                }
                                catch(e){
                                }
                                iterator++;
                            }

                        }
                    }
                let averageValLoss = temporaryLoss;
                history.vallosses.push(averageValLoss);
                if(i===49){
                    lossChart.data.labels.push(iter);
                    lossChart.data.datasets[0].data.push(history.losses[history.losses.length-1]);
                    lossChart.data.datasets[1].data.push(averageValLoss);
                    
                    const maxLoss = Math.max(
                        ...lossChart.data.datasets[0].data,
                        ...lossChart.data.datasets[1].data
                    );

                    lossChart.options.scales.y.min = 0;
                    lossChart.options.scales.y.max = maxLoss + 0.5;

                    lossChart.update();
                      await showPredictions(imgs, ysTestBatch);

                    history.modelLayersCopy = modelArr.map(obj => obj.layer);
                }
            }
            handleLayerVisualizationUpdates(history);
          iter++;
    
          tf.dispose([xsBatch, ysBatch]);
        }
        
    }
    
      async function showPredictions(imgs, labels) {
        const logits  = tf.tensor(history.batchPreds, [50, NUM_CLASSES]);
        const probs   = logits.softmax();
        const predIds = await probs.argMax(-1).data();
        const truthIds= await labels.argMax(-1).data();

        const top3 = [];
        const pb   = probs.dataSync();
        for (let i = 0; i < 50; ++i) {
          top3.push(
            Array.from(pb.slice(i*NUM_CLASSES, (i+1)*NUM_CLASSES))
                .map((v, idx) => ({v, idx}))
                .sort((a,b)=>b.v-a.v)
                .slice(0,3)
          );
        }

        const grid = document.getElementById('demoGrid');
        if (!grid.dataset.ready) {
          for (let i = 0; i < 50; ++i) {
            const cell   = document.createElement('div');
            cell.className = 'demoCell';

            const canv   = document.createElement('canvas');
            canv.width = canv.height = 28;
            canv.id    = `mn_${i}`;

            const pLbl   = document.createElement('div'); pLbl.className = 'pred';
            const tLbl   = document.createElement('div'); tLbl.className = 'truth';

            const bars   = document.createElement('div'); bars.className = 'bars';
            for (let k=0;k<3;++k){
              const bar  = document.createElement('div'); bar.className='bar';
              const fill = document.createElement('div'); fill.className='fill';
              const txt  = document.createElement('span');
              bar.append(fill, txt); bars.appendChild(bar);
            }
            cell.append(canv, pLbl, tLbl, bars); grid.appendChild(cell);
          }
          grid.dataset.ready = 1;
        }

        const tensors = tf.unstack(imgs);
        await Promise.all(tensors.map(async (t,i)=>{
          const canv = document.getElementById(`mn_${i}`);
          await tf.browser.toPixels(t.squeeze(), canv);
          t.dispose();

          const cell  = canv.parentElement;
          cell.querySelector('.pred').textContent  = `pred: ${predIds[i]}`;
          cell.querySelector('.truth').textContent = `true: ${truthIds[i]}`;

          const barDivs = cell.querySelectorAll('.bar');
          top3[i].forEach((conf,k)=>{
            const fill = barDivs[k].querySelector('.fill');
            const txt  = barDivs[k].querySelector('span');
            txt.textContent = conf.idx;
            fill.style.transform = `scaleX(${conf.v})`;
            fill.style.background = conf.idx === truthIds[i] ? 'limegreen' : 'crimson';
          });
        }));

        logits.dispose(); probs.dispose();
        await tf.nextFrame();
      }
    }
  
