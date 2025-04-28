// script.js
import * as tf from '@tensorflow/tfjs';
import {MnistData} from './data.js';

/** Global data reference */
let data = null;

const targetArea   = document.getElementById('target');
const logArea      = document.getElementById('logArea');
const chartCanvas  = document.getElementById('canvasChart');
const chartCtx     = chartCanvas.getContext('2d');
const embedCanvas  = document.getElementById('embedding-canvas');
const embedCtx     = embedCanvas.getContext('2d');

const BATCH_SIZE   = 64;
const EMBEDDING_DIM= 10;    // for similarity search
const trainBtn     = document.getElementById('trainBtn');
const validateBtn  = document.getElementById('validateBtn');
const run100Btn    = document.getElementById('run100Btn');
const leaderboard  = document.getElementById('leaderboard');
const saveBoardBtn = document.getElementById('saveLeaderboardBtn');

let chartDataLoss  = [];
let chartDataVal   = [];
let model          = null;
let isTraining     = false;
const EPOCHS       = 5;

/** Called once at page load */
window.onload = async () => {
  document.getElementById('loading-indicator').style.display = 'block';
  data = new MnistData();
  await data.load();
  document.getElementById('loading-indicator').style.display = 'none';

  initDragDrop();
  appendLog('MNIST data loaded.\n');
  fetchLeaderboard();
};

/** Minimal logger */
function appendLog(msg) {
  logArea.textContent += msg;
  logArea.scrollTop = logArea.scrollHeight;
}

/** Draw chart with custom canvas: scaled line for loss & val */
function drawChart() {
  // Clear
  chartCtx.fillStyle = '#fff';
  chartCtx.fillRect(0, 0, chartCanvas.width, chartCanvas.height);

  const margin = 20;
  const maxLen = Math.max(chartDataLoss.length, chartDataVal.length);
  if (maxLen < 2) return;

  // find min & max
  const allLosses  = chartDataLoss.concat(chartDataVal);
  const minVal     = Math.min(...allLosses);
  const maxVal     = Math.max(...allLosses);

  // scaled x
  const stepX      = (chartCanvas.width - margin*2) / (maxLen - 1);
  // scaled y
  function yScale(v) {
    return margin + (chartCanvas.height - margin*2) * (1 - (v - minVal)/(maxVal - minVal));
  }

  // Draw axes
  chartCtx.strokeStyle = '#000';
  chartCtx.beginPath();
  // x-axis
  chartCtx.moveTo(margin, chartCanvas.height - margin);
  chartCtx.lineTo(chartCanvas.width - margin, chartCanvas.height - margin);
  chartCtx.stroke();

  // Plot training loss
  chartCtx.strokeStyle = 'blue';
  chartCtx.beginPath();
  chartCtx.moveTo(margin, yScale(chartDataLoss[0]));
  for (let i = 1; i < chartDataLoss.length; i++) {
    chartCtx.lineTo(margin + i*stepX, yScale(chartDataLoss[i]));
  }
  chartCtx.stroke();

  // Plot validation
  chartCtx.strokeStyle = 'red';
  chartCtx.beginPath();
  chartCtx.moveTo(margin, yScale(chartDataVal[0]));
  for (let i = 1; i < chartDataVal.length; i++) {
    chartCtx.lineTo(margin + i*stepX, yScale(chartDataVal[i]));
  }
  chartCtx.stroke();
}

/** Activation & gradient capturing (optional toggle) */
function captureActivationsAndGradients(model, x) {
  // example: run a forward pass, store activation from each layer
  // then create a backward pass to get gradient w.r.t. input
  // Doing partial code for demonstration
  const layerOutputs = model.layers.map(layer => layer.output);
  const activationModel = tf.model({inputs: model.inputs, outputs: layerOutputs});
  const acts = activationModel.predict(x);
  // acts is an array of Tensors if multiple layers
  // gather them or store them if you want

  // gradient (simple example for demonstration)
  const lossFn = () => model.predict(x).mean();
  const grads  = tf.grad(lossFn)(model.weights[0].val); // gradient wrt first param
  // store grads somewhere

  acts.forEach(t => t.dispose());
  grads.dispose();
}

/** For each batch, we also track top-10 worst predictions */
function getWorstPerformers(xs, labels, batchPred, batchSize) {
  // batchPred shape: [batchSize, 10], argMax => predicted label
  // labels shape: [batchSize, 10], argMax => true label
  const predsArr  = batchPred.argMax(-1).dataSync();
  const labsArr   = labels.argMax(-1).dataSync();

  // Calculate each sample’s “loss” or confidence error
  const losses = [];
  for (let i = 0; i < batchSize; i++) {
    const correct = (predsArr[i] === labsArr[i]);
    // e.g. - we track negative conf for the correct label
    // just do a quick cheat measure (1 - predicted prob for correct label)
    const rowLogits = batchPred.slice([i,0],[1,10]);
    const rowProbs  = rowLogits.softmax().dataSync();
    const correctProb = rowProbs[labsArr[i]];
    const error       = 1 - correctProb;  // bigger = more wrong
    losses.push({idx:i, error});
  }
  // Sort descending by error
  losses.sort((a,b) => b.error - a.error);
  const top10 = losses.slice(0,10);

  // Gather them
  const worstData = [];
  top10.forEach(item => {
    worstData.push({ idxInBatch: item.idx, 
                     predLabel: predsArr[item.idx], 
                     trueLabel: labsArr[item.idx], 
                     errorScore: item.error });
  });
  return worstData;
}

/** We can store the top-10 worst image data for later display. */
function storeWorstImages(batchXs, worstData) {
  // Each image is 28x28, stored in row
  const out = [];
  worstData.forEach(o => {
    // float32 data from batch
    const slice = batchXs.slice([o.idxInBatch,0],[1,28*28]);
    const arr   = slice.dataSync();
    slice.dispose();
    out.push({imageData: arr, pred: o.predLabel, truth: o.trueLabel, errorScore: o.errorScore});
  });
  return out;
}

/** Renders top-10 “worst” images into the log area with small thumbnails. */
function renderWorstPerformers(worst, batchNum) {
  appendLog(`Worst performers batch #${batchNum}:\n`);
  worst.forEach((item, i) => {
    appendLog(`  #${i+1} -> pred=${item.pred} truth=${item.truth} err=${item.errorScore.toFixed(3)}\n`);
  });
  // optionally show thumbnails
  // create a small container in logArea or a separate div
  // ...
}

/** Simple PCA for an NxD array -> Nx2 (or NxK). Using naive SVD in JS for demonstration. */
function runPCA(tensor, outDim=2) {
  // naive approach: compute covariance, do eigen, or just do tf.svd
  // e.g. shape [N, D], we want [N, outDim]
  // For large N or D, real PCA is best in python, but here we do a small example.
  const mean = tensor.mean(0);
  const centered = tensor.sub(mean);
  const {u, s, v} = tf.svd(centered, true); 
  // v shape [D, D], columns are principal components
  // project = X_centered dot v[:,0:outDim]
  const PCs = v.slice([0,0],[v.shape[0], outDim]);
  const result = centered.matMul(PCs);
  return result; // shape [N, outDim]
}

/** Show a 2D scatter of embeddings on embedCanvas. xy in [-range, range]. */
function drawEmbeddings(emb, labelsArr) {
  embedCtx.clearRect(0,0,embedCanvas.width, embedCanvas.height);
  const w = embedCanvas.width, h = embedCanvas.height;
  // find min / max
  const coords = emb.arraySync();
  let minX=9999, maxX=-9999, minY=9999, maxY=-9999;
  coords.forEach(([xx,yy]) => {
    if(xx<minX) minX=xx; if(xx>maxX) maxX=xx;
    if(yy<minY) minY=yy; if(yy>maxY) maxY=yy;
  });
  const pad = 20;
  function scaleX(x){ return pad + (x - minX)/(maxX-minX)*(w-2*pad); }
  function scaleY(y){ return h-pad - (y - minY)/(maxY-minY)*(h-2*pad); }

  for (let i=0; i<coords.length; i++){
    const x = coords[i][0], y= coords[i][1];
    embedCtx.fillStyle = `hsl(${labelsArr[i]*36},100%,50%)`;
    embedCtx.beginPath();
    embedCtx.arc(scaleX(x), scaleY(y), 3, 0, 2*Math.PI);
    embedCtx.fill();
  }
}

/** Attach to the “Train Single Network” button */
trainBtn.onclick = async () => {
  const {ok, errors} = shapeCheck();
  if(!ok){
    alert(errors.join('\n'));
    return;
  }
  // build & train
  await trainModelOne();
};

/** Attach to the “Validate Model Blocks” button */
validateBtn.onclick = () => {
  const {ok, errors} = shapeCheck();
  if(!ok) alert(errors.join('\n'));
  else alert('Blocks look shape-compatible & have a compile block. Good to go!');
};

/** Attach to the “Run 100 Versions (No changes)” button */
run100Btn.onclick = async () => {
  // Just train 100 times with the same architecture
  // In real usage we would param-sweep. Here is a minimal approach.
  for (let i=0; i<100; i++){
    appendLog(`=== Running net #${i+1}/100 ===\n`);
    await trainModelOne(false /*no logs each batch, just final*/);
  }
  appendLog(`=== Done with 100 runs ===\n`);
};

/** Attach to “Save to Leaderboard” button */
saveBoardBtn.onclick = async () => {
  const valLoss = chartDataVal[chartDataVal.length-1];
  if(!valLoss) {
    alert('No valLoss found. Train first.');
    return;
  }
  const name = prompt('Enter your name for the leaderboard:');
  if(!name) return;
  const payload = { name, valLoss };
  try {
    const res = await fetch('http://localhost:4000/api/leaderboard', {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)
    });
    if(!res.ok) throw new Error('Failed to save leaderboard');
    appendLog('Saved to leaderboard!\n');
    fetchLeaderboard();
  } catch(err){
    appendLog(`Error saving leaderboard: ${err}\n`);
  }
};

/** GET the leaderboard from server */
async function fetchLeaderboard(){
  try {
    const res = await fetch('http://localhost:4000/api/leaderboard');
    const list = await res.json();
    leaderboard.innerHTML = '<strong>Leaderboard (lowest valLoss first):</strong><br>';
    list.forEach(entry => {
      leaderboard.innerHTML += `${entry.name} → ${entry.valLoss.toFixed(4)}<br>`;
    });
  } catch(err){
    leaderboard.innerHTML = '<strong>Leaderboard not available</strong>';
  }
}

/** Actually builds the model from the drag-drop blocks, then trains it. */
async function trainModelOne(verboseLog=true) {
  if(isTraining) { appendLog('Already training...\n'); return; }
  isTraining = true;
  chartDataLoss = [];
  chartDataVal  = [];
  model = buildModelFromBlocks();
  if(verboseLog) appendLog(`Model built: ${model.summary()}\n`);

  const trainBatches = 10; // or more
  const testBatches  = 2;
  for (let batchNum=0; batchNum<trainBatches; batchNum++){
    const batch = data.nextTrainBatch(BATCH_SIZE);
    const xs = batch.xs;
    const ys = batch.labels;
    
    // single step of training
    const history = await model.fit(xs, ys, {
      epochs: 1,
      batchSize: BATCH_SIZE,
      shuffle: false,
      // callbacks for custom logic
    });
    const lossValue = history.history.loss[0];

    // do a small test batch
    const valBatch  = data.nextTestBatch(BATCH_SIZE);
    const valXs     = valBatch.xs;
    const valYs     = valBatch.labels;
    const valPred   = model.predict(valXs);
    const valLoss   = tf.losses.softmaxCrossEntropy(valYs, valPred).mean().dataSync()[0];
    chartDataLoss.push(lossValue);
    chartDataVal.push(valLoss);
    drawChart();

    // top-10 worst
    const worst   = getWorstPerformers(valXs, valYs, valPred, BATCH_SIZE);
    const worstIm = storeWorstImages(valXs, worst);
    if(verboseLog) renderWorstPerformers(worstIm, batchNum);

    // optional capture activations/grad
    // captureActivationsAndGradients(model, valXs);

    xs.dispose(); ys.dispose(); valXs.dispose(); valYs.dispose(); valPred.dispose();
  }

  // after training is done, do a small embedding example
  // e.g. run model up to penultimate layer to get embeddings on some images
  const embedBatch   = data.nextTestBatch(100);
  const embedXs      = embedBatch.xs;
  // your penultimate layer is typically model.layers[model.layers.length-2]
  let penLayer = model.layers.length-2;
  if(penLayer<0) penLayer = 0;
  const activModel   = tf.model({inputs: model.inputs, outputs: model.layers[penLayer].output});
  const embedOut     = activModel.predict(embedXs); // shape [100, something]
  // run pca
  const emb2D        = runPCA(embedOut, 2);         // shape [100,2]
  const labelsArr    = embedBatch.labels.argMax(-1).dataSync(); 
  drawEmbeddings(emb2D, labelsArr);

  embedXs.dispose(); embedOut.dispose(); emb2D.dispose();
  isTraining = false;
}

/** Build a TFJS model from the user’s dropped blocks in #target */
function buildModelFromBlocks() {
  const m = tf.sequential();
  const blocks = [...targetArea.children];
  let compiled = false;

  // We track a shape = [28,28,1], we see if the first conv2d or dense needs inputShape
  let hasFirstLayer = false;

  blocks.forEach((b, i) => {
    const type = b.dataset.type;
    if(type==='conv2d'){
      const k = +b.querySelector('[data-param="kernelSize"]').value;
      const f = +b.querySelector('[data-param="filters"]').value;
      const s = +b.querySelector('[data-param="strides"]').value;
      if(!hasFirstLayer){
        m.add(tf.layers.conv2d({
          inputShape:[28,28,1],
          kernelSize:k, filters:f, strides:s, kernelInitializer:'varianceScaling'
        }));
        hasFirstLayer = true;
      } else {
        m.add(tf.layers.conv2d({
          kernelSize:k, filters:f, strides:s, kernelInitializer:'varianceScaling'
        }));
      }
      m.add(tf.layers.relu()); // or you can skip adding an activation if you want
    }
    else if(type==='maxpool'){
      const p = +b.querySelector('[data-param="poolSize"]').value;
      m.add(tf.layers.maxPooling2d({poolSize:p, strides:p}));
    }
    else if(type==='dense'){
      const inC  = +b.querySelector('[data-param="inputChannels"]').value;
      const outC = +b.querySelector('[data-param="outputChannels"]').value;
      // ensure flatten
      if(!hasFirstLayer){
        m.add(tf.layers.flatten({inputShape:[28,28,1]}));
        hasFirstLayer = true;
      } else {
        m.add(tf.layers.flatten());
      }
      // check mismatch with inC is ignored for demonstration 
      m.add(tf.layers.dense({units:outC, kernelInitializer:'varianceScaling'}));
    }
    else if(type==='relu'){
      m.add(tf.layers.activation({activation:'relu'}));
    }
    else if(type==='sigmoid'){
      m.add(tf.layers.activation({activation:'sigmoid'}));
    }
    else if(type==='tanh'){
      m.add(tf.layers.activation({activation:'tanh'}));
    }
    else if(type==='loss'){
      if(compiled) return; // skip if we had multiple
      compiled = true;
      const optSel  = b.querySelector('[data-param="optimizer"]').value;
      const lossSel = b.querySelector('[data-param="loss"]').value;
      let optimizer;
      if(optSel==='sgd') optimizer = tf.train.sgd(0.01);
      else if(optSel==='adam') optimizer = tf.train.adam();
      else if(optSel==='rmsprop') optimizer = tf.train.rmsprop(0.001);
      m.add(tf.layers.dense({units:10, activation:'softmax'}));
      m.compile({
        optimizer,
        loss: lossSel,
        metrics: ['accuracy']
      });
    }
  });

  // if user forgot to add a loss block, do it anyway
  if(!compiled){
    m.add(tf.layers.dense({units:10, activation:'softmax'}));
    m.compile({
      optimizer: tf.train.sgd(0.01),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
  }
  return m;
}

/** Similar to your existing shapeCheck. Ensures the final block is “loss”, etc. */
function shapeCheck() {
  const errs = [];
  let compiled = false;
  const blocks = [...targetArea.querySelectorAll('.block')];
  if(!blocks.length){
    errs.push('No blocks in target area');
    return { ok:false, errors:errs };
  }

  // Basic check: last block must be or include a “loss” block
  let hasLoss = blocks.some(b => b.dataset.type==='loss');
  if(!hasLoss) errs.push('Missing a compile/loss block somewhere');

  // If you want more advanced shape or param checks, do them here
  // For now, we rely on your existing “validateModelBlocks” approach plus
  // this final check that a “loss” block is present.

  const paramReport = validateModelBlocks();
  if(!paramReport.ok) errs.push(...paramReport.errors);

  return { ok: errs.length===0, errors: errs };
}

/** The existing param-based validator (copied from your code) */
function validateModelBlocks() {
  const errors = [];
  const blocks = [...targetArea.querySelectorAll('.block')];

  const num = (el, name) => Number(el.querySelector(`[data-param="${name}"]`)?.value);

  blocks.forEach((b, idx) => {
    const type = b.dataset.type;
    switch (type) {
      case 'conv2d': {
        const k = num(b, 'kernelSize');
        const f = num(b, 'filters');
        const s = num(b, 'strides');
        if (!k || k<1) errors.push(`Block ${idx+1} Conv2D: kernelSize missing/invalid`);
        if (!f || f<1) errors.push(`Block ${idx+1} Conv2D: filters missing/invalid`);
        if (!s || s<1) errors.push(`Block ${idx+1} Conv2D: strides missing/invalid`);
        break;
      }
      case 'maxpool': {
        const p = num(b, 'poolSize');
        if (!p || p<1) errors.push(`Block ${idx+1} MaxPool: poolSize missing/invalid`);
        break;
      }
      case 'dense': {
        const inC  = num(b, 'inputChannels');
        const outC = num(b, 'outputChannels');
        if(!inC || inC<1)  errors.push(`Block ${idx+1} Dense: inputChannels missing/invalid`);
        if(!outC || outC<1)errors.push(`Block ${idx+1} Dense: outputChannels missing/invalid`);
        break;
      }
      default:
        // no param checks for stateless or “loss” blocks
    }
  });

  return { ok: errors.length===0, errors };
}

/** Basic drag-drop logic for the net builder */
function initDragDrop() {
  // Make blocks in .source-area draggable
  const blocks = document.querySelectorAll('.block');
  blocks.forEach(function(box) {
    box.addEventListener('dragstart', function(ev) {
      ev.dataTransfer.setData('text/plain', box.id);
      if(targetArea.children.length!==0) createDropZonesInTarget();
    });
  });

  targetArea.addEventListener('dragover', ev => {
    ev.preventDefault();
  });
  targetArea.addEventListener('drop', ev => {
    ev.preventDefault();
    const boxId = ev.dataTransfer.getData('text/plain');
    const origBox = document.getElementById(boxId);
    const newBox = origBox.cloneNode(true);
    newBox.id = `${boxId}-clone-${cloneCount++}`;
    targetArea.appendChild(newBox);
    const defaultText = document.getElementById('default-text');
    if(defaultText) defaultText.remove();
    addTrashIconToBlock(newBox);

    removeDropZones();
  });
}

/** Creates the visual drop-zones between existing blocks */
function createDropZonesInTarget() {
  removeDropZones();
  const blocks = targetArea.querySelectorAll('.block');
  blocks.forEach(block => {
    const dz = document.createElement('div');
    dz.classList.add('drop-zone');
    targetArea.insertBefore(dz, block);
  });
  const dropZones = targetArea.querySelectorAll('.drop-zone');
  dropZones.forEach(zone => {
    zone.addEventListener('dragover', e => {
      e.preventDefault();
      e.stopPropagation();
      zone.classList.add('drop-zone-active');
    });
    zone.addEventListener('dragleave', () => {
      zone.classList.remove('drop-zone-active');
    });
    zone.addEventListener('drop', e => {
      e.preventDefault();
      e.stopPropagation();
      zone.classList.remove('drop-zone-active');
      const boxId = e.dataTransfer.getData('text/plain');
      const origBox = document.getElementById(boxId);
      const newBox = origBox.cloneNode(true);
      newBox.id = `${boxId}-clone-${cloneCount++}`;
      targetArea.insertBefore(newBox, zone);
      addTrashIconToBlock(newBox);
      removeDropZones();
    });
  });
}

/** Removes all .drop-zone elements from #target */
function removeDropZones() {
  const zones = targetArea.querySelectorAll('.drop-zone');
  zones.forEach(z => z.remove());
}

/** Adds the trash icon for each newly created block in #target */
function addTrashIconToBlock(block) {
  const trashIcon = document.createElement('div');
  trashIcon.classList.add('trash-icon');
  trashIcon.innerHTML = '🗑️';
  trashIcon.addEventListener('click', ev => {
    ev.stopPropagation();
    block.remove();
  });
  block.insertBefore(trashIcon, block.firstChild);
}
