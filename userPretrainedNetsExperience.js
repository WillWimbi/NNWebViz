const TRAIN_DATA_SIZE = 55000;
const TEST_DATA_SIZE = 10000;
const IMAGE_SIZE = 784;        // 28×28
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;
const NUM_TRAIN_ELEMENTS = 55000;
const SCALE = 1.6;

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

        img.width = img.naturalWidth;
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
          ctx.drawImage(img,0, i * chunkSize,img.width, chunkSize,0, 0,img.width, chunkSize);
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          for (let j = 0; j < imageData.data.length / 4; j++) {
            datasetBytesView[j] = imageData.data[j * 4] / 255;
          }
        }
        this.datasetImages = new Float32Array(datasetBytesBuffer);
        resolve();
      };
      img.onerror = reject;
      img.src = MNIST_IMAGES_SPRITE_PATH;
    });
    const labelsRequest = fetch(MNIST_LABELS_PATH);
    const [, labelsResponse] = await Promise.all([imgRequest, labelsRequest]);
    this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());
    this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.testLabels = this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
  }


  getImg(idx, ) {
    const images = this.testImages;
    const labels = this.testLabels;
    const image  = images.slice(idx * IMAGE_SIZE, (idx + 1) * IMAGE_SIZE);
    const label  = labels.slice(idx * NUM_CLASSES, (idx + 1) * NUM_CLASSES);
    const xs = tf.tensor2d(image,  [1, IMAGE_SIZE]);
    const ys = tf.tensor2d(label,  [1, NUM_CLASSES]);
    return { xs, ys };
  }

  getTestImageDataURL(idx) {
    const image = this.testImages.slice(idx * IMAGE_SIZE, (idx + 1)*IMAGE_SIZE);
    return floatArrayToDataURL(image);
  }
}

const mnistData = new MnistData();
mnistData.load().then(() => {
  console.log("MNIST data loaded (trainImages,testImages, etc).");
}).catch(e => console.error("Error loading MNIST data:", e));

fetch('/api/runs')
  .then(r => {
    if(!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  })
  .then(runs => {
    console.log("Runs loaded:", runs.length);
    drawFourCharts(runs);     // chart 1: loss vs val_loss
    buildAggregateGrid(runs);    // chart 2: acc vs val_acc
  })
  .catch(err => {
    console.error("/api/runs fetch error:", err);
    const ebox = document.createElement('div');
    ebox.textContent = `Error loading runs: ${err.message}`;
    ebox.style.color = 'red';
    document.body.appendChild(ebox);
  });

function buildAggregateGrid(runs) {
  // 1) For each unique idx in validationImagesTest, sum up the 10 probs
  //    across all runs. Also track min & max prob found for the correct digit.
  const uniqueBucket = {};  // idx => { sum:[10], n, min, max, trueLabel }
  for (const run of runs) {
    for (const { idx, label, probs } of run.validationImagesTest) {
      if (!uniqueBucket[idx]) {
        uniqueBucket[idx] = {sum: Array(10).fill(0),n: 0,min: 1,max: 0,label};
      }
      const b = uniqueBucket[idx];
      b.n++;
      for (let i=0; i<10; i++){b.sum[i]+= probs[i];}
      const cProb = probs[label];
      if (cProb < b.min) b.min = cProb;
      if (cProb > b.max) b.max = cProb;
    }
  }

  const items = Object.entries(uniqueBucket).map(([idx,b]) => {
    const avg = b.sum.map(s => s / b.n);
    return {
      idx: +idx,label: b.label,avg,avgCorrect: avg[b.label],min: b.min,max: b.max,pred: avg.indexOf(Math.max(...avg))   // argmax
    };
  });

  items.sort((a,b)=> a.avgCorrect - b.avgCorrect);
  const top100 = items.slice(0,100);
  const grid = document.getElementById("imgGrid");
  grid.innerHTML = '';
  grid.style.display = 'grid';
  grid.style.gridTemplateColumns = `repeat(10, ${70 + 28*(SCALE-1)}px)`;
  grid.style.gap = '8px';
  top100.forEach((obj, i) => {
    const tile = document.createElement('div');
    tile.style.border='1px solid #ccc';
    tile.style.padding='3px';
    tile.style.font='10px monospace';
    const indexInTest = obj.idx; // presumably 0..9999 range
    const url = mnistData.getTestImageDataURL(indexInTest);
    const im = new Image();
    im.src = url;


    im.width = im.height = 28 * SCALE;
    im.style.display = 'block';
    im.style.margin = 'auto'; //gotta center it
    tile.appendChild(im);
    // line: "t2 p5" eqs true=2, pred=5
    const title = document.createElement('div');
    title.style.textAlign='center';
    title.style.fontWeight='bold';
    title.textContent = `T${obj.label} P${obj.pred}`;
    tile.appendChild(title);
    // 10 barrsrr
    obj.avg.forEach((p, digit) => {
      const bar = document.createElement('div');
      bar.style.position='relative';
      bar.style.height='10px';

      bar.style.margin='1px 0';
      bar.style.background = (digit===obj.label) ? '#090' : '#c33';
      bar.style.width = (p*100).toFixed(1)+'%';
      const textSpan = document.createElement('span');
      
      textSpan.style.position='absolute';
      textSpan.style.left='2px';
      textSpan.style.top='0';
      textSpan.style.color='#fff';
      textSpan.style.fontWeight='bold';
      textSpan.textContent = `${digit} ${(p*100).toFixed(0)}%`;
      bar.appendChild(textSpan);
      tile.appendChild(bar);
    });
    // min / max for correct digit..
    const mm = document.createElement('div');
    mm.style.textAlign='center';
    mm.textContent = `min ${(obj.min*100).toFixed(1)}% max ${(obj.max*100).toFixed(1)}%`;
    mm.style.marginTop='10px'; 
    tile.appendChild(mm);

    grid.appendChild(tile);
  });
}

function floatArrayToDataURL(arr) {
  const c = document.createElement('canvas');
  c.width = 28; c.height = 28;
  const ctx = c.getContext('2d');
  const imageData = ctx.createImageData(28, 28);
  for (let i=0; i<784; i++){
    const v = Math.floor(arr[i] * 255);
    imageData.data[i*4+0] = v;    
    imageData.data[i*4+1] = v;    
    imageData.data[i*4+2] = v;    
    imageData.data[i*4+3] = 255;  
  }
  ctx.putImageData(imageData,0,0);
  return c.toDataURL('image/png');
}


function drawFourCharts(runs) {
  const EPOCHS = 3;                        // fixed
  const labels = ['Epoch 1','Epoch 2','Epoch 3'];

  function percentile(arr, p){
    // p in [0..100]; arr is NUMBER[] – does NOT mutate arr
    const sorted = [...arr].sort((a,b)=>a-b);
    const idx = (p/100)*(sorted.length-1);
    const lo = Math.floor(idx), hi = Math.ceil(idx);
    if (lo === hi) return sorted[lo];
    return sorted[lo] + (sorted[hi]-sorted[lo]) * (idx-lo);
  }

  const metrics = ['loss','val_loss','acc','val_acc'];
  const buckets = {};                      // metric -> epoch -> []
  metrics.forEach(m => buckets[m] = Array.from({length:EPOCHS},
                                              ()=> []));

  for (let r=0; r<runs.length; r++){
    for (let e=0; e<EPOCHS; e++){
      const ep = runs[r].epochs[e];
      buckets.loss     [e].push(ep.loss);
      buckets.val_loss [e].push(ep.val_loss);
      buckets.acc      [e].push(ep.acc);
      buckets.val_acc  [e].push(ep.val_acc);
    }
  }

  function summary(metric){
    const avg = [], p2=[], p10=[], p25=[], p75=[], p90=[], p98=[];
    for (let e=0; e<EPOCHS; e++){
      const arr = buckets[metric][e];
      let sum = 0;
      for (let i=0;i<arr.length;i++) sum += arr[i];
      avg .push( sum / arr.length );
      p2  .push( percentile(arr,  2));
      p10 .push( percentile(arr, 10));
      p25 .push( percentile(arr, 25));
      p75 .push( percentile(arr, 75));
      p90 .push( percentile(arr, 90));
      p98 .push( percentile(arr, 98));
    }
    return {avg,p2,p10,p25,p75,p90,p98};
  }

  const S  = summary('loss');
  const SV = summary('val_loss');
  const A  = summary('acc');
  const AV = summary('val_acc');

  function rgba(hex, a){
    const bigint = parseInt(hex.replace('#',''),16);
    const r=(bigint>>16)&255, g=(bigint>>8)&255, b=bigint&255;
    return `rgba(${r},${g},${b},${a})`;
  }

  function makeChart(canvasId, baseColor, data){
    new Chart(document.getElementById(canvasId).getContext('2d'),{
      type:'line',
      data:{
        labels,
        datasets:[
          {label:'2%',  data:data.p2,  borderColor:rgba(baseColor,0.2), fill:false},
          {label:'10%', data:data.p10, borderColor:rgba(baseColor,0.35),fill:false},
          {label:'25%', data:data.p25, borderColor:rgba(baseColor,0.55),fill:false},
          {label:'avg', data:data.avg, borderColor:rgba(baseColor,1.00), fill:false,
                       borderWidth:3},
          {label:'75%', data:data.p75, borderColor:rgba(baseColor,0.55),fill:false},
          {label:'90%', data:data.p90, borderColor:rgba(baseColor,0.35),fill:false},
          {label:'98%', data:data.p98, borderColor:rgba(baseColor,0.2), fill:false}
        ]
      },
      options:{
        scales:{ y:{ beginAtZero:true } },
        plugins:{ legend:{position:'bottom'} }
      }
    });
  }
  makeChart('lossChart',    '#0066ff', SV);   
  makeChart('valLossChart', '#800080', S);    
  makeChart('accChart',     '#008000', AV);   
  makeChart('valAccChart',  '#d2b48c', A);    
}



