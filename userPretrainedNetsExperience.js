/*  1824, 409, 4506, 4012, 3657, 2286, 1679, 8935, 1424, 9674,
  6912, 520, 488, 1535, 3582, 3811, 8279, 9863, 434, 9195,
  3257, 8928, 6873, 3611, 7359, 9654, 4557, 106, 2615, 6924,
  5574, 4552, 2547, 3527, 5514, 1674, 1519, 6224, 1584, 5881,
  5635, 9891, 4333, 711, 7527, 8785, 2045, 6201, 1291, 9044,
  4803, 5925, 9459, 3150, 1139, 750, 3733, 4741, 1307, 3814,
  1654, 6227, 4554, 7428, 5977, 2664, 6065, 5820, 3432, 4374,
  1169, 9980, 2803, 8751, 4010, 2677, 7573, 6216, 4422, 9125,
  3598, 5313, 916, 3752, 525, 5168, 6572, 4386, 1084, 3456,
  9292, 5155, 3483, 8179, 6482, 7517, 2340, 4339, 2287, 4040 */

//these values must be loaded, they madeup the 100 image validation set for the pretrained nets

//display graph of all 

//load fulldata
const fullData = [];
/* global Chart */        // assumes chart.js already loaded

// ───────────────────────────────────────────────────────────
// PUBLIC ENTRY
// call once you have fullData = Array(50) of run objects
// will mutate DOM:  <canvas id="metricsChart"></canvas><div id="imgGrid"></div>
export function visualizeRuns(fullData, imageSrcLookup){
    const epochs   = 3; //3 in this case.
    const metrics  = ["loss","val_loss","acc","val_acc"];
  
    // ---------- 1. aggregate metrics → percentile lines ----------
    const percentileLines = percentileEnvelope(fullData, epochs, metrics);
  
    renderChart(percentileLines, epochs, metrics);          // big Chart.js plot
    // ---------- 2. score images & render 10×10 grid -------------
    const worst100 = rankImages(fullData).slice(0,100);     // [{idx,score}]
    renderImageGrid(worst100, imageSrcLookup);              // expects idx→src fn
  }
  
  // ───────────────────────────────────────────────────────────
  // HELPERS
  function percentileEnvelope(runs, epochs, metrics){
    const perc = [2,10,25,50,75,90,98];
    const out  = {};                        // metric → [{label,data}]
    // Alright, so here's what's happening: we're looping over each metric in our metrics array.
    // For each metric, we're creating an entry in the 'out' object. This entry is an array of objects.
    // Each object corresponds to a specific percentile (like 2%, 10%, etc.) for that metric.
    // The 'label' of each object is a string that combines the percentile and the metric name, 
    // and 'data' is initialized as an empty array, ready to be filled with data points later.
    // The '%' symbol here represents the percentile. For each metric, we are creating an entry in the 'out' object.
    // This entry is an array of objects, where each object corresponds to a specific percentile (like 2%, 10%, etc.) for that metric.
    // The 'label' of each object is a string that combines the percentile and the metric name.
    metrics.forEach(m => out[m] = perc.map(p => ({ label: `${p}% ${m}`, data: [] })));
    //for each metric, simply give a nice label, leave data empty for now
    for(let e=0;e<epochs;e++){
      metrics.forEach(m=>{
        // This line of code extracts the values of a specific metric 'm' for a given epoch 'e' from each run in the 'runs' array.
        // It then sorts these values in ascending order. The result is an array 'epochVals' that contains the sorted metric values
        // for the specified epoch across all runs, which is used to calculate percentile values later.
        const epochVals = runs.map(r=>r.epochs[e][m]).sort((a,b)=>a-b);
        perc.forEach((p,i)=>{
          const k = Math.round((p/100)*(epochVals.length-1));
          out[m][i].data.push(epochVals[k]);
        });
      });
    }
    return out;          // {loss:[{label,data},…], …}
  }
  
  function renderChart(env, epochs, metrics){
    const colors = ["#222","#444","#666","#888","#aaa","#ccc","#eee"];
    const ctx = document.getElementById("metricsChart").getContext("2d");
    const datasets = [];
    metrics.forEach((m,mi)=>{
      env[m].forEach((d,di)=>{
        datasets.push({
          label:d.label,
          data:d.data,
          borderColor:colors[di],
          borderWidth:1+(mi===0?1:0),     // make loss thicker if you like
          fill:false,
          yAxisID:m.includes("acc")?"y1":"y0"
        });
      });
    });
    new Chart(ctx,{
      type:"line",
      data:{labels:[...Array(epochs).keys()],datasets},
      options:{
        responsive:true,
        scales:{
          y0:{type:"linear",position:"left",title:{text:"loss",display:true}},
          y1:{type:"linear",position:"right",title:{text:"accuracy",display:true},min:0,max:1,grid:{drawOnChartArea:false}}
        }
      }
    });
  }
  
  function rankImages(runs){
    const imgScore = {};                       // idx → summed error
    runs.forEach(r=>{
      r.validationImagesTest.forEach(({idx,label,probs})=>{
        const pTrue = probs[label];
        imgScore[idx] = (imgScore[idx]||0) + (1 - pTrue);
      });
    });
    return Object.entries(imgScore)
                 .map(([idx,score])=>({idx:+idx,score}))
                 .sort((a,b)=>b.score-a.score);         // worst first
  }
  
  function renderImageGrid(arr, srcLookup){
    const grid = document.getElementById("imgGrid");
    grid.style.display="grid";
    grid.style.gridTemplateColumns="repeat(10, auto)";
    grid.style.gap="4px";
    arr.forEach(({idx},i)=>{
      const img = new Image();
      img.width=28; img.height=28;
      img.src = srcLookup(idx);     // user-provided function/path
      img.title = `#${i} idx:${idx}`;
      grid.appendChild(img);
    });
  }
  