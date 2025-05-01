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

fetch('/api/runs')
  .then(r => {
    if (!r.ok) throw new Error(`API request failed: ${r.status} ${r.statusText || ''}`);
    return r.json();
  })
  .then(runs => {
    console.log(`Loaded ${runs.length} run datasets`);
    visualizeRuns(runs);
  })
  .catch(e => {
    console.error('API /api/runs failed:', e);
    const errorDiv = `<div style="color:red; padding:20px; text-align:center; border: 1px solid red; background-color: #fee; margin: 20px auto; width: 80%;">Error loading data: ${e.message}. Please check the console and ensure the API is running correctly.</div>`;
    document.getElementById("metricsChartContainer").innerHTML = errorDiv;
    document.getElementById("imgGridContainer").style.display = 'none';
  })
  .finally(() => {
      const loadingIndicator = document.getElementById("loading-indicator");
      if (loadingIndicator) loadingIndicator.remove();
  });
// Constants for MNIST data
const MNIST_IMAGES_SPRITE_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';
const SPRITE_SIZE = 28;
const IMAGE_SIZE = SPRITE_SIZE * SPRITE_SIZE;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;
const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

// Global variable to hold the loaded dataset
let mnistDataset = null;

// Function to load and process MNIST data
async function loadMnistData() {
    if (mnistDataset) {
        console.log("MNIST data already loaded.");
        return mnistDataset;
    }
    console.log("Loading MNIST data...");

    // 1. Load the sprite image
    const img = new Image();
    const imgPromise = new Promise((resolve, reject) => {
        img.crossOrigin = 'anonymous';
        img.onload = () => {
            console.log("MNIST sprite image loaded successfully.");
            img.width = img.naturalWidth;
            img.height = img.naturalHeight;
            resolve();
        };
        img.onerror = reject;
        img.src = MNIST_IMAGES_SPRITE_PATH;
    });

    // 2. Load the labels (optional for visualization)
    const labelsRequest = fetch(MNIST_LABELS_PATH);

    // Wait for both image and labels to load
    try {
        const [, labelsResponse] = await Promise.all([imgPromise, labelsRequest]);
        const datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());
        console.log("MNIST labels loaded successfully.");

        // 3. Process the image sprite into pixel data
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const datasetBytesBuffer = new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);

        // Process in chunks to avoid memory issues
        const chunkSize = 5000;
        canvas.width = img.width;
        canvas.height = chunkSize * SPRITE_SIZE / (img.width / SPRITE_SIZE);

        console.log("Processing sprite image into pixel data...");
        for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
            const datasetBytesView = new Float32Array(
                datasetBytesBuffer,
                i * IMAGE_SIZE * chunkSize * 4,
                IMAGE_SIZE * chunkSize
            );

            const sourceY = i * chunkSize * SPRITE_SIZE / (img.width / SPRITE_SIZE);

            ctx.drawImage(
                img,
                0, sourceY,
                img.width, canvas.height,
                0, 0,
                img.width, canvas.height
            );

            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

            for (let j = 0; j < imageData.data.length / 4; j++) {
                datasetBytesView[j] = imageData.data[j * 4] / 255;
            }
            console.log(`Processed chunk ${i + 1}/${Math.ceil(NUM_DATASET_ELEMENTS / chunkSize)}`);
        }

        const datasetImages = new Float32Array(datasetBytesBuffer);
        console.log("Pixel data processing complete.");

        // Store the loaded data
        mnistDataset = {
            images: datasetImages,
            labels: datasetLabels,
            testImages: datasetImages.slice(NUM_TRAIN_ELEMENTS * IMAGE_SIZE),
            testLabels: datasetLabels.slice(NUM_TRAIN_ELEMENTS * NUM_CLASSES)
        };

        return mnistDataset;
    } catch (error) {
        console.error("Error loading MNIST data:", error);
        throw error;
    }
}

// Function to get pixel data for a specific image index
function getPixelDataForIndex(idx, isTestIdx = false) {
    if (!mnistDataset || !mnistDataset.images) {
        throw new Error("MNIST data not loaded yet. Call loadMnistData first.");
    }
    // If we're using a test set index, add the training set offset (55,000)
    const globalIdx = isTestIdx ? idx + NUM_TRAIN_ELEMENTS : idx;
    const offset = globalIdx * IMAGE_SIZE;
    return mnistDataset.images.slice(offset, offset + IMAGE_SIZE);
}

// Function to decode one-hot vectors on demand
function decodeLabel(arr, start) {
    for (let c = 0; c < NUM_CLASSES; c++) {
        if (arr[start + c]) return c;
    }
    return null;          // should never hit
}

// Function to get the correct label for an index
function getLabelForIndex(idx, isTest = false) {
    if (!mnistDataset) throw new Error('loadMnistData first');
    const base = isTest ? mnistDataset.testLabels : mnistDataset.labels;
    return decodeLabel(base, idx * NUM_CLASSES);
}

// Function to draw pixel data onto a canvas and return its data URL
function drawPixelsToDataURL(pixelData, width = SPRITE_SIZE, height = SPRITE_SIZE) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(width, height);

    for (let i = 0; i < pixelData.length; i++) {
        const pixelValue = Math.round(pixelData[i] * 255);
        imageData.data[i * 4 + 0] = pixelValue; // Red
        imageData.data[i * 4 + 1] = pixelValue; // Green
        imageData.data[i * 4 + 2] = pixelValue; // Blue
        imageData.data[i * 4 + 3] = 255;        // Alpha
    }

    ctx.putImageData(imageData, 0, 0);
    return canvas.toDataURL('image/png');
}

// Main visualization function
export async function visualizeRuns(fullData) {
    try {
        await loadMnistData();

        if (!fullData?.length) {
            document.getElementById("metricsChart").innerHTML =
                '<div style="color:red;text-align:center;padding:20px;border:1px solid #f88;background:#fee">Error: No valid run data received</div>';
            return;
        }

        createChart(fullData);
        await createImageGrid(fullData, async (idx) => {
            // Pass true to indicate this is a test set index
            const pixelData = getPixelDataForIndex(idx, true);
            return drawPixelsToDataURL(pixelData);
        });

    } catch (error) {
        console.error("Error during visualization:", error);
        document.getElementById("metricsChart").innerHTML =
            `<div style="color:red;text-align:center;padding:20px;">Error visualizing data: ${error.message}</div>`;
    }
}

// Create performance chart
function createChart(runs) {
    const ctx = document.getElementById("metricsChart").getContext("2d");
    const epochs = runs[0].epochs.length;
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const colors = ['#FF6384', '#FF9F40', '#36A2EB', '#4BC0C0'];
    const yAxes = ['y-loss', 'y-loss', 'y-acc', 'y-acc'];

    // Calculate medians for each metric and epoch
    const medians = {};
    metrics.forEach(m => {
        medians[m] = Array(epochs).fill().map((_, e) => {
            const values = runs.map(r => r.epochs[e]?.[m]).filter(v => v !== undefined && v !== null);
            return median(values);
        });
    });

    // Create chart
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({ length: epochs }, (_, i) => `Epoch ${i + 1}`),
            datasets: metrics.map((m, i) => ({
                label: m.replace('val_', 'Validation ').replace(/^[a-z]/, char => char.toUpperCase()),
                data: medians[m],
                borderColor: colors[i],
                borderWidth: 2,
                fill: false,
                yAxisID: yAxes[i],
                tension: 0.1
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: 'Median Model Performance Across Runs', font: { size: 16, weight: 'bold' } },
                tooltip: { mode: 'index', intersect: false },
                legend: { position: 'top' }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Epoch' }
                },
                'y-loss': {
                    type: 'linear', position: 'left',
                    title: { display: true, text: 'Loss' },
                    grid: { color: 'rgba(0,0,0,0.05)' },
                    beginAtZero: true
                },
                'y-acc': {
                    type: 'linear', position: 'right', min: 0, max: 1,
                    title: { display: true, text: 'Accuracy' },
                    grid: { drawOnChartArea: false }
                }
            }
        }
    });
}

// Create grid of challenging images
async function createImageGrid(runs, getImageURL) {
    // Calculate difficulty scores
    const scores = {};
    const summedLogits = {}; // Store summed logits for each image
    const sampleCounts = {}; // Track how many runs include each image
    
    // Process each run's validation data
    runs.forEach(run => {
        if (run.validationImagesTest && Array.isArray(run.validationImagesTest)) {
            run.validationImagesTest.forEach(({ idx, label, probs }) => {
                if (Array.isArray(probs) && typeof label === 'number' && label >= 0 && label < probs.length) {
                    // Track difficulty score
                    scores[idx] = (scores[idx] || 0) + (1 - probs[label]);
                    
                    // Sum logits across all runs for this image
                    if (!summedLogits[idx]) {
                        summedLogits[idx] = Array(10).fill(0); // 10 classes for MNIST
                    }
                    
                    // Count samples
                    sampleCounts[idx] = (sampleCounts[idx] || 0) + 1;
                    
                    // Add this run's probabilities to the sum
                    for (let i = 0; i < probs.length; i++) {
                        summedLogits[idx][i] += probs[i];
                    }
                } else {
                    console.warn(`Skipping invalid image data for index ${idx}: label=${label}, probs=${probs}`);
                }
            });
        } else {
            console.warn("Missing or invalid validationImagesTest in run data:", run);
        }
    });

    // Get top 100 challenging images
    const images = Object.entries(scores)
        .map(([idx, score]) => ({ idx: parseInt(idx), score }))
        .sort((a, b) => b.score - a.score)
        .slice(0, 100);

    // Optional: Use the hard-coded challenge set instead
    // const CHALLENGE_SET = [1824, 409, 4506, 4012, 3657, 2286, 1679, 8935, 1424, 9674, 6912, 520, 488, 1535, 3582, 3811, 8279, 9863, 434, 9195, 3257, 8928, 6873, 3611, 7359, 9654, 4557, 106, 2615, 6924, 5574, 4552, 2547, 3527, 5514, 1674, 1519, 6224, 1584, 5881, 5635, 9891, 4333, 711, 7527, 8785, 2045, 6201, 1291, 9044, 4803, 5925, 9459, 3150, 1139, 750, 3733, 4741, 1307, 3814, 1654, 6227, 4554, 7428, 5977, 2664, 6065, 5820, 3432, 4374, 1169, 9980, 2803, 8751, 4010, 2677, 7573, 6216, 4422, 9125, 3598, 5313, 916, 3752, 525, 5168, 6572, 4386, 1084, 3456, 9292, 5155, 3483, 8179, 6482, 7517, 2340, 4339, 2287, 4040];
    // const images = CHALLENGE_SET.map(idx => ({ idx, score: 0 }));

    if (images.length === 0) {
        console.warn("No challenging images found to display.");
        document.getElementById('imgGrid').innerHTML = '<p>No challenging image data available.</p>';
        return;
    }

    // Create grid
    const grid = document.getElementById('imgGrid');
    grid.innerHTML = '';

    // Add heading to parent container
    const gridContainer = document.getElementById('imgGridContainer');
    const existingHeading = gridContainer.querySelector('h3');
    if (!existingHeading) {
        const heading = document.createElement('h3');
        heading.textContent = 'Top 100 Most Challenging Images';
        gridContainer.insertBefore(heading, grid);
    }

    // Add images to grid with logit bars
    for (let i = 0; i < images.length; i++) {
        const container = document.createElement('div');
        container.className = 'digit-container';

        // Create image 
        const imgWrapper = document.createElement('div');
        imgWrapper.className = 'img-wrapper';
        
        const img = new Image();
        img.width = SPRITE_SIZE;
        img.height = SPRITE_SIZE;
        img.style.display = 'block';
        img.style.margin = '0 auto';
        
        try {
            img.src = await getImageURL(images[i].idx);
        } catch (error) {
            console.error(`Failed to generate image for index ${images[i].idx}:`, error);
            img.alt = `Error ${images[i].idx}`;
        }

        // Get true label from MNIST dataset
        const trueLabel = getLabelForIndex(images[i].idx, true);
        
        // Get predicted label (argmax from first run)
        const scoreObj = images[i];
        const predLabel = (() => {
            const probsEntry = runs[0].validationImagesTest
                .find(x => x.idx === scoreObj.idx);
            return probsEntry ? probsEntry.probs.indexOf(Math.max(...probsEntry.probs))
                            : '?';
        })();

        // Add label overlay
        const labelOverlay = document.createElement('div');
        labelOverlay.className = 'label-overlay';
        labelOverlay.textContent = trueLabel !== undefined ? trueLabel : '?';
        
        imgWrapper.appendChild(img);
        imgWrapper.appendChild(labelOverlay);
        container.appendChild(imgWrapper);

        // Get summed logits and calculate averages
        const logits = summedLogits[scoreObj.idx] || Array(10).fill(0);
        const sampleCount = sampleCounts[scoreObj.idx] || 1;
        const avgLogits = logits.map(sum => sum / sampleCount);
        
        // Find top 3 predictions (indices and values)
        const topPreds = avgLogits
            .map((val, idx) => ({ val, idx }))
            .sort((a, b) => b.val - a.val)
            .slice(0, 3);
        
        // Create logit bars container
        const barsContainer = document.createElement('div');
        barsContainer.className = 'logit-bars';
        
        // Add top 3 prediction bars
        topPreds.forEach(pred => {
            const barWrapper = document.createElement('div');
            barWrapper.className = 'bar-wrapper';
            
            const label = document.createElement('div');
            label.className = 'bar-label';
            label.textContent = pred.idx;
            
            const barContainer = document.createElement('div');
            barContainer.className = 'bar-container';
            
            const bar = document.createElement('div');
            bar.className = 'bar';
            bar.style.width = `${Math.min(100, pred.val * 100)}%`;
            
            // Highlight correct prediction
            if (pred.idx === trueLabel) {
                bar.classList.add('correct');
            }
            
            const percent = document.createElement('span');
            percent.className = 'bar-percent';
            percent.textContent = `${(pred.val * 100).toFixed(0)}%`;
            
            barContainer.appendChild(bar);
            barContainer.appendChild(percent);
            
            barWrapper.appendChild(label);
            barWrapper.appendChild(barContainer);
            barsContainer.appendChild(barWrapper);
        });
        
        container.appendChild(barsContainer);

        // Create comprehensive label with true and predicted values
        const labelBox = document.createElement('div');
        labelBox.className = 'digit-label';
        labelBox.textContent = `#${i + 1}  idx=${scoreObj.idx}  true→${trueLabel}  pred→${predLabel}`;
        labelBox.style.color = (trueLabel === predLabel ? '#090' : '#c00');  // Green for correct, red for wrong
        container.appendChild(labelBox);
        
        grid.appendChild(container);
    }
}

// Calculate median of array
function median(arr) {
    if (!arr || !arr.length) return null;
    const sorted = arr.filter(v => typeof v === 'number' && !isNaN(v)).sort((a, b) => a - b);
    if (!sorted.length) return null;
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}
// await loadMnistData();                 // waits for sprite + labels
// console.log('3811 →', getLabelForIndex(3811, false));   // should log 3
// console.log('test idx 311 →', getLabelForIndex(311, true)); // should log 3



// function makeChart(){

// }