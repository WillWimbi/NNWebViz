// //float32 mantissa etc?
// //little-endian??
// //ES-6 is what? History of JS?
// //EXPLAIN more about proxy.
// //dataloader


// function getDataFromServer(how to get from server?){
//     //convert native image size to 32x32=1024 long array (or 28x28=784) of values, float32 (can we do float16? Prob not)
//     //find local image size, e.g how many bytes/bits per pixel brightness value? 3, 6, 9 etc? Then
//     //divide total amount of bytes the img is stored as by the amount of bytes/bits per pixel brightness value
//     //so if we used a single byte per pixel value, thats 1024B per 32x32 img, and 1024B/1B = 1024. If they
//     //were somehow int16 values and we could find that out, we'd calculate 1024*2B = 2048B.Then we know
//     //to iterate by 2Bytes for each pixel value when creating our own ArrayBuffer. So
//     //since we're using Float32, we do our own Byte amount, 4, divided by 2, and get 2, so we initilize a 
//     //4096 Byte ArrayBuffer, and do new Float32Array(4096) to get our float32 array, then assign it likewise.
//     //we then instantiate our ArrayBuffer with our locally decided float representation amount,
//     //in this case probably 32, meaning unfortunately we can't use smaller representations 
//     //as this seems somewhat unneccessarily large but that's what we're working with here.
//     //So we'd have 1024
//     //LOAD INTO ArrayBuffer(img.dataSize)


//Might use proxies to check when a certain tensor object's data was changed, and log that to a compute graph.
//Whereas currently I have this updated manually during each tensor op.

//Implementation:
//DOM manipulation.

// const neuronDiv = document.createElement('div');
// neuronDiv.className = 'neuron';  // assign a CSS class
// neuronDiv.textContent = '0.42';  // perhaps display the value as text
// document.body.appendChild(neuronDiv);
//document.getElementById, getElementsByClassName(), etc, query
//element.textContent, element.style.width, element.setAttribute('data-val', 0.42)
//Look deeper into: MUTATIONObserver:
//Use case example: If The Will decided to let React render the initial UI structure (like placeholders for layers), but then uses low-level code to fill in details, a MutationObserver could notify our code once React has finished adding those placeholder elements, so we can then populate them or attach events. Another use: if an element we’re visualizing gets removed (maybe the user closed a panel), a MutationObserver can catch that removal, and then we know to stop some corresponding process (like stop feeding data to that visualization). Be mindful that MutationObserver callbacks are asynchronous and batch multiple mutations together before calling your callback. This is good for efficiency (you won’t get spammed for every single minor change one by one in most cases; you’ll get a batch after they’ve happened). But it means in the callback, the DOM is already in its new state.
//IF its fucking in a new state, why would we use it/override it as discussed above???^^^^
//will use chart+grids for vizzing. Char.js, <cavnas>, WebGL...
//Explain requestAnimation frame.


// we can use XHR or fetch to send saved model params to the server. It will be our 'leaderboard' that shows
// best performance on mnist dataset, preventing users from training the model on a single sample to 'game'
// best loss scores, hence making it CRUD-like as a requirement for this assignment.
// if I set something like 'cannot send best score to leaderboard if training set was not like so:'
// can the users 'change' that raw javascript that runs in their browser? Fair enough. I guess I would have to check:
// was the default info i send to the server side ABOUT their training method true - as in - did they really achieve it
// with default settings, no 'one sample' train-hacking?
//Using fetch to load data: If the model is hosted on a server, you might skip FileReader and just use fetch(). For example: let resp = await fetch('mymodel.bin'); let arrayBuffer = await resp.arrayBuffer(); gives you an ArrayBuffer​stackoverflow.com. That’s an alternative path (with the advantage of not requiring user action if same-origin or proper CORS). But our focus is on the FileReader for direct user-provided files.
//Now The Will can incorporate file inputs and outputs: perhaps allowing a student to download the state of the network after each experiment, or load a configuration that sets up a particular network scenario. These file operations ensure the tool isn’t a closed box; it interacts with external data, which is a key part of any practical system. (Having dealt with external data, we turn our attention inward again – how to inspect and optimize the running system. The next part is all about using console and performance tools to profile and debug our real-time app.)
//allow user to download net params.

//This pattern drives a continuous loop that tries to update each frame. The work in computeNextState must be small enough to fit in one frame time. If it’s not, we revert to chunking or moving to a worker as discussed. Batching network calls: If the visualization pulls data from a server periodically (say to get new input data or to send telemetry), we should use async fetch calls and possibly debounce or throttle them – meaning not to bombard the server or overlap too many requests. Using Promise.all we can do parallel fetches if needed and wait for all, which is often faster than doing them one by one with awaits. For instance, to load multiple files:
//const [file1, file2] = await Promise.all([fetch(url1).then(r=>r.json()), fetch(url2).then(r=>r.json())]);
//If trainOneEpoch() is synchronous and heavy, the UI update will only happen after it finishes. The user might see the loss plot jump in steps and the page frozen during training. Instead, making trainOneEpoch() an async function that yields periodically or even just doing:
//for(let epoch=0; epoch<10; epoch++){
//     await trainOneEpoch();   // internally yields control periodically
//     updateUI();              // quick DOM update
//     await new Promise(r => setTimeout(r, 0)); // small yield after UI update too
//   }
//This ensures after each epoch’s work (or within it), the loop yields back to event loop, allowing the DOM update to render and user interactions to be processed (maybe a stop button?). Using requestAnimationFrame: If we aim for a smooth 60fps visualization (like an ongoing animation of network activity), we might not want to use setTimeout with a fixed delay, but rather rAF which syncs to actual frame rate. For example:
//for(let i=0;i<1000000;i++){
//     await doSomethingQuick();
// }
//If doSomethingQuick() resolves immediately (like it’s an async function that doesn’t actually await anything), you might end up blocking anyway because each iteration goes microtask -> next iteration quickly. To yield, you might still need an explicit await new Promise(r=> setTimeout(r,0)) occasionally. In other words, just because a function is async doesn’t mean it yields control unless it hits an await of a pending promise.
//Will use await and promise to fetch our mnist training data from our server.
/*I think we can just use this huge image google allows us to get which stores all images in it, its a 10MB image. 

But I want to be able to 'discretize' the image into files upon reaching the user. 
Do you recall all my notes and commented notes about how we'll get the worst performing images each round?
alongside potentially evaluating for 'most similar image' via condensed encodings of the images? This would require saving all the images locally. 

maybe  we'll have one huge array on client side and simply save indexes. Sound like a good plan? We need to be able to reliably get the images - but once the images are all loaded the array itself should be unchangeable, only alterable if the client were to want to get new images or something. 
We'd have parallel arrays:
one direct image array (784 elements per image, since 28x28)
label array, 1 per image
maybe an 'encoding' array, either done via latent vector encodings or perhaps a numerical method I can create, a kind of high dimensional vector, and we can 'find' the most similar number to a number we observe in our browser file by simply left clicking on a little tag that we'll show on one of its corners, which will give us some image options, like find most similar image, etc. 

We will use mongoDB for storing and loading neural net data later, for leaderboards and viewing pretained nets. */
activationNames = ["relu","sigmoid","tanh","dropout"];

const ctx = canvas.getContext("2d");
ctx.fillStyle="red";
ctx.fillRect(100,100,25,25)
function dataLoader(){
    

}

const filePath = String.raw`C:\Users\willw\Downloads\MNIST Dataset JPG format\MNIST - JPG - training`;

fetch('https://example.com/image.jpg').then(function(response) {
    if (response.ok) {
      console.log('Request succeeded!');
    } else {
      console.log('Request failed!');
    }
  });


//for loading model layers from the lego building blocks, I have an idea...
//We can simply have the dictionairy/list values equal precisely the order model.add expects them in, setting default values for most of them.



function parseInputIntoModel(){
    let currentModelInput = document.getElementById("modelBuilderInputDropdown");
    let modelArr = [];
    for(item in currentModelInput){
        modelArr.push({layerName:item.layerName,layerSpecs:item.layerSpecs});
    }
    

}


function modelBuilder(modelArr, activationNames){
    for(let i=0;i<modelArr.length;){
        
        if(modelArr[i+1] in activationNames){
            if(modelArr[i]=="conv2d"){
                model.add(tf.layers.conv2d(modelArr[i][layerSpecs].value))
            }
            
            

        }
    
    }
    


}

// Your TensorFlow.js code here
function mnistModel() {
    const model = tf.sequential();
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;  
    
    // In the first layer of our convolutional neural network we have 
    // to specify the input shape. Then we specify some parameters for 
    // the convolution operation that takes place in this layer.
    //Adding RELU effectively gives what we'd typically think of as another layer right 'below' this initial conv2d layer..
    model.add(tf.layers.conv2d({inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],kernelSize: 5,filters: 8,strides: 1,activation:'relu',kernelInitializer: 'varianceScaling'}));
    
    // The MaxPooling layer acts as a sort of downsampling using max values
    // in a region instead of averaging.  
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    
    // Repeat another conv2d + maxPooling stack. 
    // Note that we have more filters in the convolution.
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'}));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    
    // Now we flatten the output from the 2D filters into a 1D vector to prepare
    // it for input into our last layer. This is common practice when feeding
    // higher dimensional data to a final classification output layer.
    model.add(tf.layers.flatten());
    
    // Our last layer is a dense layer which has 10 output units, one for each
    // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
    const NUM_OUTPUT_CLASSES = 10;
    model.add(tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'
    }));
    
    // Choose an optimizer, loss function and accuracy metric,
    // then compile and return the model
    const optimizer = tf.train.sgd();
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });
    
    return model;
    }

function showGrads(layer,htmlEl){
    let r = document.getElementById(htmlEl);
    r.content = layer;

}


// Run training when the page loads
window.onload = mnistModel();