/* ========================================================================== *
   neuralNetMnist.js  —  spec‑compliant autodiff & NN micro‑framework
   ========================================================================== */

/* ----------------------- helpers ---------------------------------------- */

let dynamicFuncGraph = [];


class Node {
  constructor(funcType,leafA,leafB,child){
    this.funcType = funcType;
    if(funcType!== "transpose" && funcType!== "relu"){
      this.parentA = leafA;
      this.parentB = leafB;
    }
    else {this.parentA = leafA}
    this.child = child;

  }

}

//THIS LIB ASSUMES SUB ARRAYS ARE EQUAL IN SIZE (e.g no tensor of 2 inner arrays where one is 4 and the other is 5 elements)
//Don't want to put in even more performance reducing checks for this, and no one creates tensors manually most of the time.
//Ah fuck it we'll just do it anyway
const randn = () =>
  Math.sqrt(-2 * Math.log(Math.random())) * Math.cos(2 * Math.PI * Math.random());

// Recursively infer the shape of a nested array structure
// For example: [[1,2],[3,4]] -> [2,2]
// This is a neat trick using recursion to traverse the nested structure
// and build up the shape array dimension by dimension.
// We check if x is an array - if yes, we record its length as the current dimension
// and recurse into the first element to get remaining dimensions.
// When we hit a non-array (base case), we return an empty array to terminate.
const inferShape = x => (Array.isArray(x) ? [x.length, ...inferShape(x[0])] : []);
//CHANGE: should prob put into tensor class^^^^^

// Creates a nested array of zeros or any other func val with the specified shape
// This is a beautiful recursive function that builds arrays of arbitrary dimensions
// For example: zeros([2,3]) creates [[0,0,0],[0,0,0]]
// How it works:
// 1. Base case: If shape is empty (shape.length === 0), return a single 0
// 2. Recursive case: Create an array of length shape[0], and fill each element
//    by recursively calling zeros() with the remaining dimensions (shape.slice(1))
// This elegantly handles any number of dimensions through recursion
const fill = (shape, gen) =>
  shape.length === 0
    ? gen()                                         // evaluate *per element*
    : Array.from({ length: shape[0] },
        () => fill(shape.slice(1), gen));

const zeros = shape => fill(shape, () => 0);

//^^^CHANGE: should prob put into tensor class

/* broadcast‑aware element‑wise mapping */
//If they're both arrays
//EXAMPLE: Lets imagine a and b are 2dim tensors: first a.map goes to one row with b[i] going to its own row. 
//We of course pass in the func to first map2 recursive call.
//Repeats, we see they're both still arrs, but no more subArrs.
//this time, v is a single element val, and arrA and arrB are both false.
//so we return fn(a,b) which is fn(v,b[i]), which is elementwise fn application.
//EXAMPLE2: Let's imagine we have a 3dim tensor 3,2,4 and a 2dim tensor 2,4.
//We first see that they're both arrays, so we call map2(a[0],b[0],fn) --> a[0]=3,2 and b[0]=4 element arr
//then we see that a[0] is an array, so we call map2(a[0][0],b[0][0],fn) --> a[0][0]=3 and b[0][0]=single element val for second iteration
//on second iteration since b[0][0] is a single element val, we perform fn for each a element with b[0][0].
//Hence broadcasting is possible

//delete I think this is old
const padShape = (shape, len) =>
  Array(len - shape.length).fill(1).concat(shape);

const padArray = (arr, targetRank) => {
  let out = arr;
  let rank = inferShape(arr).length;          // e.g. [2,3] -> 2
  while (rank < targetRank) {                 // add one axis per step
    out = [out];
    rank++;
  }
  return out;                                 // new wrapper(s), same leaves
};

// Converts a flat index to a multi-dimensional index
// For example, if shape is [2,3] and k is 4, returns [1,1] (second row, second column)
function unravel(k, shape) {
  const idx = new Array(shape.length);
  // Work backwards through dimensions, computing each index component
  for (let i = shape.length - 1; i >= 0; i--) {
    idx[i] = k % shape[i];        // Get the remainder for this dimension
    k = Math.floor(k / shape[i]); // Integer divide to move to next dimension
  }
  return idx;
}

// Example: Let's trace through unravel(4, [2,3]) step by step
// Initial values:
// k = 4
// shape = [2,3]
// idx = new Array(2) = [undefined, undefined]

// First loop iteration (i = 1):
// idx[1] = 4 % 3 = 1       // Second column (0-indexed)
// k = Math.floor(4 / 3) = 1
// idx = [undefined, 1]

// Second loop iteration (i = 0):
// idx[0] = 1 % 2 = 1       // Second row (0-indexed)
// k = Math.floor(1 / 2) = 0
// idx = [1, 1]

// Return idx = [1, 1]
// This means the flat index 4 corresponds to position [1,1] in a 2×3 array
// In a 2×3 array, the positions are:
// [0,0] = 0, [0,1] = 1, [0,2] = 2
// [1,0] = 3, [1,1] = 4, [1,2] = 5

//Helpers:
  /* ----------  SMALL HELPERS ---------- */
// Throws a helpful error message when matrix dimensions don't match for multiplication
function throwErr() {
  throw new Error(`Inner dim mismatch: [${SA}] x [${SB}]`);
}
//[[1,2]] is treated as shape 1,2 in js.
const checkDimsCompatible = (a,b,aShape,bShape,aShapeLength,bShapeLength,minShapeLength) => {
  let truey = true;
  let incompatDim = "";
  for(let i = 1; i < minShapeLength+1; i++){
    if((aShape[aShapeLength-i] !== bShape[bShapeLength-i]) && (aShape[aShapeLength-i] !== 1 && bShape[bShapeLength-i] !== 1)){
      truey = false;
      incompatDim = `aShape: ${aShape}, bShape: ${bShape}`;
      break; // This stops the current loop, not the entire function

    }
  }
  return [truey, incompatDim];
}

const mapBroadcast = (a, b, fn) => {
  const isArrA = Array.isArray(a), isArrB = Array.isArray(b);

  // both arrays ⇒ recurse over the *max* length; modulo‑index to broadcast 1‑length
  if (isArrA && isArrB) {
    const len = Math.max(a.length, b.length);
    return Array.from({length: len}, (_, i) =>
      mapBroadcast(
        a[i % a.length],           // repeats if a.length === 1
        b[i % b.length],           // repeats if b.length === 1
        fn));
  }

  // one array, one scalar ⇒ map scalar across array
  if (isArrA) return a.map(v => mapBroadcast(v, b, fn));
  if (isArrB) return b.map(v => mapBroadcast(a, v, fn));

  // both scalars ⇒ apply the op
  return fn(a, b);
};

//However this in theory would work for unequal dimensionns....
//CHANGE BELOW: MUST not work for unequal dimensions. Also, if this is being used later on, for Add() funcs, 
const map2 = (a, b, fn) => {
  const isArrA = Array.isArray(a), isArrB = Array.isArray(b);
  if(!isArrA || !isArrB) throw Error("map2: at least one input must be an array");
  const aShape = inferShape(a);
  const bShape = inferShape(b);
  const aShapeLength = aShape.length;
  const bShapeLength = bShape.length;
  maxShapeLength = Math.max(aShape.length,bShape.length);
  minShapeLength = Math.min(aShape.length,bShape.length);
  // diff = aShape.length - bShape.length;
  const [truey, incompatDims] = checkDimsCompatible(a,b,aShape,bShape,aShapeLength,bShapeLength,minShapeLength);
  if(!truey) throw Error(`map2: incompatible dimensions: ${incompatDims}`);
  // returnBroadCastedTensors(a,b,diff);
  let aPadded = padArray(a,maxShapeLength);
  let bPadded = padArray(b,maxShapeLength);
  console.log('aPadded:', aPadded);
  console.log('bPadded:', bPadded);
  // both arrays ⇒ recurse over the *max* length; modulo‑index to broadcast 1‑length
  return mapBroadcast(aPadded,bPadded,fn);
}
const addArrays = (a, b) => map2(a, b, (x, y) => x + y);
//^^^^ applying recursive func applier to addition. 


console.log("sumAlongDim:",sumAlongDim([[[1,2],[3,4],[[3,6],[2,1]]]],0));





// This function extracts a specific slice from nested arrays using indices
// For example, if data is [[[1,2],[3,4]],[[5,6],[7,8]]] and idx is [0,1]
// it will return [3,4] (first outer array, second inner array)
function recurSlice(data, idx) {
  let cur = data;  // Start with the full data structure
  let d = 0;       // Track which dimension we're currently indexing
  // Navigate through the nested structure one index at a time
  while (d < idx.length) { 
    cur = cur[idx[d]];  // Move to the next level of nesting
    d++;                // Move to the next dimension
  }
  return cur;  // Return the extracted slice
}


// Adjusts indices for broadcasting
// If a dimension is 1, it's broadcasted by using index 0 repeatedly
// Otherwise, use the actual index from the batch dimension
function broadcastIdx(idx, targetShape, outBatch) {
  // Calculate offset due to different batch dimension lengths
  const off = outBatch.length - targetShape.length;
  return targetShape.map((dim, i) =>
    // If dimension is 1, use index 0 (broadcast), otherwise use actual index
    dim === 1 ? 0 : idx[off + i]
  );
}
//So if idx=[0,0] and targetShape=[3,7] then off = 2-2 = 0.
//first iter: So we return [3,7].map(dim, i) => dim === 1 ? 0 : idx[off + i]
//which is [3,7].map(dim, i) => dim === 1 ? 0 : [0,0][0] = 0.
//So we return 0 for first element and 0 for second element (since idx[1] = 0).
//So we return [0,0]

//another example: idx=[2,4] and targetShape=[3,7] then off = 2-2 = 0.
//first iter: we return [3,7].map(dim, i) => dim === 1 ? 0 : idx[off + i]
//which is [3,7].map(dim, i) => dim === 1 ? 0 : [2,4][0] = 2.
//second iter: we do [3,7].map(dim, i) => dim === 1 ? 0 : [2,4][1] = 4.
//So we return [2,4]
//^^^^^that function works well if one tensor has a 1dim and the other has a >1dim.
//By the time we're at this point in the code, we're assuming the dims are EITHER 1 or equal to the other tensor's dim.


const transposeF = (tensor) => {
  let x = tensor.data;
  let shapeN = tensor.shape;
  let shapeLength = shapeN.length;
  let out = [];

  if (shapeLength > 2) {
    // recurse on each leading‑dim slice, drop the first dim’s size
    out = x.map(v => transposeF(v, shapeN.slice(1), shapeLength - 1));
  } else if (shapeLength === 2) {
    // allocate target [cols][rows] container
    out = Array.from({ length: shapeN[1] }, () => Array(shapeN[0]));
    for (let i = 0; i < shapeN[0]; i++) {
      for (let j = 0; j < shapeN[1]; j++) {
        out[j][i] = x[i][j];        // write transposed element
      }
    }
  }

  return new Tensor(out, {requiresGrad: tensor.requiresGrad});
};
//naively just calculates gradient assuming they're the same length. 
//won't error tho since map2 pads everything and handles scalars fine.
// Calculate gradient for a tensor that might have been broadcasted
function calcGrad(tensor, otherTensor, outputTensor, gradFunc) {
  // Apply the gradient function to compute raw gradients
  let rawGrad = map2(outputTensor.grad, otherTensor.data, gradFunc);
  console.log("rawGrad:",rawGrad);
  // If shapes match, no broadcasting occurred for this tensor
  const rawGradShape = inferShape(rawGrad);
  const tensorShape = tensor.shape;
  
  // If the shapes are the same, no reduction needed, simply checks dim equality
  if (rawGradShape.length === tensorShape.length && 
      rawGradShape.every((dim, i) => dim === tensorShape[i])) {
    return rawGrad;
  }
  
  // Need to reduce along the broadcasted dimensions
  return reduceAlongBroadcastedDims(rawGrad, rawGradShape, tensorShape);
}

const flatten = arr =>
  Array.isArray(arr) ? arr.flatMap(flatten) : [arr];

// Helper function to reduce gradient along dimensions that were broadcasted
function reduceAlongBroadcastedDims(grad, gradShape, targetShape) {
  // If target is a scalar, sum everything
  if (targetShape.length === 0 || (targetShape.length === 1 && targetShape[0] === 1)) {
    return [flatten(grad).reduce((a, b) => a + b, 0)]; 
  }
  
  // For higher dimensions, recursively reduce
  let result = grad;
  
  // Start from the right (least significant dimensions)
  for (let i = gradShape.length - 1; i >= 0; i--) {
    // If this dimension was broadcasted (target has fewer dims or dim size is 1)
    const targetDim = i - (gradShape.length - targetShape.length);
    if (targetDim < 0 || (targetDim >= 0 && targetShape[targetDim] === 1)) {
      // Sum along this dimension
      result = sumAlongDim(result, i);
    }
  }
  
  return result;
}

function sumAlongDim(arr, dim) {
  // Handle base case: summing along the last dimension
  const shape = inferShape(arr);
  if (dim === shape.length - 1) {
    return arr.map(row => row.reduce((a, b) => a + b, 0));
  }
  
  // For dimensions other than the last one, we need to transpose or use recursion
  
  const result = [];
  
  // For inner dimensions, we need to recursively sum along the dimension
  if (dim === 0) {
    // For the first dimension, just sum all arrays at that level
    const firstElem = arr[0];
    const resultShape = shape.slice(1);  // Shape without first dimension
    let sum = zeros(resultShape);        // Initialize with zeros
    
    for (let i = 0; i < arr.length; i++) {
      sum = addArrays(sum, arr[i]);      // Add each "slice" to the sum
    }
    return sum;
  } else {
    // For middle dimensions, apply recursively to each slice
    return arr.map(slice => sumAlongDim(slice, dim - 1));
  }
}




// This is incorrect syntax. To create a new array with nested structure:
const newArray = [[2,4],[1,3]]; // Fixed syntax with proper comma between inner arrays
const newArray2 = [[[4,6,2,1],[1,2,3,4]],
[[9,8,1,1],[1,2,3,4]],
[[9,8,1,1],[1,2,3,4]]]; //3,2,4
const newArray3 = [[[4,5,6,7],[1,2,3,4]]]; //1,2,4
const newArray3Shape = inferShape(newArray3);
function isArrayRecursive(arr) {
  if (Array.isArray(arr)) {
    console.log(`${arr} is an array, here's the length: ${arr.length}`);
    arr.forEach(item => isArrayRecursive(item));
  } else {
    console.log(`${arr} is not an array`);
  }
}
// Creating arrays with specified dimensions
const nArr1 = [[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], 
               [[[13, 14], [15, 16], [17, 18]], [[19, 20], [21, 22], [23, 24]]], 
               [[[25, 26], [27, 28], [29, 30]], [[31, 32], [33, 34], [35, 36]]]]; // 3×2×3×2

const nArr2 = [[[41, 42], [43, 44], [45, 46]], 
               [[47, 48], [49, 50], [51, 52]]]; // 2×3×2

const nArr3 = [[[60], [61], [62]], 
               [[63], [64], [65]], 
               [[66], [67], [68]], 
               [[69], [70], [71]]]; // 4×3×1

const nArr4 = [[[80, 81], [82, 83], [84, 85]], 
               [[86, 87], [88, 89], [90, 91]], 
               [[92, 93], [94, 95], [96, 97]]]; // 3×3×2

// Verify shapes
console.log('nArr1 shape:', inferShape(nArr1)); // Should be [3, 2, 3, 2]
console.log('nArr2 shape:', inferShape(nArr2)); // Should be [2, 3, 2]
console.log('nArr3 shape:', inferShape(nArr3)); // Should be [4, 3, 1]
console.log('nArr4 shape:', inferShape(nArr4)); // Should be [3, 3, 2]

console.log('nArr1 + nArr2:', map2(nArr1, nArr2, (x, y) => x + y));


let padArr3=padShape(newArray3Shape,4);
console.log('newArray3 padded:',padArr3);
console.log('arr shape:',inferShape(newArray3));
console.log("Recursively checking newArray3:");
isArrayRecursive(newArray3);
const newArray4 = [[4,5,6,7],[1,2,3,4,4]]; //2,4
console.log("diff length",newArray.length)
// console.log('addArrs 3,2,4 and 2,4',addArrays(newArray2,newArray3));
console.log("dims compatible",checkDimsCompatible(newArray2,newArray3,inferShape(newArray2),inferShape(newArray3),inferShape(newArray2).length,inferShape(newArray3).length,Math.min(inferShape(newArray2).length,inferShape(newArray3).length)));
// Make each input to checkDimsCompatible printable
console.log("Input 1 to checkDimsCompatible:", newArray2);
console.log("Input 2 to checkDimsCompatible:", newArray3);
console.log("Input 3 to checkDimsCompatible (inferShape(newArray2)):", inferShape(newArray2));
console.log("Input 4 to checkDimsCompatible (inferShape(newArray3)):", inferShape(newArray3));
console.log("Input 5 to checkDimsCompatible (Math.min(...)):", Math.min(inferShape(newArray2).length, inferShape(newArray3).length));
console.log("Input 6 to checkDimsCompatible (inferShape(newArray2).length):", inferShape(newArray2).length);
console.log("Input 7 to checkDimsCompatible (inferShape(newArray3).length):", inferShape(newArray3).length);



/* ----------------------- global registries ----------------------------- */
const WEIGHTS = [];
let   LR = 0.01;

/* ----------------------- initialisers ---------------------------------- */
function initWeights(shape, scheme = "default") {
  const fanIn = shape.slice(-1)[0] || 1; // Get the size of the input dimension (last element in shape array)
  let scale;
  switch (scheme) {
    case "xavier": scale = Math.sqrt(1 / fanIn); break; // Xavier/Glorot initialization
    case "he":     scale = Math.sqrt(2 / fanIn); break; // He initialization for ReLU networks
    default:       scale = 0.01;                        // Default small scale
  }
  
  // The fill function recursively creates a nested array structure matching the given shape
  // and fills it with random values scaled appropriately
  // How it works:
  // 1. Base case: If s is empty (s.length === 0), we've reached the deepest level,
  //    so return a single random number scaled by our initialization factor
  // 2. Recursive case: Create an array of length s[0], and fill each element
  //    by recursively calling fill() with the remaining dimensions (s.slice(1))
  // This  handles any number of dimensions through recursion
  return fill(shape, () => randn() * scale);
  //map2 operates for 2 tensors (arrs of arrs...) so we can't use it here....
  //CHANGE: I wonder if we can merge the logic for this fill() and with zeros() to have a recursive func applier, with func
  //being logic provided here or simply returning a 0 at the base cap, with a map() like func.
}

/* ======================================================================= *
   Tensor with chooser‑routed ops
 * ======================================================================= */
//Big question: CHANGE: Should everything we use be a tensor? I originally planned on having everything
//have requiresGrad = true, but we probably want to initialize a Tensor object to store our grads, so that would be just a pure nested arr but also eats memory..
//hmmm....
class Tensor {
  constructor(data, { requiresGrad = false, name = "", hasWeights=false} = {}, opname=null,parents=null) {
    this.data = data;
    this.shape = inferShape(data);
    this.requiresGrad = requiresGrad;
    this.grad = requiresGrad ? zeros(this.shape) : null;
    this._parents = parents;
    this._backward = () => {}; //CHANGE: what? I don't yet understand
    this.name = name;
    if(hasWeights){
      if(hasWeights && !WEIGHTS.includes(this)) WEIGHTS.push(this);
    }
  }


  /* -------- element‑wise op factory -------- */
  static eltwise(a, b, f, dfa, dfb, opName, optChild=null) {
    //apply func, and give the output tensor
    
    if((dfa === null && dfb === null)){
      const out = new Tensor(map2(a.data, b.data, f),
                           { requiresGrad: a.requiresGrad || b.requiresGrad }, opName,[a, b]);
      const n = new Node(opName, a, b, out);
      dynamicFuncGraph.push(n);
      return out;
                          }

    else if(f===null && optChild!==null){
    
      
      
      if (a.requiresGrad) {
        a.grad = addArrays(a.grad , calcGrad(a,b,optChild, dfa));
        //a.grad = addArrays(a.grad, map2(optChild.grad, b.data, dfa));
      }
      if (b.requiresGrad) {
        b.grad = addArrays(b.grad , calcGrad(b,a,optChild, dfb));
        //b.grad = addArrays(b.grad, map2(optChild.grad, a.data, dfb));
      }
      // else if(a.shape!==Tensor.inferShape(a.grad)){
      //   //essentially the only option we have is to check and see if perhaps the initial initialized grad for this tensor was created is different than its actual shape.
      //   //e.g a might have been created as a scalar tensor of value [3], but through our default process have been given a grad array matching the values of
      //   //its b counterpart which is now something like [[7,10][8,9]]. Here is a red flag.
        

      // }
    
    }
  }
  //Call elementwise functions
  static addForward(a, b) { return Tensor.eltwise(a, b, (x,y)=>x+y, null, null, "add"); } //CHANGE: I'm confused.... what is g here? We're essentially passing it as a function 
  // to map2 yet I don't
  //know what it is yet.
  static subForward(a, b) { return Tensor.eltwise(a, b, (x,y)=>x-y, null, null, "sub"); }
  static mulForward(a, b) { return Tensor.eltwise(a, b, (x,y)=>x*y, null, null, "mul"); }
  static divForward(a, b) {
    return Tensor.eltwise(
      a, b, (x,y)=>x/y, null, null, "div"); //CHANGE:I don't understand this logic!
  }
  //other opnames will be "matmul","relu","conv2d", etc.
  /* ---------------- relu ------------------- */
  //v=>v>0?v:0 is a compact max(0,x) statement
  static reluForward(x) {
    const out = new Tensor(map2(x.data, x.data, v=>v>0?v:0),
                           { requiresGrad: x.requiresGrad });
    out._parents=[x];
    out._backward=()=>{
      if(!x.requiresGrad) return;
      const upd = map2(out.grad,x.data,(g,v)=>v>0?g:0); //just propagating gradient, upd=update func, add grads to previous parent x tensor
      x.grad = addArrays(x.grad, upd);
    };
    return out;
  }
  

  //CHANGE: DON'T GET DA LOGIC!!!
  /* ---------------- matmul (2‑D) ----------- */
  static matmulForward(a,b) {
  const A = a.data; 
  const B = b.data;
  const SA = a.shape;
  const RA = SA.length;
  const SB = b.shape;
  const RB = SB.length;

    /* ----------  INNER LOW‑RANK KERNELS  ---------- */
    function dot1d1d(a, b) {           // (n)·(n)->number
      let acc = 0; 
      for (let i = 0; i < a.length; i++) {
        acc += a[i] * b[i]; 
      }
      return acc;
    }
    function mat2d1d(M, v) {           // (m,n)x(n)->(m)
      const m = M.length;
      const n = M[0].length;
      const out = new Array(m).fill(0);
      
      for (let i = 0; i < m; i++) {
        let s = 0;
        for (let k = 0; k < n; k++) {
          s += M[i][k] * v[k];
        }
        out[i] = s;
      } 
      return out;
    }
    function mat1d2d(v, M) {           // (n)x(n,p)->(p)
      const n = v.length;
      const p = M[0].length;
      const out = new Array(p).fill(0);
      
      for (let j = 0; j < p; j++) {
        for (let k = 0; k < n; k++) {
          out[j] += v[k] * M[k][j];
        }
      }
      return out;
    }
    function mat2d2d(A, B) {           // (m,n)x(n,p)->(m,p)
      const m = A.length;
      const n = A[0].length;
      const p = B[0].length;
      const O = new Array(m).fill(0);;
      
      for (let i = 0; i < m; i++) {
        const row = new Array(p);
        for (let j = 0; j < p; j++) {
          let s = 0;
          for (let k = 0; k < n; k++) {
            s += A[i][k] * B[k][j];
          }
          row[j] = s;
        }
        O[i] = row;
      }
      return O;

    }

  /* ----------  BASE‑CASE SHORTCUTS  ---------- */
  if(RA<=2 && RB<=2){
    if(RA===1 && RB===1){
      if(SA[0]!==SB[0])throwErr();
      return new Tensor(dot1d1d(A,B));
    }
    if(RA===2 && RB===1){
      if(SA[1]!==SB[0])throwErr();
      return new Tensor(mat2d1d(A,B));
    }
    if(RA===1 && RB===2){
      if(SA[0]!==SB[0])throwErr();
      return new Tensor(mat1d2d(A,B));
    }
    if(RA===2 && RB===2){
      if(SA[1]!==SB[0])throwErr();
      return new Tensor(mat2d2d(A,B));
    }
  }

  /* ----------  BROADCAST BATCH DIMS  ---------- */
  //If the tensor(s) have more than 2 dimensions, we take the last 2 dimensions,
  //otherwise we take the entire shape 
  const MatA = RA>=2 ? SA.slice(-2) : SA; 
  const MatB = RB>=2 ? SB.slice(-2) : SB;
  //Checking for row-col compatibility
  if(MatA[1] !== MatB[0]) throwErr();
  //If the tensor(s) have more than 2 dimensions, we take the entire shape except the last 2 dimensions
  const batchA = RA>=2 ? SA.slice(0,-2) : [];
  const batchB = RB>=2 ? SB.slice(0,-2) : [];
  //Finding maximum amount of dims BEYOND the last 2 reserved matmul dims
  const maxBatchLen = Math.max(batchA.length, batchB.length);
  const outBatch = new Array(maxBatchLen);
  for(let i=0;i<maxBatchLen;i++){
    //if the batchA or batchB is empty, we set it to 1
    // This line gets the value from batchA at position [batchA.length-1-i]
    // The ?? is the nullish coalescing operator which returns the right-hand value (1)
    // if the left-hand expression is null or undefined, otherwise returns the left-hand value
    // In this case, if the index is out of bounds (undefined), it defaults to 1
    const a = batchA[batchA.length-1-i] ?? 1;
    const b = batchB[batchB.length-1-i] ?? 1;
    //if the batchA or batchB is not the same, we throw an error
    if(a!==b && a!==1 && b!==1)
      throw new Error('Batch dims not broadcastable');
    //we set the outBatch to the maximum of the batchA or batchB
    outBatch[maxBatchLen-1-i] = Math.max(a,b);
  }
  // ^^^^^ so lets walk through the above function ^^^^^
  //Imagine we have tensor 3,7,5,4 and tensor 2,7,4,5.
  //maxBatchLen = 2.
  //outBatch is initialized to length 2, then filled in with (a=7 b=7) so just outBatch[1] = 7.
  //Then we go to 3,2. a gets set to 3 and b gets set to 2. neither are 1 and they're unequal so we throw an error.
  //If they had've been equal, say 3 or 2, then outBatch would've been [3,7] or [2,7].



  // Calculate the total number of batch operations needed
  // If outBatch is [2,3], we need 2*3=6 separate matrix multiplications
  const totalBatch = outBatch.reduce((p, c) => p * c, 1);
  
  // Extract the dimensions for the output matrices
  // These are the dimensions after matrix multiplication: (m,n)×(n,p)→(m,p)
  const outMatRows = MatA[0], outMatCols = MatB[1];
  
  // Recursively builds a nested array structure with the given shape
  // For example, buildNested([2,3], 0) creates a 2×3 array filled with zeros
  // REPLACED BY ZEROS FUNCTION function buildNested(shape, fill = 0) {
  //   if (shape.length === 0) return fill;  // Base case: return the fill value
    
  //   const len = shape[0];           // Get the size of the current dimension
  //   const rest = shape.slice(1);    // Get the remaining dimensions
  //   const arr = new Array(len);     // Create an array of the current dimension
    
  //   // Recursively build each element of the current dimension
  //   for (let i = 0; i < len; i++) {
  //     arr[i] = buildNested(rest, fill);
  //   }
  //   return arr;
  // }
  //^^^testing case imagine we have tensor 3,7,5,4. We're here using only 3,7 since prior matmul logic
  //took care of 5,4. We're building a nested array with the shape 3,7. Therefore comes into 
  //Buildnested. we do loop 3 times and call buildNested on rest each time, where rest = 7.
  //What does calling buildNested(7) do? Is it returned as a single [7] array or a standalone value?
  //I'm curious about potential issues here, it seems problematic potentially. FUNCTION HAS BEEN FLAGGED BY ME.

  // Create the output tensor structure with the correct batch and matrix dimensions
  // The last dimension is actually a placeholder for the matrix that will be computed
  // The spread operator (...) unpacks the outBatch array elements into individual elements
  // For example, if outBatch is [3,7], [...outBatch, outMatRows, outMatCols] becomes [3,7,outMatRows,outMatCols]
  // REDUNDANT FUNC CALL and FUNC --> const OUT = buildNested([...outBatch, outMatRows, outMatCols], null); 
  const OUT = zeros([...outBatch, outMatRows, outMatCols]);
  //^^^^But wait, wouldn't we need outMatCols as well? a 3,7,5,4 x 3,7,4,5 would be a 3,7,5,5 tensor. Change made.





  // Process each batch element
  for (let flat = 0; flat < totalBatch; flat++) {
    // Convert flat index to multi-dimensional batch index
    const idx = unravel(flat, outBatch); 
    
    // Extract the appropriate slices from input tensors, handling broadcasting
    const subA = recurSlice(A, broadcastIdx(idx, batchA, outBatch));
    const subB = recurSlice(B, broadcastIdx(idx, batchB, outBatch));
    // const subA = broadcastIdx(idx, batchA)
    //          .reduce((cur, i) => cur[i], A.data);
    // const subB = broadcastIdx(idx, batchB)
    //          .reduce((cur, i) => cur[i], B.data);
    //alternative way to do this:
    //const subA = SB.length > 1 ? (SB.length > 2 ? (SB.length > 3 ? A[idx[0]][idx[1]][idx[2]][idx[3]] : A[idx[0]][idx[1]][idx[2]]) : A[idx[0]][idx[1]]) : A[idx[0]];
    //const subB = SA.length > 1 ? (SA.length > 2 ? (SA.length > 3 ? B[idx[0]][idx[1]][idx[2]][idx[3]] : B[idx[0]][idx[1]][idx[2]]) : B[idx[0]][idx[1]]) : B[idx[0]];
    
    // Compute matrix multiplication for this batch element
    // let finalChoice = [0,0];
    let result;
    const rA = Array.isArray(subA[0]) ? 2 : 1;
    const rB = Array.isArray(subB[0]) ? 2 : 1;
    
    if      (rA === 1 && rB === 1) result = dot1d1d(subA, subB);
    else if (rA === 2 && rB === 1) result = mat2d1d(subA, subB);
    else if (rA === 1 && rB === 2) result = mat1d2d(subA, subB);
    else                           result = mat2d2d(subA, subB);
    
    //If we want to handle vectors @ larger matrices than 2dims, we can do this:

    // Store the result in the output structure
    // Navigate to the correct position and assign the result
    let cur = OUT;
    for (let i = 0; i < idx.length; i++) {
      if (i === idx.length - 1) {
        cur[idx[i]] = result;
      } else {
        cur = cur[idx[i]];
      }
    }
  }

  dynamicFuncGraph.push(new Node("matmul", a, b, OUT));

  return new Tensor(OUT, {requiresGrad: A.requiresGrad || B.requiresGrad});
    //lets imagine A=3,7,5,4 @ B=3,7,4,5.
    //outbatch was calculated to be [3,7] earlier in the code.
    //totalBatch = 3*7 = 21 was also calculated earlier in the code.
    //1st loop (flat=0, continues for 20 more cycles after): outBatch = [3,7]
    //so lets say idx = unravel(0, [3,7]) = [0,0]
    //subA = recurSlice(A, [3,4], broadcastIdx([0,0], [3,7])) ....
    //broadcastIdx([0,0], [3,7]) evals to [0,0]
    //subA = recurSlice(A, [3,4], [0,0]) = A[0][0] = 3,7 batch dims are evaled and we're left with A[0][0] = 5x4 matrix  
    
    //subB = recurSlice(B, [4,5], broadcastIdx([0,0], [3,7])) = 
    //B[0][0] = 3,7 batch dims are evaled and we're left with B[0][0] = 4x5 matrix
    //result = mat2d2d(subA, subB) = 5x5 matrix
    //OUT[0] = 5x5 matrix
    //But wait. For first loop where i = 0... cur is a 3,7,5,5 tensor.
    //cur[idx[0]] is cur[0]] = 7,5,5 tensor because i was not yet == idx.length - 1 (i.e i was not yet 1, would be in next loop)
    //now this second (inner) loop is i=1, so we do a 7,5,5 cur tensor accessed via cur[0] = 5x5 matrix.
    //We repeat 20 more times (21 cycles total)). We get a 3,7,5,5 matmul result. 
  }


  /* ------ convenience wrappers ---------- */
  add(o){ return Tensor.addForward(this, o);}
  sub(o){ return Tensor.subForward(this, o); }
  mul(o){ return Tensor.mulForward(this, o); }
  div(o){ return Tensor.divForward(this, o); }
  matmul(o){ return Tensor.matmulForward(this, o); }
  transpose(){ return transposeF(this); }
  relu(){ return Tensor.reluForward(this); }

  static transpose(x){return transposeF(x);}
  static add(a, b) { return Tensor.addForward(a, b); }
  static sub(a, b) { return Tensor.subForward(a, b); }
  static mul(a, b) { return Tensor.mulForward(a, b); }
  static div(a, b) { return Tensor.divForward(a, b); }
  static matmul(a, b) { return Tensor.matmulForward(a, b); }
  static relu(x) { return Tensor.eltwise(x, null, x => Math.max(0, x), null, null, "relu"); }


  static addBackward(parent0,parent1,child){return Tensor.eltwise(parent0,parent1,null,g=>g, g=>g, "add",child)};
  static subBackward(parent0,parent1,child){return Tensor.eltwise(parent0,parent1,null,g=>g, g=>-g, "sub", child)};
  static mulBackward(parent0,parent1,child){return Tensor.eltwise(parent0,parent1,null,(g,y)=>g*y,(g,x)=>g*x,"mul",child)};
  static divBackward(parent0,parent1,child){return Tensor.eltwise(parent0,parent1,null,(g,y)=>g/y, (g,x,y)=>-g*x/(y*y), "div", child)};
  static matmulBackward(parent0,parent1,child){
    


    return parent0,parent1,null,g=>g, g=>g, "matmul",child
  };
  static transposeBackward(parent0,parent1){return Tensor.eltwise(parent0,parent1,null,g=>g, g=>g, "transpose")};
  static reluBackward(parent0){return Tensor.eltwise(parent0,null,null,g=>g, g=>g, "relu")}; //not sure yet here


  // static addForward(a, b) { return Tensor.eltwise(a, b, (x,y)=>x+y, g=>g, g=>g, "add"); } //CHANGE: I'm confused.... what is g here? We're essentially passing it as a function 
  // // to map2 yet I don't
  // //know what it is yet.
  // static subForward(a, b) { return Tensor.eltwise(a, b, (x,y)=>x-y, g=>g, g=>-g,"sub"); }
  // static mulForward(a, b) { return Tensor.eltwise(a, b, (x,y)=>x*y,(g,y)=>g*y,(g,x)=>g*x,"mul"); }
  // static divForward(a, b) {
  //   return Tensor.eltwise(
  //     a, b, (x,y)=>x/y, (g,y)=>g/y, (g,x,y)=>-g*x/(y*y), "div"); //CHANGE:I don't understand this logic!
  // }

  //Here's the big challenge:
  //CHANGE:I DON'T UNDERSTAND THE LOGIC!!!!
  //Imagine we calculated the loss with an MSELoss function that took in the last output layer as input. The loss is then calculated from MSELoss and returned...
  //It must be returned as a tensor I think.
  //Reasonable assumption --> everything is a tensor. The final tensor is simply the 'loss value'. It's grad must be calculated once we call '.backward()'. 
  //It's default backwards grad calculating function is there and we can immediately start the chain.
  /* -------------- backward driver -------- */

  
  starterBackward() {
    // If this is a scalar (0D) and no grad set, init grad=1
    if (this.shape.length === 0 && this.grad === null) {
      this.grad = 1;
    }
    for(let i = dynamicFuncGraph.length-1; i >= 0; i--){
      const node = dynamicFuncGraph[i];
      
      switch(node.funcType) {
        case "matmul":
          Tensor.matmulBackward(node.parentA, node.parentB, node.child);
          break;
        case "add":
          Tensor.addBackward(node.parentA, node.parentB, node.child);
          break;
        case "sub":
          Tensor.subBackward(node.parentA, node.parentB, node.child);
          break;
        case "mul":
          Tensor.mulBackward(node.parentA, node.parentB, node.child);
          break;
        case "div":
          Tensor.divBackward(node.parentA, node.parentB, node.child);
          break;
        case "relu":
          Tensor.reluBackward(node.parentA, node.child); //not sure yet here
          break;
        default:
          // For any other operations or if _op is not set
          // throw new Error("Testing testing, this shouldn't be happening, tensor should have an op!!");
          console.log("node.op:",node.funcType);
          //v._backward();
      }
    }
    
  }
}

/* ======================================================================= *
   Losses
 * ======================================================================= */
class MSELoss {
  forward(pred,targ){
    const diff=pred.sub(targ), sq=diff.mul(diff);
    const loss= new Tensor(
      diff.data.reduce((s,row,i)=>s+row.reduce((a,c)=>a+c,0),0) /
      (pred.shape[0]*pred.shape[1]),
      {requiresGrad:true});
    loss._parents=[sq];
    loss._backward=()=>{
      const coef=1/(pred.shape[0]*pred.shape[1]);
      sq.grad = addArrays(sq.grad, map2(sq.data,sq.data,_=>coef*loss.grad));
    };
    return loss;
  }
}

class CrossEntropyLoss {
  forward(logits,targetIdx){
    const m=Math.max(...logits.data);
    const exps=logits.data.map(v=>Math.exp(v-m)); //basically doing softmax, right?
    const Z=exps.reduce((a,b)=>a+b,0);
    const loss=new Tensor(-Math.log(exps[targetIdx]/Z),{requiresGrad:true});
    loss._parents=[logits];
    loss._backward=()=>{
      logits.data.forEach((v,i)=>{
        const soft=Math.exp(v-m)/Z;
        logits.grad[i]+= (soft-(i===targetIdx?1:0))*loss.grad;
      });
    };
    return loss;
  }
}

/* ======================================================================= *
   Optimizer (SGD)
 * ======================================================================= */
class Optimizer{
    static step(lr = LR){
      WEIGHTS.forEach(p => {
        // in‑place update is faster & safe because .data is a fresh JS array
        p.data = map2(p.data, p.grad, (w,g) => w - lr * g);
      });
    }
  
    static zero_grad_and_graph_clear(){
      WEIGHTS.forEach(p => p.grad = zeros(p.shape));
      dynamicFuncGraph = [];
    }
}

/* ======================================================================= *
   Example Linear Layer (weight/bias params)
 * ======================================================================= */
class Linear{
  //outF is number of neurons, inF is how many FC ops to expect to go to one neuron
  constructor(inF,outF,{bias=true,init="xavier"}={}){
    this.W=new Tensor(initWeights([outF,inF],init),{requiresGrad:true});
    this.b=bias?new Tensor(initWeights([outF],"default"),{requiresGrad:true}):null;
  }
  forward(x){
    let y=x.matmul(this.W.transpose()); //Default pytorch rules I think, initializing layers with in & Out feats that are flipped 
    //wrt normal matrix multiplication, e.g they might do r1 = nn.linear(2,10), r2 = nn.linear(5,10), when in reality its 2,10 @ 10,5.
    //Might change syntax^^
    if(this.b) y=y.add(this.b);
    return y;
  }
}


console.log("addForward:",Tensor.addForward(new Tensor([[2,1],[1,3],[2,4]]),new Tensor([2,4])));
console.log("matmul:",new Tensor([[2,1]]).matmul(new Tensor([[2,4],[1,3]])));
console.log("matmul:",new Tensor([[2,1],[4,5]]).matmul(new Tensor([[2,4],[1,3]])));
console.log("divForward:",Tensor.divForward(new Tensor([[2,1],[4,5]]),new Tensor([[2,4],[1,3]])));
console.log("divForward:",Tensor.divForward(new Tensor(nArr1),new Tensor(nArr2)));

let r1 = new Tensor([[[4,5],[6,2],[4,9]],[[4,5],[6,2],[4,9]],[[4,5],[6,2],[4,9]]]);
let r2 = new Tensor([[4,5],[6,2],[4,9]]);
let scalarSizeTest = new Tensor([4]);
console.log("brief test on mulwise: ", r1.mul(r2));
console.log("infer scalar size test",inferShape(scalarSizeTest.data));
console.log("trans test",r2.transpose());
console.log("test on matmul", r1.matmul(r2.transpose()));
//first grad test=

let b1 = new Tensor([[[4,5],[6,2],[4,9]],[[4,5],[6,2],[4,9]],[[4,5],[6,2],[4,9]]], {requiresGrad: true}); 
let b2 = new Tensor([[4,5],[6,2],[4,9]], {requiresGrad: true});
let out = b1.mul(b2);out.requiresGrad=true; // Shape [3,3,2]
out.grad = zeros([3,3,2]); // Initialize gradient

out.grad.every((v,i) => v = 1);

settingGradOnes = fill(inferShape(out.grad),()=>1);
out.grad = settingGradOnes;
console.log("out.grad:",out.grad);
out.starterBackward();
console.log("out.data:",out.data);
console.log("out.grad:",out.grad);
console.log("b2.grad:",b2.grad);
console.log("b1.grad:",b1.grad);
Optimizer.zero_grad_and_graph_clear();

let l1 = new Tensor([[5,6],[3,4]],{requiresGrad:true,hasWeights:true});
let l2 = new Tensor([[1,2],[3,4]],{requiresGrad:true,hasWeights:true});
let l3 = new Tensor([[1,2],[2,1]],{requiresGrad:true,hasWeights:true});
let l4 = new Tensor([3],{requiresGrad:true,hasWeights:true});
let in1=l2.add(l1); in1.requiresGrad=true;
let in2=l3.add(in1); in2.requiresGrad=true;
let in3=l4.mul(in2); in3.requiresGrad=true;
console.log("l1.data:",l1.data);
console.log("l2.data:",l2.data);
console.log("l3.data:",l3.data);
console.log("l4.data:",l4.data);
console.log("in1.data:",in1.data);
console.log("in2.data:",in2.data);
console.log("in3.data:",in3.data);

console.log("l1.op:",l1);
console.log("l2.op:",l2);
console.log("l3.op:",l3);
console.log("l4.op:",l4);
console.log("in1.op:",in1);
console.log("in2.op:",in2);
console.log("in3.op:",in3);
in3.grad = [1];
in3.starterBackward();

// Log the gradients of all tensors
console.log("l1.grad:", l1.grad);
console.log("l2.grad:", l2.grad);
console.log("l3.grad:", l3.grad);
console.log("l4.grad:", l4.grad);
console.log("in1.grad:", in1.grad);
console.log("in2.grad:", in2.grad);


console.log("in3.grad:",in3.grad);
Optimizer.zero_grad_and_graph_clear();

// Usage examples:
console.log("testing flatten",flatten([[[[[[[[1, 2]]]]]]], [[3], 4]]));
console.log("testing more lil nets:\n\n\n\n");
let ui1 = new Tensor(nArr1, {requiresGrad:true,hasWeights:true});
let ui2 = new Tensor(nArr2, {requiresGrad:true,hasWeights:true});
let ui3 = new Tensor([5], {requiresGrad:true,hasWeights:true});
let uia1 = ui1.add(ui2);
let uia2 = uia1.mul(ui3); //eltwise static function 
//inherits requiresGrad true if its set for either parent. Therefore this is safe.
uia2.grad = [1];
uia2.starterBackward();
console.log("ui1.grad:",ui1.grad);
console.log("ui2.grad:",ui2.grad);
console.log("ui3.grad:",ui3.grad);
console.log("uia1.grad:",uia1.grad);
console.log("uia2.grad:",uia2.grad);
let bia1 = ui1.add(ui2);
let bia2 = bia1.div(ui3);
bia2.grad = [1];
bia2.starterBackward();
console.log("bia1.grad:",bia1.grad);
console.log("bia2.grad:",bia2.grad);
console.log("ui1.grad:",ui1.grad);
console.log("ui2.grad:",ui2.grad);
console.log("ui3.grad:",ui3.grad);




// //must run:
// fc1=Linear(2,10);
// fc2=Linear(10,5);
// rl3=fc2.relu();
// fc4=Linear(5,1);
// //Will need to calculate softmax I think?
// CrossEntropyLoss(fc4);

////something like this:
// conv2d({inCh:1, outCh:6, k:5, stride:1, pad:0}),
// relu(),
// maxPool2d({k:2, stride:2}),
// conv2d({inCh:6, outCh:16, k:5, stride:1, pad:0}),
// relu(),
// maxPool2d({k:2, stride:2}),
// conv2d({inCh:16, outCh:64, k:5, stride:1, pad:0}), // 64×1×1 output
// relu(),
// flatten(),
// linear({inFeat:64, outFeat:84}),
// relu(),
// linear({inFeat:84, outFeat:10})

////or even:
// conv2d({inCh:1,  outCh:8,  k:5, stride:1, pad:2}),
// relu(),
// maxPool2d({k:2,  stride:2}),          // → 8×12×12

// conv2d({inCh:8,  outCh:16, k:5, stride:1, pad:2}),
// relu(),
// maxPool2d({k:3,  stride:3}),          // → 16×4×4

// flatten(),                            // 256 features
// linear({inFeat:256, outFeat:10}),
// softmax()

//Then, must run something like:




/* ======================================================================= */
if(typeof module!=="undefined") //CHANGE:What the fuck is this? If type of module is undefined? Does this apply only to NODE or smth? WHATTTTT
  module.exports={Tensor,Linear,MSELoss,CrossEntropyLoss,Optimizer,WEIGHTS,setLR:v=>LR=v};

// =================== TEST BATTERY FOR TENSOR OPERATIONS ===================

// --------- TEST ARRAYS ---------
// 1D arrays (vectors)
const vec1 = [1, 2];                                // shape: [2]
const vec2 = [10, 20, 30];                         // shape: [3]
const vec3 = [5];                                  // shape: [1] (scalar-like)

// 2D arrays (matrices)
const mat1 = [[1, 2], [3, 4]];                      // shape: [2, 2]
const mat2 = [[10, 20], [30, 40], [50, 60]];        // shape: [3, 2]
const mat3 = [[100, 200, 300], [400, 500, 600]];    // shape: [2, 3]
const mat4 = [[1, 2, 3]];                          // shape: [1, 3]
const mat5 = [[1], [2]];                           // shape: [2, 1]

// 3D arrays (batched matrices)
const batch1 = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];           // shape: [2, 2, 2]
const batch2 = [[[10, 20, 30], [40, 50, 60]]];                 // shape: [1, 2, 3]
const batch3 = [[[1, 2]], [[3, 4]], [[5, 6]]];                 // shape: [3, 1, 2]
const batch4 = [[[1], [2]], [[3], [4]]];                       // shape: [2, 2, 1]

// 4D arrays
const tensor4d1 = [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                   [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]; // shape: [2, 2, 2, 2]
const tensor4d2 = [[[[1]]], [[[2]]]];                           // shape: [2, 1, 1, 1]

// --------- SHAPE VERIFICATION ---------
console.log("===== SHAPE VERIFICATION =====");
console.log("vec1 shape:", inferShape(vec1));
console.log("mat1 shape:", inferShape(mat1));
console.log("batch1 shape:", inferShape(batch1));
console.log("tensor4d1 shape:", inferShape(tensor4d1));

// --------- ELEMENT-WISE OPERATIONS ---------
console.log("\n===== ELEMENT-WISE OPERATIONS =====");

// 1. Same shape addition
console.log("\n-- Same shape addition --");
console.log("mat1 + mat1:", map2(mat1, mat1, (a, b) => a + b));

// 2. Broadcasting smaller to larger
console.log("\n-- Broadcasting: 2D to 3D --");
console.log("batch1 + mat1:", map2(batch1, mat1, (a, b) => a + b));

// 3. Matrix-vector multiplication (2D×1D)
console.log("\n-- Matrix-vector multiplication (2D×1D) --");
console.log("mat2 × vec1:", new Tensor(mat2).matmul(new Tensor(vec1)));  // [3,2] × [2]

// 4. 1D vector broadcast to matrix
console.log("\n-- 1D vector broadcast to matrix --");
console.log("mat1 + vec1:", map2(mat1, vec1, (a, b) => a + b));

// 5. Broadcasting across 4D tensor
console.log("\n-- Broadcasting to 4D tensor --");
console.log("tensor4d1 + vec1:", map2(tensor4d1, vec1, (a, b) => a + b));

// 6. Scalar-like broadcasting
console.log("\n-- Scalar-like broadcasting --");
console.log("batch1 + vec3:", map2(batch1, vec3, (a, b) => a + b));

// 7. Mixed operations (multiply + add)
console.log("\n-- Mixed operations --");
console.log("mat1 * vec1 + mat5:", map2(map2(mat1, vec1, (a, b) => a * b), mat5, (a, b) => a + b));

// --------- MATRIX MULTIPLICATION ---------
console.log("\n===== MATRIX MULTIPLICATION =====");

// 1. Basic matrix multiplication
console.log("\n-- Basic matrix multiplication (2D×2D) --");
console.log("mat1 × mat3:", new Tensor(mat1).matmul(new Tensor(mat3)));  // [2,2] × [2,3]

// 2. Vector-matrix multiplication (1D×2D)
console.log("\n-- Vector-matrix multiplication (1D×2D) --");
console.log("vec1 × mat3:", new Tensor(vec1).matmul(new Tensor(mat3)));  // [2] × [2,3]

// 3. Matrix-vector multiplication (2D×1D)
console.log("\n-- Matrix-vector multiplication (2D×1D) --");
console.log("mat2 × vec1:", new Tensor(mat2).matmul(new Tensor(vec1)));  // [3,2] × [2]

// 4. Batch matrix multiplication (3D×2D)
console.log("\n-- Batch matrix multiplication (3D×2D) --");
console.log("batch1 × mat3:", new Tensor(batch1).matmul(new Tensor(mat3)));  // [2,2,2] × [2,3]



// 6. Higher dimension multiplication (4D×3D)
console.log("\n-- Higher dimension multiplication (4D×3D) --");
console.log("tensor4d1 × batch4:", new Tensor(tensor4d1).matmul(new Tensor(batch4)));  // [2,2,2,2] × [2,2,1]

// 7. Extreme broadcasting in matmul (4D×2D)
console.log("\n-- Extreme broadcasting in matmul (4D×2D) --");
console.log("tensor4d1 × mat5:", new Tensor(tensor4d1).matmul(new Tensor(mat5)));  // [2,2,2,2] × [2,1]

// 8. Single element dimensions
console.log("\n-- Single element dimensions --");
console.log("tensor4d2 × vec3:", new Tensor(tensor4d2).matmul(new Tensor(vec3)));  // [2,1,1,1] × [1]

// --------- EDGE CASES ---------
console.log("\n===== EDGE CASES =====");

// 2. Incompatible dimensions - should throw error
console.log("\n-- Incompatible dimensions --");
try {
  console.log("mat1 × mat2:", new Tensor(mat1).matmul(new Tensor(mat2)));  // [2,2] × [3,2] should fail
} catch (e) {
  console.log("Correctly caught error:", e.message);
}

// 3. Extreme broadcasting
console.log("\n-- Extreme broadcasting --");
console.log("tensor4d1 + vec3:", map2(tensor4d1, vec3, (a, b) => a + b));  // [2,2,2,2] + [1]

// console.log("tensor4d1.transpose():", new Tensor(tensor4d1).transpose());
