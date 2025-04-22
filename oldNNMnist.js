//neuralNetMnist.js
//Sanity loading check (more dataset loading functionality tba ofc)
function loadImage1() {
    const canvas = document.getElementById('c');      // get canvas
    const ctx = canvas.getContext('2d');              // get 2D context
    let img = new Image();
    img.src = "mnistFolder/MNIST Dataset JPG format/MNIST - JPG - training/1/14.jpg";
        img.onload = () => {
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            console.log(`loaded public_html/${img.src}`);
        }
}

// // A brief first attempt. A few but certanly not all Neural network components (will also try to simplify later)
// function minMaxNorm(array) {

//   //take in array -->
//   max = array[0];
//   min = array[0];
//     for(let i = 0; i < array.length; i++){

//         if (array[i] > max){
//             max = array[i];
//         }
//         if (array[i+1] < min){
//             min = array[i];
//         }
//         }   
//         let diff = max - min;
//     for(let i = 0; i < array.length; i++){

//         array[i] -= min; array[i] /= diff;
//     }
//     return array;
// }

// function ReLU(array) {
//   // Implementation of ReLU activation
//   // max(0, x)
//     for(let i = 0; i < array.length; i++){
//         if (array[i] < 0){array[i] = 0;}
//     }return array;
// }

// //i realize I will have to adjust this such that we have seperate modes for initialization and layer activation,
// //similar to init() and forward()
// function conv2d(input, filters, stride = 1, padding = 0) {
//   // Implementation of 2D convolution

//   if(padding > 0){
//     let newTens = tensor(input.length,input[0].length);
    
//     for(let i = 0; i < padding; i++){

//     }
//   }
// }

// function linear(input, weights, bias) {
//   // Implementation of fully connected layer
//   // output = input * weights + bias.....
// }

// // Loss function
// function crossEntropyLoss(predictions, targets) {
//   // Need to implementat cross-entropy loss...
// }

// // .backward(), softmax... and others etc....

// //will add randomization possibility for initilization
// function tensor(a,b,) {
//     //new array of size b fill with zeros
//     //make a instances filled with duplicates of b
//     //return the array of arrays...

//     let newArray = [];
//     for (let i = 0; i < a; i++) {
//       newArray.push(Array(b).fill(0));
//     }
//     return newArray;

// }

// function transpose(tens){
//     let newTens = [];
//     for(let i = 0; i < tens.length; i++){
//         for(let j = 0; j < tens[0].length; j++){
//             newTens[j][i] = tens[i][j];
//     }
//     return newTens;
// }
// }

// //one must tranpose() the second matrix first...
// function matMul(tens1,tens2){
//     let m = tens1.length;         // rows of tens1
//     let n = tens1[0].length;      // cols of tens1
//     let p = tens2[0].length;      // cols of tens2
   
//     for (let i = 0; i < m; i++) {         // rows of tens1
//         for (let j = 0; j < p; j++) {       // columns of tens2
//           for (let k = 0; k < n; k++) {     // shared dimension
//             result[i][j] += tens1[i][k] * tens2[k][j];
//           }
//         }
//       }
//     return result;
// }


class Tensor {
  constructor(data, requires_grad = false) {
    this.data = data;
    this.shape = this.shape();
    this.requires_grad = requires_grad;
    this.grad = null;
    this._backward = null; // Function to run during backprop
    this._children = []; // Dependencies - tensors used to pass into this one through our forward()
  }

  add(other){
    // Check if shapes match
    // For simplicity, only handling 2D case now
    if (this.data.length !== other.data.length || 
        this.data[0].length !== other.data[0].length) {
      throw new Error('Tensor shapes must match for addition');
    }
    
    let result = [];
    for (let i = 0; i < this.data.length; i++) {
      let row = [];
      for (let j = 0; j < this.data[0].length; j++) {
        row.push(this.data[i][j] + other.data[i][j]);
      }
      result.push(row);
    }
    return new Tensor(result);
  }

  sub(other) {
    // element‑wise subtraction: this.data – other.data
    if (this.data.length !== other.data.length ||
        this.data[0].length !== other.data[0].length) {
      throw new Error('Tensor shapes must match for subtraction');
    }
    const result = [];
    for (let i = 0; i < this.data.length; i++) {
      const row = [];
      for (let j = 0; j < this.data[0].length; j++) {
        row.push(this.data[i][j] - other.data[i][j]);
      }
      result.push(row);
    }
    return new Tensor(result);
  }

  mul(other) {
    // element‑wise multiplication: this.data * other.data
    if (this.data.length !== other.data.length ||
        this.data[0].length !== other.data[0].length) {
      throw new Error('Tensor shapes must match for multiplication');
    }
    const result = [];
    for (let i = 0; i < this.data.length; i++) {
      const row = [];
      for (let j = 0; j < this.data[0].length; j++) {
        row.push(this.data[i][j] * other.data[i][j]);
      }
      result.push(row);
    }
    return new Tensor(result);
  }
  
  div(other) {
    // element‑wise division: this.data / other.data
    if (this.data.length !== other.data.length ||
        this.data[0].length !== other.data[0].length) {
      throw new Error('Tensor shapes must match for division');
    }
    const result = [];
    for (let i = 0; i < this.data.length; i++) {
      const row = [];
      for (let j = 0; j < this.data[0].length; j++) {
        if (other.data[i][j] === 0) {
          throw new Error('Division by zero in Tensor.div()');
        }
        row.push(this.data[i][j] / other.data[i][j]);
      }
      result.push(row);
    }
    return new Tensor(result);
  }

  matmul(other) {
    const A = this.data; 
    const B = other.data;
    const SA = this.shape;
    const RA = SA.length;
    const SB = other.shape;
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
      const out = new Array(m);
      
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
      const out = new Array(p);
      
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
      const O = new Array(m);
      
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

  /* ----------  BASE‑CASE SHORT‑CIRCUITS  ---------- */
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



    
    // return new Tensor(result);
  
  }
  
  shape() {
    // Simple recursive shape inference
    const shape = [];
    let cur = this.data;
    while (Array.isArray(cur)) {
      shape.push(cur.length);
      cur = cur[0];
    }
    return shape;
  }
} 

console.log(new Tensor([[1,2,3],[4,5,6]]).add(new Tensor([[7,8,9],[10,11,12]])));
console.log(new Tensor([[1,2,3],[1,2,3]]).matmul(new Tensor([[1,2],[2,3],[3,4]])));

//2,3,4
  const a = [
    [
      [1, 2, 3, 1],
      [2, 3, 1, 2],
      [3, 1, 2, 3]
    ],
    [
      [2, 1, 3, 2],
      [1, 3, 2, 1],
      [3, 2, 1, 3]
    ]
  ];
//2,4,5
  const b = [
    [[1, 3, 2, 1, 3],
    [2, 1, 3, 2, 1],
    [3, 2, 1, 3, 2],
    [1, 3, 2, 1, 3]],[    [1, 3, 2, 1, 3],
    [2, 1, 3, 2, 1],
    [3, 2, 1, 3, 2],
    [1, 3, 2, 1, 3]]
  ];

// class Module {
//   constructor(){}

//   forward(){}

//   backward(){}

  
// }
// class Conv2d(){}
// class ReLU(){}
// class xxx(){}

// class Linear extends Module {
//   constructor(in_features, out_features, bias = true){

//   }
// }
 
function broadcastShapes(shapeA, shapeB) {
  // Right‑align shorter shape with leading 1s
  const len = Math.max(shapeA.length, shapeB.length);
  const out = new Array(len);
  console.log("this is shapeA", shapeA);
  console.log("this is shapeB", shapeB);
  for (let i = 0; i < len; i++) {
    // Get the value at the current position from the end of shapeA, or use 1 if undefined
    // The ?? operator is the nullish coalescing operator that returns the right-hand value (1)
    // if the left-hand value is null or undefined, which happens when we access beyond the array bounds
    const a = shapeA[shapeA.length - 1 - i] ?? 1;
    const b = shapeB[shapeB.length - 1 - i] ?? 1;
    console.log("this is i", i);
    console.log("this is a", a);
    console.log("this is b", b);
    console.log("this is out", out);
    if (a !== b && a !== 1 && b !== 1)
      
      throw new Error('Batch dims not broadcast‑compatible');
    out[len - 1 - i] = Math.max(a, b);
    
  }
  
  return out;               // plain array of ints
}


//?function autograd(){}

function mat2d2d(A, B) {
  const m = A.length;
  const n = A[0].length;
  const p = B[0].length;
  const out = new Array(m);
  for (let i = 0; i < m; i++) {
    const row = new Array(p);
    for (let j = 0; j < p; j++) {
      let acc = 0;
      for (let k = 0; k < n; k++) {
        acc += A[i][k] * B[k][j];
      }
      row[j] = acc;
    }
    out[i] = row;
  }
  return out;
}

function mat3d2d(A, B) {
  const l = A.length;
  const m = A[0].length;
  const n = A[0][0].length;
  const p = B[0].length;
  
  const out2 = new Array(l);
  for(let r = 0; r < l; r++){
    const out1 = new Array(m);
    for (let i = 0; i < m; i++) {
        const row = new Array(p);
        for (let j = 0; j < p; j++) {
          let acc = 0;
          for (let k = 0; k < n; k++) {
            acc += A[r][i][k] * B[k][j];
          }
          row[j] = acc;
        }
        out1[i] = row;
      }
      out2[r] = out1;
  }
  return out2;
}

// function mat3d3d(A, B){

// }
// function mat2d3d(A, B){}
// func mat 3d1d(){}
// function mat1d3d(){}
//AI PLEASE implement all versions of mat4dxd and matxd4d!!!!


  // console.log("this is mat2d2d", mat2d2d(a[0], b[0]));
  // console.log("this is mat3d2d", mat3d2d(a, b[0]));



  
  // // 2. Or call the shape function directly on the arrays:
  // console.log(broadcastShapes(getArrayShape(a), getArrayShape(b)));
  
  // Where getArrayShape would be a function like:
  function getArrayShape(arr) {
    let shape = [];
    let current = arr;
    while (Array.isArray(current)) {
      shape.push(current.length);
      current = current[0];
    }
    return shape;
  }


  // Examples of how slice() works in JavaScript arrays
  
  