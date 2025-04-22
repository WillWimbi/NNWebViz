const randn = () => Math.sqrt(-2 * Math.log(Math.random())) * Math.cos(2 * Math.PI * Math.random());

const inferShape = x => Array.isArray(x) ? [x.length, ...inferShape(x[0])] : [];

const fill = (shape, gen) => shape.length === 0 ? gen() : Array.from({length: shape[0]}, () => fill(shape.slice(1), gen));

const zeros = shape => fill(shape, () => 0);

const padShape = (shape, len) => Array(len - shape.length).fill(1).concat(shape);

const padArray = (arr, targetRank) => {
  let out = arr;
  let rank = inferShape(arr).length;
  while (rank < targetRank) {
    out = [out];
    rank++;
  }
  return out
};

function unravel(k, shape) {
  const idx = new Array(shape.length);
  for (let i = shape.length - 1; i >= 0; i--) {
    idx[i] = k % shape[i];
    k = Math.floor(k / shape[i]);
  }
  return idx;
}

function throwErr() {
  throw new Error(`Inner dim mismatch: [${SA}] x [${SB}]`);
}

function recurSlice(data, idx) {
  let cur = data;
  let d = 0;
  while (d < idx.length) {
    cur = cur[idx[d]];
    d++;
  }
  return cur;
}

function broadcastIdx(idx, targetShape, outBatch) {
  const off = outBatch.length - targetShape.length;
  return targetShape.map((dim, i) => dim === 1 ? 0 : idx[off + i]);
}

const transposeF = (x, shapeN, shapeLength) => {
  let out = [];
  if (shapeLength > 2) {
    out = x.map(v => transposeF(v, shapeN.slice(1), shapeLength - 1));
  } else if (shapeLength === 2) {
    out = Array.from({length: shapeN[1]}, () => Array(shapeN[0]));
    for (let i = 0; i < shapeN[0]; i++) {
      for (let j = 0; j < shapeN[1]; j++) {
        out[j][i] = x[i][j];
      }
    }
  }
  return out;
};

const mapBroadcast = (a, b, fn) => {
  const isArrA = Array.isArray(a), isArrB = Array.isArray(b);
  if (isArrA && isArrB) {
    const len = Math.max(a.length, b.length);
    return Array.from({length: len}, (_, i) => mapBroadcast(a[i % a.length], b[i % b.length], fn));
  }
  if (isArrA) return a.map(v => mapBroadcast(v, b, fn));
  if (isArrB) return b.map(v => mapBroadcast(a, v, fn));
  return fn(a, b);
};

const checkDimsCompatible = (a, b, aShape, bShape, aShapeLength, bShapeLength, minShapeLength) => {
  let truey = true;
  let incompatDim = "";
  for (let i = 1; i < minShapeLength + 1; i++) {
    if (aShape[aShapeLength - i] !== bShape[bShapeLength - i] && 
        aShape[aShapeLength - i] !== 1 && 
        bShape[bShapeLength - i] !== 1) {
      truey = false;
      incompatDim = `aShape: ${aShape}, bShape: ${bShape}`;
      break;
    }
  }
  return [truey, incompatDim];
};

const map2 = (a, b, fn) => {
  const isArrA = Array.isArray(a), isArrB = Array.isArray(b);
  if (!isArrA || !isArrB) throw Error("map2: at least one input must be an array");
  
  const aShape = inferShape(a);
  const bShape = inferShape(b);
  const aShapeLength = aShape.length;
  const bShapeLength = bShape.length;
  maxShapeLength = Math.max(aShape.length, bShape.length);
  minShapeLength = Math.min(aShape.length, bShape.length);
  diff = aShape.length - bShape.length;
  
  const [truey, incompatDims] = checkDimsCompatible(a, b, aShape, bShape, aShapeLength, bShapeLength, minShapeLength);
  if (!truey) throw Error(`map2: incompatible dimensions: ${incompatDims}`);
  
  let aPadded = padArray(a, maxShapeLength);
  let bPadded = padArray(b, maxShapeLength);
  return mapBroadcast(aPadded, bPadded, fn);
};

const addArrays = (a, b) => map2(a, b, (x, y) => x + y);

function isArrayRecursive(arr) {
  if (Array.isArray(arr)) {
    console.log(`${arr} is an array, here's the length: ${arr.length}`);
    arr.forEach(item => isArrayRecursive(item));
  } else {
    console.log(`${arr} is not an array`);
  }
} 


const WEIGHTS = [];
let LR = 0.01;

function initWeights(shape, scheme = "default") {
  const fanIn = shape.slice(-1)[0] || 1;
  let scale;
  switch (scheme) {
    case "xavier":
      scale = Math.sqrt(1 / fanIn);
      break;
    case "he":
      scale = Math.sqrt(2 / fanIn);
      break;
    default:
      scale = 0.01;
      break;
  }
  return fill(shape, () => randn() * scale);
}

console.log("transpose:", transposeF(nArr1, inferShape(nArr1), inferShape(nArr1).length));

class Tensor {
  constructor(data, {requiresGrad = false, name = ""} = {}) {
    this.data = data;
    this.shape = inferShape(data);
    this.requiresGrad = requiresGrad;
    this.grad = requiresGrad ? zeros(this.shape) : null;
    this._op = null;
    this._parents = [];
    this._backward = () => {};
    this.name = name;
    if (requiresGrad) WEIGHTS.push(this);
  }

  fillGradWithZeros() {
    if (this.grad === null) this.grad = zeros(this.shape);
  }

  static eltwise(a, b, f, dfa, dfb, opName) {
    const out = new Tensor(map2(a.data, b.data, f), {requiresGrad: a.requiresGrad || b.requiresGrad});
    out._op = opName;
    out._parents = [a, b];
    out._backward = () => {
      if (a.requiresGrad) {
        a.fillGradWithZeros();
        a.grad = addArrays(a.grad, map2(out.grad, b.data, dfa));
      }
      if (b.requiresGrad) {
        b.fillGradWithZeros();
        b.grad = addArrays(b.grad, map2(out.grad, a.data, dfb));
      }
    };
    return out;
  }
  static addForward(a, b) {
    return Tensor.eltwise(a, b, (x, y) => x + y, g => g, g => g, "add");
  }
  static subForward(a, b) {
    return Tensor.eltwise(a, b, (x, y) => x - y, g => g, g => -g, "sub");
  }
  static mulForward(a, b) {
    return Tensor.eltwise(a, b, (x, y) => x * y, (g, y) => g * y, (g, x) => g * x, "mul");
  }
  static divForward(a, b) {
    return Tensor.eltwise(a, b, (x, y) => x / y, (g, y) => g / y, (g, x, y) => -g * x / (y * y), "div");
  }
  static reluForward(x) {
    const out = new Tensor(map2(x.data, x.data, v => v > 0 ? v : 0), {requiresGrad: x.requiresGrad});
    out._op = "relu";
    out._parents = [x];
    out._backward = () => {
      if (!x.requiresGrad) return;
      x.fillGradWithZeros();
      const upd = map2(out.grad, x.data, (g, v) => v > 0 ? g : 0);
      x.grad = addArrays(x.grad, upd);
    };
    return out;
  }

  relu() {
    return this.chooser("relu");
  }

  static matmulForward(a, b) {
    const A = a.data;
    const B = b.data;
    const SA = a.shape;
    const RA = SA.length;
    const SB = b.shape;
    const RB = SB.length;

    function dot1d1d(a, b) {
      let acc = 0;
      for (let i = 0; i < a.length; i++) {
        acc += a[i] * b[i];
      }
      return acc;
    }

    function mat2d1d(M, v) {
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

    function mat1d2d(v, M) {
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

    function mat2d2d(A, B) {
      const m = A.length;
      const n = A[0].length;
      const p = B[0].length;
      const O = new Array(m).fill(0);
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

    function backward() {
      return 0;
    }

    if (RA <= 2 && RB <= 2) {
      if (RA === 1 && RB === 1) {
        if (SA[0] !== SB[0]) throwErr();
        return new Tensor(dot1d1d(A, B));
      }
      if (RA === 2 && RB === 1) {
        if (SA[1] !== SB[0]) throwErr();
        return new Tensor(mat2d1d(A, B));
      }
      if (RA === 1 && RB === 2) {
        if (SA[0] !== SB[0]) throwErr();
        return new Tensor(mat1d2d(A, B));
      }
      if (RA === 2 && RB === 2) {
        if (SA[1] !== SB[0]) throwErr();
        return new Tensor(mat2d2d(A, B));
      }
    }

    const MatA = RA >= 2 ? SA.slice(-2) : SA;
    const MatB = RB >= 2 ? SB.slice(-2) : SB;
    if (MatA[1] !== MatB[0]) throwErr();

    const batchA = RA >= 2 ? SA.slice(0, -2) : [];
    const batchB = RB >= 2 ? SB.slice(0, -2) : [];
    const maxBatchLen = Math.max(batchA.length, batchB.length);
    const outBatch = new Array(maxBatchLen);

    for (let i = 0; i < maxBatchLen; i++) {
      const a = batchA[batchA.length - 1 - i] ?? 1;
      const b = batchB[batchB.length - 1 - i] ?? 1;
      if (a !== b && a !== 1 && b !== 1) throw new Error('Batch dims not broadcastable');
      outBatch[maxBatchLen - 1 - i] = Math.max(a, b);
    }

    const totalBatch = outBatch.reduce((p, c) => p * c, 1);
    const outMatRows = MatA[0], outMatCols = MatB[1];
    const OUT = zeros([...outBatch, outMatRows, outMatCols]);

    for (let flat = 0; flat < totalBatch; flat++) {
      const idx = unravel(flat, outBatch);
      const subA = recurSlice(A, broadcastIdx(idx, batchA, outBatch));
      const subB = recurSlice(B, broadcastIdx(idx, batchB, outBatch));
      const rA = Array.isArray(subA[0]) ? 2 : 1;
      const rB = Array.isArray(subB[0]) ? 2 : 1;
      
      let result;
      if (rA === 1 && rB === 1) result = dot1d1d(subA, subB);
      else if (rA === 2 && rB === 1) result = mat2d1d(subA, subB);
      else if (rA === 1 && rB === 2) result = mat1d2d(subA, subB);
      else result = mat2d2d(subA, subB);
      
      let cur = OUT;
      for (let i = 0; i < idx.length; i++) {
        if (i === idx.length - 1) {
          cur[idx[i]] = result;
        } else {
          cur = cur[idx[i]];
        }
      }
    }
    return new Tensor(OUT);
  }

  add(o){ return Tensor.addForward(this, o); }
  sub(o){ return Tensor.subForward(this, o); }
  mul(o){ return Tensor.mulForward(this, o); }
  div(o){ return Tensor.divForward(this, o); }
  matmul(o){ return Tensor.matmulForward(this, o); }
  transpose(){ return Tensor.transposeF(this.data, this.shape, this.shape.length); }
  relu(){ return Tensor.reluForward(this); }

  backward() {
    if (this.shape.length === 0 && this.grad === null) {
      this.grad = 1;
    }
    if (this.grad === null) {
      this.fillGradWithZeros();
    }
    
    const topo = [];
    const visited = new Set();
    
    function buildTopo(v) {
      if (!visited.has(v)) {
        visited.add(v);
        for (const p of v._prev) {
          buildTopo(p);
        }
        topo.push(v);
      }
    }
    
    buildTopo(this);
    
    for (let i = topo.length - 1; i >= 0; i--) {
      topo[i]._backward();
    }
  }
}

class MSELoss {
  forward(pred, targ) {
    const diff = pred.sub(targ),
          sq = diff.mul(diff);
    
    const loss = new Tensor(
      diff.data.reduce((s, row, i) => s + row.reduce((a, c) => a + c, 0), 0) / (pred.shape[0] * pred.shape[1]),
      {requiresGrad: true}
    );
    
    loss._parents = [sq];
    loss._backward = () => {
      sq.fillGradWithZeros();
      const coef = 1 / (pred.shape[0] * pred.shape[1]);
      sq.grad = addArrays(sq.grad, map2(sq.data, sq.data, _ => coef * loss.grad));
    };
    
    return loss;
  }
}

class CrossEntropyLoss {
  forward(logits, targetIdx) {
    const m = Math.max(...logits.data);
    const exps = logits.data.map(v => Math.exp(v - m));
    const Z = exps.reduce((a, b) => a + b, 0);
    
    const loss = new Tensor(-Math.log(exps[targetIdx] / Z), {requiresGrad: true});
    
    loss._parents = [logits];
    loss._backward = () => {
      logits.fillGradWithZeros();
      logits.data.forEach((v, i) => {
        const soft = Math.exp(v - m) / Z;
        logits.grad[i] += (soft - (i === targetIdx ? 1 : 0)) * loss.grad;
      });
    };
    
    return loss;
  }
}

class Optimizer {
  static step() {
    WEIGHTS.forEach(p => {
      p.data = map2(p.data, p.grad, (w, g) => w - LR * g);
    });
  }
  
  static zero_grad() {
    WEIGHTS.forEach(p => p.grad = zeros(p.shape));
  }
}

class Linear {
  constructor(inF, outF, {bias = true, init = "xavier"} = {}) {
    this.W = new Tensor(initWeights([outF, inF], init), {requiresGrad: true});
    this.b = bias ? new Tensor(initWeights([outF], "default"), {requiresGrad: true}) : null;
  }
  
  forward(x) {
    let y = x.matmul(this.W.transpose());
    if (this.b) y = y.add(this.b);
    return y;
  }
}
///start of testing

const newArray = [[2, 4], [1, 3]];
const newArray2 = [[[4, 6, 2, 1], [1, 2, 3, 4]], [[9, 8, 1, 1], [1, 2, 3, 4]], [[9, 8, 1, 1], [1, 2, 3, 4]]];
const newArray3 = [[[4, 5, 6, 7], [1, 2, 3, 4]]];
const newArray3Shape = inferShape(newArray3);
const nArr1 = [[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], [[[13, 14], [15, 16], [17, 18]], [[19, 20], [21, 22], [23, 24]]], [[[25, 26], [27, 28], [29, 30]], [[31, 32], [33, 34], [35, 36]]]];
const nArr2 = [[[41, 42], [43, 44], [45, 46]], [[47, 48], [49, 50], [51, 52]]];
const nArr3 = [[[60], [61], [62]], [[63], [64], [65]], [[66], [67], [68]], [[69], [70], [71]]];
const nArr4 = [[[80, 81], [82, 83], [84, 85]], [[86, 87], [88, 89], [90, 91]], [[92, 93], [94, 95], [96, 97]]];

console.log('nArr1 shape:', inferShape(nArr1));
console.log('nArr2 shape:', inferShape(nArr2));
console.log('nArr3 shape:', inferShape(nArr3));
console.log('nArr4 shape:', inferShape(nArr4));
console.log('nArr1 + nArr2:', map2(nArr1, nArr2, (x, y) => x + y));

let padArr3 = padShape(newArray3Shape, 4);
console.log('newArray3 padded:', padArr3);
console.log('arr shape:', inferShape(newArray3));
console.log("Recursively checking newArray3:");
isArrayRecursive(newArray3);

const newArray4 = [[4, 5, 6, 7], [1, 2, 3, 4, 4]];
console.log("diff length", newArray.length);
console.log("dims compatible", checkDimsCompatible(
  newArray2, 
  newArray3, 
  inferShape(newArray2), 
  inferShape(newArray3), 
  inferShape(newArray2).length, 
  inferShape(newArray3).length, 
  Math.min(inferShape(newArray2).length, inferShape(newArray3).length)
));

console.log("Input 1 to checkDimsCompatible:", newArray2);
console.log("Input 2 to checkDimsCompatible:", newArray3);
console.log("Input 3 to checkDimsCompatible (inferShape(newArray2)):", inferShape(newArray2));
console.log("Input 4 to checkDimsCompatible (inferShape(newArray3)):", inferShape(newArray3));
console.log("Input 5 to checkDimsCompatible (Math.min(...)):", Math.min(inferShape(newArray2).length, inferShape(newArray3).length));
console.log("Input 6 to checkDimsCompatible (inferShape(newArray2).length):", inferShape(newArray2).length);
console.log("Input 7 to checkDimsCompatible (inferShape(newArray3).length):", inferShape(newArray3).length);
console.log("addForward:", Tensor.addForward(new Tensor([[2, 1], [1, 3], [2, 4]]), new Tensor([2, 4])));
console.log("matmul:", new Tensor([[2, 1]]).matmul(new Tensor([[2, 4], [1, 3]])));
console.log("matmul:", new Tensor([[2, 1], [4, 5]]).matmul(new Tensor([[2, 4], [1, 3]])));
console.log("divForward:", Tensor.divForward(new Tensor([[2, 1], [4, 5]]), new Tensor([[2, 4], [1, 3]])));
console.log("divForward:", Tensor.divForward(new Tensor(nArr1), new Tensor(nArr2)));

let l1 = new Tensor([[5, 6], [3, 4]]);
let l2 = new Tensor([[1, 2], [3, 4]]);
let l3 = new Tensor([[1, 2], [2, 1]]);
let l4 = new Tensor([3]);
let in1 = l2.add(l1);
let in2 = l3.add(in1);
let in3 = l4.mul(in2);
console.log("in1:", in1);
console.log("in2:", in2);
console.log("in3:", in3);
in3.backward();
console.log("in3.grad:", in3.grad);

if (typeof module !== "undefined") module.exports = {
  Tensor,
  Linear,
  MSELoss,
  CrossEntropyLoss,
  Optimizer,
  WEIGHTS,
  setLR: v => LR = v
};

const vec1 = [1, 2];
const vec2 = [10, 20, 30];
const vec3 = [5];

const mat1 = [[1, 2], [3, 4]];
const mat2 = [[10, 20], [30, 40], [50, 60]];
const mat3 = [[100, 200, 300], [400, 500, 600]];
const mat4 = [[1, 2, 3]];
const mat5 = [[1], [2]];

const batch1 = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
const batch2 = [[[10, 20, 30], [40, 50, 60]]];
const batch3 = [[[1, 2]], [[3, 4]], [[5, 6]]];
const batch4 = [[[1], [2]], [[3], [4]]];

const tensor4d1 = [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]];
const tensor4d2 = [[[[1]]], [[[2]]]];

console.log("===== SHAPE VERIFICATION =====");
console.log("vec1 shape:", inferShape(vec1));
console.log("mat1 shape:", inferShape(mat1));
console.log("batch1 shape:", inferShape(batch1));
console.log("tensor4d1 shape:", inferShape(tensor4d1));

console.log("\n===== ELEMENT-WISE OPERATIONS =====");
console.log("\n-- Same shape addition --");
console.log("mat1 + mat1:", map2(mat1, mat1, (a, b) => a + b));

console.log("\n-- Broadcasting: 2D to 3D --");
console.log("batch1 + mat1:", map2(batch1, mat1, (a, b) => a + b));

console.log("\n-- Matrix-vector multiplication (2D×1D) --");
console.log("mat2 × vec1:", new Tensor(mat2).matmul(new Tensor(vec1)));

console.log("\n-- 1D vector broadcast to matrix --");
console.log("mat1 + vec1:", map2(mat1, vec1, (a, b) => a + b));

console.log("\n-- Broadcasting to 4D tensor --");
console.log("tensor4d1 + vec1:", map2(tensor4d1, vec1, (a, b) => a + b));

console.log("\n-- Scalar-like broadcasting --");
console.log("batch1 + vec3:", map2(batch1, vec3, (a, b) => a + b));

console.log("\n-- Mixed operations --");
console.log("mat1 * vec1 + mat5:", map2(map2(mat1, vec1, (a, b) => a * b), mat5, (a, b) => a + b));

console.log("\n===== MATRIX MULTIPLICATION =====");
console.log("\n-- Basic matrix multiplication (2D×2D) --");
console.log("mat1 × mat3:", new Tensor(mat1).matmul(new Tensor(mat3)));

console.log("\n-- Vector-matrix multiplication (1D×2D) --");
console.log("vec1 × mat3:", new Tensor(vec1).matmul(new Tensor(mat3)));

console.log("\n-- Matrix-vector multiplication (2D×1D) --");
console.log("mat2 × vec1:", new Tensor(mat2).matmul(new Tensor(vec1)));

console.log("\n-- Batch matrix multiplication (3D×2D) --");
console.log("batch1 × mat3:", new Tensor(batch1).matmul(new Tensor(mat3)));

console.log("\n-- Higher dimension multiplication (4D×3D) --");
console.log("tensor4d1 × batch4:", new Tensor(tensor4d1).matmul(new Tensor(batch4)));

console.log("\n-- Extreme broadcasting in matmul (4D×2D) --");
console.log("tensor4d1 × mat5:", new Tensor(tensor4d1).matmul(new Tensor(mat5)));

console.log("\n-- Single element dimensions --");
console.log("tensor4d2 × vec3:", new Tensor(tensor4d2).matmul(new Tensor(vec3)));

console.log("\n===== EDGE CASES =====");
console.log("\n-- Incompatible dimensions --");
try {
  console.log("mat1 × mat2:", new Tensor(mat1).matmul(new Tensor(mat2)));
} catch (e) {
  console.log("Correctly caught error:", e.message);
}

console.log("\n-- Extreme broadcasting --");
console.log("tensor4d1 + vec3:", map2(tensor4d1, vec3, (a, b) => a + b));
console.log("tensor4d1.transpose():", new Tensor(tensor4d1).transpose());
