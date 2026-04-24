// ======== Drag and Drop Upload ========
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const uploadForm = document.getElementById("uploadForm");
const fileNameDisplay = document.getElementById("fileName");
const dropText = document.getElementById("dropText");
const spinner = document.getElementById("spinner");
const toast = document.getElementById("toast");

// Click to select file
dropZone.addEventListener("click", () => fileInput.click());

// File selected
fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) {
    const name = fileInput.files[0].name;
    dropZone.classList.add("selected");
    fileNameDisplay.textContent = `📁 ${name}`;
    dropText.textContent = "File selected:";
  }
});

// Drag events
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  fileInput.files = e.dataTransfer.files;
  if (fileInput.files.length > 0) {
    const name = fileInput.files[0].name;
    dropZone.classList.add("selected");
    fileNameDisplay.textContent = `📁 ${name}`;
    dropText.textContent = "File selected:";
  }
});


// ======== Loading Spinner and Toast ========
uploadForm.addEventListener("submit", () => {
  spinner.style.display = "block";
});

window.addEventListener("load", () => {
  const params = new URLSearchParams(window.location.search);
  if (params.has("success")) {
    showToast();
  }
});

function showToast() {
  spinner.style.display = "none";
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 4000);
}

function solveCircuit() {
  const input = document.getElementById("netlistInput").value;
  const result = nodalAnalysis(input);
  document.getElementById("output").textContent = result;
}

function nodalAnalysis(netlist) {
  const lines = netlist
    .split('\n')
    .map(l => l.trim())
    .filter(l => l && !l.startsWith('*') && !l.startsWith('.') && !l.startsWith('//'));

  const components = [];
  const nodeSet = new Set();

  for (const line of lines) {
    // Split by whitespace, but voltage source may have "DC" before value
    const parts = line.split(/\s+/); 
    const name = parts[0];
    const type = name[0].toUpperCase();

    if (!['R', 'V', 'I', 'C', 'L'].includes(type)) continue;

    const n1 = parts[1];
    const n2 = parts[2];

    // For voltage source, value can be 4th or 5th token (handle 'DC' keyword)
    let valueStr = parts[3];
    if (type === 'V' && parts[3] && parts[3].toUpperCase() === 'DC') {
      valueStr = parts[4];
    }

    const value = parseFloat(valueStr);
    if (isNaN(value)) {
      return `Error parsing value for component ${name}`;
    }

    components.push({ name, type, n1, n2, value });
    nodeSet.add(n1);
    nodeSet.add(n2);
  }

  // Ground node '0' is index -1, others indexed from 0
  nodeSet.delete('0');
  const nodes = Array.from(nodeSet);
  const nodeIndex = { '0': -1 };
  nodes.forEach((node, i) => nodeIndex[node] = i);

  const N = nodes.length;
  const Vsources = components.filter(c => c.type === 'V');
  const M = Vsources.length;
  const size = N + M;

  // Initialize conductance matrix G and current vector I
  const G = Array.from({ length: size }, () => Array(size).fill(0));
  const I = Array(size).fill(0);

  let vsCount = 0;
  const vsOffset = N;

  for (const comp of components) {
    const i = nodeIndex[comp.n1];
    const j = nodeIndex[comp.n2];

    switch (comp.type) {
      case 'R': {
        const g = 1 / comp.value;
        if (i >= 0) G[i][i] += g;
        if (j >= 0) G[j][j] += g;
        if (i >= 0 && j >= 0) {
          G[i][j] -= g;
          G[j][i] -= g;
        }
        break;
      }
      case 'C':
        // Capacitor open circuit in DC → skip
        break;
      case 'L': {
        // Inductor short circuit in DC → very large conductance
        const g = 1e9;
        if (i >= 0) G[i][i] += g;
        if (j >= 0) G[j][j] += g;
        if (i >= 0 && j >= 0) {
          G[i][j] -= g;
          G[j][i] -= g;
        }
        break;
      }
      case 'I': {
        if (i >= 0) I[i] -= comp.value;
        if (j >= 0) I[j] += comp.value;
        break;
      }
      case 'V': {
        const k = vsOffset + vsCount;
        if (i >= 0) {
          G[i][k] += 1;
          G[k][i] += 1;
        }
        if (j >= 0) {
          G[j][k] -= 1;
          G[k][j] -= 1;
        }
        I[k] = comp.value;
        vsCount++;
        break;
      }
    }
  }

  // Solve G * x = I
  const x = solveLinearSystem(G, I);
  if (!x) return "Error: Matrix singular or circuit invalid.";

  // Output voltages and currents
  const out = [];
  nodes.forEach((node, idx) => {
    voltage_value = Math.abs(x[idx].toFixed(6))
    out.push(`V(${node}) = ${voltage_value} V`);
  });
  Vsources.forEach((v, idx) => {
    current_value = Math.abs(x[N + idx].toExponential(6))
    out.push(`I(${v.name}) = ${current_value} A`);
  });

  return out.join('\n');
}

// Gaussian elimination solver
function solveLinearSystem(A, b) {
  const n = A.length;
  const x = Array(n).fill(0);

  for (let i = 0; i < n; i++) {
    // Partial pivot
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(A[k][i]) > Math.abs(A[maxRow][i])) maxRow = k;
    }

    if (Math.abs(A[maxRow][i]) < 1e-15) return null; // Singular matrix

    [A[i], A[maxRow]] = [A[maxRow], A[i]];
    [b[i], b[maxRow]] = [b[maxRow], b[i]];

    for (let k = i + 1; k < n; k++) {
      const factor = A[k][i] / A[i][i];
      for (let j = i; j < n; j++) {
        A[k][j] -= factor * A[i][j];
      }
      b[k] -= factor * b[i];
    }
  }

  for (let i = n - 1; i >= 0; i--) {
    x[i] = b[i];
    for (let j = i + 1; j < n; j++) {
      x[i] -= A[i][j] * x[j];
    }
    x[i] /= A[i][i];
  }

  return x;
}

const nodeCoords = {
  '0': { x: 100, y: 400 },
  '1': { x: 300, y: 250 },
  '2': { x: 500, y: 250 },
  '3': { x: 300, y: 100 }
};

function drawCircuit() {
  const canvas = document.getElementById("circuitCanvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const lines = document.getElementById("netlistInput").value
    .split("\n")
    .map(line => line.trim())
    .filter(line => line && !line.startsWith("*") && !line.startsWith("."));

  lines.forEach(line => {
    const tokens = line.split(/\s+/);
    const name = tokens[0];
    const type = name[0].toUpperCase();
    const n1 = tokens[1];
    const n2 = tokens[2];
    const value = tokens.slice(3).join(" ");

    const p1 = nodeCoords[n1];
    const p2 = nodeCoords[n2];
    if (!p1 || !p2) return;

    drawWire(ctx, p1, p2);
    drawComponent(ctx, type, name, value, p1, p2);
  });

  drawNodes(ctx);
}

function drawWire(ctx, p1, p2) {
  ctx.beginPath();
  ctx.moveTo(p1.x, p1.y);
  ctx.lineTo(p2.x, p2.y);
  ctx.strokeStyle = "black";
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawComponent(ctx, type, label, value, p1, p2) {
  const midX = (p1.x + p2.x) / 2;
  const midY = (p1.y + p2.y) / 2;
  const angle = Math.atan2(p2.y - p1.y, p2.x - p1.x);

  ctx.save();
  ctx.translate(midX, midY);
  ctx.rotate(angle);

  ctx.fillStyle = "black";
  ctx.font = "12px monospace";

  switch (type) {
    case 'R':
      ctx.strokeRect(-20, -10, 40, 20);
      break;
    case 'C':
      ctx.beginPath();
      ctx.moveTo(-10, -15);
      ctx.lineTo(-10, 15);
      ctx.moveTo(10, -15);
      ctx.lineTo(10, 15);
      ctx.stroke();
      break;
    case 'L':
      ctx.beginPath();
      for (let i = -20; i <= 20; i += 10) {
        ctx.arc(i, 0, 5, 0, Math.PI);
      }
      ctx.stroke();
      break;
    case 'V':
      ctx.beginPath();
      ctx.arc(0, 0, 15, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, -8);
      ctx.lineTo(0, 8);
      ctx.moveTo(-8, 0);
      ctx.lineTo(8, 0);
      ctx.stroke();
      break;
    default:
      ctx.fillText("?", -5, 5);
  }

  ctx.fillText(label, -25, -20);
  ctx.fillText(value, -25, 30);

  ctx.restore();
}

function drawNodes(ctx) {
  for (const [node, pos] of Object.entries(nodeCoords)) {
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, 4, 0, 2 * Math.PI);
    ctx.fillStyle = "blue";
    ctx.fill();
    ctx.fillStyle = "black";
    ctx.font = "12px sans-serif";
    ctx.fillText(`Node ${node}`, pos.x + 5, pos.y - 5);
  }
}