const canvas = document.getElementById('board');
const ctx = canvas.getContext('2d');
const overlay = document.getElementById('overlay');
const overlayCtx = overlay.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const brushSize = document.getElementById('brushSize');
const expression = document.getElementById('expression');
const result = document.getElementById('result');
const eqCount = document.getElementById('eqCount');
const raw = document.getElementById('raw');

let drawing = false;
let lastX = 0;
let lastY = 0;
let solveTimer = null;
const solveDelayMs = 350;
const HAND_FONT = "'Patrick Hand', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";

function setupCanvas() {
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'black';
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
}

function drawDot(x, y) {
  const r = Number(brushSize.value);
  ctx.beginPath();
  ctx.arc(x, y, r / 2, 0, Math.PI * 2);
  ctx.fillStyle = 'black';
  ctx.fill();
}

function getCanvasPos(evt) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = (evt.clientX - rect.left) * scaleX;
  const y = (evt.clientY - rect.top) * scaleY;
  return { x, y };
}

function startDraw(evt) {
  drawing = true;
  const pos = getCanvasPos(evt);
  lastX = pos.x;
  lastY = pos.y;
  drawDot(lastX, lastY);
}

function moveDraw(evt) {
  if (!drawing) return;
  const pos = getCanvasPos(evt);
  const x = pos.x;
  const y = pos.y;
  ctx.lineWidth = Number(brushSize.value);
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.stroke();
  lastX = x;
  lastY = y;
}

function endDraw() {
  drawing = false;
  scheduleAutoSolve();
}

function clearCanvas() {
  setupCanvas();
  setOutputList(expression, ['-']);
  setOutputList(result, ['-']);
  eqCount.textContent = '0';
  raw.textContent = '';
  if (solveTimer) {
    clearTimeout(solveTimer);
    solveTimer = null;
  }
}

function scheduleAutoSolve() {
  if (solveTimer) {
    clearTimeout(solveTimer);
  }
  solveTimer = setTimeout(() => {
    solveTimer = null;
    solve({ showDebug: false });
  }, solveDelayMs);
}

function renderOverlay(data) {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  if (!data || !Array.isArray(data.equations)) return;

  overlayCtx.fillStyle = '#1e4bd6';
  for (const eq of data.equations) {
    if (!eq || !Array.isArray(eq.symbols)) continue;
    const eqSymbol = eq.symbols.find((s) => s.label === '=');
    if (!eqSymbol) continue;

    let resultText = null;
    if (eq.result && eq.result.status === 'ok') {
      if (typeof eq.result.value === 'string') {
        resultText = eq.result.value;
      } else if (typeof eq.result.left === 'string') {
        resultText = eq.result.left;
      }
    }
    if (!resultText) continue;

    const box = eqSymbol.box;
    if (!Array.isArray(box) || box.length !== 4) continue;
    const x1 = box[2];
    const y0 = box[1];
    const y1 = box[3];
    const eqHeight = Math.max(16, y1 - y0);
    const symbolBoxes = (eq.symbols || []).filter(
      (s) => Array.isArray(s.box) && s.box.length === 4
    );
    const digitHeights = symbolBoxes
      .filter((s) => typeof s.label === 'string' && /^[0-9]$/.test(s.label))
      .map((s) => Math.max(1, s.box[3] - s.box[1]));
    const otherHeights = symbolBoxes
      .filter((s) => s.label !== '=')
      .map((s) => Math.max(1, s.box[3] - s.box[1]));
    const maxHeight = (arr) => (arr.length ? Math.max(...arr) : null);
    const baseHeight = maxHeight(digitHeights) ?? maxHeight(otherHeights) ?? eqHeight;
    const targetHeight = Math.max(18, Math.round(baseHeight * 0.82));
    let fontSize = Math.max(18, Math.round(targetHeight));
    overlayCtx.font = `bold ${fontSize}px ${HAND_FONT}`;
    const metrics = overlayCtx.measureText(resultText);
    const actualHeight =
      (metrics.actualBoundingBoxAscent || 0) +
      (metrics.actualBoundingBoxDescent || 0);
    if (actualHeight > 0) {
      fontSize = Math.max(18, Math.round(fontSize * (targetHeight / actualHeight)));
      overlayCtx.font = `bold ${fontSize}px ${HAND_FONT}`;
    }
    overlayCtx.textBaseline = 'middle';
    overlayCtx.fillText(resultText, x1 + 10, (y0 + y1) / 2);
  }
}

function formatEquationResult(eq) {
  if (!eq || !eq.result) return '-';
  if (eq.result.status === 'ok') {
    if (typeof eq.result.value === 'string') {
      return eq.result.value;
    }
    if (eq.result.match === true) {
      return eq.result.left + ' = ' + eq.result.right + ' (OK)';
    }
    if (eq.result.match === false) {
      return eq.result.left + ' = ' + eq.result.right + ' (nesúlad)';
    }
    if (typeof eq.result.left === 'string' && typeof eq.result.right === 'string') {
      return eq.result.left + ' = ' + eq.result.right;
    }
  }
  if (eq.result.message) {
    return eq.result.message;
  }
  return '-';
}

function formatEquationExpression(eq) {
  return (eq && eq.expression) || '-';
}

function setOutputList(container, items) {
  container.innerHTML = '';
  if (!items || items.length === 0) {
    container.textContent = '-';
    return;
  }
  for (const item of items) {
    const line = document.createElement('div');
    line.textContent = item;
    container.appendChild(line);
  }
}

async function solve(options = {}) {
  const showDebug = options.showDebug === true;
  if (showDebug) {
    raw.textContent = '';
  }
  const blob = await new Promise((resolve) =>
    canvas.toBlob(resolve, 'image/png')
  );
  if (!blob) return;
  const form = new FormData();
  form.append('file', blob, 'canvas.png');

  const response = await fetch('/solve', { method: 'POST', body: form });
  if (!response.ok) {
    const err = await response.text();
    setOutputList(expression, ['-']);
    setOutputList(result, ['Chyba']);
    if (showDebug) {
      raw.textContent = err;
    }
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
    return;
  }

  const data = await response.json();
  eqCount.textContent = String(data.equation_count || 0);
  if (!data.equations || data.equations.length === 0) {
    setOutputList(expression, ['(nič nezistené)']);
    setOutputList(result, ['-']);
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
    return;
  }

  const equations = data.equations;
  if (equations.length === 1) {
    const first = equations[0];
    setOutputList(expression, [formatEquationExpression(first)]);
    setOutputList(result, [formatEquationResult(first)]);
  } else {
    setOutputList(
      expression,
      equations.map((eq, idx) => `${idx + 1}) ${formatEquationExpression(eq)}`)
    );
    setOutputList(
      result,
      equations.map((eq, idx) => `${idx + 1}) ${formatEquationResult(eq)}`)
    );
  }

  if (showDebug) {
    raw.textContent = JSON.stringify(data, null, 2);
  }
  renderOverlay(data);
}

canvas.addEventListener('pointerdown', startDraw);
canvas.addEventListener('pointermove', moveDraw);
canvas.addEventListener('pointerup', endDraw);
canvas.addEventListener('pointerleave', endDraw);
clearBtn.addEventListener('click', clearCanvas);

setupCanvas();

if (document.fonts && document.fonts.load) {
  document.fonts.load(`32px ${HAND_FONT}`);
}
