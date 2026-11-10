  const arrowHead = (svg, id, color) => {
    const d = svg.querySelector('defs') || svg.insertBefore(el('defs'), svg.firstChild);
    const m = el('marker', { id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' });
    m.appendChild(el('path', { d: 'M0,0 L10,5 L0,10 z', fill: color })); d.appendChild(m);
    return `url(#${id})`;
  };
  const GRAY = '#9a9a9a', BLUE = '#2F6FD0', GOLD = '#C98A1B', RED = '#B03A3A', GREEN = '#2E7D4F', INK = '#1A1A1A';
  const LBLUE = '#C9DBF4', LGOLD = '#F6EAD2', LRED = '#F6DADA', LGREEN = '#D6EBDC';
  const syncButtons = (root, attr, val) => root.querySelectorAll(`[data-${attr}]`).forEach(b => b.classList.toggle('on', b.dataset[attr] === val));
  const grid = (svg, x0, y0, n, m, cs, fillFn, opts = {}) => {
    const cells = [];
    for (let r = 0; r < n; r++) for (let c = 0; c < m; c++) cells.push(R(svg, x0 + c * cs, y0 + r * cs, cs - (opts.gap ?? 1), cs - (opts.gap ?? 1), { fill: fillFn ? fillFn(r, c) : '#fff', stroke: opts.stroke || '#d0d0d0', 'stroke-width': .6 }));
    return cells;
  };
  const tileLines = (svg, x0, y0, n, cs, every) => { for (let k = 0; k <= n; k += every) { L(svg, x0 + k * cs, y0, x0 + k * cs, y0 + n * cs, { stroke: INK, 'stroke-width': 1.6 }); L(svg, x0, y0 + k * cs, x0 + n * cs, y0 + k * cs, { stroke: INK, 'stroke-width': 1.6 }); } };

  // ------------------------------------------------------------ thread → output element → input patch
  ANIM.mapping = (root) => {
    const svg = newSvg(root, '0 0 1000 460', 'A 16 by 16 output computed by 4 by 4 blocks of 4 by 4 threads; each thread reads the input patch centred on its output element');
    const N = 16, cs = 22, xi = 70, xo = 590, y0 = 70, r = 2;
    T(svg, xi - 50, 20, 'input N (global memory)', { 'font-size': 13, class: 'lbl' }); T(svg, xo, 20, 'output P', { 'font-size': 13, class: 'lbl' });
    const inC = grid(svg, xi, y0, N, N, cs), outC = grid(svg, xo, y0, N, N, cs);
    tileLines(svg, xi, y0, N, cs, 4); tileLines(svg, xo, y0, N, cs, 4);
    const ghost = svg.appendChild(el('g'));
    const hA = arrowHead(svg, 'mp-a', BLUE);
    const arrow = L(svg, 0, 0, 0, 0, { stroke: BLUE, 'stroke-width': 2.5, 'marker-end': hA });
    const info = T(svg, 20, 452, '', { 'font-size': 14 });
    const S = [[1, 1, 1, 1], [0, 0, 0, 0], [3, 2, 3, 1], [1, 3, 2, 3]];   // blockY, blockX, threadY, threadX
    const cap = root.querySelector('[data-cap]');
    return stepper(root, () => S.length, (k) => {
      const [by, bx, ty, tx] = S[k], row = by * 4 + ty, col = bx * 4 + tx;
      inC.forEach((e, i) => { const rr = Math.floor(i / N), cc = i % N; const inP = Math.abs(rr - row) <= r && Math.abs(cc - col) <= r; e.setAttribute('fill', rr === row && cc === col ? BLUE : inP ? '#9DB7E3' : '#fff'); });
      outC.forEach((e, i) => { const rr = Math.floor(i / N), cc = i % N; e.setAttribute('fill', rr === row && cc === col ? GREEN : (Math.floor(rr / 4) === by && Math.floor(cc / 4) === bx ? LGREEN : '#fff')); });
      ghost.innerHTML = '';
      let g = 0;
      for (let dr = -r; dr <= r; dr++) for (let dc = -r; dc <= r; dc++) {
        const rr = row + dr, cc = col + dc;
        if (rr < 0 || cc < 0 || rr >= N || cc >= N) { g++; ghost.appendChild(el('rect', { x: xi + cc * cs, y: y0 + rr * cs, width: cs - 1, height: cs - 1, fill: LRED, stroke: RED, 'stroke-dasharray': '3 2' })); }
      }
      arrow.setAttribute('x1', xi + col * cs + cs / 2); arrow.setAttribute('y1', y0 + row * cs + cs / 2);
      arrow.setAttribute('x2', xo + col * cs + cs / 2 - 6); arrow.setAttribute('y2', y0 + row * cs + cs / 2);
      info.textContent = `block (${bx},${by}), thread (${tx},${ty}) → P[${row}][${col}] · reads rows ${row - r}…${row + r}, cols ${col - r}…${col + r}${g ? ` · ${g} ghost cells skipped` : ''}`;
      cap.innerHTML = k === 0 ? 'The same mapping as the grayscale kernel of session 11: <code>outCol = blockIdx.x*blockDim.x + threadIdx.x</code>, and the same for rows. Each thread owns <b>one output element</b>.'
        : k === 1 ? 'Thread (0,0) of block (0,0): its patch hangs over the top-left corner. The missing input elements are <b>ghost cells</b>; with value 0 the thread simply skips them.'
        : k === 2 ? 'The patch always starts at <code>(outRow − r, outCol − r)</code>: two nested loops over the (2r + 1)² filter positions give the input element of each multiplication.'
        : 'Neighbouring threads read overlapping patches: every input element is read by up to (2r + 1)² threads. That reuse is what tiling will exploit.';
    }, 2800);
  };

  // ------------------------------------------------------------ ghost cells and divergence
  ANIM.ghost = (root) => {
    const svg = newSvg(root, '0 0 540 320', 'Output elements coloured by how many filter positions fall on ghost cells');
    const q = (n) => root.querySelector(`[data-in=${n}]`), out = root.querySelector('[data-out]');
    const draw = () => {
      svg.innerHTML = '';
      const W = Math.max(4, Math.min(64, +q('w').value | 0)), H = Math.max(4, Math.min(40, +q('h').value | 0)), r = Math.max(1, Math.min(7, +q('r').value | 0));
      const cs = Math.min(20, Math.floor(500 / W), Math.floor(270 / H));
      const m = 2 * r + 1, valid = (i, n) => Math.min(i + r, n - 1) - Math.max(i - r, 0) + 1;
      let edge = 0, skipped = 0;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const v = valid(x, W) * valid(y, H), miss = m * m - v; skipped += miss; if (miss) edge++;
        const a = Math.min(1, miss / (m * m) * 2.2);
        R(svg, 20 + x * cs, 20 + y * cs, cs - 1, cs - 1, { fill: miss ? `rgba(176,58,58,${.15 + .85 * a})` : LBLUE });
      }
      T(svg, 20, 20 + H * cs + 18, 'blue: no ghost cells · red: skips some positions (darker = more)', { 'font-size': 13, class: 'lbl' });
      out.innerHTML = `${W} × ${H} output, ${m} × ${m} filter: <b>${edge}</b> of ${W * H} elements (${(edge / (W * H) * 100).toFixed(1)} %) touch ghost cells · skipped multiplications: ${skipped} of ${W * H * m * m} (${(skipped / (W * H * m * m) * 100).toFixed(1)} %)` +
        `<br>a 4096 × 4096 image with this filter: ${(100 * (1 - (4096 - 2 * r) ** 2 / 4096 ** 2)).toFixed(2)} % of the elements touch ghost cells`;
    };
    root.querySelectorAll('input').forEach(i => i.addEventListener('input', draw));
    return { enter: draw, leave() {} };
  };

  // ------------------------------------------------------------ arithmetic intensity of the kernels
  ANIM.ai = (root) => {
    const svg = newSvg(root, '0 0 1000 300', 'Arithmetic intensity of four convolution kernels against the H100 ridge of 20 FLOP per byte');
    const q = (n) => root.querySelector(`[data-in=${n}]`), out = root.querySelector('[data-out]');
    const tiled = (t, m) => (t * t * m * m * 2) / (((t + m - 1) ** 2 + t * t) * 4);
    const draw = () => {
      svg.innerHTML = '';
      const m = Math.max(1, (+q('m').value | 0)), t = Math.max(1, +q('t').value | 0);
      const rows = [['basic kernel', .25, '2 FLOP per 8 B: N and F from global memory'], ['constant memory for F', .5, '2 FLOP per 4 B: only N from global'], [`tiled, ${t} × ${t} output tile`, tiled(t, m), 'each input loaded once per block'], ['ideal (each element once)', m * m / 4, 'm² / 4']];
      const max = Math.max(25, ...rows.map(r => r[1])) * 1.05, sc = 640 / max, X0 = 260;
      rows.forEach(([name, v, note], i) => {
        const y = 30 + i * 60;
        T(svg, 20, y + 22, name, { 'font-size': 13, 'font-weight': 600 }); T(svg, 20, y + 40, note, { 'font-size': 13, class: 'lbl' });
        R(svg, X0, y, Math.max(3, v * sc), 30, { fill: v < 20 ? '#E7B467' : '#86C495', rx: 2 });
        T(svg, X0 + Math.max(3, v * sc) + 8, y + 21, `${v.toFixed(2)} FLOP/B`, { 'font-size': 13 });
      });
      const rx = X0 + 20 * sc;
      L(svg, rx, 18, rx, 270, { stroke: RED, 'stroke-width': 2, 'stroke-dasharray': '6 4' }); T(svg, rx + 6, 286, 'H100 ridge: 20 FLOP/B', { 'font-size': 13, fill: RED });
      out.innerHTML = `${m} × ${m} filter (r = ${(m - 1) / 2}): the basic kernel reaches at most 3.35 TB/s × 0.25 = <b>0.84 TFLOPS</b>; the tiled one ${(3.35 * Math.min(20, tiled(t, m))).toFixed(1)} TFLOPS` +
        ` (${tiled(t, m) < 20 ? 'still memory-bound' : 'compute-bound: up to 66.9'}).`;
      syncButtons(root, 'p', `${m},${t}`);
    };
    root.querySelectorAll('input').forEach(i => i.addEventListener('input', draw));
    root.querySelectorAll('[data-p]').forEach(b => b.addEventListener('click', () => { const [m, t] = b.dataset.p.split(','); q('m').value = m; q('t').value = t; draw(); }));
    return { enter: draw, leave() {} };
  };

  // ------------------------------------------------------------ constant cache broadcast
  ANIM.bcast = (root) => {
    const svg = newSvg(root, '0 0 1000 330', 'Eight threads of a warp each compute their own output; in every loop iteration they all read the same filter element, served by one broadcast from the constant cache');
    const hB = arrowHead(svg, 'bc-b', BLUE), hG = arrowHead(svg, 'bc-g', GOLD);
    T(svg, 20, 26, 'F in constant memory (3 × 3)', { 'font-size': 13, class: 'lbl' });
    const fc = grid(svg, 20, 36, 3, 3, 40, null, { stroke: '#9a9a9a' });
    const fT = fc.map((_, i) => T(svg, 40 + (i % 3) * 40, 61 + Math.floor(i / 3) * 40, `F${Math.floor(i / 3)}${i % 3}`, { ...mid, 'font-size': 13 }));
    const cc = R(svg, 20, 170, 120, 44, { class: 'mem', rx: 3 }); T(svg, 80, 197, 'constant cache', { ...mid, 'font-size': 13 });
    T(svg, 300, 26, 'input row of N (global memory, L1/L2)', { 'font-size': 13, class: 'lbl' });
    const nRow = Array.from({ length: 10 }, (_, i) => { const e = R(svg, 300 + i * 62, 36, 58, 34, { fill: '#fff', stroke: '#9a9a9a' }); T(svg, 329 + i * 62, 58, `N[${i}]`, { ...mid, 'font-size': 13 }); return e; });
    const lanes = Array.from({ length: 8 }, (_, t) => { const e = R(svg, 362 + t * 62, 240, 58, 34, { fill: LBLUE, rx: 3 }); T(svg, 391 + t * 62, 262, `t${t}`, { ...mid, 'font-size': 13 }); return e; });
    const lines = svg.appendChild(el('g'));
    const cnt = T(svg, 20, 316, '', { 'font-size': 14 });
    const cap = root.querySelector('[data-cap]');
    return stepper(root, () => 3, (k) => {
      lines.innerHTML = '';
      fc.forEach((e, i) => e.setAttribute('fill', i === 3 + k ? BLUE : '#fff')); fT.forEach((e, i) => e.setAttribute('fill', i === 3 + k ? '#fff' : INK));
      nRow.forEach((e, i) => e.setAttribute('fill', i >= k && i < k + 8 ? '#9DB7E3' : '#fff'));
      lines.appendChild(el('line', { x1: 140, y1: 192, x2: 356, y2: 230, stroke: BLUE, 'stroke-width': 3, 'marker-end': hB }));
      lines.appendChild(el('rect', { x: 362, y: 226, width: 8 * 62 - 4, height: 6, fill: BLUE, rx: 2 }));
      for (let t = 0; t < 8; t++) lines.appendChild(el('line', { x1: 329 + (t + k) * 62, y1: 72, x2: 391 + t * 62, y2: 238, stroke: GOLD, 'stroke-width': 1.5, opacity: .8, 'marker-end': hG }));
      cnt.textContent = `fCol = ${k}: 1 constant-cache read, broadcast to 8 threads · 8 different N elements from global/L1`;
      cap.innerHTML = ['Each thread computes a different output, but in the first inner iteration <b>all</b> of them need <code>F[1][0]</code>. One read from the constant cache is broadcast to the whole warp.',
        'Next iteration: all threads move to <code>F[1][1]</code> together, while each one reads its own, shifted element of N.',
        'F is small, never changes during the kernel and is read in the same order by every thread: exactly what constant memory and its cache are built for. The N reads still go to global memory (or L1/L2).'][k];
    }, 2600);
  };

  // ------------------------------------------------------------ cache hierarchy
  ANIM.hier = (root) => {
    const svg = newSvg(root, '0 0 1000 350', 'Cores with registers, per-SM L1 and constant caches, a shared L2 cache, and DRAM; and where a load is served from');
    const hG = arrowHead(svg, 'hi-g', GOLD);
    R(svg, 10, 10, 980, 230, { class: 'frame', rx: 5 }); T(svg, 26, 32, 'the GPU chip', { 'font-size': 13, class: 'lbl' });
    const sms = [0, 1, 2].map(i => {
      const x = 40 + i * 320;
      R(svg, x, 44, 290, 110, { fill: '#fff', stroke: '#6f6f6f', rx: 4 }); T(svg, x + 10, 62, `SM ${i}`, { 'font-size': 13, class: 'lbl' });
      const core = R(svg, x + 14, 70, 80, 36, { class: 'sp', rx: 3 }); T(svg, x + 54, 93, 'regs', { ...mid, 'font-size': 13 });
      const l1 = R(svg, x + 104, 70, 170, 36, { class: 'mem', rx: 3 }); T(svg, x + 189, 93, 'L1 / shared memory', { ...mid, 'font-size': 13 });
      const cc = R(svg, x + 104, 112, 170, 32, { class: 'mem', rx: 3 }); T(svg, x + 189, 133, 'constant cache', { ...mid, 'font-size': 13 });
      return { core, l1, cc };
    });
    const l2 = R(svg, 40, 172, 930, 50, { class: 'mem', rx: 4 }); T(svg, 505, 202, 'L2 cache (shared by all SMs)', { ...mid, 'font-size': 14 });
    const dram = R(svg, 40, 264, 930, 56, { class: 'mem', rx: 4 }); T(svg, 505, 297, 'DRAM: global and constant memory', { ...mid, 'font-size': 14 });
    const path = svg.appendChild(el('g'));
    const S = [
      { on: [], cap: 'Every SM has registers, an L1 cache (whose capacity it shares with shared memory) and a small <b>constant cache</b>. All SMs share one L2. Behind it sits DRAM. Closer means faster and smaller: L1 runs near core speed but holds up to about 128 KB; L2 holds hundreds of KB to tens of MB and takes tens to hundreds of cycles.' },
      { on: ['l1'], cap: 'A load from global memory first looks in the SM\'s <b>L1</b>. If a recent load brought that data, it is served on chip.' },
      { on: ['l1', 'l2'], cap: 'If not, <b>L2</b>: maybe another SM (another block) loaded it recently. This is why the halo cells of a tile are often cheap: the neighbouring block has just read them.' },
      { on: ['l1', 'l2', 'dram'], cap: 'Only on a miss in both does the request reach <b>DRAM</b>, with its long latency and shared bandwidth.' },
      { on: ['cc'], cap: 'Constant memory lives in DRAM too, but kernels cannot write it, so its cache needs no write logic: small, cheap in area and power, and able to <b>broadcast</b> one value to a whole warp. With a filter of a few hundred bytes, practically every read of F hits it.' },
      { on: ['shared'], cap: 'Caches are <b>transparent</b>: the program uses the original address and the hardware decides what to keep. Shared memory is a <b>scratchpad</b>: the program declares <code>__shared__</code> variables and copies data in explicitly, so it controls what stays on chip.' }
    ];
    const cap = root.querySelector('[data-cap]');
    return stepper(root, () => S.length, (k) => {
      const on = S[k].on; path.innerHTML = '';
      sms.forEach((s, i) => { s.l1.classList.toggle('on', i === 0 && (on.includes('l1') || on.includes('shared'))); s.cc.classList.toggle('on', on.includes('cc')); s.core.classList.toggle('on', i === 0 && k > 0); });
      l2.classList.toggle('on', on.includes('l2')); dram.classList.toggle('on', on.includes('dram'));
      const pts = [[134, 88]]; if (on.includes('l1')) pts.push([229, 106]); if (on.includes('l2')) pts.push([229, 172]); if (on.includes('dram')) pts.push([229, 264]);
      for (let i = 0; i + 1 < pts.length; i++) path.appendChild(el('line', { x1: pts[i][0], y1: pts[i][1], x2: pts[i + 1][0], y2: pts[i + 1][1], stroke: GOLD, 'stroke-width': 3, 'marker-end': hG }));
      cap.innerHTML = S[k].cap;
    }, 3400);
  };

  // ------------------------------------------------------------ input tile vs output tile
  ANIM.tiles = (root) => {
    const svg = newSvg(root, '0 0 600 330', 'An output tile and the larger input tile, extended by the filter radius on every side');
    const q = (n) => root.querySelector(`[data-in=${n}]`), out = root.querySelector('[data-out]');
    const draw = () => {
      svg.innerHTML = '';
      const t = Math.max(1, Math.min(12, +q('t').value | 0)), r = Math.max(1, Math.min(4, +q('r').value | 0)), inT = t + 2 * r;
      const n = Math.max(inT + 4, 16), cs = Math.min(17, Math.floor(280 / n));
      const ox = 2 + r, oy = 2 + r;
      T(svg, 10, 18, 'input', { 'font-size': 13, class: 'lbl' }); T(svg, 310, 18, 'output', { 'font-size': 13, class: 'lbl' });
      grid(svg, 10, 24, n, n, cs, (rr, cc) => (rr >= oy && rr < oy + t && cc >= ox && cc < ox + t) ? BLUE : (rr >= oy - r && rr < oy + t + r && cc >= ox - r && cc < ox + t + r) ? '#9DB7E3' : '#fff');
      grid(svg, 310, 24, n, n, cs, (rr, cc) => (rr >= oy && rr < oy + t && cc >= ox && cc < ox + t) ? GREEN : '#fff');
      out.innerHTML = `output tile ${t} × ${t} = <b>${t * t}</b> elements · filter ${2 * r + 1} × ${2 * r + 1} · input tile (${t} + 2·${r})² = ${inT} × ${inT} = <b>${inT * inT}</b> elements · ratio <b>${(inT * inT / (t * t)).toFixed(2)}</b>` +
        `<br>halo cells (light blue): ${inT * inT - t * t} · with a 16 × 16 output tile and the same filter: ratio ${((16 + 2 * r) ** 2 / 256).toFixed(2)}`;
      syncButtons(root, 'p', `${t},${r}`);
    };
    root.querySelectorAll('input').forEach(i => i.addEventListener('input', draw));
    root.querySelectorAll('[data-p]').forEach(b => b.addEventListener('click', () => { const [t, r] = b.dataset.p.split(','); q('t').value = t; q('r').value = r; draw(); }));
    return { enter: draw, leave() {} };
  };

  // ------------------------------------------------------------ two thread organizations
  ANIM.orgs = (root) => {
    let mode = 'input';
    const svg = newSvg(root, '0 0 1000 320', 'Block size equal to the input tile versus equal to the output tile, in the load phase and the compute phase');
    const IN = 8, OUT = 6, cs = 30;
    const draw = (k) => {
      svg.innerHTML = '';
      const phase = k === 0 ? 'load' : 'compute';
      [0, 1].forEach(p => {
        const x0 = 40 + p * 480, ph = p === 0 ? 'load' : 'compute';
        T(svg, x0, 22, ph === 'load' ? 'phase 1: load the 8 × 8 input tile' : 'phase 2: compute the 6 × 6 output tile', { 'font-size': 13, 'font-weight': 600, fill: ph === phase ? INK : GRAY });
        for (let r = 0; r < IN; r++) for (let c = 0; c < IN; c++) {
          const inner = r >= 1 && r < 7 && c >= 1 && c < 7;
          let f = '#fff';
          if (mode === 'input') { f = ph === 'load' ? BLUE : (inner ? BLUE : '#E2E2E2'); }
          else {
            if (ph === 'load') { const idx = r * IN + c; f = idx < 36 ? BLUE : '#9DB7E3'; }
            else { f = inner ? BLUE : '#fff'; }
          }
          R(svg, x0 + c * cs, 34 + r * cs, cs - 2, cs - 2, { fill: f, rx: 2, opacity: ph === phase ? 1 : .35 });
        }
        const legend = [];
        if (mode === 'input' && ph === 'compute') legend.push(['#E2E2E2', 'idle thread']);
        if (mode === 'output' && ph === 'load') legend.push([BLUE, '1st round of loads'], ['#9DB7E3', '2nd round of loads']);
        if (ph === phase) legend.forEach(([color, text], n) => {
          R(svg, x0 + IN * cs + 14, 44 + n * 28, 18, 18, { fill: color, rx: 2 });
          T(svg, x0 + IN * cs + 40, 58 + n * 28, text, { 'font-size': 13, fill: INK });
        });
      });
      T(svg, 40, 300, mode === 'input' ? '64 threads per block: each loads one element, 36 of them compute (44 % idle in phase 2)' : '36 threads per block: they load 64 elements in two rounds (28 threads load twice), all 36 compute', { 'font-size': 13 });
      root.querySelector('[data-cap]').innerHTML = mode === 'input'
        ? (k === 0 ? '<b>Block = input tile</b> (8 × 8 threads for a 3 × 3 filter): loading is trivial, one element per thread.' : 'But only the inner 6 × 6 threads have an output element; the outer ring sits idle while the others compute. This is the organization of the kernel on the next slides.')
        : (k === 0 ? '<b>Block = output tile</b> (6 × 6 threads): the input tile has more elements than threads, so the threads loop to load it.' : 'In return every thread computes, and the block, input tile and output tile can have the same power-of-two width.');
    };
    const st = stepper(root, () => 2, draw, 2600);
    modeButtons(root, 'mode', (m) => { mode = m; st.reset(); });
    return st;
  };

  // ------------------------------------------------------------ tiled kernel walkthrough
  ANIM.tiled = (root) => {
    const svg = newSvg(root, '0 0 1000 350', 'An 8 by 8 block over an 8 by 8 input tile with a 3 by 3 filter: all threads load, the inner 6 by 6 threads compute the output tile');
    const IN = 8, OUT = 6, cs = 34, X0 = 40, Y0 = 30;
    T(svg, X0, 20, 'input tile in shared memory N_s = thread block (8 × 8)', { 'font-size': 13, class: 'lbl' });
    const cells = [];
    for (let r = 0; r < IN; r++) for (let c = 0; c < IN; c++) { cells.push({ r, c, e: R(svg, X0 + c * cs, Y0 + r * cs, cs - 2, cs - 2, { fill: '#fff', stroke: '#c9c9c9', rx: 2 }), t: T(svg, X0 + c * cs + 16, Y0 + r * cs + 21, '', { ...mid, 'font-size': 13 }) }); }
    const box = svg.appendChild(el('rect', { x: X0 + cs - 1, y: Y0 + cs - 1, width: OUT * cs, height: OUT * cs, fill: 'none', stroke: INK, 'stroke-width': 3 }));
    const patch = svg.appendChild(el('rect', { fill: 'none', stroke: GOLD, 'stroke-width': 3, 'stroke-dasharray': '6 3' }));
    T(svg, 600, 20, 'output tile (6 × 6)', { 'font-size': 13, class: 'lbl' });
    const outC = []; for (let r = 0; r < OUT; r++) for (let c = 0; c < OUT; c++) { outC.push(R(svg, 600 + c * cs, Y0 + r * cs, cs - 2, cs - 2, { fill: '#fff', stroke: '#c9c9c9', rx: 2 })); }
    const hA = arrowHead(svg, 'tl-a', GREEN); const arrow = L(svg, 0, 0, 0, 0, { stroke: GREEN, 'stroke-width': 2.5, 'marker-end': hA });
    const info = T(svg, 600, 260, '', { 'font-size': 13 }), info2 = T(svg, 600, 282, '', { 'font-size': 13 });
    const ghostCol = 0;   // this block sits at the left edge of the image: column 0 of the tile is a ghost column
    const S = [
      { cap: 'One thread per input-tile element. <code>col = blockIdx.x*OUT_TILE_DIM + threadIdx.x − r</code>: the block starts r elements before its output tile. This block is at the left edge of the image.' },
      { cap: 'Every thread loads one element into N_s. Elements outside the image (here column 0) are <b>ghost cells</b>: the thread stores 0 instead of loading.' },
      { cap: '<code>__syncthreads()</code>: no thread may start computing until the whole input tile, halo included, is in shared memory.' },
      { cap: 'Thread (1,1) has <code>tileCol = tileRow = 0</code>: it computes output element (0,0) of the tile from the 3 × 3 patch starting at N_s[0][0].', t: [1, 1] },
      { cap: 'Thread (6,6) computes output element (5,5) from the patch starting at N_s[5][5]. All patch reads now hit shared memory.', t: [6, 6] },
      { cap: 'The outer ring of threads (tileCol or tileRow outside 0…5) does no computation: they existed only to load the halo. For a 5 × 5 filter and 32 × 32 blocks, 28 × 28 compute and 23 % of the threads are idle.' }
    ];
    const cap = root.querySelector('[data-cap]');
    return stepper(root, () => S.length, (k) => {
      cells.forEach(o => {
        const ghost = o.c === ghostCol, inner = o.r >= 1 && o.r < 7 && o.c >= 1 && o.c < 7;
        let f = '#fff';
        if (k === 0) f = '#EEF3FB';
        if (k >= 1) f = ghost ? LRED : '#9DB7E3';
        if (k === 5 && !inner) f = '#E2E2E2';
        o.e.setAttribute('fill', f); o.t.textContent = k >= 1 ? (ghost ? '0' : 'N') : `${o.c},${o.r}`;
      });
      box.setAttribute('opacity', k >= 3 ? 1 : .25);
      const s = S[k];
      if (s.t) {
        const [tx, ty] = s.t; patch.setAttribute('x', X0 + (tx - 1) * cs - 2); patch.setAttribute('y', Y0 + (ty - 1) * cs - 2); patch.setAttribute('width', 3 * cs + 2); patch.setAttribute('height', 3 * cs + 2); patch.setAttribute('opacity', 1);
        outC.forEach((e, i) => e.setAttribute('fill', i === (ty - 1) * OUT + (tx - 1) ? GREEN : LGREEN));
        arrow.setAttribute('x1', X0 + tx * cs + 16); arrow.setAttribute('y1', Y0 + ty * cs + 16); arrow.setAttribute('x2', 600 + (tx - 1) * cs + 10); arrow.setAttribute('y2', Y0 + (ty - 1) * cs + 16); arrow.setAttribute('opacity', 1);
        info.textContent = `thread (${tx},${ty}): tileCol = ${tx - 1}, tileRow = ${ty - 1}`; info2.textContent = `patch N_s[${ty - 1}…${ty + 1}][${tx - 1}…${tx + 1}] → output (${tx - 1},${ty - 1})`;
      } else {
        patch.setAttribute('opacity', 0); arrow.setAttribute('opacity', 0); info.textContent = k === 2 ? '__syncthreads()' : ''; info2.textContent = '';
        outC.forEach(e => e.setAttribute('fill', k === 5 ? LGREEN : '#fff'));
      }
      cap.innerHTML = s.cap;
    }, 2800);
  };

  // ------------------------------------------------------------ arithmetic intensity table
  ANIM.aitable = (root) => {
    const out = root.querySelector('[data-out]'), inp = root.querySelector('[data-in=ridge]');
    const ts = [8, 16, 28, 32, 64], ms = [3, 5, 7, 9, 11, 13];
    const f = (t, m) => (t * t * m * m * 2) / (((t + m - 1) ** 2 + t * t) * 4);
    const draw = () => {
      const ridge = Math.max(0.1, +inp.value || 20);
      let h = '<table><tr><th>filter</th>' + ts.map(t => `<th>t = ${t}</th>`).join('') + '<th>ideal</th></tr>';
      ms.forEach(m => {
        h += `<tr><td><b>${m} × ${m}</b></td>` + ts.map(t => { const v = f(t, m); return `<td style="background:${v < ridge ? '#F6DADA' : '#D6EBDC'}">${v.toFixed(2)}</td>`; }).join('');
        const id = m * m / 4; h += `<td style="background:${id < ridge ? '#F6DADA' : '#D6EBDC'}"><b>${id.toFixed(2)}</b></td></tr>`;
      });
      out.innerHTML = h + '</table>' + `<p class="small" style="margin:.4em 0 0">FLOP/B of internal blocks of the tiled kernel; red: below the ridge (${ridge} FLOP/B) → memory-bound, green: compute-bound.</p>`;
    };
    inp.addEventListener('input', draw);
    return { enter: draw, leave() {} };
  };

  // ------------------------------------------------------------ idle threads and output tile size
  ANIM.idle = (root) => {
    const q = (n) => root.querySelector(`[data-in=${n}]`), out = root.querySelector('[data-out]');
    const update = () => {
      const IN = Math.max(2, Math.min(32, +q('in').value | 0)), r = Math.max(0, Math.min(15, +q('r').value | 0)), W = Math.max(1, +q('w').value | 0);
      const OUT = IN - 2 * r;
      if (OUT <= 0) { out.innerHTML = '<span style="color:#B03A3A">the filter does not fit: OUT_TILE_DIM = IN_TILE_DIM − 2r ≤ 0</span>'; return; }
      const m = 2 * r + 1, t = OUT, ai = (t * t * m * m * 2) / (((t + m - 1) ** 2 + t * t) * 4), blocks = Math.ceil(W / OUT) ** 2;
      const pow2 = (OUT & (OUT - 1)) === 0;
      out.innerHTML = `OUT_TILE_DIM = ${IN} − 2·${r} = <b>${OUT}</b>${pow2 ? ' (power of two)' : ' (not a power of two: output rows start at unaligned addresses)'}` +
        `<br>threads that only load: ${IN * IN - OUT * OUT} of ${IN * IN} = <b>${((1 - OUT * OUT / (IN * IN)) * 100).toFixed(0)} %</b>` +
        `<br>blocks for a ${W} × ${W} image: ⌈${W} / ${OUT}⌉² = <b>${fmt(blocks)}</b> · intensity of internal blocks: <b>${ai.toFixed(2)} FLOP/B</b> (ideal ${(m * m / 4).toFixed(2)})`;
    };
    root.querySelectorAll('input').forEach(i => i.addEventListener('input', update));
    root.querySelectorAll('[data-p]').forEach(b => b.addEventListener('click', () => { const [a, r] = b.dataset.p.split(','); q('in').value = a; q('r').value = r; update(); }));
    return { enter: update, leave() {} };
  };

  // ------------------------------------------------------------ cached halo kernel
  ANIM.halo = (root) => {
    const svg = newSvg(root, '0 0 1000 280', 'With block equal to output tile, interior patch elements come from shared memory and halo elements directly from global memory, usually hitting L2');
    const T0 = 6, cs = 34, X0 = 40, Y0 = 30, r = 1, n = T0 * 2;   // a 12 × 12 image of 4 tiles; we look at tile (0,1)
    const tileX = 6, tileY = 0;
    T(svg, X0, 20, 'image (12 × 12) with four 6 × 6 tiles; the outlined tile is this block', { 'font-size': 13, class: 'lbl' });
    const cells = []; for (let rr = 0; rr < n; rr++) for (let cc = 0; cc < n; cc++) cells.push({ rr, cc, e: R(svg, X0 + cc * cs * .6, Y0 + rr * cs * .6, cs * .6 - 2, cs * .6 - 2, { fill: '#fff', stroke: '#d0d0d0', rx: 1 }) });
    svg.appendChild(el('rect', { x: X0 + tileX * cs * .6 - 1, y: Y0 + tileY * cs * .6 - 1, width: T0 * cs * .6, height: T0 * cs * .6, fill: 'none', stroke: INK, 'stroke-width': 3 }));
    const legend = [['shared memory (N_s)', BLUE], ['global memory, likely an L2 hit', GOLD], ['ghost cell: skipped', LRED]];
    legend.forEach(([s, c], i) => { R(svg, 520, 40 + i * 30, 22, 18, { fill: c, rx: 2 }); T(svg, 552, 54 + i * 30, s, { 'font-size': 13 }); });
    const info = T(svg, 520, 160, '', { 'font-size': 14, 'font-weight': 600 }), info2 = T(svg, 520, 186, '', { 'font-size': 13 }), info3 = T(svg, 520, 210, '', { 'font-size': 13 });
    const S = [[2, 2], [0, 3], [5, 0], [0, 0]];   // threadIdx (x, y) inside the tile
    const cap = root.querySelector('[data-cap]');
    return stepper(root, () => S.length + 1, (k) => {
      if (k === 0) {
        cells.forEach(o => { const inTile = o.cc >= tileX && o.cc < tileX + T0 && o.rr >= tileY && o.rr < tileY + T0; o.e.setAttribute('fill', inTile ? '#9DB7E3' : '#fff'); });
        info.textContent = 'load phase'; info2.textContent = '36 threads load exactly the 36 elements of their tile'; info3.textContent = 'no halo in shared memory, no idle threads';
        cap.innerHTML = 'Block = output tile = input tile, all 32 × 32 on a real GPU (a power of two). Each thread loads its own element; shared memory holds <b>only the tile</b>, no halo.';
        return;
      }
      const [tx, ty] = S[k - 1], row = tileY + ty, col = tileX + tx;
      let sh = 0, gl = 0, gh = 0;
      cells.forEach(o => {
        const inP = Math.abs(o.rr - row) <= r && Math.abs(o.cc - col) <= r, inTile = o.cc >= tileX && o.cc < tileX + T0 && o.rr >= tileY && o.rr < tileY + T0;
        o.e.setAttribute('fill', inTile ? '#EEF3FB' : '#fff');
        if (inP) { if (inTile) { o.e.setAttribute('fill', BLUE); sh++; } else { o.e.setAttribute('fill', GOLD); gl++; } }
      });
      for (let dr = -r; dr <= r; dr++) for (let dc = -r; dc <= r; dc++) { const rr = row + dr, cc = col + dc; if (rr < 0 || cc < 0 || rr >= n || cc >= n) gh++; }
      info.textContent = `thread (${tx},${ty}) → output (${col},${row})`;
      info2.textContent = `from shared memory: ${sh} · from global (halo): ${gl}`; info3.textContent = gh ? `ghost cells skipped: ${gh}` : '';
      cap.innerHTML = [
        'An interior thread: its whole 3 × 3 patch lies inside the tile, every read comes from <b>shared memory</b>.',
        'A thread on the left edge of the tile: three patch elements belong to the tile on the left. They are read from <b>global memory</b>, but that block has just loaded them, so they are probably in <b>L2</b>.',
        'A thread on the right edge, top row: the patch crosses the tile edge (global reads) and the image edge (ghost cells, skipped).',
        'The price: every patch element needs a test (inside the tile? inside the image?), so the compute loop has more branches and more divergence. In exchange, no idle threads and power-of-two tiles.'][k - 1];
    }, 3000);
  };
