  const arrowHead = (svg, id, color) => {
    const d = svg.querySelector('defs') || svg.insertBefore(el('defs'), svg.firstChild);
    const m = el('marker', { id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' });
    m.appendChild(el('path', { d: 'M0,0 L10,5 L0,10 z', fill: color })); d.appendChild(m);
    return `url(#${id})`;
  };
  const GRAY = '#9a9a9a', BLUE = '#2F6FD0', GOLD = '#C98A1B', RED = '#B03A3A', GREEN = '#2E7D4F', PINK = '#F6DADA', INK = '#1A1A1A';
  const vis = (e, on) => e.setAttribute('opacity', on ? 1 : 0);
  const fill = (e, c) => { e.style.fill = c || ''; };
  const gcd = (a, b) => b ? gcd(b, a % b) : a;
  const syncButtons = (root, attr, val) => root.querySelectorAll(`[data-${attr}]`).forEach(b => b.classList.toggle('on', b.dataset[attr] === val));

  // ------------------------------------------------------------ why DRAM is slow
  ANIM.dram = (root) => {
    const svg = newSvg(root, '0 0 1000 380', 'A DRAM array: a decoder drives a horizontal line, the selected cell shares its charge with a long vertical line read by a sense amplifier');
    const hA = arrowHead(svg, 'dr-a', GOLD);
    const dec = R(svg, 20, 40, 70, 250, { class: 'mem', rx: 3 });
    const dt = T(svg, 55, 165, 'decoder', { ...mid, 'font-size': 13, transform: 'rotate(-90 55 165)' });
    const ROWS = 6, COLS = 10, x0 = 130, y0 = 50, dx = 78, dy = 40;
    const word = [], bit = [], cells = [];
    for (let r = 0; r < ROWS; r++) word.push(L(svg, 90, y0 + r * dy + 8, x0 + COLS * dx - 20, y0 + r * dy + 8, { class: 'wire' }));
    for (let c = 0; c < COLS; c++) bit.push(L(svg, x0 + c * dx + 30, y0 - 12, x0 + c * dx + 30, 300, { class: 'wire' }));
    for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) {
      const x = x0 + c * dx + 36, y = y0 + r * dy + 14;
      cells.push({ r, c, e: R(svg, x, y, 16, 14, { fill: '#C9DBF4', rx: 2 }) });
    }
    const amps = [];
    for (let c = 0; c < COLS; c++) amps.push(R(svg, x0 + c * dx + 14, 300, 32, 22, { class: 'mem', rx: 2 }));
    T(svg, x0 + COLS * dx / 2, 346, 'sense amplifiers', { ...mid, 'font-size': 13, class: 'lbl' });
    T(svg, 920, 30, 'about 1000 cells', { 'font-size': 11, class: 'lbl', 'text-anchor': 'end' });
    T(svg, 920, 44, 'on each vertical line', { 'font-size': 11, class: 'lbl', 'text-anchor': 'end' });
    const clock = T(svg, 500, 372, '', { ...mid, 'font-size': 13, fill: GOLD });
    const pulse = L(svg, x0 + 3 * dx + 30, y0 + 2 * dy + 20, x0 + 3 * dx + 30, 298, { stroke: GOLD, 'stroke-width': 4, 'marker-end': hA });
    const R0 = 2, C0 = 3;
    const caps = [
      'Global memory is <b>DRAM</b>. Each bit is a tiny capacitor (a cell): charge present means 1, absent means 0.',
      'To read, the <b>decoder</b> drives one horizontal line. It reaches the gates of thousands of cells, so charging it to the right level takes time.',
      'The selected cell\'s gate opens and its tiny charge flows onto a <b>long vertical line</b> shared with about 1000 cells, a large capacitance.',
      'The <b>sense amplifier</b> must notice that small change before it can decide 0 or 1. This takes <b>tens of nanoseconds</b>, while a processor clock tick is under one nanosecond.',
      'Like smelling a small cup of coffee from the far end of a long hallway. Cells keep shrinking to fit more bits per chip, so they get weaker: DRAM latency has not gone down over the years.'
    ];
    const cap = root.querySelector('[data-cap]');
    return stepper(root, () => caps.length, (k) => {
      word.forEach((w, r) => w.classList.toggle('on', k >= 1 && r === R0));
      bit.forEach((b, c) => b.classList.toggle('on', k >= 2 && c === C0));
      cells.forEach(o => fill(o.e, (o.r === R0 && o.c === C0) ? (k >= 0 ? BLUE : '') : (k >= 1 && o.r === R0 ? '#9DB7E3' : '')));
      dec.classList.toggle('on', k === 1); amps.forEach((a, c) => a.classList.toggle('on', k >= 3 && c === C0));
      vis(pulse, k === 2 || k === 3);
      clock.textContent = k >= 3 ? 'one access ≈ tens of ns  ·  one clock cycle < 1 ns' : '';
      cap.innerHTML = caps[k];
    }, 3000);
  };

  // ------------------------------------------------------------ DRAM bursts
  ANIM.burst = (root) => {
    let mode = 'seq';
    const svg = newSvg(root, '0 0 1000 300', 'Locations of a DRAM row grouped in bursts; consecutive requests are served by one burst, scattered ones need many');
    const N = 32, B = 8, cw = 29, x0 = 20;
    T(svg, 20, 28, 'DRAM locations (bursts of 8 consecutive locations)', { 'font-size': 13, class: 'lbl' });
    const cells = [], bursts = [];
    for (let b = 0; b < N / B; b++) bursts.push(R(svg, x0 + b * B * cw - 3, 38, B * cw + 2, 48, { fill: 'none', stroke: '#bdbdbd', 'stroke-width': 1.5, rx: 4 }));
    for (let i = 0; i < N; i++) { cells.push(R(svg, x0 + i * cw, 44, cw - 3, 36, { class: 'mem', rx: 2 })); T(svg, x0 + i * cw + 13, 67, i, { ...mid, 'font-size': 10 }); }
    const reqs = [];
    for (let t = 0; t < 8; t++) { const g = svg.appendChild(el('g')); g.appendChild(el('rect', { x: 0, y: 0, width: 46, height: 26, rx: 3, fill: '#C9DBF4' })); g.appendChild(el('text', { x: 23, y: 17, 'text-anchor': 'middle', 'font-size': 11 }, `t${t}`)); reqs.push(g); }
    const lines = svg.appendChild(el('g'));
    const bus = R(svg, 20, 240, 940, 26, { fill: '#F4F4F4', stroke: '#c9c9c9', rx: 3 }); T(svg, 490, 258, 'bus to the processor', { ...mid, 'font-size': 12, class: 'lbl' });
    const out = T(svg, 20, 292, '', { 'font-size': 14 });
    const A = { seq: [0, 1, 2, 3, 4, 5, 6, 7], scat: [3, 9, 14, 20, 26, 31, 5, 17] };
    const caps = {
      seq: ['Eight threads ask for locations 0–7: they are <b>consecutive</b>.', 'Every access reads a whole <b>burst</b> anyway: many sensors work in parallel on consecutive locations.', 'One burst delivers all eight values at high speed. Every byte transferred is used.'],
      scat: ['Now the eight requests are <b>scattered</b> over the row.', 'Each one needs its own burst: the DRAM reads four bursts and most of what it reads is not needed.', 'Four times the time for the same eight values. Focused access to consecutive locations is what DRAM rewards.']
    };
    const cap = root.querySelector('[data-cap]');
    const st = stepper(root, () => 3, (k) => {
      const a = A[mode], used = new Set(a), bset = new Set(a.map(x => Math.floor(x / B)));
      reqs.forEach((g, t) => g.setAttribute('transform', `translate(${40 + t * 116},150)`));
      lines.innerHTML = '';
      a.forEach((addr, t) => lines.appendChild(el('line', { x1: 63 + t * 116, y1: 150, x2: x0 + addr * cw + 13, y2: 82, stroke: BLUE, 'stroke-width': 1.5, opacity: .8 })));
      cells.forEach((c, i) => { const inB = bset.has(Math.floor(i / B)); fill(c, k >= 1 && inB ? (used.has(i) ? '#9DB7E3' : '#F6EAD2') : (used.has(i) ? '#DCE7F8' : '')); });
      bursts.forEach((b, i) => { b.setAttribute('stroke', k >= 1 && bset.has(i) ? GOLD : '#bdbdbd'); b.setAttribute('stroke-width', k >= 1 && bset.has(i) ? 3 : 1.5); });
      fill(bus, k === 2 ? '#F6EAD2' : '');
      out.textContent = k >= 1 ? `bursts read: ${bset.size} · bytes moved: ${bset.size * B * 4} · bytes used: ${a.length * 4} (${Math.round(a.length / (bset.size * B) * 100)} %)` : '';
      cap.innerHTML = caps[mode][k];
    }, 2600);
    modeButtons(root, 'mode', (m) => { mode = m; st.reset(); });
    return st;
  };

  // ------------------------------------------------------------ coalescing of one warp load
  ANIM.coalesce = (root) => {
    const svg = newSvg(root, '0 0 1000 330', 'The 32 addresses of one warp load and the 128-byte memory segments they fall into');
    let stride = 1, off = 0;
    const draw = () => {
      svg.innerHTML = '';
      const addr = Array.from({ length: 32 }, (_, t) => off + t * stride);
      const seg = new Set(addr.map(a => Math.floor(a / 32)));
      const segs = [...seg].sort((a, b) => a - b);
      T(svg, 20, 26, `thread t reads a[${off ? off + ' + ' : ''}${stride === 1 ? '' : stride + ' × '}t]`, { 'font-size': 15, 'font-weight': 600 });
      for (let t = 0; t < 32; t++) { R(svg, 60 + t * 29, 40, 25, 22, { fill: '#C9DBF4', rx: 2 }); T(svg, 72 + t * 29, 55, t, { ...mid, 'font-size': 10 }); }
      const show = segs.slice(0, 4);
      show.forEach((s, j) => {
        const y = 130 + j * 40;
        R(svg, 60, y, 930, 26, { fill: '#F6EAD2', stroke: GOLD, rx: 2 });
        for (let w = 0; w < 32; w++) {
          const a = s * 32 + w, hit = addr.includes(a);
          R(svg, 60 + w * 29 + 1, y + 2, 27, 22, { fill: hit ? BLUE : '#fff', rx: 1 });
        }
        T(svg, 54, y + 18, `seg ${s}`, { 'font-size': 11, class: 'lbl', 'text-anchor': 'end' });
        addr.forEach((a, t) => { if (Math.floor(a / 32) === s) svg.appendChild(el('line', { x1: 72 + t * 29, y1: 62, x2: 74 + (a % 32) * 29, y2: y, stroke: BLUE, 'stroke-width': 1, opacity: .55 })); });
      });
      if (segs.length > 4) T(svg, 500, 300, `… and ${segs.length - 4} more segments`, { ...mid, 'font-size': 13, fill: RED });
      const eff = 128 / (segs.length * 128);
      root.querySelector('[data-out]').innerHTML = `segments of 128 B touched: <b>${segs.length}</b> · bytes moved ${segs.length * 128} for 128 used · efficiency <b>${Math.round(eff * 100)} %</b>`;
      root.querySelector('[data-cap]').innerHTML = segs.length === 1 ? 'All 32 addresses fall in one aligned 128-byte segment: the hardware <b>coalesces</b> them into a single request.'
        : stride === 1 ? 'Consecutive but <b>not aligned</b>: the 32 values straddle a segment boundary, so one extra segment is moved.'
        : `With a stride of ${stride}, the warp's addresses spread over ${segs.length} segments. Each needs its own transfer: the accesses are <b>not coalesced</b>.`;
      syncButtons(root, 'p', `${stride},${off}`);
    };
    root.querySelectorAll('[data-p]').forEach(b => b.addEventListener('click', () => { [stride, off] = b.dataset.p.split(',').map(Number); draw(); }));
    return { enter: draw, leave() {} };
  };

  // ------------------------------------------------------------ row-major placement
  ANIM.rowmajor = (root) => {
    const svg = newSvg(root, '0 0 1000 300', 'A 4 by 4 matrix placed row after row into a linear array');
    const shade = ['#FFFFFF', '#DCDCDC', '#9A9A9A', '#3A3A3A'], ink = ['#1A1A1A', '#1A1A1A', '#fff', '#fff'];
    const grid = [];
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
      const g = R(svg, 380 + c * 60, 10 + r * 40, 58, 38, { fill: shade[r], stroke: '#6f6f6f' }); T(svg, 409 + c * 60, 35 + r * 40, `M${r},${c}`, { ...mid, 'font-size': 13, fill: ink[r] });
      grid.push(g);
    }
    T(svg, 20, 236, 'linear memory, increasing address →', { 'font-size': 13, class: 'lbl' });
    const lin = [];
    for (let i = 0; i < 16; i++) {
      const r = Math.floor(i / 4), c = i % 4;
      const g = svg.appendChild(el('g', { opacity: 0 }));
      g.appendChild(el('rect', { x: 20 + i * 60, y: 180, width: 58, height: 38, fill: shade[r], stroke: '#6f6f6f' }));
      g.appendChild(el('text', { x: 49 + i * 60, y: 205, 'text-anchor': 'middle', 'font-size': 13, fill: ink[r] }, `M${r},${c}`));
      lin.push(g);
    }
    const br = svg.appendChild(el('path', { d: 'M 49 222 L 49 250 L 289 250 L 289 222', fill: 'none', stroke: RED, 'stroke-width': 2.5 }));
    const brT = T(svg, 169, 270, 'M0,0 and M1,0: 4 locations apart', { ...mid, 'font-size': 13, fill: RED });
    const caps = ['Row 0 is placed first, in order: M0,0 M0,1 M0,2 M0,3.', 'Then row 1 …', '… row 2 …', '… and row 3. This is <b>row-major</b> order, used by C and CUDA.',
      'Elements next to each other in a <b>column</b> are <b>Width</b> locations apart in memory: here 4, in a real matrix thousands.'];
    const cap = root.querySelector('[data-cap]');
    return stepper(root, () => 5, (k) => {
      lin.forEach((g, i) => g.setAttribute('opacity', Math.floor(i / 4) <= k ? 1 : 0));
      grid.forEach((g, i) => { const r = Math.floor(i / 4), c = i % 4; g.setAttribute('stroke', (k === 4 && c === 0 && r <= 1) || (k < 4 && r === k) ? RED : '#6f6f6f'); g.setAttribute('stroke-width', (k === 4 && c === 0 && r <= 1) || (k < 4 && r === k) ? 3 : 1); });
      vis(br, k === 4); vis(brT, k === 4);
      cap.innerHTML = caps[k];
    }, 2200);
  };

  // ------------------------------------------------------------ coalesced vs un-coalesced pattern of the matmul loop
  ANIM.pattern = (root) => {
    let mode = 'row';
    const svg = newSvg(root, '0 0 1000 360', 'Four threads iterating down the rows of N; the elements they read in one iteration are adjacent in row-major layout and far apart in column-major layout');
    const hB = arrowHead(svg, 'pt-b', BLUE);
    const code = T(svg, 20, 30, '', { 'font-size': 15, 'font-weight': 600 });
    T(svg, 20, 56, 'logical view: thread Tc walks down column c of N', { 'font-size': 12, class: 'lbl' });
    const shade = ['#FFFFFF', '#DCDCDC', '#9A9A9A', '#3A3A3A'], ink = ['#1A1A1A', '#1A1A1A', '#fff', '#fff'];
    const logical = [];
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) { logical.push(R(svg, 40 + c * 58, 100 + r * 36, 56, 34, { stroke: '#6f6f6f' })); T(svg, 68 + c * 58, 122 + r * 36, `N${r},${c}`, { ...mid, 'font-size': 12 }); }
    for (let c = 0; c < 4; c++) { T(svg, 68 + c * 58, 84, `T${c}`, { ...mid, 'font-size': 13, 'font-weight': 600 }); }
    const ltext = [...svg.querySelectorAll('text')].slice(-20, -4);
    T(svg, 14, 176, 'k', { 'font-size': 13, class: 'lbl' });
    L(svg, 26, 100, 26, 244, { stroke: GRAY, 'stroke-width': 2, 'marker-end': arrowHead(svg, 'pt-g', GRAY) });
    T(svg, 340, 332, '', {}); const layoutT = svg.lastChild;
    const phys = [], ptxt = [];
    for (let i = 0; i < 16; i++) { phys.push(R(svg, 340 + i * 40, 280, 38, 34, { stroke: '#6f6f6f' })); ptxt.push(T(svg, 359 + i * 40, 302, '', { ...mid, 'font-size': 10 })); }
    const arr = svg.appendChild(el('g'));
    const req = T(svg, 340, 354, '', { 'font-size': 13 });
    const cap = root.querySelector('[data-cap]');
    const st = stepper(root, () => 4, (k) => {
      code.textContent = mode === 'row' ? 'Pvalue += M[row*Width + k] * N[k*Width + col];' : 'Pvalue += M[row*Width + k] * N[col*Width + k];';
      layoutT.textContent = mode === 'row' ? 'physical view: N in row-major layout' : 'physical view: N in column-major layout';
      layoutT.setAttribute('font-size', 12); layoutT.setAttribute('class', 'lbl');
      for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
        const s = mode === 'row' ? r : c, e = logical[r * 4 + c];
        e.setAttribute('fill', shade[s]); ltext[r * 4 + c].setAttribute('fill', ink[s]);
        e.setAttribute('stroke', r === k ? BLUE : '#6f6f6f'); e.setAttribute('stroke-width', r === k ? 3 : 1);
      }
      const pos = (r, c) => mode === 'row' ? r * 4 + c : c * 4 + r;
      for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) { const p = pos(r, c); const s = mode === 'row' ? r : c; phys[p].setAttribute('fill', shade[s]); ptxt[p].setAttribute('fill', ink[s]); ptxt[p].textContent = `N${r},${c}`; }
      arr.innerHTML = '';
      for (let c = 0; c < 4; c++) { const p = pos(k, c); arr.appendChild(el('line', { x1: 68 + c * 58, y1: 100 + k * 36 + 34, x2: 359 + p * 40, y2: 278, stroke: BLUE, 'stroke-width': 2, 'marker-end': hB })); }
      const ps = [0, 1, 2, 3].map(c => pos(k, c));
      req.textContent = mode === 'row' ? `iteration ${k}: addresses ${ps.join(', ')} are adjacent → one coalesced access` : `iteration ${k}: addresses ${ps.join(', ')} are Width apart → separate accesses`;
      req.setAttribute('fill', mode === 'row' ? GREEN : RED);
      cap.innerHTML = mode === 'row'
        ? 'Index <code>k*Width + col</code>: k and Width are the same for all threads, col = blockIdx.x*blockDim.x + threadIdx.x. Consecutive threads read <b>consecutive addresses</b>.'
        : 'Index <code>col*Width + k</code>: col is multiplied by Width, so consecutive threads read addresses <b>Width apart</b>. In a real matrix that is thousands of elements: no coalescing.';
    }, 2000);
    modeButtons(root, 'mode', (m) => { mode = m; st.reset(); });
    return st;
  };

  // ------------------------------------------------------------ corner turning
  ANIM.corner = (root) => {
    let mode = 'no';
    const svg = newSvg(root, '0 0 1000 340', 'Loading a 4 by 4 tile of a column-major matrix N into shared memory, without and with corner turning');
    const hB = arrowHead(svg, 'cn-b', BLUE);
    T(svg, 20, 26, 'tile of N (column-major)', { 'font-size': 12, class: 'lbl' });
    const tile = [];
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) { tile.push(R(svg, 30 + c * 56, 40 + r * 36, 54, 34, { fill: '#fff', stroke: '#6f6f6f' })); T(svg, 57 + c * 56, 62 + r * 36, `N${r},${c}`, { ...mid, 'font-size': 12 }); }
    T(svg, 300, 26, 'global memory: column 0, then column 1, …', { 'font-size': 12, class: 'lbl' });
    const mem = [];
    for (let i = 0; i < 16; i++) { mem.push(R(svg, 300 + i * 43, 40, 41, 34, { fill: '#fff', stroke: '#6f6f6f' })); T(svg, 320 + i * 43, 62, `N${i % 4},${Math.floor(i / 4)}`, { ...mid, 'font-size': 10 }); }
    const thr = [0, 1, 2, 3].map(t => { const g = R(svg, 320 + t * 120, 150, 90, 30, { fill: '#C9DBF4', rx: 3 }); T(svg, 365 + t * 120, 170, `thread ${t}`, { ...mid, 'font-size': 12 }); return g; });
    const arr = svg.appendChild(el('g'));
    T(svg, 30, 222, 'shared memory tile Nds', { 'font-size': 12, class: 'lbl' });
    const sh = [];
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) { sh.push(R(svg, 30 + c * 56, 232 + r * 24, 54, 22, { fill: '#F4F4F4', stroke: '#c9c9c9' })); }
    const shT = sh.map((_, i) => T(svg, 57 + (i % 4) * 56, 248 + Math.floor(i / 4) * 24, '', { ...mid, 'font-size': 11 }));
    const res = T(svg, 300, 260, '', { 'font-size': 14 }), res2 = T(svg, 300, 286, '', { 'font-size': 13 });
    const cap = root.querySelector('[data-cap]');
    const st = stepper(root, () => 5, (k) => {
      arr.innerHTML = '';
      const step = Math.min(k, 3);
      const elems = [0, 1, 2, 3].map(t => mode === 'no' ? [step, t] : [t, step]);   // [row, col] loaded by thread t
      tile.forEach((e, i) => { const r = Math.floor(i / 4), c = i % 4; const on = k < 4 && elems.some(([a, b]) => a === r && b === c); e.setAttribute('fill', on ? '#9DB7E3' : '#fff'); });
      const addrs = elems.map(([r, c]) => c * 4 + r);
      mem.forEach((e, i) => e.setAttribute('fill', k < 4 && addrs.includes(i) ? '#9DB7E3' : '#fff'));
      if (k < 4) addrs.forEach((a, t) => arr.appendChild(el('line', { x1: 320 + a * 43, y1: 74, x2: 365 + t * 120, y2: 150, stroke: BLUE, 'stroke-width': 2, 'marker-start': hB })));
      sh.forEach((e, i) => { const r = Math.floor(i / 4), c = i % 4; const done = mode === 'no' ? r <= Math.min(k, 3) : c <= Math.min(k, 3); shT[i].textContent = done ? `N${r},${c}` : ''; e.setAttribute('fill', done ? '#DCE7F8' : '#F4F4F4'); });
      const contiguous = Math.max(...addrs) - Math.min(...addrs) === 3;
      res.textContent = k < 4 ? `load ${k}: addresses ${addrs.join(', ')} → ${contiguous ? 'adjacent, coalesced' : '4 apart, not coalesced'}` : (mode === 'no' ? 'every load instruction was un-coalesced' : 'every load instruction was coalesced');
      res.setAttribute('fill', mode === 'no' ? RED : GREEN);
      res2.textContent = mode === 'yes' && k >= 1 ? 'threadIdx.x now walks down a column: swap tx and ty in the load index' : '';
      cap.innerHTML = k === 4 ? (mode === 'no' ? 'The tile ends up correct, but each load instruction of the warp touched addresses Width apart. With real tiles, consecutive threads hit 32 different places in memory.'
        : 'Same tile, same shared memory contents. Shared memory is SRAM and does not need coalescing, so the threads can then read Nds in any order. Real tiles are 32 wide: 32 consecutive threads read 32 consecutive elements of a column.')
        : (mode === 'no' ? 'Without corner turning, the four threads of a row load the <b>top edge</b> of the tile, as for a row-major matrix. In column-major layout those elements are far apart.'
        : 'With <b>corner turning</b>, the same threads load the <b>left edge</b> (one column). In column-major layout a column is contiguous: one coalesced access.');
    }, 2200);
    modeButtons(root, 'mode', (m) => { mode = m; st.reset(); });
    return st;
  };

  // ------------------------------------------------------------ carpooling
  ANIM.carpool = (root) => {
    const svg = newSvg(root, '0 0 1000 330', 'A road full of single-occupant cars versus a few full carpool vehicles, and the schedules that make carpooling possible');
    const road = R(svg, 20, 30, 600, 150, { fill: '#EDEDED', rx: 4 });
    for (let l = 1; l < 4; l++) L(svg, 20, 30 + l * 37.5, 620, 30 + l * 37.5, { stroke: '#fff', 'stroke-width': 2, 'stroke-dasharray': '14 10' });
    const cars = [];
    for (let i = 0; i < 32; i++) { const g = svg.appendChild(el('g', { class: 'tok' })); g.appendChild(el('rect', { x: 0, y: 0, width: 34, height: 20, rx: 5, fill: BLUE })); cars.push(g); }
    const bus = svg.appendChild(el('g', { class: 'tok' })); bus.appendChild(el('rect', { x: 0, y: 0, width: 150, height: 26, rx: 6, fill: GREEN })); bus.appendChild(el('text', { x: 75, y: 18, 'text-anchor': 'middle', 'font-size': 12, fill: '#fff' }, '32 passengers'));
    const dram = R(svg, 640, 30, 140, 150, { class: 'mem', rx: 4 }); T(svg, 710, 100, 'DRAM', { ...mid, 'font-size': 15 }); const q = T(svg, 710, 124, '', { ...mid, 'font-size': 12 });
    const tl = svg.appendChild(el('g'));
    const rowT = (y, name, items, ok) => {
      tl.appendChild(el('text', { x: 20, y: y + 14, 'font-size': 12 }, name));
      items.forEach(([a, b, s]) => { tl.appendChild(el('rect', { x: 120 + a * 80, y, width: (b - a) * 80 - 4, height: 20, rx: 3, fill: s === 'work' ? (ok ? '#86C495' : '#E7B467') : '#DCE7F8' })); tl.appendChild(el('text', { x: 120 + (a + b) * 40, y: y + 14, 'text-anchor': 'middle', 'font-size': 11 }, s)); });
    };
    const caps = [
      '<b>Traffic congestion</b>: too many cars, each with one person, on a road built for fewer. Every DRAM request is a car; with one request per thread, the queue at DRAM grows and the cores wait.',
      '<b>Carpooling</b>: the same people in far fewer vehicles. Coalescing does this: the accesses of a warp to adjacent locations travel as one request.',
      'Carpooling only works if the riders keep <b>similar schedules</b>: worker A and worker B sleep, work and eat at the same times.',
      'With <b>very different schedules</b> they cannot share a car. Threads in the same warp execute the same load at the same time (SIMD): they are the ideal carpool.'
    ];
    const cap = root.querySelector('[data-cap]');
    return stepper(root, () => 4, (k) => {
      cars.forEach((c, i) => { const lane = i % 4, col = Math.floor(i / 4); c.setAttribute('transform', `translate(${30 + col * 70 + (lane % 2) * 20},${38 + lane * 37.5})`); vis(c, k === 0); });
      bus.setAttribute('transform', `translate(${k === 1 ? 440 : 60},${82})`); vis(bus, k >= 1);
      q.textContent = k === 0 ? 'queue: 32 requests' : 'queue: 1 request'; q.setAttribute('fill', k === 0 ? RED : GREEN);
      dram.classList.toggle('on', k === 0);
      tl.innerHTML = '';
      if (k >= 2) {
        tl.appendChild(el('text', { x: 20, y: 206, 'font-size': 13, 'font-weight': 600, fill: k === 2 ? GREEN : RED }, k === 2 ? 'similar schedules: one car' : 'different schedules: no carpool'));
        rowT(218, 'worker A', k === 2 ? [[0, 3, 'sleep'], [3, 7, 'work'], [7, 9, 'dinner']] : [[0, 2, 'party'], [2, 6, 'sleep'], [6, 9, 'work']], true);
        rowT(246, 'worker B', [[0, 3, 'sleep'], [3, 7, 'work'], [7, 9, 'dinner']], k === 2);
        tl.appendChild(el('text', { x: 120, y: 290, 'font-size': 12, class: 'lbl' }, 'time →'));
      }
      cap.innerHTML = caps[k];
    }, 3200);
  };

  // ------------------------------------------------------------ channels and banks
  ANIM.channels = (root) => {
    const svg = newSvg(root, '0 0 1000 330', 'A processor with four memory channels, each with a bus connecting four DRAM banks');
    const hG = arrowHead(svg, 'ch-g', GRAY);
    const proc = R(svg, 20, 200, 960, 120, { class: 'frame', rx: 5 }); T(svg, 500, 310, 'processor', { ...mid, 'font-size': 14, 'font-weight': 600 });
    const chans = [], buses = [], banks = [];
    for (let c = 0; c < 4; c++) {
      const x = 60 + c * 235;
      chans.push(R(svg, x, 216, 130, 64, { class: 'mem', rx: 3 })); T(svg, x + 65, 254, `channel ${c}`, { ...mid, 'font-size': 13 });
      buses.push(L(svg, x + 110, 216, x + 110, 30, { class: 'wire', 'marker-end': hG, 'marker-start': hG }));
      T(svg, x + 118, 26, 'bus', { 'font-size': 12, class: 'lbl' });
      for (let b = 3; b >= 0; b--) { const r = R(svg, x - 6 + b * 12, 50 + b * 22, 90, 50, { fill: '#fff', stroke: '#9a9a9a', rx: 3 }); T(svg, x + 39 + b * 12, 80 + b * 22, b === 0 ? 'bank' : '', { ...mid, 'font-size': 11 }); banks.push({ c, b, r }); L(svg, x + 84 + b * 12, 70 + b * 22, x + 110, 70 + b * 22, { class: 'wire' }); }
    }
    const caps = [
      'DRAM is parallel at several levels. A <b>channel</b> is a memory controller with its own <b>bus</b>; a processor has one to eight of them.',
      'Each bus connects several <b>banks</b>. A bank holds a cell array, its sense amplifiers and the interface that sends bursts onto the bus.',
      'A bus\'s bandwidth is width × transfers per cycle × clock. 64-bit DDR5-6400: 8 B × 2 × 3.2 GHz = <b>51.2 GB/s</b>. A GPU needing over 1000 GB/s would need about 20 such buses.',
      'That many buses on a circuit board cost too much power, hence <b>HBM</b>: DRAM packaged next to the processor, with many fast buses (HBM3E: 16 64-bit buses at 4.9 GHz, about 1 TB/s).'
    ];
    const cap = root.querySelector('[data-cap]');
    return stepper(root, () => 4, (k) => {
      chans.forEach(c => c.classList.toggle('on', k === 0)); buses.forEach(b => b.classList.toggle('on', k === 0 || k === 2));
      banks.forEach(o => { o.r.setAttribute('stroke', k === 1 ? GOLD : '#9a9a9a'); o.r.setAttribute('stroke-width', k === 1 ? 2.5 : 1); });
      proc.classList.toggle('on', k === 3);
      cap.innerHTML = caps[k];
    }, 3200);
  };

  ANIM.buscalc = (root) => {
    const q = (n) => root.querySelector(`[data-in=${n}]`), out = root.querySelector('[data-out]');
    const update = () => {
      const bits = Math.max(1, +q('w').value), ghz = Math.max(0, +q('f').value), need = Math.max(0, +q('n').value);
      const bw = bits / 8 * 2 * ghz;
      out.innerHTML = `one bus: ${bits / 8} B × 2 × ${ghz} GHz = <b>${bw.toFixed(1)} GB/s</b><br>buses for ${need} GB/s: ⌈${need} / ${bw.toFixed(1)}⌉ = <b>${bw > 0 ? Math.ceil(need / bw - 1e-9) : '∞'}</b>`;
    };
    root.querySelectorAll('input').forEach(i => i.addEventListener('input', update));
    root.querySelectorAll('[data-p]').forEach(b => b.addEventListener('click', () => { const [w, f, n] = b.dataset.p.split(','); q('w').value = w; q('f').value = f; q('n').value = n; update(); }));
    return { enter: update, leave() {} };
  };

  // ------------------------------------------------------------ banking timeline
  ANIM.banking = (root) => {
    const svg = newSvg(root, '0 0 1000 320', 'Timelines of DRAM banks sharing one channel: each access is a long latency followed by a short burst transfer');
    const qn = (n) => root.querySelector(`[data-in=${n}]`), out = root.querySelector('[data-out]');
    const W = 860, X0 = 120, SLOTS = 44;
    let Rr = 3, B = 2;
    const draw = (k) => {
      svg.innerHTML = '';
      Rr = Math.max(1, Math.min(40, +qn('r').value | 0)); B = Math.max(1, Math.min(64, +qn('b').value | 0));
      const u = W / SLOTS, rowsB = Math.min(B, 6), period = Rr + 1;
      const act = Math.min(B, period);                          // banks that can be kept busy
      const chanBusy = new Array(SLOTS).fill(-1);
      for (let b = 0; b < act; b++) for (let t = b + Rr; t < SLOTS; t += Math.max(period, act)) chanBusy[t] = b;
      const cur = Math.min(SLOTS, Math.round((k + 1) / 20 * SLOTS));
      for (let b = 0; b < rowsB; b++) {
        const y = 20 + b * 34;
        T(svg, 20, y + 17, `bank ${b}`, { 'font-size': 12 });
        R(svg, X0, y, W, 24, { fill: '#F7F7F7', rx: 2 });
        if (b < act) for (let t0 = b; t0 < SLOTS; t0 += Math.max(period, act)) {
          const a = Math.min(cur, t0), e = Math.min(cur, t0 + Rr), te = Math.min(cur, t0 + Rr + 1);
          if (e > a) R(svg, X0 + a * u, y, (e - a) * u, 24, { fill: '#E2E2E2' });
          if (te > e) R(svg, X0 + e * u, y, (te - e) * u, 24, { fill: INK });
        } else if (cur > 0) T(svg, X0 + 8, y + 17, 'idle: the bus is already full', { 'font-size': 11, class: 'lbl' });
      }
      if (B > rowsB) T(svg, 20, 20 + rowsB * 34 + 14, `… ${B - rowsB} more banks`, { 'font-size': 12, class: 'lbl' });
      const yc = 20 + Math.max(rowsB, B > rowsB ? rowsB + 1 : rowsB) * 34 + 16;
      svg.setAttribute('viewBox', `0 0 1000 ${yc + 56}`);
      T(svg, 20, yc + 17, 'channel', { 'font-size': 12, 'font-weight': 600 });
      R(svg, X0, yc, W, 24, { fill: '#F7F7F7', rx: 2, stroke: '#c9c9c9' });
      for (let t = 0; t < cur; t++) if (chanBusy[t] >= 0) R(svg, X0 + t * u + 1, yc, u - 2, 24, { fill: INK });
      L(svg, X0 + cur * u, 10, X0 + cur * u, yc + 30, { stroke: GOLD, 'stroke-width': 1.5, 'stroke-dasharray': '4 3' });
      T(svg, X0, yc + 44, 'light = cell-array access latency · dark = burst on the bus · time →', { 'font-size': 11, class: 'lbl' });
      const util = Math.min(1, B / period);
      out.innerHTML = `latency : transfer = <b>${Rr} : 1</b> · banks <b>${B}</b> → channel busy at most ${B} / ${period} = <b>${(util * 100).toFixed(1)} %</b>` +
        `<br>a 16 GB/s channel delivers about <b>${(16 * util).toFixed(2)} GB/s</b> · banks needed for full use: R + 1 = <b>${period}</b>`;
      root.querySelector('[data-cap]').innerHTML = B === 1 ? 'One bank: each access waits the whole latency before its short burst, and the bus sits idle in between.'
        : B < period ? `With ${B} banks, the next bank starts its access while the previous one is still waiting, so their latencies <b>overlap</b>. Still not enough to fill the bus.`
        : `${B} banks ≥ R + 1 = ${period}: a burst is ready in every slot, the bus is <b>fully used</b>. More banks also make two requests to the same bank (a bank conflict) less likely.`;
    };
    const st = stepper(root, () => 20, draw, 220);
    root.querySelectorAll('input').forEach(i => i.addEventListener('input', () => st.reset()));
    root.querySelectorAll('[data-p]').forEach(b => b.addEventListener('click', () => { const [r, n] = b.dataset.p.split(','); qn('r').value = r; qn('b').value = n; st.reset(); }));
    return st;
  };

  // ------------------------------------------------------------ interleaved distribution
  ANIM.interleave = (root) => {
    const svg = newSvg(root, '0 0 1000 320', 'Elements of an array M distributed two at a time over four channels and two banks per channel');
    const hG = arrowHead(svg, 'il-g', GRAY);
    const boxes = {};
    for (let c = 0; c < 4; c++) {
      const x = 40 + c * 240;
      R(svg, x + 40, 250, 120, 50, { class: 'mem', rx: 3 }); T(svg, x + 100, 280, `channel ${c}`, { ...mid, 'font-size': 13 });
      L(svg, x + 175, 250, x + 175, 30, { class: 'wire', 'marker-end': hG });
      for (let b = 1; b >= 0; b--) {
        const bx = x + (1 - b) * 30, by = 40 + (1 - b) * 0 + b * 0 + (b === 1 ? 20 : 110);
        const r = R(svg, bx, by, 140, 70, { fill: '#fff', stroke: '#9a9a9a', rx: 3 });
        T(svg, bx + 8, by + 16, `bank ${b}`, { 'font-size': 11, class: 'lbl' });
        const t = T(svg, bx + 70, by + 46, '', { ...mid, 'font-size': 15, 'font-weight': 600 });
        boxes[`${c}${b}`] = { r, t };
        L(svg, bx + 140, by + 35, x + 175, by + 35, { class: 'wire' });
      }
    }
    const note = T(svg, 20, 18, '', { 'font-size': 13 });
    const cap = root.querySelector('[data-cap]');
    return stepper(root, () => 10, (k) => {
      Object.values(boxes).forEach(o => { o.t.textContent = ''; o.r.setAttribute('stroke', '#9a9a9a'); o.r.setAttribute('stroke-width', 1); fill(o.r, ''); });
      const shown = Math.min(k + 1, 8);
      for (let p = 0; p < shown; p++) {
        const c = p % 4, b = Math.floor(p / 4) % 2, o = boxes[`${c}${b}`];
        o.t.textContent = `M[${2 * p}] M[${2 * p + 1}]`;
        if (p === shown - 1 && k < 8) { o.r.setAttribute('stroke', GOLD); o.r.setAttribute('stroke-width', 3); fill(o.r, '#F6EAD2'); }
      }
      if (k >= 8) { const o = boxes['00']; o.r.setAttribute('stroke', BLUE); o.r.setAttribute('stroke-width', 3); }
      note.textContent = k < 8 ? `burst = 2 elements (8 B) · placing M[${2 * Math.min(k, 7)}] and M[${2 * Math.min(k, 7) + 1}]` : 'M[16], M[17] wrap around to bank 0 of channel 0, and so on';
      cap.innerHTML = k < 4 ? 'The hardware fills one burst in bank 0 of channel 0, then moves to <b>the next channel</b>: consecutive bursts go to different channels.'
        : k < 8 ? 'After the last channel, it wraps to channel 0 again, now using <b>bank 1</b>.'
        : k === 8 ? 'This <b>interleaved</b> distribution spreads even a small array over all channels and banks: here any 16 elements use them all.'
        : 'So a warp reading consecutive data, or many warps reading nearby data, keep several channels and banks busy <b>at the same time</b>.';
    }, 1800);
  };

  // ------------------------------------------------------------ which channels each phase of the tiled matmul uses
  ANIM.tilebanks = (root) => {
    const svg = newSvg(root, '0 0 1000 330', 'The elements of M loaded by four thread blocks in two phases, colored by the DRAM channel that holds them');
    const chColor = ['#86C495', '#E7B467', '#9DB7E3', '#D9A0C8'];
    const loc = (L) => { const p = Math.floor(L / 2); return { c: p % 4, b: Math.floor(p / 4) % 2 }; };
    const cells = [];
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
      const L = r * 4 + c, e = R(svg, 30 + c * 64, 50 + r * 46, 62, 44, { stroke: '#6f6f6f' });
      T(svg, 61 + c * 64, 70 + r * 46, `M${r},${c}`, { ...mid, 'font-size': 12 }); T(svg, 61 + c * 64, 86 + r * 46, `[${L}]`, { ...mid, 'font-size': 10, class: 'lbl' });
      cells.push({ e, L, r, c });
    }
    T(svg, 30, 36, 'M (linear index in brackets)', { 'font-size': 12, class: 'lbl' });
    const chans = [0, 1, 2, 3].map(c => { const r = R(svg, 330 + c * 160, 250, 140, 56, { fill: '#fff', stroke: '#9a9a9a', rx: 4 }); T(svg, 400 + c * 160, 284, `channel ${c}`, { ...mid, 'font-size': 13 }); return r; });
    const rows = svg.appendChild(el('g'));
    const cap = root.querySelector('[data-cap]');
    const P = [
      { ph: 0, blocks: [['block (0,0) and (0,1)', [0, 1, 4, 5]], ['block (1,0) and (1,1)', [8, 9, 12, 13]]] },
      { ph: 1, blocks: [['block (0,0) and (0,1)', [2, 3, 6, 7]], ['block (1,0) and (1,1)', [10, 11, 14, 15]]] }
    ];
    return stepper(root, () => 3, (k) => {
      rows.innerHTML = '';
      const p = P[Math.min(k, 1)], used = new Set(p.blocks.flatMap(b => b[1]));
      cells.forEach(o => { const ch = loc(o.L).c; o.e.setAttribute('fill', k < 2 && used.has(o.L) ? chColor[ch] : '#fff'); o.e.setAttribute('stroke-width', k < 2 && used.has(o.L) ? 2.5 : 1); });
      const usedCh = new Set([...used].map(L => loc(L).c));
      chans.forEach((r, c) => { r.setAttribute('fill', k < 2 && usedCh.has(c) ? chColor[c] : '#fff'); r.setAttribute('stroke-width', k < 2 && usedCh.has(c) ? 2.5 : 1); });
      if (k < 2) p.blocks.forEach(([name, ls], i) => {
        rows.appendChild(el('text', { x: 330, y: 60 + i * 60, 'font-size': 13, 'font-weight': 600 }, `phase ${p.ph}, ${name}`));
        rows.appendChild(el('text', { x: 330, y: 84 + i * 60, 'font-size': 13 }, ls.map(L => { const l = loc(L); return `M[${L}]→ch${l.c} b${l.b}`; }).join('  ')));
      });
      else rows.appendChild(el('text', { x: 330, y: 80, 'font-size': 14 }, 'phase 0 → channels 0 and 2 · phase 1 → channels 1 and 3'));
      cap.innerHTML = k === 0 ? 'Phase 0 of the 4 × 4 tiled product with 2 × 2 blocks. Each block makes two coalesced accesses; together they reach <b>channels 0 and 2</b> at the same time. Blocks (0,0) and (0,1) load the same M elements, and the caches merge those requests.'
        : k === 1 ? 'Phase 1 moves to the elements of <b>channels 1 and 3</b>, again in parallel.'
        : 'Many threads accessing memory at once are what keeps channels and banks busy, one more reason to maximise <b>occupancy</b>. Larger matrices would use all four channels.';
    }, 3200);
  };

  // ------------------------------------------------------------ vector loads
  ANIM.vecload = (root) => {
    let mode = 'scalar';
    const svg = newSvg(root, '0 0 1000 300', 'Vector addition of 16 elements: 16 threads with 4-byte loads, or 4 threads with 16-byte float4 loads');
    const hB = arrowHead(svg, 'vl-b', BLUE);
    const arrs = ['x', 'y', 'z'].map((n, j) => { T(svg, 20, 60 + j * 60, n, { 'font-size': 15, 'font-weight': 600 }); return Array.from({ length: 16 }, (_, i) => R(svg, 50 + i * 58, 40 + j * 60, 54, 32, { fill: '#fff', stroke: '#9a9a9a', rx: 2 })); });
    const lanes = svg.appendChild(el('g')), lines = svg.appendChild(el('g'));
    const cnt = T(svg, 20, 290, '', { 'font-size': 14 });
    const caps = {
      scalar: ['Each of 16 threads loads <b>x[i]</b>: one 4-byte load instruction per thread.', 'Then <b>y[i]</b>: another load each.', 'And stores <b>z[i]</b>. 16 threads × 3 instructions = 48 memory instructions for 16 elements.'],
      vec: ['With <code>float4</code>, each of 4 threads loads 16 bytes of x in <b>one</b> instruction.', 'Then 16 bytes of y.', 'And stores 16 bytes of z. 4 threads × 3 = 12 memory instructions: the loads drop from 32 to 8, a 75 % cut in instruction overhead.']
    };
    const cap = root.querySelector('[data-cap]');
    const st = stepper(root, () => 3, (k) => {
      lanes.innerHTML = ''; lines.innerHTML = '';
      const nT = mode === 'scalar' ? 16 : 4, w = mode === 'scalar' ? 1 : 4;
      for (let t = 0; t < nT; t++) {
        const x = 50 + t * w * 58, wd = w * 58 - 4;
        lanes.appendChild(el('rect', { x, y: 220, width: wd, height: 26, rx: 3, fill: '#C9DBF4' }));
        lanes.appendChild(el('text', { x: x + wd / 2, y: 238, 'text-anchor': 'middle', 'font-size': 11 }, `t${t}`));
        const yTop = 40 + k * 60 + 32;
        lines.appendChild(el('line', { x1: x + wd / 2, y1: 220, x2: x + wd / 2, y2: yTop + 2, stroke: BLUE, 'stroke-width': mode === 'scalar' ? 1.5 : 4, 'marker-end': k < 2 ? '' : hB, 'marker-start': k < 2 ? hB : '' }));
      }
      arrs.forEach((row, j) => row.forEach((e, i) => { fill(e, j === k ? '#9DB7E3' : j < k ? '#DCE7F8' : ''); e.setAttribute('stroke', mode === 'vec' && i % 4 === 0 ? INK : '#9a9a9a'); }));
      const per = mode === 'scalar' ? 16 : 4;
      cnt.textContent = `memory instructions so far: ${per * (k + 1)} (${mode === 'scalar' ? '4 B' : '16 B'} each) · threads launched: ${nT}`;
      cap.innerHTML = caps[mode][k];
    }, 2400);
    modeButtons(root, 'mode', (m) => { mode = m; st.reset(); });
    return st;
  };

  // ------------------------------------------------------------ shared memory banks
  ANIM.banks = (root) => {
    const svg = newSvg(root, '0 0 1000 320', 'The 32 threads of a warp accessing a[threadIdx.x * stride] and the shared-memory bank each access falls into');
    const inp = root.querySelector('[data-in=s]');
    const draw = () => {
      const s = Math.max(1, Math.min(64, +inp.value | 0));
      svg.innerHTML = '';
      const hits = new Array(32).fill(0);
      for (let t = 0; t < 32; t++) hits[(t * s) % 32]++;
      const deg = Math.max(...hits), used = hits.filter(h => h).length;
      T(svg, 20, 22, `thread t accesses a[t × ${s}] → bank (t × ${s}) mod 32`, { 'font-size': 14, 'font-weight': 600 });
      for (let t = 0; t < 32; t++) { R(svg, 20 + t * 30, 34, 26, 20, { fill: '#C9DBF4', rx: 2 }); T(svg, 33 + t * 30, 48, t, { ...mid, 'font-size': 9 }); }
      for (let t = 0; t < 32; t++) { const b = (t * s) % 32; svg.appendChild(el('line', { x1: 33 + t * 30, y1: 54, x2: 33 + b * 30, y2: 120, stroke: hits[b] > 1 ? RED : BLUE, 'stroke-width': 1.2, opacity: .7 })); }
      for (let b = 0; b < 32; b++) {
        R(svg, 20 + b * 30, 120, 26, 24, { fill: '#EDEDED', stroke: '#bdbdbd', rx: 2 }); T(svg, 33 + b * 30, 136, b, { ...mid, 'font-size': 9 });
        const h = hits[b] * 4;
        if (hits[b]) R(svg, 21 + b * 30, 150, 24, h, { fill: hits[b] > 1 ? RED : GREEN, rx: 1 });
        if (hits[b] > 1) T(svg, 33 + b * 30, 162 + h, hits[b], { ...mid, 'font-size': 10, fill: RED });
      }
      T(svg, 20, 116, 'banks (4 B wide)', { 'font-size': 11, class: 'lbl' });
      root.querySelector('[data-out]').innerHTML = `banks used: <b>${used}</b> · conflict degree: <b>${deg}-way</b> → the hardware splits the access into <b>${deg}</b> pass${deg > 1 ? 'es' : ''} (gcd(${s}, 32) = ${gcd(s, 32)})`;
      root.querySelector('[data-cap]').innerHTML = deg === 1 ? 'Every thread reaches a <b>different bank</b>: all 32 accesses are served in parallel.'
        : `${32 / deg} bank${used > 1 ? 's' : ''} receive${used > 1 ? '' : 's'} ${deg} requests each: a <b>${deg}-way bank conflict</b>. The bank can serve one request at a time, so the access takes ${deg} times as long.`;
      syncButtons(root, 'p', String(s));
    };
    inp.addEventListener('input', draw);
    root.querySelectorAll('[data-p]').forEach(b => b.addEventListener('click', () => { inp.value = b.dataset.p; draw(); }));
    return { enter: draw, leave() {} };
  };

  // ------------------------------------------------------------ padding (toy: 8 banks, 8 x 8 tile)
  ANIM.padding = (root) => {
    let mode = 'nopad';
    const svg = newSvg(root, '0 0 1000 330', 'An 8 by 8 shared-memory tile in a toy machine with 8 banks; a warp of 8 threads writes one column, without and with one padding column');
    const NB = 8, cell = 34;
    const pal = ['#86C495', '#E7B467', '#9DB7E3', '#D9A0C8', '#F2C230', '#A7D3D0', '#C9B79C', '#F4A6A6'];
    const draw = (k) => {
      svg.innerHTML = '';
      const cols = mode === 'pad' ? NB + 1 : NB, ty = 2;
      T(svg, 20, 22, mode === 'pad' ? '__shared__ float a[8][8 + 1];  a[tx][ty] = …;' : '__shared__ float a[8][8];  a[tx][ty] = …;', { 'font-size': 14, 'font-weight': 600 });
      for (let r = 0; r < NB; r++) for (let c = 0; c < cols; c++) {
        const lin = r * cols + c, b = lin % NB, pad = c === NB;
        const hit = c === ty && k >= 1;
        R(svg, 20 + c * cell, 36 + r * cell, cell - 2, cell - 2, { fill: pad ? '#fff' : pal[b], 'fill-opacity': pad ? 1 : (hit ? 1 : .35), stroke: hit ? INK : (pad ? '#bdbdbd' : 'none'), 'stroke-width': hit ? 2.5 : 1, 'stroke-dasharray': pad ? '3 3' : '' });
        T(svg, 20 + c * cell + 16, 36 + r * cell + 21, pad ? '' : b, { ...mid, 'font-size': 11 });
      }
      T(svg, 20, 36 + NB * cell + 18, 'number = bank of each element; dashed column = padding, never used', { 'font-size': 11, class: 'lbl' });
      const X = 380;
      T(svg, X, 52, `the warp (tx = 0 … 7, ty = ${ty}) writes column ${ty}:`, { 'font-size': 13 });
      const banks = [];
      for (let tx = 0; tx < NB; tx++) { const lin = tx * cols + ty, b = lin % NB; banks.push(b); if (k >= 1) T(svg, X, 80 + tx * 22, `tx ${tx}: a[${tx}][${ty}] = index ${tx}×${cols}+${ty} = ${lin} → bank ${b}`, { 'font-size': 12, fill: k >= 2 ? (new Set(banks).size === banks.length || mode === 'pad' ? INK : RED) : INK }); }
      const deg = Math.max(...Array.from({ length: NB }, (_, b) => banks.filter(x => x === b).length));
      if (k >= 2) T(svg, X, 270, deg === 1 ? 'all different banks: no conflict' : `all in bank ${banks[0]}: ${deg}-way conflict`, { 'font-size': 15, 'font-weight': 600, fill: deg === 1 ? GREEN : RED });
      const caps = {
        nopad: ['Toy machine: <b>8 banks</b>, consecutive 4-byte words in consecutive banks. The real one has 32 banks and 32 × 32 tiles; the pattern is the same.',
          'A warp has the same threadIdx.y and consecutive threadIdx.x. Writing <code>a[tx][ty]</code> walks down a <b>column</b>: indices 8 apart.',
          'Index tx × 8 + ty always lands in the same bank: an <b>8-way conflict</b> (32-way on the real GPU). The writes are served one after another.'],
        pad: ['Add one unused column: each row is now <b>9</b> words long.',
          'The same column write now has indices 9 apart, so each row shifts the bank by one.',
          'Index tx × 9 + ty → bank (tx + ty) mod 8: all different, <b>no conflict</b>. The price: extra shared memory, and an index multiply by 33 instead of a shift by 5 on the real GPU.']
      };
      root.querySelector('[data-cap]').innerHTML = caps[mode][k];
    };
    const st = stepper(root, () => 3, draw, 2800);
    modeButtons(root, 'mode', (m) => { mode = m; st.reset(); });
    return st;
  };

  // ------------------------------------------------------------ thread coarsening
  ANIM.coarsen = (root) => {
    let mode = 'fine4';
    const svg = newSvg(root, '0 0 1000 300', 'Timelines of execution slots running 16 units of work: one unit per thread or four units per thread, on hardware with 4 or 16 slots');
    const draw = (k) => {
      svg.innerHTML = '';
      const slots = mode.endsWith('16') ? 16 : 4, coarse = mode.startsWith('coarse');
      const threads = coarse ? 4 : 16, per = coarse ? 4 : 1;
      const ov = 1, wk = 1, u = 100, X0 = 120;
      const rows = Math.min(slots, 8);
      const tasks = [];
      for (let t = 0; t < threads; t++) {
        const s = t % slots, wave = Math.floor(t / slots);
        const start = wave * (ov + per * wk);
        tasks.push({ s, start, t });
      }
      const total = Math.max(...tasks.map(o => o.start + ov + per * wk));
      for (let s = 0; s < rows; s++) { T(svg, 20, 30 + s * 30 + 14, `slot ${s}`, { 'font-size': 11 }); R(svg, X0, 30 + s * 30, 8 * u, 22, { fill: '#F7F7F7', rx: 2 }); }
      if (slots > rows) T(svg, 20, 30 + rows * 30 + 14, `… ${slots - rows} more slots`, { 'font-size': 11, class: 'lbl' });
      const lim = (k + 1) / 8 * total;
      tasks.forEach(o => {
        if (o.s >= rows) return;
        const y = 30 + o.s * 30;
        const a = Math.min(lim, o.start), b = Math.min(lim, o.start + ov);
        if (b > a) R(svg, X0 + a * u, y, (b - a) * u - 1, 22, { fill: '#9a9a9a', rx: 2 });
        for (let j = 0; j < per; j++) { const c = o.start + ov + j * wk, d = Math.min(lim, c + wk); if (d > c) R(svg, X0 + c * u, y, (d - c) * u - 1, 22, { fill: BLUE, rx: 2 }); }
        if (lim >= o.start + ov) T(svg, X0 + o.start * u + 4, y + 15, `t${o.t}`, { 'font-size': 10, fill: '#fff' });
      });
      const yb = 30 + (rows + (slots > rows ? 1 : 0)) * 30 + 22;
      svg.setAttribute('viewBox', `0 0 1000 ${yb + 14}`);
      T(svg, X0, yb, `total time: ${total} units · overhead paid ${threads} times · gray = parallelization overhead, blue = useful work`, { 'font-size': 13 });
      const c = {
        fine4: 'One unit per thread, but only 4 slots: the hardware runs the 16 threads in <b>4 waves</b> and pays the overhead 16 times. Total 8 units.',
        coarse4: 'Coarsening factor 4: 4 threads, each doing 4 units. One wave, overhead paid 4 times. Total <b>5</b> units instead of 8.',
        fine16: 'With 16 slots, the fine-grained version runs everything at once: total <b>2</b> units.',
        coarse16: 'Coarsened on the big machine: 12 of the 16 slots sit <b>idle</b> and the total grows to 5. Coarsening only pays when the hardware would serialize the threads anyway.'
      };
      root.querySelector('[data-cap]').innerHTML = c[mode];
    };
    const st = stepper(root, () => 8, draw, 600);
    modeButtons(root, 'mode', (m) => { mode = m; st.reset(); });
    return st;
  };

  ANIM.waves = (root) => {
    const svg = newSvg(root, '0 0 600 170', 'Waves of thread blocks over the GPU, before and after coarsening');
    const q = (n) => root.querySelector(`[data-in=${n}]`), out = root.querySelector('[data-out]');
    const update = () => {
      const B = Math.max(1, +q('b').value | 0), C = Math.max(1, +q('c').value | 0), f = Math.max(1, +q('f').value | 0);
      svg.innerHTML = '';
      const row = (y, blocks, len, label) => {
        const waves = blocks / C, full = Math.ceil(waves - 1e-9);
        T(svg, 20, y + 17, label, { 'font-size': 13 });
        const u = Math.min(60, 440 / Math.max(1, full * len));
        for (let w = 0; w < full; w++) {
          const frac = Math.min(1, waves - w);
          R(svg, 140 + w * len * u, y, len * u - 3, 26, { fill: '#F4F4F4', stroke: '#c9c9c9', rx: 2 });
          R(svg, 140 + w * len * u, y, (len * u - 3) * frac, 26, { fill: frac < 1 ? GOLD : BLUE, rx: 2 });
        }
        return { waves, time: full * len };
      };
      const a = row(30, B, 1, `${fmt(B)} blocks`);
      const nb = Math.ceil(B / f);
      const b = row(90, nb, f, `${fmt(nb)} blocks × ${f}`);
      T(svg, 20, 140, 'one bar per wave; length = time (a coarsened block runs f times longer)', { 'font-size': 11, class: 'lbl' });
      T(svg, 20, 158, 'blue = full wave, gold = partial wave', { 'font-size': 11, class: 'lbl' });
      out.innerHTML = `before: ${a.waves.toFixed(2)} waves → time <b>${a.time}</b> · after coarsening by ${f}: ${b.waves.toFixed(2)} waves → time <b>${b.time}</b>` +
        `<br>${b.waves < 1 ? `<span style="color:#B03A3A">only ${Math.round(b.waves * 100)} % of the GPU is used</span>` : Math.abs(b.waves - Math.round(b.waves)) > 1e-9 ? '<span style="color:#B03A3A">partial last wave: tail effect</span>' : 'whole waves only'} (ignores the overhead saved by coarsening)`;
    };
    root.querySelectorAll('input').forEach(i => i.addEventListener('input', update));
    root.querySelectorAll('[data-p]').forEach(bn => bn.addEventListener('click', () => { const [B, f] = bn.dataset.p.split(','); q('b').value = B; q('f').value = f; update(); }));
    return { enter: update, leave() {} };
  };

  // ------------------------------------------------------------ loop unrolling and instruction scheduling
  ANIM.unroll = (root) => {
    let mode = 'rolled';
    const svg = newSvg(root, '0 0 1000 240', 'Issue timeline of one thread for four iterations: rolled loop, unrolled loop, and unrolled loop with instructions reordered');
    const seqs = {
      rolled: [].concat(...[0, 1, 2, 3].map(i => [`A${i}`, '·', '·', `B${i}`, 'br', '·', '·'])),
      unrolled: [].concat(...[0, 1, 2, 3].map(i => [`A${i}`, '·', '·', `B${i}`]), ['br', '·', '·']),
      sched: ['A0', 'A1', 'A2', 'A3', 'B0', 'B1', 'B2', 'B3', 'br', '·', '·']
    };
    const draw = (k) => {
      svg.innerHTML = '';
      const s = seqs[mode], u = 34;
      const shown = Math.min(s.length, Math.round((k + 1) / 12 * 28));
      for (let i = 0; i < 28; i++) R(svg, 20 + i * u, 50, u - 3, 40, { fill: '#F7F7F7', rx: 2 });
      for (let i = 0; i < shown; i++) {
        const x = s[i], c = x === '·' ? PINK : x === 'br' ? '#9a9a9a' : x[0] === 'A' ? BLUE : GOLD;
        R(svg, 20 + i * u, 50, u - 3, 40, { fill: c, rx: 2 });
        T(svg, 20 + i * u + (u - 3) / 2, 75, x === '·' ? 'stall' : x, { ...mid, 'font-size': x === '·' ? 8 : 12, fill: x === '·' ? RED : '#fff' });
      }
      T(svg, 20, 36, 'cycles →', { 'font-size': 12, class: 'lbl' });
      T(svg, 20, 130, `cycles for 4 iterations: ${s.length} · branches: ${s.filter(x => x === 'br').length} · stall cycles: ${s.filter(x => x === '·').length}`, { 'font-size': 14, 'font-weight': 600 });
      T(svg, 20, 160, 'toy timing: B(i) needs the result of A(i), ready 3 cycles after A starts; a branch stalls the thread 2 cycles (no branch prediction)', { 'font-size': 11, class: 'lbl' });
      const c = {
        rolled: 'The plain loop: after each A(i) the thread waits for its result before B(i), and every iteration ends with a branch that stalls it again.',
        unrolled: 'Unrolled by 4: the body is copied 4 times, so there is one branch instead of four. The waits between A and B remain.',
        sched: 'If A(i+1) does not depend on A(i) or B(i), the copies can be reordered: all A first, then all B. The waits are filled with useful work (<b>instruction scheduling</b>). The compiler does both automatically.'
      };
      root.querySelector('[data-cap]').innerHTML = c[mode];
    };
    const st = stepper(root, () => 12, draw, 350);
    modeButtons(root, 'mode', (m) => { mode = m; st.reset(); });
    return st;
  };

  // ------------------------------------------------------------ double buffering
  ANIM.dbuf = (root) => {
    let mode = 'one';
    const svg = newSvg(root, '0 0 1000 300', 'Timelines of four threads over three iterations, with two barriers per iteration and one buffer, or one barrier and two buffers');
    const rd = [1.0, 1.6, 0.8, 1.2], wr = [1.1, 0.7, 1.5, 0.9];
    const sim = () => {
      const ev = rd.map(() => []); let t = 0; const bars = [];
      for (let it = 0; it < 3; it++) {
        if (mode === 'one') {
          const e1 = rd.map((r, j) => { ev[j].push(['R', t, t + r, it]); return t + r; }); const b1 = Math.max(...e1); e1.forEach((e, j) => ev[j].push(['w', e, b1])); bars.push(b1);
          const e2 = wr.map((w, j) => { ev[j].push(['W', b1, b1 + w, it]); return b1 + w; }); const b2 = Math.max(...e2); e2.forEach((e, j) => ev[j].push(['w', e, b2])); bars.push(b2); t = b2;
        } else {
          const e = rd.map((r, j) => { ev[j].push(['R', t, t + r, it]); ev[j].push(['W', t + r, t + r + wr[j], it]); return t + r + wr[j]; }); const b = Math.max(...e); e.forEach((x, j) => ev[j].push(['w', x, b])); bars.push(b); t = b;
        }
      }
      return { ev, bars, total: t };
    };
    const draw = (k) => {
      svg.innerHTML = '';
      const { ev, bars, total } = sim(), u = 760 / 9.5, X0 = 120, lim = (k + 1) / 10 * total;
      ev.forEach((list, j) => {
        const y = 30 + j * 40;
        T(svg, 20, y + 18, `thread ${j}`, { 'font-size': 12 });
        list.forEach(([kind, a, b, it]) => {
          const e = Math.min(b, lim); if (e <= a) return;
          const c = kind === 'w' ? PINK : kind === 'R' ? BLUE : GOLD;
          R(svg, X0 + a * u, y, (e - a) * u - 1, 26, { fill: c, rx: 2 });
          if (kind !== 'w' && e - a > .5) T(svg, X0 + a * u + 4, y + 17, kind === 'R' ? (mode === 'two' ? `read ${'AB'[it % 2]}` : 'read') : (mode === 'two' ? `write ${'BA'[it % 2]}` : 'write'), { 'font-size': 10, fill: '#fff' });
        });
      });
      bars.forEach(b => { if (b <= lim) L(svg, X0 + b * u, 24, X0 + b * u, 192, { stroke: INK, 'stroke-width': 2.5 }); });
      T(svg, 20, 220, `barriers: ${bars.length} · time for 3 iterations: ${total.toFixed(1)} · pink = waiting at a barrier`, { 'font-size': 14, 'font-weight': 600 });
      T(svg, 20, 248, mode === 'one' ? 'one buffer: read buffer[other]; barrier; write buffer[mine]; barrier' : 'two buffers: read in[other]; write out[mine]; barrier; swap(in, out)', { 'font-size': 12, class: 'lbl' });
      root.querySelector('[data-cap]').innerHTML = mode === 'one'
        ? 'With one buffer, each iteration needs <b>two barriers</b>: after the reads (write-after-read: do not overwrite what others still read) and after the writes (read-after-write: do not read before the value is there). Every barrier makes the fast threads wait for the slowest.'
        : 'Reads come from one buffer and writes go to the other, so a write can never clobber a value someone is still reading: the write-after-read barrier is gone. One barrier per iteration (the true dependence) and the buffers swap roles: <b>double buffering</b>.';
    };
    const st = stepper(root, () => 10, draw, 500);
    modeButtons(root, 'mode', (m) => { mode = m; st.reset(); });
    return st;
  };

  // ------------------------------------------------------------ checklist of optimizations
  ANIM.checklist = (root) => {
    const rows = {
      compute: [
        ['Occupancy tuning', 'more warps to hide pipeline latency', 'more memory accesses in flight to hide DRAM latency', 'tune threads per block, shared memory per block and registers per thread'],
        ['Loop unrolling', 'fewer branches, more independent instructions, fewer stalls', 'local arrays may move to registers, less global traffic', 'mostly done by the compiler; use loops with constant bounds'],
        ['Less control divergence', 'higher SIMD efficiency, fewer idle cores', '—', 'rearrange which thread gets which work or data; rearrange the data layout']],
      memory: [
        ['Coalesced global accesses', 'fewer stalls waiting for global memory', 'less traffic, bursts and cache lines fully used', 'remap threads to data; change the layout (transpose, array of structures → structure of arrays); load coalesced into shared memory and do the irregular part there (corner turning, packing)'],
        ['Shared memory tiling', 'fewer stalls waiting for global memory', 'less global traffic', 'data reused inside a block goes through DRAM only once'],
        ['Register tiling', 'fewer stalls waiting for shared memory', 'less shared memory traffic', 'data reused inside a warp or thread goes to registers once'],
        ['Vector loads and stores', 'fewer instructions per byte moved', 'more accesses in flight', 'each thread moves 16 B chunks of contiguous data'],
        ['No shared memory bank conflicts', 'fewer stalls waiting for shared memory', 'better use of the banks', 'remap threads to data; change the layout (padding)'],
        ['Privatization (later sessions)', 'fewer stalls on atomic updates', 'less contention on shared outputs', 'update a private copy, merge into the public one at the end']],
      sync: [
        ['Warp-level primitives', 'fewer block-wide barriers', 'less global and shared traffic', 'do the synchronizing work inside each warp, then combine warp results'],
        ['Double buffering', 'removes barriers that only enforce false dependences', '—', 'write to a different buffer than the one being read, then swap']],
      general: [
        ['Thread coarsening', 'depends on the overhead of parallelizing', 'depends on the overhead of parallelizing', 'give each thread several units of work when the hardware would serialize them anyway']]
    };
    const names = { compute: 'Compute utilization: keep the cores busy', memory: 'Memory utilization: use the memory system efficiently', sync: 'Synchronization latency: wait less at barriers', general: 'General: category depends on the context' };
    const out = root.querySelector('[data-out]');
    const draw = (cat) => {
      syncButtons(root, 'cat', cat);
      out.innerHTML = `<p style="margin:0 0 .4em"><b>${names[cat]}</b></p><table><tr><th>Optimization</th><th>Benefit for the cores</th><th>Benefit for memory</th><th>How</th></tr>` +
        rows[cat].map(r => `<tr><td><b>${r[0]}</b></td><td>${r[1]}</td><td>${r[2]}</td><td>${r[3]}</td></tr>`).join('') + '</table>';
    };
    root.querySelectorAll('[data-cat]').forEach(b => b.addEventListener('click', () => draw(b.dataset.cat)));
    return { enter: () => draw(root.querySelector('[data-cat].on')?.dataset.cat || 'compute'), leave() {} };
  };

  // ------------------------------------------------------------ optimization strategy
  ANIM.strategy = (root) => {
    let bn = 'bw';
    const svg = newSvg(root, '0 0 1000 300', 'The optimization cycle: measure, find the bottleneck, apply an optimization that relieves it, measure again');
    const nodes = [['measure', 'profiler, roofline'], ['find the bottleneck', 'which resource limits it?'], ['pick an optimization', 'that relieves that resource'], ['check side effects', 'does it overload another?'], ['measure again', 'close to the roof? stop']];
    const pos = nodes.map((_, i) => { const a = -Math.PI / 2 + i * 2 * Math.PI / 5; return [220 + 170 * Math.cos(a), 150 + 120 * Math.sin(a)]; });
    const hG = arrowHead(svg, 'st-g', GRAY);
    pos.forEach((p, i) => { const q = pos[(i + 1) % 5]; svg.appendChild(el('line', { x1: p[0] + (q[0] - p[0]) * .3, y1: p[1] + (q[1] - p[1]) * .3, x2: p[0] + (q[0] - p[0]) * .7, y2: p[1] + (q[1] - p[1]) * .7, stroke: GRAY, 'stroke-width': 2, 'marker-end': hG })); });
    const nb = nodes.map(([a, b], i) => { const [x, y] = pos[i]; const r = R(svg, x - 85, y - 24, 170, 48, { class: 'mem', rx: 6 }); T(svg, x, y - 4, a, { ...mid, 'font-size': 13, 'font-weight': 600 }); T(svg, x, y + 14, b, { ...mid, 'font-size': 10, class: 'lbl' }); return r; });
    const side = svg.appendChild(el('g'));
    const B = {
      bw: ['global memory bandwidth (memory-bound, far below the roof)', ['coalesced accesses', 'shared memory tiling', 'register tiling', 'vector loads'], ['watch shared memory and registers: they can cut occupancy']],
      occ: ['occupancy, already limited by shared memory per block', ['smaller tiles or less shared memory per block', 'tune threads per block'], ['adding more shared memory tiling makes it worse']],
      ins: ['instruction overhead (address math, loop control)', ['vector loads and stores', 'loop unrolling', 'thread coarsening'], ['too much coarsening leaves the GPU partly empty']],
      sync: ['waiting at barriers', ['double buffering', 'warp-level primitives'], ['removing a barrier that guards a true dependence is a bug']],
      div: ['control divergence (idle lanes in warps)', ['remap work and data to threads so a warp takes one path', 'rearrange the data layout'], ['check that coalescing survives the remapping']]
    };
    const cap = root.querySelector('[data-cap]');
    const st = stepper(root, () => 5, (k) => {
      syncButtons(root, 'bn', bn);
      nb.forEach((r, i) => r.classList.toggle('on', i === k));
      side.innerHTML = '';
      const [name, good, bad] = B[bn];
      const add = (y, s, c, w) => side.appendChild(el('text', { x: 470, y, 'font-size': 13, fill: c, 'font-weight': w || 400 }, s));
      add(40, `bottleneck: ${name}`, INK, 600);
      add(74, 'try:', GREEN, 600); good.forEach((g, i) => add(98 + i * 22, `✓ ${g}`, GREEN));
      add(110 + good.length * 22, 'careful:', RED, 600); bad.forEach((g, i) => add(134 + good.length * 22 + i * 22, `✗ ${g}`, RED));
      cap.innerHTML = ['<b>Measure</b> first: a profiler (Nsight Compute) reports how close each resource is to its limit.',
        'The <b>bottleneck</b> is the resource that limits this kernel on this GPU. Another GPU may have a different one.',
        'Choose an optimization that relieves <b>that</b> resource. Optimizations trade one resource for another; relieving one that is not the bottleneck gains nothing.',
        'Check that it does not overload a resource that then becomes the new bottleneck (for example shared memory tiling when occupancy is already limited by shared memory).',
        'Measure again and repeat. Stop when the kernel is close to the hardware limit of the roofline: there is little left to gain.'][k];
    }, 2600);
    root.querySelectorAll('[data-bn]').forEach(b => b.addEventListener('click', () => { bn = b.dataset.bn; st.reset(); }));
    return st;
  };
