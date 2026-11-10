  // ============================================================ memory architecture (Chapter 5, 5.1–5.2)
  const arrowHead = (svg, id, color) => {
    const d = svg.querySelector('defs') || svg.insertBefore(el('defs'), svg.firstChild);
    const m = el('marker', { id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' });
    m.appendChild(el('path', { d: 'M0,0 L10,5 L0,10 z', fill: color })); d.appendChild(m);
    return `url(#${id})`;
  };
  const GRAY = '#9a9a9a', BLUE = '#2F6FD0', GOLD = '#C98A1B', RED = '#B03A3A', GREEN = '#2E7D4F';
  const place = (e, x, y) => e.setAttribute('transform', `translate(${x},${y})`);
  const vis = (e, on) => e.setAttribute('opacity', on ? 1 : 0);

  // ------------------------------------------------------------ the memory wall: latency and congestion
  ANIM.wall = (root) => {
    const svg = newSvg(root, '0 0 1000 380', 'An SM whose warps send load requests over a narrow path to off-chip DRAM; when requests exceed the bandwidth a queue forms and the cores sit idle');
    const head = arrowHead(svg, 'wl-a', GRAY);
    R(svg, 10, 10, 380, 300, { class: 'frame', rx: 5 }); T(svg, 24, 34, 'SM (on chip)', { 'font-size': 15, 'font-weight': 600 });
    const warps = [];
    for (let i = 0; i < 8; i++) {
      const y = 50 + i * 30;
      warps.push(R(svg, 30, y, 150, 22, { class: 'chip', rx: 2 })); T(svg, 105, y + 16, `warp ${i}`, { ...mid, 'font-size': 12 });
    }
    const onchip = R(svg, 200, 50, 170, 232, { class: 'mem', rx: 3 });
    T(svg, 285, 150, 'registers +', { ...mid, 'font-size': 13 }); T(svg, 285, 170, 'shared memory', { ...mid, 'font-size': 13 });
    T(svg, 285, 194, '(on chip)', { ...mid, 'font-size': 12, class: 'lbl' });
    R(svg, 440, 150, 300, 30, { fill: '#F4F4F4', stroke: '#c9c9c9', rx: 3 });
    T(svg, 590, 140, 'path to global memory: limited bandwidth', { ...mid, 'font-size': 12, class: 'lbl' });
    L(svg, 390, 165, 440, 165, { class: 'wire', 'marker-end': head }); L(svg, 740, 165, 790, 165, { class: 'wire', 'marker-end': head });
    const dram = R(svg, 790, 60, 200, 210, { class: 'mem', rx: 4 });
    T(svg, 890, 150, 'Global memory', { ...mid, 'font-size': 15 }); T(svg, 890, 172, 'off-chip DRAM', { ...mid, 'font-size': 13, class: 'lbl' });
    T(svg, 890, 194, '3.35 TB/s on H100', { ...mid, 'font-size': 12, class: 'lbl' });
    const lat = T(svg, 590, 222, '', { ...mid, 'font-size': 13, fill: GOLD });
    const toks = [];
    for (let i = 0; i < 12; i++) { const g = svg.appendChild(el('g', { class: 'tok' })); g.appendChild(el('circle', { cx: 0, cy: 0, r: 8, fill: GOLD })); toks.push(g); }
    R(svg, 30, 330, 340, 18, { fill: '#EDEDED', rx: 2 }); const gauge = R(svg, 30, 330, 0, 18, { class: 'bar', rx: 2 });
    const gtxt = T(svg, 380, 344, '', { 'font-size': 13 }); T(svg, 30, 322, 'compute units busy', { 'font-size': 12, class: 'lbl' });
    const qtxt = T(svg, 590, 300, '', { ...mid, 'font-size': 13 });
    // per step: busy warps, tokens [x, y] (null = hidden), utilisation, queue text
    const lane = (i) => [460 + i * 37, 165];
    const qpos = [[415, 118], [415, 138], [415, 192], [415, 212]];
    const full = [...Array.from({ length: 8 }, (_, i) => lane(i)), ...qpos];
    const S = [
      { busy: [], wait: [0], tok: [lane(3)], util: .05, lat: '≈ hundreds of clock cycles each way', q: '' },
      { busy: [1, 2, 3, 4, 5], wait: [0, 6, 7], tok: [lane(1), lane(4), lane(7)], util: .62, lat: 'latency hidden by switching warps', q: 'requests in flight: 3, within bandwidth' },
      { busy: [3], wait: [0, 1, 2, 4, 5, 6, 7], tok: full, util: .12, lat: '', q: 'requests exceed the bandwidth: a queue forms' },
      { busy: [3], wait: [0, 1, 2, 4, 5, 6, 7], tok: full, util: .12, lat: 'cores idle, waiting for data', q: 'congestion: the kernel is memory-bound' },
      { busy: [0, 1, 2, 3, 4, 5, 6], wait: [7], tok: [lane(2), lane(6)], util: .85, lat: '', q: 'reused data served on chip: few DRAM requests' }
    ];
    const caps = [
      'One warp loads an operand from <b>global memory</b>. It is off-chip DRAM: the request and the data travel for <b>hundreds of cycles</b>.',
      'Many resident warps hide that latency (session 12): while warp 0 waits, the SM runs warps that already have their operands.',
      'But each thread of our kernels so far loads data for almost every operation. All warps send requests at once and the path to DRAM <b>saturates</b>.',
      '<b>Traffic congestion</b>: most warps wait for memory, the cores sit idle, and the kernel reaches only a tiny fraction of the hardware peak.',
      'The fix of this chapter: keep data that is <b>reused</b> in on-chip memories (registers, shared memory). Redundant DRAM traffic disappears and the cores work again.'
    ];
    const cap = root.querySelector('[data-cap]');
    return stepper(root, () => S.length, (k) => {
      const s = S[k];
      warps.forEach((w, i) => { w.classList.toggle('run', s.busy.includes(i)); if (s.wait.includes(i)) w.setAttribute('style', 'fill:#F6DADA;stroke:#B03A3A'); else w.removeAttribute('style'); });
      toks.forEach((t, i) => { const p = s.tok[i]; vis(t, !!p); if (p) place(t, p[0], p[1]); });
      gauge.setAttribute('width', 340 * s.util); gtxt.textContent = Math.round(s.util * 100) + ' %';
      lat.textContent = s.lat; qtxt.textContent = s.q; qtxt.setAttribute('fill', k === 2 || k === 3 ? RED : '#1A1A1A');
      dram.classList.toggle('on', k === 2 || k === 3); onchip.classList.toggle('on', k === 4);
      cap.innerHTML = caps[k];
    }, 3000);
  };

  // ------------------------------------------------------------ compute-bound or memory-bound: who waits?
  ANIM.bound = (root) => {
    const PEAK = 66.9, BW = 3.35, RIDGE = PEAK / BW;
    const kernels = {
      va: { name: 'vector addition', I: 1 / 12, note: '1 FLOP per 12 B' },
      mm: { name: 'naive matrix product', I: 0.25, note: '2 FLOP per 8 B' },
      tile: { name: 'tiled product, 32 × 32', I: 8, note: '0.25 × 32' },
      hi: { name: 'kernel with 40 FLOP/B', I: 40, note: 'above the ridge' }
    };
    let mode = 'va';
    const svg = newSvg(root, '0 0 1000 300', 'Timelines of the memory channel and the compute cores for kernels of different compute-to-global-memory-access ratio');
    T(svg, 20, 30, '', { 'font-size': 15, 'font-weight': 600 });
    const title = svg.lastChild;
    T(svg, 20, 92, 'memory channel', { 'font-size': 13, class: 'lbl' }); T(svg, 20, 172, 'compute cores', { 'font-size': 13, class: 'lbl' });
    const N = 20, X0 = 170, W = 780;
    const mCells = [], cCells = [];
    for (let i = 0; i < N; i++) {
      mCells.push(R(svg, X0 + i * W / N, 70, W / N - 3, 34, { fill: '#EDEDED', rx: 2 }));
      cCells.push(R(svg, X0 + i * W / N, 150, W / N - 3, 34, { fill: '#EDEDED', rx: 2 }));
    }
    const cursor = L(svg, X0, 56, X0, 198, { stroke: '#1A1A1A', 'stroke-width': 1.5, 'stroke-dasharray': '4 3' });
    T(svg, X0, 216, 'time →', { 'font-size': 12, class: 'lbl' });
    const res = T(svg, 20, 252, '', { 'font-size': 14 }), res2 = T(svg, 20, 278, '', { 'font-size': 14 });
    const cap = root.querySelector('[data-cap]');
    const draw = (k) => {
      const kn = kernels[mode], I = kn.I;
      const mU = Math.min(1, RIDGE / I), cU = Math.min(1, I / RIDGE);
      const on = (u, i) => Math.floor((i + 1) * u + 1e-9) > Math.floor(i * u + 1e-9);
      for (let i = 0; i < N; i++) {
        const vis = i <= k;
        mCells[i].setAttribute('fill', !vis ? '#F7F7F7' : on(mU, i) ? GOLD : '#F6DADA');
        cCells[i].setAttribute('fill', !vis ? '#F7F7F7' : on(cU, i) ? BLUE : '#F6DADA');
      }
      const x = X0 + (k + 1) * W / N; cursor.setAttribute('x1', x); cursor.setAttribute('x2', x);
      title.textContent = `${kn.name}: ${I < 1 ? I.toFixed(3) : I} FLOP/B (${kn.note})`;
      const perf = Math.min(PEAK, BW * I);
      const pc = (u) => u < .095 ? (u * 100).toFixed(1) : Math.round(u * 100);
      res.textContent = `memory busy ${pc(mU)} % · compute busy ${pc(cU)} % · idle = pink`;
      res2.textContent = `${I < RIDGE ? 'memory-bound' : 'compute-bound'}: at most ${perf < 1 ? perf.toFixed(2) : perf.toFixed(1)} TFLOPS = ${(perf / PEAK * 100).toFixed(perf / PEAK < .01 ? 2 : 0)} % of the 66.9 TFLOPS peak`;
      res2.setAttribute('fill', I < RIDGE ? RED : GREEN);
      cap.innerHTML = I < RIDGE
        ? `Ratio ${I < 1 ? I.toFixed(3) : I} &lt; ${RIDGE.toFixed(1)} FLOP/B (the H100 ridge, 66.9 / 3.35): the <b>memory channel is always busy</b> and the cores wait. Speed = bandwidth × ratio.`
        : `Ratio ${I} &gt; ${RIDGE.toFixed(1)} FLOP/B: the <b>cores are always busy</b> and memory has spare time. Only the compute peak limits the kernel.`;
    };
    const st = stepper(root, () => N, draw, 260);
    modeButtons(root, 'mode', (m) => { mode = m; st.reset(); });
    return st;
  };

  // ------------------------------------------------------------ Fig. 5.1: who can read and write each memory
  ANIM.reach = (root) => {
    const svg = newSvg(root, '0 0 1000 420', 'CUDA device memory model with the memories each role can read or write: host, a thread of block 0, another thread of block 0, a thread of block 1');
    const hB = arrowHead(svg, 'rc-b', BLUE);
    R(svg, 6, 6, 150, 408, { class: 'frame', rx: 4 });
    const host = R(svg, 20, 318, 122, 76, { fill: '#fff', stroke: '#6f6f6f', rx: 3 }); T(svg, 81, 352, 'Host', { ...mid, 'font-size': 15, 'font-weight': 600 }); T(svg, 81, 372, '(CPU code)', { ...mid, 'font-size': 12, class: 'lbl' });
    const list = [0, 1, 2, 3, 4, 5, 6].map(i => T(svg, 18, 34 + i * 22, '', { 'font-size': 12 }));
    R(svg, 166, 6, 828, 408, { class: 'frame', rx: 4 }); T(svg, 180, 26, 'Device (grid)', { 'font-size': 13, 'font-weight': 600 });
    const M = {}, who = {}, TX = {};
    [0, 1].forEach(b => {
      const bx = 182 + b * 404;
      R(svg, bx, 36, 392, 196, { fill: 'none', stroke: '#6f6f6f', 'stroke-width': 1.5, rx: 4 }); T(svg, bx + 12, 54, `Block (${b}, 0)`, { 'font-size': 12 });
      M[`sh${b}`] = R(svg, bx + 14, 62, 364, 34, { class: 'mem', rx: 3 }); T(svg, bx + 196, 84, 'Shared memory', { ...mid, 'font-size': 13 });
      [0, 1].forEach(t => {
        const tx = bx + 16 + t * 188; TX[`${b}${t}`] = tx;
        M[`rg${b}${t}`] = R(svg, tx, 116, 112, 30, { class: 'mem', rx: 3 }); T(svg, tx + 56, 136, 'Registers', { ...mid, 'font-size': 12 });
        who[`t${b}${t}`] = R(svg, tx, 176, 168, 40, { fill: '#C9DBF4', stroke: 'none', rx: 3 }); T(svg, tx + 84, 201, `Thread (${t}, 0)`, { ...mid, 'font-size': 12 });
      });
    });
    M.gl = R(svg, 182, 262, 796, 58, { class: 'mem', rx: 3 }); T(svg, 580, 288, 'Global memory', { ...mid, 'font-size': 15 }); T(svg, 580, 308, 'large, off chip, slow', { ...mid, 'font-size': 12, class: 'lbl' });
    M.co = R(svg, 182, 346, 796, 58, { class: 'mem', rx: 3 }); T(svg, 580, 372, 'Constant memory', { ...mid, 'font-size': 15 }); T(svg, 580, 392, '64 KB, cached, read-only for the device', { ...mid, 'font-size': 12, class: 'lbl' });
    const arrows = svg.appendChild(el('g'));
    const arrow = (x1, y1, x2, y2, label, lx, ly, dash) => {
      const l = arrows.appendChild(el('line', { x1, y1, x2, y2, stroke: BLUE, 'stroke-width': 2.5, 'marker-end': hB }));
      if (label === 'R/W') l.setAttribute('marker-start', hB);
      if (dash) l.setAttribute('stroke-dasharray', '6 4');
      arrows.appendChild(el('rect', { x: lx - 3, y: ly - 12, width: label.length * 8.5 + 6, height: 16, fill: '#fff', rx: 2 }));
      arrows.appendChild(el('text', { x: lx, y: ly, 'font-size': 12, 'font-weight': 600, fill: BLUE }, label));
    };
    const threadArrows = (b, t) => {
      const tx = TX[`${b}${t}`];
      arrow(tx + 56, 174, tx + 56, 148, 'R/W', tx + 62, 166);
      arrow(tx + 140, 174, tx + 140, 98, 'R/W', tx + 146, 140);
      arrow(tx + 30, 218, tx + 30, 260, 'R/W', tx + 36, 246);
      arrow(tx + 100, 218, tx + 100, 344, 'R', tx + 106, 338, true);
    };
    const roles = [
      { id: 'host', label: ['Host'], rw: { gl: 'R/W', co: 'R/W' }, no: ['sh0', 'sh1', 'rg00', 'rg01', 'rg10', 'rg11'],
        draw: () => { arrow(142, 336, 180, 296, 'R/W', 146, 306); arrow(142, 380, 180, 376, 'R/W', 146, 366); },
        cap: 'The <b>host</b> reads and writes global memory (<code>cudaMemcpy</code>) and writes constant memory (<code>cudaMemcpyToSymbol</code>). It cannot touch registers or shared memory: they exist only while a kernel runs.' },
      { id: 't00', label: ['Thread (0,0)', 'of block 0'], rw: { rg00: 'R/W', sh0: 'R/W', gl: 'R/W', co: 'R' }, no: ['rg01', 'sh1', 'rg10', 'rg11'], draw: () => threadArrows(0, 0),
        cap: 'A <b>thread</b> reads and writes its own registers, the shared memory of <b>its</b> block and global memory, and only <b>reads</b> constant memory.' },
      { id: 't01', label: ['Thread (1,0)', 'of block 0'], rw: { rg01: 'R/W', sh0: 'R/W', gl: 'R/W', co: 'R' }, no: ['rg00', 'sh1', 'rg10', 'rg11'], draw: () => threadArrows(0, 1),
        cap: 'Its neighbour in the same block sees the <b>same shared memory</b>, which is how the two cooperate, but never the other thread\'s registers.' },
      { id: 't10', label: ['Thread (0,0)', 'of block 1'], rw: { rg10: 'R/W', sh1: 'R/W', gl: 'R/W', co: 'R' }, no: ['sh0', 'rg00', 'rg01', 'rg11'], draw: () => threadArrows(1, 0),
        cap: 'A thread of <b>another block</b> has its own copy of the shared variables. To exchange data with block 0 it must go through <b>global memory</b>.' }
    ];
    const cap = root.querySelector('[data-cap]');
    const draw = (k) => {
      const r = roles[k]; arrows.innerHTML = '';
      root.querySelectorAll('[data-role]').forEach(b => b.classList.toggle('on', b.dataset.role === r.id));
      Object.entries(M).forEach(([key, rect]) => { rect.classList.toggle('on', key in r.rw); rect.setAttribute('opacity', r.no.includes(key) ? .35 : 1); });
      Object.entries(who).forEach(([key, rect]) => rect.setAttribute('fill', key === r.id ? BLUE : '#C9DBF4'));
      host.setAttribute('stroke', r.id === 'host' ? BLUE : '#6f6f6f'); host.setAttribute('stroke-width', r.id === 'host' ? 3 : 1);
      r.draw();
      const acc = (p) => { const v = Object.entries(r.rw).find(([key]) => key.startsWith(p)); return v ? v[1] : 'none'; };
      const lines = [...r.label, '', `registers: ${acc('rg')}`, `shared: ${acc('sh')}`, `global: ${acc('gl')}`, `constant: ${acc('co')}`];
      list.forEach((t, i) => { const s2 = lines[i] || ''; t.textContent = s2; t.setAttribute('font-weight', i < r.label.length ? 600 : 400); t.setAttribute('fill', s2.endsWith('none') ? RED : '#1A1A1A'); });
      cap.innerHTML = r.cap;
    };
    const st = stepper(root, () => roles.length, draw, 3600);
    modeButtons(root, 'role', (m) => { st.leave(); draw(roles.findIndex(r => r.id === m)); });
    return st;
  };

  // ------------------------------------------------------------ memory hierarchy of an H100
  ANIM.pyramid = (root) => {
    const svg = newSvg(root, '0 0 1000 400', 'Memory hierarchy of an H100 from registers at the top to global memory at the bottom, with capacity, scope and approximate latency');
    const levels = [
      { name: 'Registers', size: '256 KB per SM (65,536 × 4 B)', scope: 'one thread', lat: '≈ 1 cycle', w: 220 },
      { name: 'Shared memory / L1', size: 'up to 228 KB shared per SM', scope: 'one block', lat: '≈ tens of cycles', w: 380 },
      { name: 'L2 cache', size: '50 MB for the whole GPU', scope: 'all SMs (hardware-managed)', lat: '≈ a few hundred cycles', w: 560 },
      { name: 'Global memory (HBM3)', size: '80 GB · 3.35 TB/s', scope: 'every thread + the host', lat: '≈ hundreds of cycles', w: 740 }
    ];
    const H = 70, cx = 400, rects = [], txt = [];
    levels.forEach((lv, i) => {
      const y = 16 + i * (H + 10), wTop = i === 0 ? 120 : levels[i - 1].w, wB = lv.w;
      const p = svg.appendChild(el('polygon', { points: `${cx - wTop / 2},${y} ${cx + wTop / 2},${y} ${cx + wB / 2},${y + H} ${cx - wB / 2},${y + H}`, class: 'mem' }));
      rects.push(p); txt.push(T(svg, cx, y + 30, lv.name, { ...mid, 'font-size': 15, 'font-weight': 600 }));
      T(svg, cx, y + 52, lv.size, { ...mid, 'font-size': 12, class: 'lbl' });
    });
    const side = [
      { name: 'Constant memory', note: '64 KB, in DRAM, cached; read-only for kernels', y: 186 },
      { name: 'Local memory', note: 'per thread, lives in DRAM (arrays, spills)', y: 266 }
    ].map(s => { const r = R(svg, 790, s.y, 200, 62, { class: 'mem', rx: 3, 'stroke-dasharray': '4 3' }); T(svg, 890, s.y + 26, s.name, { ...mid, 'font-size': 13, 'font-weight': 600 }); T(svg, 890, s.y + 46, s.note.split(';')[0], { ...mid, 'font-size': 10, class: 'lbl' }); return r; });
    const up = arrowHead(svg, 'py-u', GRAY);
    L(svg, 40, 340, 40, 30, { stroke: GRAY, 'stroke-width': 2, 'marker-end': up }); T(svg, 52, 40, 'faster, smaller', { 'font-size': 12, class: 'lbl' });
    T(svg, 52, 336, 'larger, slower', { 'font-size': 12, class: 'lbl' });
    const info = T(svg, cx, 386, '', { ...mid, 'font-size': 13 });
    const caps = [
      'The memories form a <b>hierarchy</b>: the closer to the cores, the faster and the smaller. H100 (SXM) figures; latencies are rough orders of magnitude.',
      '<b>Registers</b>: private to a thread, used directly as instruction operands. 65,536 per SM: at 2048 threads that is only 32 registers per thread.',
      '<b>Shared memory</b>: on chip, shared by the threads of a block, managed by the program (a scratchpad). It shares the SM\'s 256 KB with the L1 cache.',
      '<b>L2 cache</b>: on chip, shared by all SMs and managed by the hardware. It absorbs part of the global traffic, but the program does not control it.',
      '<b>Global memory</b>: off-chip DRAM, visible to all threads and to the host. Everything starts and ends here, so its bandwidth bounds memory-bound kernels.',
      '<b>Constant</b> and <b>local</b> memory are regions of the same DRAM: constant is cached and read-only; local is private per thread but as slow as global.'
    ];
    const cap = root.querySelector('[data-cap]');
    return stepper(root, () => caps.length, (k) => {
      rects.forEach((r, i) => r.classList.toggle('on', k === i + 1));
      side.forEach(r => r.classList.toggle('on', k === 5));
      info.textContent = k >= 1 && k <= 4 ? `scope: ${levels[k - 1].scope} · latency: ${levels[k - 1].lat}` : '';
      cap.innerHTML = caps[k];
    }, 3200);
  };

  // ------------------------------------------------------------ Fig. 5.2: von Neumann processor, fetch and operands
  ANIM.fetch = (root) => {
    const svg = newSvg(root, '0 0 1000 420', 'Von Neumann model: memory and I/O outside the processor; inside, a processing unit with ALU and register file and a control unit with PC and IR');
    const hG = arrowHead(svg, 'fe-g', '#bdbdbd'), hO = arrowHead(svg, 'fe-o', GOLD);
    const mem = R(svg, 250, 10, 420, 64, { class: 'mem', rx: 3 }); T(svg, 460, 36, 'Memory', { ...mid, 'font-size': 16, 'font-weight': 600 });
    const memT = T(svg, 460, 60, '', { ...mid, 'font-size': 12 });
    R(svg, 740, 10, 140, 64, { class: 'mem', rx: 3 }); T(svg, 810, 48, 'I/O', { ...mid, 'font-size': 16 });
    L(svg, 670, 32, 740, 32, { class: 'wire', 'marker-end': hG }); L(svg, 740, 52, 670, 52, { class: 'wire', 'marker-end': hG });
    R(svg, 140, 104, 760, 306, { class: 'frame', rx: 5 }); T(svg, 520, 400, 'Processor', { ...mid, 'font-size': 15, 'font-weight': 600 });
    const pu = R(svg, 320, 124, 360, 120, { fill: 'none', stroke: '#6f6f6f', rx: 3 }); T(svg, 500, 144, 'Processing unit', { ...mid, 'font-size': 14 });
    const alu = svg.appendChild(el('polygon', { points: '340,160 470,160 450,232 360,232', class: 'mem' }));
    T(svg, 405, 196, 'ALU', { ...mid, 'font-size': 15 }); const aluT = T(svg, 405, 218, '', { ...mid, 'font-size': 12, 'font-weight': 600 });
    const rf = R(svg, 500, 156, 164, 80, { class: 'sp', rx: 3 }); T(svg, 582, 174, 'Register file', { ...mid, 'font-size': 13 });
    const regT = [0, 1, 2].map(i => T(svg, 582, 194 + i * 15, '', { ...mid, 'font-size': 12 }));
    const cu = R(svg, 200, 280, 640, 92, { fill: 'none', stroke: '#6f6f6f', rx: 3 }); T(svg, 520, 300, 'Control unit', { ...mid, 'font-size': 14 });
    const pc = R(svg, 230, 316, 200, 40, { class: 'mem', rx: 3 }); const pcT = T(svg, 330, 341, '', { ...mid, 'font-size': 13 });
    const ir = R(svg, 590, 316, 220, 40, { class: 'mem', rx: 3 }); const irT = T(svg, 700, 341, '', { ...mid, 'font-size': 12 });
    const wPC = L(svg, 290, 316, 290, 74, { class: 'wire', 'marker-end': hG });
    const wIR = L(svg, 720, 74, 720, 316, { class: 'wire', 'marker-end': hG });
    const wData = L(svg, 600, 74, 600, 156, { class: 'wire', 'marker-end': hG });
    const wCtl = L(svg, 500, 280, 500, 244, { class: 'wire', 'stroke-dasharray': '5 4', 'marker-end': hG });
    const wOp = L(svg, 500, 196, 470, 196, { class: 'wire blue', 'marker-end': hG });
    const S = [
      { pc: 'PC = 100', mem: '100: fadd r1, r2, r3   104: load r2, r4, 8', ir: '', on: [wPC, pc], r: ['r1 = ?', 'r2 = 2.5', 'r3 = 1.0'] },
      { pc: 'PC = 100', mem: 'returns the instruction at 100', ir: 'IR: fadd r1, r2, r3', on: [wIR, ir, mem], r: ['r1 = ?', 'r2 = 2.5', 'r3 = 1.0'] },
      { pc: 'PC = 104', mem: '', ir: 'IR: fadd r1, r2, r3', on: [wCtl], r: ['r1 = ?', 'r2 = 2.5', 'r3 = 1.0'] },
      { pc: 'PC = 104', mem: '', ir: 'IR: fadd r1, r2, r3', on: [wOp, rf, alu], alu: '2.5 + 1.0', r: ['r1 = 3.5', 'r2 = 2.5', 'r3 = 1.0'] },
      { pc: 'PC = 104', mem: 'returns the instruction at 104', ir: 'IR: load r2, r4, 8', on: [wPC, wIR, mem, ir], r: ['r1 = 3.5', 'r2 = 2.5', 'r4 = 2000'] },
      { pc: 'PC = 108', mem: 'reads address 2000 + 8 → 7.0', ir: 'IR: load r2, r4, 8', on: [wData, mem, rf], r: ['r1 = 3.5', 'r2 = 7.0', 'r4 = 2000'] },
      { pc: 'PC = 108', mem: '', ir: '', on: [rf, pu], r: ['r1 = 3.5', 'r2 = 7.0', 'r4 = 2000'] }
    ];
    const caps = [
      'Fig. 5.2. Every instruction starts in <b>memory</b>: the control unit sends the address in the <b>program counter (PC)</b>.',
      'Memory returns the instruction, and it is stored in the <b>instruction register (IR)</b>.',
      'The control unit decodes it and drives the processing unit (dashed line). The PC moves on to the next instruction.',
      '<b>fadd r1, r2, r3</b>: both operands are already in the <b>register file</b>, inside the processor. The ALU adds them with no further memory access.',
      'The next instruction is fetched the same way: <b>load r2, r4, 8</b>.',
      'A load is a second trip to memory, this time for <b>data</b>: address r4 + 8, and the value lands in r2. Only now can an ALU instruction use it.',
      'Registers are <b>inside</b> the processor; memory is <b>outside</b>. An operand in memory costs an extra instruction and a slow trip. In a GPU, global memory plays the role of this memory.'
    ];
    const all = [wPC, wIR, wData, wCtl, wOp, pc, ir, mem, rf, alu];
    const cap = root.querySelector('[data-cap]');
    return stepper(root, () => S.length, (k) => {
      const s = S[k];
      all.forEach(e => e.classList.toggle('on', s.on.includes(e)));
      [wPC, wIR, wData].forEach(w => w.setAttribute('marker-end', s.on.includes(w) ? hO : hG));
      pu.setAttribute('stroke', s.on.includes(pu) ? BLUE : '#6f6f6f'); pu.setAttribute('stroke-width', s.on.includes(pu) ? 3 : 1);
      pcT.textContent = s.pc; memT.textContent = s.mem; irT.textContent = s.ir || 'IR'; aluT.textContent = s.alu || '';
      regT.forEach((t, i) => { t.textContent = s.r[i]; t.setAttribute('fill', s.on.includes(rf) ? '#fff' : '#1A1A1A'); });
      cap.innerHTML = caps[k];
    }, 2900);
  };

  // ------------------------------------------------------------ three reasons for registers, and their cost
  ANIM.regwin = (root) => {
    const svg = newSvg(root, '0 0 1000 300', 'Bars comparing registers and global memory in latency, instructions and energy');
    const panels = [
      { t: 'latency (log scale)', a: ['register', 1, '≈ 1 cycle'], b: ['global', 500, '≈ hundreds'], log: true },
      { t: 'instructions for one fadd', a: ['register', 1, '1: fadd'], b: ['global', 2, '2: load + fadd'], log: false, max: 2 },
      { t: 'energy per access', a: ['register', 1, '1 ×'], b: ['global', 10, '≥ 10 ×'], log: false, max: 10 }
    ];
    const bars = [];
    panels.forEach((p, i) => {
      const x = 30 + i * 330;
      T(svg, x, 30, p.t, { 'font-size': 14, 'font-weight': 600 });
      const sc = (v) => p.log ? Math.log10(v) / Math.log10(p.b[1]) : v / p.max;
      [p.a, p.b].forEach((d, j) => {
        const y = 56 + j * 60;
        T(svg, x, y + 24, d[0], { 'font-size': 13, class: 'lbl' });
        R(svg, x + 70, y, 220, 34, { fill: '#F4F4F4', rx: 2 });
        const b = R(svg, x + 70, y, 0, 34, { class: j ? 'bar2' : 'bar', rx: 2 });
        const lbl = T(svg, x + 76, y + 23, '', { 'font-size': 13, fill: '#fff', 'font-weight': 600 });
        bars.push({ b, lbl, w: Math.max(14, 220 * sc(d[1])), text: d[2], panel: i });
      });
    });
    const note = T(svg, 30, 222, '', { 'font-size': 14 }), note2 = T(svg, 30, 250, '', { 'font-size': 14 });
    T(svg, 30, 290, 'Latency values are typical orders of magnitude, not exact figures; energy ratio from the book (“at least an order of magnitude”).', { 'font-size': 11, class: 'lbl' });
    const out = root.querySelector('[data-out]'), inp = root.querySelector('[data-in=r]');
    const occ = () => {
      const r = Math.max(1, Math.min(255, +inp.value | 0));
      const threads = Math.min(2048, Math.floor(65536 / (r * 32)) * 32);
      out.innerHTML = `→ <b>${fmt(threads)}</b> threads per SM, occupancy <b>${Math.round(threads / 2048 * 100)} %</b>`;
    };
    inp.addEventListener('input', occ);
    const caps = [
      '<b>1. Speed</b>: registers are on chip, inside the processing unit. Global memory is off-chip DRAM, with long latency and a bandwidth all threads share.',
      '<b>2. Fewer instructions</b>: arithmetic instructions name registers directly. An operand in global memory needs a load first, and the SM can issue only a few instructions per cycle.',
      '<b>3. Energy</b>: reading a register costs at least an order of magnitude less energy than reading global memory.',
      '<b>The catch</b>: registers are scarce. 65,536 per SM shared by up to 2048 threads. Try the calculator: above 32 per thread, occupancy drops (session 12).'
    ];
    const cap = root.querySelector('[data-cap]');
    const st = stepper(root, () => 4, (k) => {
      bars.forEach(o => { const vis = o.panel < k + 1 && k < 3 || k === 3; o.b.setAttribute('width', vis ? o.w : 0); o.lbl.textContent = vis ? o.text : ''; o.lbl.setAttribute('fill', o.w < 120 ? '#1A1A1A' : '#fff'); if (o.w < 120) o.lbl.setAttribute('x', +o.b.getAttribute('x') + o.w + 6); });
      note.textContent = k === 3 ? '65,536 registers / 2048 threads = 32 registers per thread at full occupancy' : '';
      note2.textContent = k === 3 ? 'Every extra register a kernel needs can cost resident threads, and with them latency hiding.' : '';
      cap.innerHTML = caps[k];
    }, 3000);
    return { enter() { occ(); st.enter(); }, leave: st.leave };
  };

  // ------------------------------------------------------------ distributed shared memory (Hopper clusters)
  ANIM.dsmem = (root) => {
    const svg = newSvg(root, '0 0 1000 380', 'Four blocks of a thread block cluster on four SMs; block 0 reads data from the shared memory of block 2 either through global memory or directly');
    const hG = arrowHead(svg, 'ds-g', GOLD), hB = arrowHead(svg, 'ds-b', BLUE);
    const cl = R(svg, 10, 10, 980, 230, { class: 'frame', rx: 5 }); T(svg, 26, 234, 'Thread block cluster: 4 blocks, one per SM, in the same GPC', { 'font-size': 13, class: 'lbl' });
    const sms = [0, 1, 2, 3].map(i => {
      const x = 30 + i * 240;
      R(svg, x, 50, 220, 170, { fill: '#fff', stroke: '#6f6f6f', rx: 4 }); T(svg, x + 110, 72, `SM ${i} · block ${i}`, { ...mid, 'font-size': 13 });
      const sh = R(svg, x + 20, 90, 180, 60, { class: 'mem', rx: 3 }); T(svg, x + 110, 116, 'shared memory', { ...mid, 'font-size': 13 });
      const v = T(svg, x + 110, 138, '', { ...mid, 'font-size': 12, 'font-weight': 600 });
      R(svg, x + 20, 166, 180, 38, { fill: '#C9DBF4', rx: 3 }); T(svg, x + 110, 190, 'threads', { ...mid, 'font-size': 12 });
      return { x, sh, v };
    });
    const gm = R(svg, 30, 300, 940, 60, { class: 'mem', rx: 3 }); T(svg, 500, 336, 'Global memory (off chip)', { ...mid, 'font-size': 15 });
    const gv = T(svg, 860, 336, '', { ...mid, 'font-size': 12, 'font-weight': 600 });
    const down = L(svg, 590, 150, 590, 300, { stroke: GOLD, 'stroke-width': 3, 'marker-end': hG });
    const up = L(svg, 140, 300, 140, 150, { stroke: GOLD, 'stroke-width': 3, 'marker-end': hG });
    const direct = svg.appendChild(el('path', { d: 'M 530 96 C 430 40, 240 40, 200 96', fill: 'none', stroke: BLUE, 'stroke-width': 3.5, 'marker-end': hB }));
    const dlab = T(svg, 365, 36, 'SM-to-SM network', { ...mid, 'font-size': 12, fill: BLUE });
    const caps = [
      'Block 2 has produced partial results in <b>its</b> shared memory. Block 0 needs them. Shared memory normally has block scope, so block 0 cannot see it.',
      'Classic route: block 2 writes the data to <b>global memory</b>, a trip off chip …',
      '… and block 0 reads it back, a second off-chip trip (and usually a second kernel launch, so that the write is finished).',
      'Hopper: blocks can be grouped in a <b>thread block cluster</b> placed on SMs of the same GPC. Block 0 reads block 2\'s shared memory <b>directly</b>: <b>distributed shared memory</b>.',
      'The scope of fast on-chip memory grows from a block to a cluster: threads cooperate across blocks without going through global memory, with a larger pool (4 × up to 228 KB here).'
    ];
    const cap = root.querySelector('[data-cap]');
    return stepper(root, () => caps.length, (k) => {
      sms[2].v.textContent = 'partial = 42'; sms[0].v.textContent = k >= 2 ? 'got 42' : '';
      sms[2].sh.classList.toggle('on', k === 0 || k === 1 || k === 3); sms[0].sh.classList.toggle('on', k >= 2);
      vis(down, k === 1 || k === 2); down.setAttribute('opacity', k === 1 ? 1 : k === 2 ? .3 : 0);
      vis(up, k === 2); gm.classList.toggle('on', k === 1 || k === 2); gv.textContent = k === 1 || k === 2 ? '42' : '';
      vis(direct, k >= 3); vis(dlab, k >= 3); cl.classList.toggle('on', k >= 3);
      cap.innerHTML = caps[k];
    }, 3000);
  };

  // ------------------------------------------------------------ lifetime: grid versus application
  ANIM.lifetime = (root) => {
    const svg = newSvg(root, '0 0 1000 330', 'Timeline of an application with two kernel launches, showing when register, shared, global and constant variables exist');
    const X0 = 250, X1 = 980, tx = (f) => X0 + f * (X1 - X0);
    const rows = [
      { name: '__constant__ coeff[64]', y: 40 }, { name: '__device__ int count', y: 90 },
      { name: '__shared__ float tile[256]', y: 140 }, { name: 'float acc (register)', y: 190 }, { name: 'kernel launches', y: 240 }
    ];
    rows.forEach(r => { T(svg, 10, r.y + 22, r.name, { 'font-size': 13 }); R(svg, X0, r.y, X1 - X0, 32, { fill: '#F7F7F7', rx: 2 }); });
    const k1 = [0.22, 0.44], k2 = [0.62, 0.84];
    const kb = [k1, k2].map((k, i) => { const b = R(svg, tx(k[0]), 240, tx(k[1]) - tx(k[0]), 32, { class: 'chip', rx: 2 }); T(svg, (tx(k[0]) + tx(k[1])) / 2, 261, `kernel ${i + 1}`, { ...mid, 'font-size': 12 }); return b; });
    const cBar = R(svg, X0, 40, 0, 32, { class: 'bar', rx: 2 }), cT = T(svg, X0 + 8, 61, '', { 'font-size': 12, fill: '#fff' });
    const gBar = R(svg, X0, 90, 0, 32, { class: 'bar', rx: 2 }), gT = T(svg, X0 + 8, 111, '', { 'font-size': 12, fill: '#fff' });
    const sB = [k1, k2].map(k => R(svg, tx(k[0]), 140, 0, 32, { class: 'bar2', rx: 2 }));
    const sT = [k1, k2].map(k => T(svg, tx(k[0]) + 6, 161, '', { 'font-size': 12, fill: '#fff' }));
    const rB = [k1, k2].map(k => R(svg, tx(k[0]), 190, 0, 32, { class: 'bar2', rx: 2 }));
    const rT = [k1, k2].map(k => T(svg, tx(k[0]) + 6, 211, '', { 'font-size': 12, fill: '#fff' }));
    const cur = L(svg, X0, 28, X0, 284, { stroke: '#1A1A1A', 'stroke-width': 1.5, 'stroke-dasharray': '4 3' });
    const ct = T(svg, X0, 304, '', { ...mid, 'font-size': 12 });
    const tl = [0.06, 0.14, 0.38, 0.5, 0.8, 0.92, 1];
    const S = [
      { t: 'program starts', c: '', g: 'count = 0' },
      { t: 'host: cudaMemcpyToSymbol(coeff, …)', c: 'coeff written by the host', g: 'count = 0' },
      { t: 'kernel 1 running: 3 blocks × 256 threads', c: 'read by all threads', g: 'count = 5', s1: '3 copies', r1: '768 copies' },
      { t: 'kernel 1 ends', c: 'still there', g: 'count = 5 (kept)', s1: 'gone', r1: 'gone', end1: true },
      { t: 'kernel 2 running', c: 'read by all threads', g: 'count = 5 → 8', s1: 'gone', r1: 'gone', end1: true, s2: 'new copies', r2: 'new copies' },
      { t: 'kernel 2 ends', c: 'still there', g: 'count = 8', end1: true, end2: true, s1: 'gone', r1: 'gone', s2: 'gone', r2: 'gone' },
      { t: 'application ends', c: 'freed', g: 'freed', end1: true, end2: true, s1: 'gone', r1: 'gone', s2: 'gone', r2: 'gone' }
    ];
    const caps = [
      'Lifetime is how long a variable exists. <code>__constant__</code> and <code>__device__</code> variables are declared outside any function and exist for the <b>whole application</b>.',
      'The host fills constant memory before launching: kernels can only read it.',
      'Kernel 1 runs. Each <b>block</b> gets its own <code>tile</code> (3 copies) and each <b>thread</b> its own <code>acc</code> (768 copies). Lifetime: one grid.',
      'When kernel 1 ends, its shared and register copies disappear. <code>count</code> keeps its value: global variables pass information between kernel launches.',
      'Kernel 2 gets <b>new</b> shared and register copies. They do not remember kernel 1: every launch must initialise them again.',
      'Kernel 2 ends; global and constant variables are still there for the host or the next kernel.',
      'Only when the application ends are they released. Global variables are handy but hurt modularity: use them sparingly.'
    ];
    const cap = root.querySelector('[data-cap]');
    return stepper(root, () => S.length, (k) => {
      const s = S[k], t = tl[k];
      cur.setAttribute('x1', tx(t)); cur.setAttribute('x2', tx(t)); ct.setAttribute('x', Math.min(800, Math.max(330, tx(t)))); ct.textContent = s.t;
      cBar.setAttribute('width', tx(t) - X0); cT.textContent = s.c; gBar.setAttribute('width', tx(t) - X0); gT.textContent = s.g;
      [k1, k2].forEach((kk, i) => {
        const end = i === 0 ? s.end1 : s.end2, running = t > kk[0];
        const w = !running ? 0 : end ? tx(kk[1]) - tx(kk[0]) : Math.min(tx(t), tx(kk[1])) - tx(kk[0]);
        [sB[i], rB[i]].forEach(b => { b.setAttribute('width', w); b.setAttribute('opacity', end ? .35 : 1); });
        sT[i].textContent = running ? (i === 0 ? s.s1 : s.s2) || '' : ''; rT[i].textContent = running ? (i === 0 ? s.r1 : s.r2) || '' : '';
        [sT[i], rT[i]].forEach(t => t.setAttribute('fill', end ? '#1A1A1A' : '#fff'));
        kb[i].classList.toggle('run', running && !end); kb[i].classList.toggle('done', !!end);
      });
      cap.innerHTML = caps[k];
    }, 3000);
  };

  // ------------------------------------------------------------ local memory: arrays and spills
  ANIM.local = (root) => {
    let mode = 'const';
    const svg = newSvg(root, '0 0 1000 340', 'Where the compiler puts a per-thread array: in registers when every index is a constant, in local memory in DRAM when the index is computed at run time');
    const hG = arrowHead(svg, 'lo-g', GOLD);
    R(svg, 10, 10, 420, 250, { fill: '#F5F5F5', rx: 4 });
    const code = [0, 1, 2, 3, 4, 5].map(i => T(svg, 26, 40 + i * 30, '', { 'font-size': 14 }));
    R(svg, 470, 10, 520, 170, { class: 'frame', rx: 5 }); T(svg, 486, 32, 'one thread, inside the SM', { 'font-size': 13, class: 'lbl' });
    const regs = [0, 1, 2, 3].map(i => { const r = R(svg, 490 + i * 122, 50, 110, 46, { class: 'sp', rx: 3 }); return { r, t: T(svg, 545 + i * 122, 79, '', { ...mid, 'font-size': 13 }) }; });
    T(svg, 730, 122, 'registers', { ...mid, 'font-size': 13, class: 'lbl' });
    const l1 = R(svg, 490, 132, 480, 34, { class: 'mem', rx: 3 }); T(svg, 730, 154, 'L1 cache', { ...mid, 'font-size': 12 });
    const dram = R(svg, 470, 230, 520, 90, { class: 'mem', rx: 4 }); T(svg, 730, 252, 'DRAM (global memory)', { ...mid, 'font-size': 14 });
    const loc = R(svg, 500, 264, 460, 40, { fill: '#fff', stroke: '#bdbdbd', 'stroke-dasharray': '4 3', rx: 3 });
    const locT = T(svg, 730, 290, 'local memory of this thread', { ...mid, 'font-size': 12 });
    const path = L(svg, 730, 230, 730, 168, { stroke: GOLD, 'stroke-width': 3, 'marker-end': hG });
    const verdict = T(svg, 26, 300, '', { 'font-size': 15, 'font-weight': 600 });
    const P = {
      const: { code: ['float w[4];', 'w[0] = a; w[1] = b;', 'w[2] = c; w[3] = d;', 'sum = w[0] + w[1]', '    + w[2] + w[3];', '// every index is a constant'],
        regs: ['w[0]', 'w[1]', 'w[2]', 'w[3]'], local: false, v: ['the compiler can map w[0..3] to registers', 'no memory traffic at all', 'automatic array, but fast'],
        caps: ['An <b>automatic array</b> is private to the thread, like a scalar. Here every index is known at compile time.',
          'So the compiler can give each element its own register: w[0] → one register, w[1] → another, and so on.',
          'Result: the array behaves like four scalars, with register speed. (Chapter 6 returns to this.)'] },
      dyn: { code: ['float w[4];', 'for (int i = 0; i < 4; ++i)', '  w[i] = in[base + i];', 'sum += w[idx];', '// idx known only at run time', ''],
        regs: ['sum', 'idx', 'base', 'i'], local: true, v: ['registers cannot be indexed by a run-time value', 'so w goes to local memory, in DRAM', 'each w[idx] is a load from memory'],
        caps: ['Now the index <code>idx</code> is computed at run time. Instructions name registers by fixed numbers, so <b>w[idx]</b> cannot be a register.',
          'The compiler places <code>w</code> in <b>local memory</b>: a private area per thread, but physically in <b>global memory</b> (DRAM), cached in L1/L2.',
          'Every <code>w[idx]</code> becomes a load with global-memory latency. Local means private, not fast. Register <b>spills</b> (too many live values) end up here too.'] }
    };
    const cap = root.querySelector('[data-cap]');
    const st = stepper(root, () => 3, (k) => {
      const p = P[mode];
      code.forEach((t, i) => { t.textContent = p.code[i]; t.setAttribute('fill', (mode === 'dyn' && i === 3 && k >= 1) ? RED : '#1A1A1A'); });
      regs.forEach((r, i) => { r.t.textContent = k >= 1 || mode === 'dyn' ? p.regs[i] : ''; r.r.classList.toggle('on', mode === 'const' && k >= 1); r.t.setAttribute('fill', mode === 'const' && k >= 1 ? '#fff' : '#1A1A1A'); });
      loc.setAttribute('stroke', p.local && k >= 1 ? GOLD : '#bdbdbd'); loc.setAttribute('stroke-width', p.local && k >= 1 ? 2.5 : 1);
      locT.textContent = p.local && k >= 1 ? 'local memory: w[0] w[1] w[2] w[3]' : 'local memory of this thread';
      vis(path, p.local && k === 2); dram.classList.toggle('on', p.local && k === 2); l1.classList.toggle('on', p.local && k === 2);
      verdict.textContent = p.v[k]; verdict.setAttribute('fill', p.local ? RED : GREEN);
      cap.innerHTML = p.caps[k];
    }, 3000);
    modeButtons(root, 'mode', (m) => { mode = m; st.reset(); });
    return st;
  };

  // ------------------------------------------------------------ constant memory: one value for every thread
  ANIM.cmem = (root) => {
    let mode = 'same';
    const svg = newSvg(root, '0 0 1000 360', 'Host writes constant memory; a warp reads it through the constant cache, in one broadcast when all threads read the same address');
    const hG = arrowHead(svg, 'cm-g', GOLD), hB = arrowHead(svg, 'cm-b', BLUE);
    const host = R(svg, 10, 270, 150, 70, { fill: '#fff', stroke: '#6f6f6f', rx: 3 }); T(svg, 85, 300, 'Host', { ...mid, 'font-size': 14, 'font-weight': 600 }); T(svg, 85, 320, 'cudaMemcpyToSymbol', { ...mid, 'font-size': 10 });
    const cm = R(svg, 200, 270, 400, 70, { class: 'mem', rx: 3 }); T(svg, 400, 298, 'constant memory (DRAM)', { ...mid, 'font-size': 14 }); T(svg, 400, 320, 'coeff[0..63] · at most 64 KB in total', { ...mid, 'font-size': 12, class: 'lbl' });
    const wH = L(svg, 160, 305, 200, 305, { stroke: GOLD, 'stroke-width': 3, 'marker-end': hG });
    R(svg, 200, 20, 790, 210, { class: 'frame', rx: 5 }); T(svg, 216, 42, 'SM', { 'font-size': 13, class: 'lbl' });
    const cc = R(svg, 220, 140, 260, 60, { class: 'mem', rx: 3 }); T(svg, 350, 166, 'constant cache', { ...mid, 'font-size': 13 });
    const ccT = T(svg, 350, 188, '', { ...mid, 'font-size': 12 });
    const wD = L(svg, 350, 270, 350, 200, { stroke: GOLD, 'stroke-width': 3, 'marker-end': hG });
    const lanes = [];
    for (let i = 0; i < 16; i++) {
      const x = 540 + (i % 8) * 54, y = 60 + Math.floor(i / 8) * 80;
      const r = R(svg, x, y, 46, 40, { fill: '#C9DBF4', rx: 3 }); const t = T(svg, x + 23, y + 25, `t${i * 2}`, { ...mid, 'font-size': 11 });
      lanes.push({ r, t, x: x + 23, y: y + 40 });
    }
    T(svg, 756, 216, 'one warp (16 of its 32 threads drawn)', { ...mid, 'font-size': 11, class: 'lbl' });
    const fan = svg.appendChild(el('g'));
    const cnt = T(svg, 640, 300, '', { 'font-size': 14 }), cnt2 = T(svg, 640, 326, '', { 'font-size': 14 });
    const caps = {
      same: ['Before the launch, the <b>host</b> writes the coefficients into constant memory. Kernels can only read it; its scope is every thread of every grid.',
        'The warp executes <code>x * coeff[0]</code>. The first access misses, and the value is brought from DRAM into the SM\'s <b>constant cache</b>.',
        'All 32 threads read the <b>same address</b>: the cache serves them with <b>one broadcast</b>.',
        'Later warps and later iterations hit the cache. That is the access pattern constant memory is built for (Chapter 7 uses it for convolution filters).'],
      diff: ['Same data, but now thread t reads <code>coeff[t]</code>: different addresses inside one warp.',
        'The constant cache serves one address at a time, so the warp\'s request is <b>split</b> into one access per distinct address.',
        'With 32 distinct addresses the read is serialised 32 times: the “appropriate access pattern” is missing.',
        'Rule: use constant memory for small, read-only data that all threads of a warp read <b>at the same time</b>.']
    };
    const cap = root.querySelector('[data-cap]');
    const st = stepper(root, () => 4, (k) => {
      vis(wH, k === 0); cm.classList.toggle('on', k <= 1); vis(wD, k === 1); cc.classList.toggle('on', k >= 1);
      ccT.textContent = k >= 1 ? (mode === 'same' ? 'coeff[0] = 0.25' : 'coeff[0..31]') : 'empty';
      fan.innerHTML = '';
      const nLit = mode === 'same' ? (k >= 2 ? 16 : 0) : (k >= 2 ? 1 : 0);
      if (k >= 2) {
        fan.appendChild(el('line', { x1: 480, y1: 170, x2: 528, y2: 120, stroke: BLUE, 'stroke-width': 3 }));
        fan.appendChild(el('rect', { x: 528, y: 114, width: mode === 'same' ? 440 : 60, height: 12, fill: BLUE, rx: 3 }));
        fan.appendChild(el('text', { x: 748, y: 111, 'text-anchor': 'middle', 'font-size': 11, fill: BLUE }, mode === 'same' ? 'one value, broadcast to every lane' : 'one address per access: t0 now, t2 next, …'));
      }
      lanes.forEach((l, i) => {
        const lit = i < nLit; l.r.setAttribute('fill', lit ? BLUE : '#C9DBF4'); l.t.setAttribute('fill', lit ? '#fff' : '#1A1A1A');
        if (lit) fan.appendChild(el('line', { x1: l.x, y1: i < 8 ? 114 : 126, x2: l.x, y2: i < 8 ? 100 : 140, stroke: BLUE, 'stroke-width': 2 }));
      });
      cnt.textContent = k >= 2 ? (mode === 'same' ? 'cache accesses for the warp: 1' : 'cache accesses for the warp: 32 (one after another)') : '';
      cnt.setAttribute('fill', mode === 'same' ? GREEN : RED);
      cnt2.textContent = k === 3 ? (mode === 'same' ? 'next warps: cache hits' : 'same address for all → 1 access') : '';
      cap.innerHTML = caps[mode][k];
    }, 3000);
    modeButtons(root, 'mode', (m) => { mode = m; st.reset(); });
    return st;
  };

  // ------------------------------------------------------------ choosing where a variable lives
  ANIM.choose = (root) => {
    const svg = newSvg(root, '0 0 1000 380', 'Decision tree for choosing registers, local, shared, constant or global memory for a variable');
    const hG = arrowHead(svg, 'ch-g', '#bdbdbd'), hB = arrowHead(svg, 'ch-b', BLUE);
    const box = (x, y, w, h, s, cls) => { const r = R(svg, x, y, w, h, { class: cls, rx: 4 }); const lines = s.split('\n'); lines.forEach((ln, i) => T(svg, x + w / 2, y + h / 2 + 5 + (i - (lines.length - 1) / 2) * 17, ln, { ...mid, 'font-size': 13 })); return r; };
    const N = {
      q1: box(330, 10, 340, 52, 'Used by one thread only?', 'mem'),
      q2: box(40, 110, 330, 52, 'Scalar, or array with\nconstant indices?', 'mem'),
      q3: box(600, 110, 360, 52, 'Shared and reused by the\nthreads of one block?', 'mem'),
      q4: box(600, 210, 360, 60, 'Read-only, ≤ 64 KB, same value\nread by all threads?', 'mem'),
      reg: box(20, 220, 170, 50, 'Registers', 'chip'), loc: box(210, 220, 170, 50, 'Local memory\n(slow, avoid)', 'chip'),
      sh: box(420, 220, 150, 50, 'Shared memory', 'chip'),
      co: box(560, 310, 190, 50, 'Constant memory', 'chip'), gl: box(790, 310, 190, 50, 'Global memory', 'chip')
    };
    const E = {}, mk = (id, x1, y1, x2, y2, lab, lx, ly) => { E[id] = L(svg, x1, y1, x2, y2, { class: 'wire', 'marker-end': hG }); T(svg, lx, ly, lab, { 'font-size': 12, class: 'lbl' }); };
    mk('1y', 420, 62, 260, 110, 'yes', 318, 80); mk('1n', 580, 62, 740, 110, 'no', 668, 80);
    mk('2y', 160, 162, 105, 220, 'yes', 108, 196); mk('2n', 260, 162, 295, 220, 'no', 288, 196);
    mk('3y', 650, 162, 520, 220, 'yes', 570, 196); mk('3n', 800, 162, 800, 210, 'no', 808, 192);
    mk('4y', 700, 270, 655, 310, 'yes', 652, 296); mk('4n', 880, 270, 885, 310, 'no', 892, 296);
    const ex = [
      { id: 'i', name: 'loop counter i, Row, Col', path: ['q1', '1y', 'q2', '2y', 'reg'], why: 'Private and scalar: an automatic variable, kept in a register.' },
      { id: 'w', name: 'float w[8] indexed with w[k], k from data', path: ['q1', '1y', 'q2', '2n', 'loc'], why: 'Private, but indexed at run time: local memory, which lives in DRAM. Restructure if it is hot.' },
      { id: 'tile', name: 'tile of M used by the whole block', path: ['q1', '1n', 'q3', '3y', 'sh'], why: 'Every thread of the block reads it several times: load it once into <code>__shared__</code>.' },
      { id: 'coef', name: 'filter coefficients (small, read-only)', path: ['q1', '1n', 'q3', '3n', 'q4', '4y', 'co'], why: 'Small, never written by the kernel, read by all threads alike: <code>__constant__</code>.' },
      { id: 'mat', name: 'input and output matrices', path: ['q1', '1n', 'q3', '3n', 'q4', '4n', 'gl'], why: 'Large and shared by the whole grid and the host: global memory (then tile the parts that are reused).' }
    ];
    const lbl = T(svg, 20, 30, '', { 'font-size': 14, 'font-weight': 600 });
    const cap = root.querySelector('[data-cap]');
    const draw = (k) => {
      const e = ex[k];
      root.querySelectorAll('[data-ex]').forEach(b => b.classList.toggle('on', b.dataset.ex === e.id));
      Object.entries(N).forEach(([id, r]) => { const on = e.path.includes(id); r.classList.toggle(r.classList.contains('chip') ? 'run' : 'on', on); if (!on) { r.classList.remove('run'); r.classList.remove('on'); } });
      Object.entries(E).forEach(([id, l]) => { const on = e.path.includes(id); l.classList.toggle('on', on); l.classList.toggle('blue', on); l.setAttribute('marker-end', on ? hB : hG); });
      lbl.textContent = ''; cap.innerHTML = `<b>${e.name}</b>: ${e.why}`;
    };
    const st = stepper(root, () => ex.length, draw, 3400);
    modeButtons(root, 'ex', (m) => { st.leave(); draw(ex.findIndex(e => e.id === m)); });
    return st;
  };

  // ------------------------------------------------------------ CPU cache versus GPU shared memory
  ANIM.cache = (root) => {
    let mode = 'cpu';
    const svg = newSvg(root, '0 0 1000 330', 'A CPU thread reusing its data from a cache, many GPU warps evicting each other from a shared cache, and GPU blocks keeping their tiles in shared memory');
    const title = T(svg, 20, 30, '', { 'font-size': 15, 'font-weight': 600 });
    const thr = [], lines = [];
    for (let i = 0; i < 8; i++) { const r = R(svg, 20, 50 + i * 32, 140, 26, { class: 'chip', rx: 2 }); const t = T(svg, 90, 68 + i * 32, '', { ...mid, 'font-size': 12 }); thr.push({ r, t }); }
    const cbox = R(svg, 260, 46, 420, 266, { class: 'frame', rx: 4 }); const ctitle = T(svg, 470, 70, '', { ...mid, 'font-size': 13 });
    for (let i = 0; i < 8; i++) { const r = R(svg, 280 + (i % 4) * 98, 90 + Math.floor(i / 4) * 100, 88, 80, { class: 'mem', rx: 3 }); const t = T(svg, 324 + (i % 4) * 98, 136 + Math.floor(i / 4) * 100, '', { ...mid, 'font-size': 13 }); lines.push({ r, t }); }
    const stat = T(svg, 720, 120, '', { 'font-size': 14 }), stat2 = T(svg, 720, 150, '', { 'font-size': 14 }), stat3 = T(svg, 720, 180, '', { 'font-size': 13 });
    const colors = ['#86C495', '#E7B467', '#2F6FD0', '#F2C230', '#B03A3A', '#7A5FB0', '#2E9AA8', '#6f6f6f'];
    const P = {
      cpu: { title: 'CPU core: one or two threads', ct: 'cache: 8 lines', n: 1, steps: [
        { slots: ['A0', 'A1', 'A2', 'A3', '', '', '', ''], hits: 0, miss: 4 }, { slots: ['A0', 'A1', 'A2', 'A3', '', '', '', ''], hits: 4, miss: 4 },
        { slots: ['A0', 'A1', 'A2', 'A3', '', '', '', ''], hits: 12, miss: 4 }],
        caps: ['A CPU thread reads its block of data A0–A3: four misses, and the lines stay in the cache.',
          'When it reuses the block, the data is still there: <b>hits</b>. Blocking (tiling) on a CPU relies on this.',
          'Few threads per core means the cache keeps each thread\'s data long enough. The reuse is <b>implicit</b>.'] },
      gpu: { title: 'GPU SM: many warps share one cache', ct: 'L1 cache: 8 lines', n: 8, steps: [
        { slots: ['A0', 'A1', 'B0', 'B1', 'C0', 'C1', 'D0', 'D1'], hits: 0, miss: 8 }, { slots: ['E0', 'E1', 'F0', 'F1', 'G0', 'G1', 'H0', 'H1'], hits: 0, miss: 16 },
        { slots: ['A0', 'A1', 'B0', 'B1', 'E0', 'E1', 'F0', 'F1'], hits: 0, miss: 24 }],
        caps: ['Eight warps, each with its own data, start together: the cache fills at once.',
          'Warps E–H arrive and <b>evict</b> the lines of A–D before A–D could reuse them.',
          'When A comes back, its data is gone: <b>misses</b> again. With thousands of threads per SM, the cache cannot be trusted to keep reused data.'] },
      shm: { title: 'GPU SM: each block keeps its tile in shared memory', ct: 'shared memory: tiles owned by blocks', n: 8, steps: [
        { slots: ['A0', 'A1', 'B0', 'B1', 'C0', 'C1', 'D0', 'D1'], hits: 0, miss: 8 }, { slots: ['A0', 'A1', 'B0', 'B1', 'C0', 'C1', 'D0', 'D1'], hits: 8, miss: 8 },
        { slots: ['A0', 'A1', 'B0', 'B1', 'C0', 'C1', 'D0', 'D1'], hits: 24, miss: 8 }],
        caps: ['With tiling, each block loads its tile into <code>__shared__</code> arrays: one global load per element.',
          'Nothing evicts it: the data stays until the program itself overwrites it (after the barrier).',
          'Every reuse is served on chip. GPUs make the reuse <b>explicit</b>: shared memory and registers instead of trusting a cache.'] }
    };
    const cap = root.querySelector('[data-cap]');
    const st = stepper(root, () => 3, (k) => {
      const p = P[mode], s = p.steps[k];
      title.textContent = p.title; ctitle.textContent = p.ct;
      thr.forEach((t, i) => { const here = i < p.n; vis(t.r, here); t.t.textContent = here ? (mode === 'cpu' ? 'thread 0' : `warp ${'ABCDEFGH'[i]}`) : ''; t.r.classList.toggle("run", here && (mode === 'cpu' || (k === 1 ? i >= 4 : i < 4) || mode === 'shm')); });
      lines.forEach((l, i) => { l.t.textContent = s.slots[i]; const c = s.slots[i] ? colors['ABCDEFGH'.indexOf(s.slots[i][0])] : null; l.r.style.fill = c || ''; l.r.style.fillOpacity = c ? .45 : 1; });
      cbox.classList.toggle('on', mode === 'shm');
      stat.textContent = `hits: ${s.hits}`; stat.setAttribute('fill', GREEN); stat2.textContent = `misses: ${s.miss}`; stat2.setAttribute('fill', RED);
      stat3.textContent = s.hits + s.miss ? `hit rate ${Math.round(s.hits / (s.hits + s.miss) * 100)} %` : '';
      cap.innerHTML = p.caps[k];
    }, 2900);
    modeButtons(root, 'mode', (m) => { mode = m; st.reset(); });
    return st;
  };

  // ------------------------------------------------------------ strip-mining
  ANIM.strip = (root) => {
    const W = 8, TW = 4;
    const svg = newSvg(root, '0 0 1000 340', 'The k loop of the dot product, first as one long loop, then strip-mined into phases of TILE_WIDTH iterations separated by barriers');
    T(svg, 20, 34, 'original loop: for k in 0 … Width − 1', { 'font-size': 14 });
    const orig = [], strip = [];
    for (let k = 0; k < W; k++) {
      orig.push({ r: R(svg, 20 + k * 118, 50, 108, 46, { class: 'chip', rx: 2 }), t: T(svg, 74 + k * 118, 79, `k = ${k}`, { ...mid, 'font-size': 13 }) });
    }
    T(svg, 20, 140, 'strip-mined: for ph … { load tile; barrier; for k in 0 … TILE_WIDTH − 1; barrier }', { 'font-size': 14 });
    const phaseBox = [0, 1].map(ph => { const r = R(svg, 14 + ph * (TW * 118 + 10), 152, TW * 118 + 2, 120, { class: 'frame', rx: 4 }); T(svg, 22 + ph * (TW * 118 + 10), 172, `phase ${ph}`, { 'font-size': 13, class: 'lbl' }); return r; });
    for (let k = 0; k < W; k++) {
      const ph = Math.floor(k / TW), x = 20 + k * 118 + ph * 10;
      strip.push({ r: R(svg, x, 210, 108, 46, { class: 'chip', rx: 2 }), t: T(svg, x + 54, 239, `k = ${k % TW}`, { ...mid, 'font-size': 13 }) });
    }
    const loads = [0, 1].map(ph => { const x = 20 + ph * (TW * 118 + 10); return { r: R(svg, x, 180, TW * 118 - 10, 22, { class: 'mem', rx: 2 }), t: T(svg, x + (TW * 118 - 10) / 2, 196, 'load tile of M and N into shared · __syncthreads()', { ...mid, 'font-size': 11 }) }; });
    const counter = T(svg, 20, 300, '', { 'font-size': 14 }), counter2 = T(svg, 20, 326, '', { 'font-size': 14 });
    // steps: 0..7 original, 8 strip intro, 9..: phase 0 load, k0..3, phase1 load, k0..3
    const seq = [];
    for (let k = 0; k < W; k++) seq.push({ o: k });
    seq.push({ intro: true });
    for (let ph = 0; ph < 2; ph++) { seq.push({ load: ph }); for (let k = 0; k < TW; k++) seq.push({ s: ph * TW + k }); }
    const cap = root.querySelector('[data-cap]');
    return stepper(root, () => seq.length, (i) => {
      const s = seq[i];
      orig.forEach((o, k) => { o.r.classList.toggle('run', s.o === k); o.r.classList.toggle('done', s.o != null ? k < s.o : true); });
      const sIdx = s.s != null ? s.s : s.load != null ? s.load * TW - 1 : -1;
      strip.forEach((o, k) => { o.r.classList.toggle('run', s.s === k); o.r.classList.toggle('done', k <= sIdx && s.s !== k); });
      loads.forEach((l, ph) => l.r.classList.toggle('on', s.load === ph));
      phaseBox.forEach((b, ph) => b.classList.toggle('on', (s.load === ph) || (s.s != null && Math.floor(s.s / TW) === ph)));
      if (s.o != null) {
        counter.textContent = `global loads by this thread so far: ${2 * (s.o + 1)} (one M and one N per iteration)`; counter2.textContent = '';
        cap.innerHTML = `The dot product of one thread is a single loop of <b>Width</b> iterations; each one reads M and N from global memory.`;
      } else if (s.intro) {
        counter.textContent = `without tiling: ${2 * W} global loads per thread`; counter2.textContent = '';
        cap.innerHTML = '<b>Strip-mining</b> cuts the long loop into strips of TILE_WIDTH iterations. The outer loop walks over the strips (phases); the inner loop runs the iterations of one strip, in the original order.';
      } else if (s.load != null) {
        counter.textContent = `global loads by this thread so far: ${2 * (s.load + 1)}`; counter2.textContent = 'each thread loads 1 M and 1 N element of the tile, then waits at the barrier';
        cap.innerHTML = `Phase ${s.load}: the block loads one tile of M and one of N <b>together</b>, then a barrier makes sure the whole tile is in shared memory.`;
      } else {
        const ph = Math.floor(s.s / TW);
        counter.textContent = `global loads by this thread so far: ${2 * (ph + 1)} · iteration k = ${s.s} of the original loop`; counter2.textContent = 'operands read from shared memory';
        cap.innerHTML = s.s % TW === TW - 1 ? `End of phase ${ph}: a second barrier, so nobody overwrites the tile while others still read it. ${ph === 1 ? `Total: <b>${2 * W / TW}</b> global loads instead of ${2 * W}, the same ${W} iterations, the same result.` : ''}`
          : `Inner loop, iteration ${s.s % TW} of the strip = iteration ${s.s} of the original loop, now served from <b>shared memory</b>.`;
      }
    }, 1500);
  };
