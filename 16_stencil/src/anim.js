  const arrowHead = (svg, id, color) => {
    const d = svg.querySelector('defs') || svg.insertBefore(el('defs'), svg.firstChild);
    const m = el('marker', { id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' });
    m.appendChild(el('path', { d: 'M0,0 L10,5 L0,10 z', fill: color })); d.appendChild(m);
    return `url(#${id})`;
  };
  const GRAY = '#9a9a9a', BLUE = '#2F6FD0', GOLD = '#C98A1B', RED = '#B03A3A', GREEN = '#2E7D4F', INK = '#1A1A1A';
  const LBLUE = '#C9DBF4', MBLUE = '#9DB7E3', LGOLD = '#F6EAD2', LRED = '#F6DADA', LGREEN = '#D6EBDC', BGRAY = '#C4C4C4';
  const syncButtons = (root, attr, val) => root.querySelectorAll(`[data-${attr}]`).forEach(b => b.classList.toggle('on', b.dataset[attr] === String(val)));
  const grid = (svg, x0, y0, n, m, cs, fillFn, opts = {}) => {
    const cells = [];
    for (let r = 0; r < n; r++) for (let c = 0; c < m; c++) cells.push(R(svg, x0 + c * cs, y0 + r * cs, cs - (opts.gap ?? 1), cs - (opts.gap ?? 1), { fill: fillFn ? fillFn(r, c) : '#fff', stroke: opts.stroke || '#d0d0d0', 'stroke-width': .6 }));
    return cells;
  };
  const C = (p, cx, cy, r, a = {}) => p.appendChild(el('circle', { cx, cy, r, ...a }));
  const P = (p, d, a = {}) => p.appendChild(el('path', { d, ...a }));
  const clampIn = (inp, lo, hi, def) => { const v = Math.round(+inp.value); return Number.isFinite(v) ? Math.max(lo, Math.min(hi, v)) : def; };

  // ------------------------------------------------------------ sin(x) on a grid, central differences
  ANIM.sine = (root) => {
    const svg = newSvg(root, '0 0 560 396', 'sin x sampled on a regular grid, and the central-difference derivative at each interior grid point against cos x');
    const q = (n) => root.querySelector(`[data-in=${n}]`), out = root.querySelector('[data-out]');
    const X0 = 46, XW = 490, px = (x) => X0 + x / Math.PI * XW;
    const draw = () => {
      svg.innerHTML = '';
      const n = clampIn(q('n'), 3, 81, 7), h = Math.PI / (n - 1);
      // top plot: sin and the grid points
      const ty = (v) => 170 - v * 135;
      T(svg, X0, 18, 'f(x) = sin x and its grid values F[i]', { 'font-size': 13, class: 'lbl' });
      L(svg, X0, ty(0), X0 + XW, ty(0), { stroke: GRAY });
      let d = ''; for (let s = 0; s <= 200; s++) { const x = s / 200 * Math.PI; d += (s ? 'L' : 'M') + px(x).toFixed(1) + ',' + ty(Math.sin(x)).toFixed(1); }
      P(svg, d, { fill: 'none', stroke: '#9a9a9a', 'stroke-width': 1.5 });
      for (let i = 0; i < n; i++) {
        const x = i * h;
        if (n <= 25) L(svg, px(x), ty(0), px(x), ty(Math.sin(x)), { stroke: MBLUE, 'stroke-dasharray': '3 3' });
        C(svg, px(x), ty(Math.sin(x)), n <= 25 ? 5 : 3, { fill: i === 0 || i === n - 1 ? '#fff' : BLUE, stroke: BLUE, 'stroke-width': 1.5 });
      }
      if (n <= 13) [['0', 0], ['π/2', Math.PI / 2], ['π', Math.PI]].forEach(([s, x]) => T(svg, px(x), ty(0) + 16, s, { ...mid, 'font-size': 12, class: 'lbl' }));
      // h marker
      L(svg, px(0), 28, px(h), 28, { stroke: GOLD, 'stroke-width': 3 }); T(svg, X0 + XW, 18, `h = π/${n - 1} = ${h.toFixed(3)}`, { 'font-size': 12, fill: GOLD, 'text-anchor': 'end' });
      // bottom plot: cos and central differences
      const by = (v) => 300 - v * 75;
      T(svg, X0, 212, 'f′(x) = cos x (gray) and FD[i] = (F[i+1] − F[i−1]) / 2h', { 'font-size': 13, class: 'lbl' });
      L(svg, X0, by(0), X0 + XW, by(0), { stroke: GRAY });
      d = ''; for (let s = 0; s <= 200; s++) { const x = s / 200 * Math.PI; d += (s ? 'L' : 'M') + px(x).toFixed(1) + ',' + by(Math.cos(x)).toFixed(1); }
      P(svg, d, { fill: 'none', stroke: '#9a9a9a', 'stroke-width': 1.5 });
      let maxErr = 0;
      for (let i = 1; i < n - 1; i++) {
        const x = i * h, fd = (Math.sin(x + h) - Math.sin(x - h)) / (2 * h), e = Math.abs(fd - Math.cos(x)); maxErr = Math.max(maxErr, e);
        L(svg, px(x), by(Math.cos(x)), px(x), by(fd), { stroke: RED, 'stroke-width': 2 });
        C(svg, px(x), by(fd), n <= 25 ? 4.5 : 2.8, { fill: GREEN });
      }
      T(svg, X0, 392, 'green: FD[i] · red: error · end points lack a neighbour', { 'font-size': 11, class: 'lbl' });
      const F = Array.from({ length: n }, (_, i) => Math.sin(i * h).toFixed(2));
      const h2 = h / 2, err2 = Math.cos(h2) * (1 - Math.sin(h2) / h2);
      out.innerHTML = `h = π/${n - 1} = <b>${h.toFixed(3)}</b>` +
        `<br>F = {${n <= 13 ? F.join(', ') : F.slice(0, 5).join(', ') + ', …, ' + F[n - 1]}}` +
        `<br>max |FD[i] − cos x<sub>i</sub>| = <b>${maxErr.toExponential(2)}</b> (≈ h²/6 = ${(h * h / 6).toExponential(2)})` +
        `<br>with h/2: ${err2.toExponential(2)}, ${(maxErr / err2).toFixed(2)} times smaller`;
      syncButtons(root, 'p', n);
    };
    root.querySelectorAll('input').forEach(i => i.addEventListener('input', draw));
    root.querySelectorAll('[data-p]').forEach(b => b.addEventListener('click', () => { q('n').value = b.dataset.p; draw(); }));
    return { enter: draw, leave() {} };
  };

  // ------------------------------------------------------------ stencil shapes
  ANIM.shapes = (root) => {
    const svg = newSvg(root, '0 0 520 380', 'The points of a stencil of the chosen dimension and order, with the box a convolution of the same radius would read');
    const out = root.querySelector('[data-out]');
    let dim = 3, ord = 1;
    const draw = () => {
      svg.innerHTML = '';
      const r = ord, s = dim === 1 ? 62 : dim === 2 ? 44 : 34;
      const cx = 260, cy = dim === 3 ? 200 : 180;
      const proj = (x, y, z) => [cx + x * s - z * s * .55, cy - y * s + z * s * .42];
      const box = [];
      const rng = (k) => Array.from({ length: 2 * k + 1 }, (_, i) => i - k);
      const ys = dim >= 2 ? rng(r) : [0], zs = dim === 3 ? rng(r) : [0];
      for (const z of zs) for (const y of ys) for (const x of rng(r)) box.push([x, y, z]);
      box.sort((a, b) => b[2] - a[2]);
      box.forEach(([x, y, z]) => { const [a, b] = proj(x, y, z); C(svg, a, b, 2.2, { fill: '#b9b9b9' }); });
      // axes lines of the star
      const axes = [[1, 0, 0]]; if (dim >= 2) axes.push([0, 1, 0]); if (dim === 3) axes.push([0, 0, 1]);
      axes.forEach(([ax, ay, az]) => { const [a1, b1] = proj(-r * ax, -r * ay, -r * az), [a2, b2] = proj(r * ax, r * ay, r * az); L(svg, a1, b1, a2, b2, { stroke: '#8a8a8a', 'stroke-width': 2 }); });
      const pts = [[0, 0, 0]];
      axes.forEach(([ax, ay, az]) => { for (let k = 1; k <= r; k++) { pts.push([k * ax, k * ay, k * az]); pts.push([-k * ax, -k * ay, -k * az]); } });
      pts.sort((a, b) => b[2] - a[2]);
      pts.forEach(([x, y, z]) => { const [a, b] = proj(x, y, z), c = x === 0 && y === 0 && z === 0; C(svg, a, b, c ? 12 : 10, { fill: c ? BLUE : LBLUE, stroke: BLUE, 'stroke-width': 1.5 }); });
      // axis legend
      const lx = 40, ly = 340;
      const hA = arrowHead(svg, 'sh-a', INK);
      L(svg, lx, ly, lx + 34, ly, { stroke: INK, 'marker-end': hA }); T(svg, lx + 40, ly + 4, 'x', { 'font-size': 12 });
      if (dim >= 2) { L(svg, lx, ly, lx, ly - 34, { stroke: INK, 'marker-end': hA }); T(svg, lx - 4, ly - 40, 'y', { 'font-size': 12 }); }
      if (dim === 3) { L(svg, lx, ly, lx - 20, ly + 16, { stroke: INK, 'marker-end': hA }); T(svg, lx - 34, ly + 24, 'z', { 'font-size': 12 }); }
      const Pn = 2 * dim * r + 1, m = 2 * r + 1, boxN = m ** dim;
      T(svg, 20, 24, `${dim}D, order ${r}: ${Pn}-point stencil`, { 'font-size': 15, 'font-weight': 600 });
      out.innerHTML = `points P = 2·${dim}·${r} + 1 = <b>${Pn}</b> · FLOP per output = 2P − 1 = <b>${2 * Pn - 1}</b>` +
        `<br>convolution box of the same radius: ${m}<sup>${dim}</sup> = ${boxN} points` +
        `<br>the stencil uses <b>${(Pn / boxN * 100).toFixed(Pn === boxN ? 0 : 1)} %</b> of the box`;
      syncButtons(root, 'dim', dim); syncButtons(root, 'ord', ord);
    };
    root.querySelectorAll('[data-dim]').forEach(b => b.addEventListener('click', () => { dim = +b.dataset.dim; draw(); }));
    root.querySelectorAll('[data-ord]').forEach(b => b.addEventListener('click', () => { ord = +b.dataset.ord; draw(); }));
    return { enter: draw, leave() {} };
  };

  // ------------------------------------------------------------ a 2D stencil sweep with fixed boundary cells
  ANIM.sweep = (root) => {
    const svg = newSvg(root, '0 0 1000 400', 'A 12 by 12 input and output grid; the 5-point stencil reads the input and writes interior output points; boundary cells are not computed');
    const N = 12, cs = 26, xi = 60, xo = 560, y0 = 44;
    T(svg, xi, 26, 'input (sweep s)', { 'font-size': 13, class: 'lbl' }); T(svg, xo, 26, 'output (sweep s + 1)', { 'font-size': 13, class: 'lbl' });
    const bnd = (r, c) => r === 0 || c === 0 || r === N - 1 || c === N - 1;
    const inC = grid(svg, xi, y0, N, N, cs, (r, c) => bnd(r, c) ? BGRAY : LBLUE);
    const outC = grid(svg, xo, y0, N, N, cs, () => '#fff');
    for (let k = 0; k <= N; k += 4) { L(svg, xo + k * cs, y0, xo + k * cs, y0 + N * cs, { stroke: INK, 'stroke-width': 1.6 }); L(svg, xo, y0 + k * cs, xo + N * cs, y0 + k * cs, { stroke: INK, 'stroke-width': 1.6 }); }
    const star = svg.appendChild(el('g'));
    const hA = arrowHead(svg, 'sw-a', BLUE); const arrow = L(svg, 0, 0, 0, 0, { stroke: BLUE, 'stroke-width': 2.5, 'marker-end': hA, opacity: 0 });
    const info = T(svg, 60, 386, '', { 'font-size': 14 });
    const pos = [[1, 1], [1, 2], [1, 3], [6, 8]];
    const S = [
      { cap: 'Gray: <b>boundary cells</b>. They hold the boundary conditions of the equation and are never computed. Light blue: interior points of the current sweep.' },
      ...pos.map((p, i) => ({ p, cap: i === 0 ? 'The thread of output (1,1) reads its point and its 4 neighbours from the input: a 2D 5-point stencil. Two of the neighbours are boundary cells: real values, not ghost zeros.' : i < 3 ? 'The next thread does the same one point to the right. Every output uses the same coefficients.' : 'Any interior point, anywhere: the outputs only depend on the input grid, so their order does not matter.' })),
      { all: true, cap: 'So all interior points are computed <b>in parallel</b>, one thread each, blocks of 4 × 4 outputs (thick lines). The boundary of the output is still empty.' },
      { all: true, bnd: true, cap: 'Boundary cells keep their values: copied once into both buffers before the first sweep. Only the inner 10 × 10 points change.' },
      { all: true, bnd: true, swap: true, cap: 'Next sweep: the output becomes the input. The host swaps the two pointers and launches the kernel again.' }
    ];
    const cap = root.querySelector('[data-cap]');
    return stepper(root, () => S.length, (k) => {
      const s = S[k], done = new Set();
      if (s.p) for (let j = 1; j < k; j++) done.add(pos[j - 1].join());
      star.innerHTML = '';
      inC.forEach((e, i) => { const r = Math.floor(i / N), c = i % N; e.setAttribute('fill', bnd(r, c) ? BGRAY : (s.swap ? LGREEN : LBLUE)); });
      outC.forEach((e, i) => {
        const r = Math.floor(i / N), c = i % N;
        let f = '#fff';
        if (bnd(r, c)) f = s.bnd ? BGRAY : '#fff';
        else if (s.all) f = s.swap ? '#fff' : LGREEN;
        else if (done.has(`${r},${c}`)) f = LGREEN;
        if (s.p && r === s.p[0] && c === s.p[1]) f = GREEN;
        e.setAttribute('fill', f);
      });
      if (s.p) {
        const [r, c] = s.p;
        [[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]].forEach(([dr, dc]) => star.appendChild(el('rect', { x: xi + (c + dc) * cs, y: y0 + (r + dr) * cs, width: cs - 1, height: cs - 1, fill: dr || dc ? MBLUE : BLUE, stroke: BLUE, 'stroke-width': 1.5 })));
        arrow.setAttribute('x1', xi + c * cs + cs / 2); arrow.setAttribute('y1', y0 + r * cs + cs / 2); arrow.setAttribute('x2', xo + c * cs + cs / 2 - 6); arrow.setAttribute('y2', y0 + r * cs + cs / 2); arrow.setAttribute('opacity', 1);
        info.textContent = `out[${r}][${c}] = c0·in[${r}][${c}] + c1·in[${r}][${c - 1}] + c2·in[${r}][${c + 1}] + c3·in[${r - 1}][${c}] + c4·in[${r + 1}][${c}]`;
      } else {
        arrow.setAttribute('opacity', 0);
        info.textContent = k === 0 ? '44 boundary cells, 100 interior points' : s.swap ? 'std::swap(d_in, d_out): the new input (green) is the old output' : s.bnd ? 'boundary cells: same values in both buffers' : '100 threads, one per interior point, no dependence between them';
      }
      cap.innerHTML = s.cap;
    }, 2600);
  };

  // ------------------------------------------------------------ many sweeps: Jacobi for Laplace's equation
  ANIM.heat = (root) => {
    const svg = newSvg(root, '0 0 560 360', 'A 2D grid whose top edge is held hot and other edges cold; repeated Jacobi sweeps spread the heat into the interior');
    const N = 22, cs = 13, X0 = 20, Y0 = 34;
    let a = new Float64Array(N * N);
    const init = () => { a = new Float64Array(N * N); for (let c = 0; c < N; c++) a[c] = 1; };
    const sweep = () => { const b = a.slice(); let dmax = 0; for (let r = 1; r < N - 1; r++) for (let c = 1; c < N - 1; c++) { const v = .25 * (a[(r - 1) * N + c] + a[(r + 1) * N + c] + a[r * N + c - 1] + a[r * N + c + 1]); dmax = Math.max(dmax, Math.abs(v - a[r * N + c])); b[r * N + c] = v; } a = b; return dmax; };
    const color = (v) => { const t = Math.max(0, Math.min(1, Math.pow(v, .6))); const lerp = (x, y, u) => Math.round(x + (y - x) * u); return t < .5 ? `rgb(${lerp(255, 233, t * 2)},${lerp(255, 180, t * 2)},${lerp(255, 103, t * 2)})` : `rgb(${lerp(233, 176, t * 2 - 1)},${lerp(180, 58, t * 2 - 1)},${lerp(103, 58, t * 2 - 1)})`; };
    T(svg, X0, 22, 'top edge held at 1, other edges at 0', { 'font-size': 14, class: 'lbl' });
    const cells = []; for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) cells.push(R(svg, X0 + c * cs, Y0 + r * cs, cs - .5, cs - .5, { fill: '#fff' }));
    R(svg, X0 - 1, Y0 - 1, N * cs + 1, N * cs + 1, { fill: 'none', stroke: INK, 'stroke-width': 1.2 });
    const t1 = T(svg, 330, 60, '', { 'font-size': 20, 'font-weight': 600 }), t2 = T(svg, 330, 90, '', { 'font-size': 13 }), t2b = T(svg, 330, 110, '', { 'font-size': 13 }), t3 = T(svg, 330, 132, '', { 'font-size': 13 });
    for (let i = 0; i <= 10; i++) R(svg, 330 + i * 18, 170, 18, 14, { fill: color(i / 10) });
    T(svg, 330, 200, '0', { 'font-size': 14, class: 'lbl' }); T(svg, 528, 200, '1', { 'font-size': 14, class: 'lbl', 'text-anchor': 'end' });
    const plan = [0, 1, 2, 5, 10, 25, 50, 100, 200, 400];
    const cap = root.querySelector('[data-cap]');
    let done = 0, last = 0;
    return stepper(root, () => plan.length, (k) => {
      if (k === 0 || plan[k] < done) { init(); done = 0; last = 0; }
      while (done < plan[k]) { last = sweep(); done++; }
      cells.forEach((e, i) => e.setAttribute('fill', color(a[i])));
      t1.textContent = `after ${done} sweep${done === 1 ? '' : 's'}`;
      t2.textContent = done ? 'largest change in the' : 'initial guess:'; t2b.textContent = done ? `last sweep: ${last.toFixed(4)}` : 'interior = 0';
      t3.textContent = `centre value: ${a[(N >> 1) * N + (N >> 1)].toFixed(3)}`;
      cap.innerHTML = k === 0 ? 'Laplace\'s equation, solved by <b>Jacobi iteration</b>: every interior point becomes the average of its 4 neighbours.'
        : k < 4 ? 'Each sweep moves information <b>one point</b> further from the hot edge: the stencil only reaches its neighbours.'
          : k < 8 ? 'Hundreds of sweeps are normal. Each one is a full kernel launch over the whole grid, so its memory traffic is what matters.'
            : 'The changes shrink towards zero: the field converges to the solution. A solver stops when the largest change is below a tolerance.';
    }, 1800);
  };

  // ------------------------------------------------------------ where the 7 neighbours live in memory
  ANIM.neigh = (root) => {
    const svg = newSvg(root, '0 0 1000 380', 'Three 8 by 8 planes of a 3D grid and the linear memory that stores them; the seven loads of one output are highlighted');
    const N = 8, cs = 22, gx = [70, 390, 710], Y0 = 40;
    const ci = 1, cj = 3, ck = 4;   // centre point: plane 1 of the three shown, row 3, column 4
    const labels = ['plane i − 1', 'plane i', 'plane i + 1'];
    const planes = gx.map((x, p) => { T(svg, x, 28, labels[p], { 'font-size': 13, class: 'lbl' }); return grid(svg, x, Y0, N, N, cs, () => '#fff'); });
    const MS = 3 * N * N, mw = 920 / MS, MY = 268;
    T(svg, 40, MY - 10, 'linear memory: in[i*N*N + j*N + k], one cell per float, the three planes one after another', { 'font-size': 15, class: 'lbl' });
    const mem = Array.from({ length: MS }, (_, m) => R(svg, 40 + m * mw, MY, mw - .4, 26, { fill: '#fff', stroke: '#e0e0e0', 'stroke-width': .3 }));
    for (let p = 0; p <= 3; p++) L(svg, 40 + p * N * N * mw, MY - 4, 40 + p * N * N * mw, MY + 30, { stroke: INK, 'stroke-width': 1.5 });
    const marks = svg.appendChild(el('g'));
    const t1 = T(svg, 40, 330, '', { 'font-size': 14, 'font-weight': 600 }), t2 = T(svg, 40, 354, '', { 'font-size': 13 });
    const L7 = [
      ['in[i][j][k]', 0, 0, 0, 'the centre itself'],
      ['in[i][j][k−1]', 0, 0, -1, '1 float before: same 32-byte sector'],
      ['in[i][j][k+1]', 0, 0, 1, '1 float after'],
      ['in[i][j−1][k]', 0, -1, 0, `N floats before: ${N * 4} B here, 2 KB for N = 512`],
      ['in[i][j+1][k]', 0, 1, 0, 'N floats after'],
      ['in[i−1][j][k]', -1, 0, 0, `N² floats before: ${N * N * 4} B here, 1 MB for N = 512`],
      ['in[i+1][j][k]', 1, 0, 0, 'N² floats after: a different DRAM page']
    ];
    const cap = root.querySelector('[data-cap]');
    return stepper(root, () => L7.length + 2, (k) => {
      const shown = Math.min(k, 7);
      planes.forEach(g => g.forEach(e => e.setAttribute('fill', '#fff')));
      mem.forEach(e => e.setAttribute('fill', '#fff'));
      marks.innerHTML = '';
      for (let q = 0; q < Math.max(1, shown); q++) {
        const [, di, dj, dk] = L7[q], p = ci + di, idx = (cj + dj) * N + ck + dk, cur = q === shown - 1 && k >= 1 && k <= 7;
        planes[p][idx].setAttribute('fill', cur ? GOLD : (di || dj || dk ? MBLUE : BLUE));
        mem[p * N * N + idx].setAttribute('fill', cur ? GOLD : BLUE);
        marks.appendChild(el('line', { x1: 40 + (p * N * N + idx + .5) * mw, y1: MY + 28, x2: 40 + (p * N * N + idx + .5) * mw, y2: MY + 38, stroke: cur ? GOLD : BLUE, 'stroke-width': 2 }));
      }
      if (k >= 1 && k <= 7) { const [s, , , , note] = L7[k - 1]; t1.textContent = `load ${k}: ${s}`; t2.textContent = note; }
      else if (k === 0) { t1.textContent = 'one thread, one output: out[i][j][k]'; t2.textContent = 'centre of the stencil: plane i, row j = 3, column k = 4'; }
      else { t1.textContent = '13 FLOP / (7 loads × 4 B + 1 store × 4 B) = 13/32 = 0.41 FLOP/B'; t2.textContent = 'ideal, each input loaded once and each output stored once: 13 / (4 + 4) = 1.625 FLOP/B'; }
      cap.innerHTML = k === 0 ? 'The basic kernel: each thread computes one output and reads its 7 inputs straight from global memory.'
        : k <= 3 ? 'Neighbours along x (k ± 1) are next to each other in memory, and next to what the neighbouring threads of the warp read.'
          : k <= 5 ? 'Along y the neighbour is a whole row away: N floats.'
            : k <= 7 ? 'Along z it is a whole <b>plane</b> away: N² floats. The 7 loads land in three far-apart places in memory.'
              : 'Every point is loaded by up to 7 threads, and each load is counted again. The basic kernel reaches only <b>0.41 FLOP/B</b> (0.46 = 13/28 if the store is not counted). L1 and L2 catch some of the repeats, but shared memory makes the reuse reliable.';
    }, 2200);
  };

  // ------------------------------------------------------------ ideal intensity: stencil against convolution
  ANIM.aicalc = (root) => {
    const svg = newSvg(root, '0 0 540 360', 'Ideal arithmetic intensity of stencils and convolutions of order 1 to 3 for the chosen dimension');
    const out = root.querySelector('[data-out]');
    let dim = 2;
    const st = (d, r) => (2 * (2 * d * r + 1) - 1) / 8, cv = (d, r) => 2 * (2 * r + 1) ** d / 8;
    const draw = () => {
      svg.innerHTML = '';
      const max = cv(dim, 3) * 1.12, H = 250, Y0 = 290, sc = H / max;
      T(svg, 20, 24, `${dim}D: ideal FLOP/B, order 1 to 3`, { 'font-size': 14, 'font-weight': 600 });
      L(svg, 40, Y0, 520, Y0, { stroke: INK });
      [1, 2, 3].forEach((r, i) => {
        const x = 70 + i * 160, a = st(dim, r), b = cv(dim, r);
        R(svg, x, Y0 - Math.max(2, a * sc), 54, Math.max(2, a * sc), { fill: BLUE });
        R(svg, x + 62, Y0 - b * sc, 54, b * sc, { fill: '#E7B467' });
        T(svg, x + 27, Y0 - Math.max(2, a * sc) - 6, a.toFixed(3).replace(/0+$/, '').replace(/\.$/, ''), { ...mid, 'font-size': 12 });
        T(svg, x + 89, Y0 - b * sc - 6, b.toFixed(3).replace(/0+$/, '').replace(/\.$/, ''), { ...mid, 'font-size': 12 });
        T(svg, x + 58, Y0 + 18, `order ${r}`, { ...mid, 'font-size': 12 });
        T(svg, x + 58, Y0 + 34, `${2 * dim * r + 1} pts vs ${2 * r + 1}${dim > 1 ? '^' + dim : ''}`, { ...mid, 'font-size': 11, class: 'lbl' });
      });
      R(svg, 40, 40, 14, 14, { fill: BLUE }); T(svg, 60, 52, 'stencil (2P − 1)/8', { 'font-size': 12 });
      R(svg, 40, 62, 14, 14, { fill: '#E7B467' }); T(svg, 60, 74, 'convolution 2m^d/8', { 'font-size': 12 });
      let h = '<table><tr><th></th><th>order 1</th><th>order 2</th><th>order 3</th></tr>';
      [1, 2, 3].forEach(d => { h += `<tr${d === dim ? ' style="background:#EEF3FB"' : ''}><td><b>${d}D</b></td>` + [1, 2, 3].map(r => `<td>${st(d, r)} / ${cv(d, r)}</td>`).join('') + '</tr>'; });
      out.innerHTML = h + '</table><span class="small muted">stencil / convolution, FLOP/B</span>' +
        `<br>${dim}D order 3: the convolution reaches <b>${(cv(dim, 3) / st(dim, 3)).toFixed(1)}×</b> the stencil's intensity`;
      syncButtons(root, 'dim', dim);
    };
    root.querySelectorAll('[data-dim]').forEach(b => b.addEventListener('click', () => { dim = +b.dataset.dim; draw(); }));
    return { enter: draw, leave() {} };
  };

  // ------------------------------------------------------------ tiled intensity as a function of the tile size
  ANIM.tcurve = (root) => {
    const svg = newSvg(root, '0 0 1000 330', 'Arithmetic intensity of the 3D tiled kernel and of the z-coarsened kernel against the input tile width t, with the 1024-thread limits');
    const inp = root.querySelector('[data-in=t]'), out = root.querySelector('[data-out]');
    const tiled = (t) => 13 * (t - 2) ** 3 / (4 * t ** 3 + 4 * (t - 2) ** 3);
    const coarse = (t) => 13 * (t - 2) ** 2 / (4 * t * t + 4 * (t - 2) ** 2);
    const X0 = 70, XW = 880, Y0 = 280, YH = 250, tMax = 40, vMax = 1.8;
    const px = (t) => X0 + (t - 2) / (tMax - 2) * XW, py = (v) => Y0 - v / vMax * YH;
    const draw = () => {
      svg.innerHTML = '';
      const t = clampIn(inp, 3, 40, 8);
      L(svg, X0, Y0, X0 + XW, Y0, { stroke: INK }); L(svg, X0, Y0, X0, Y0 - YH, { stroke: INK });
      [0, .5, 1, 1.5].forEach(v => { T(svg, X0 - 8, py(v) + 4, v.toFixed(1), { 'font-size': 12, 'text-anchor': 'end' }); L(svg, X0, py(v), X0 + XW, py(v), { stroke: '#eeeeee' }); });
      [4, 8, 10, 16, 20, 24, 32, 40].forEach(x => T(svg, px(x), Y0 + 18, x, { ...mid, 'font-size': 12 }));
      T(svg, X0 + XW + 14, Y0 + 4, 't', { 'font-size': 13, 'font-style': 'italic' });
      T(svg, 20, 18, 'FLOP/B', { 'font-size': 12, class: 'lbl' });
      L(svg, X0, py(1.625), X0 + XW, py(1.625), { stroke: GREEN, 'stroke-dasharray': '6 4', 'stroke-width': 1.5 }); T(svg, X0 + XW - 4, py(1.625) - 6, 'ideal 13/8 = 1.625', { 'font-size': 12, fill: GREEN, 'text-anchor': 'end' });
      const curve = (f, a, b, attrs) => { let d = ''; for (let x = a; x <= b + 1e-9; x += .25) d += (d ? 'L' : 'M') + px(x).toFixed(1) + ',' + py(f(x)).toFixed(1); P(svg, d, { fill: 'none', ...attrs }); };
      curve(tiled, 3, 10, { stroke: BLUE, 'stroke-width': 3 }); curve(tiled, 10, 40, { stroke: BLUE, 'stroke-width': 2, 'stroke-dasharray': '4 4', opacity: .45 });
      curve(coarse, 3, 32, { stroke: GOLD, 'stroke-width': 3 }); curve(coarse, 32, 40, { stroke: GOLD, 'stroke-width': 2, 'stroke-dasharray': '4 4', opacity: .45 });
      L(svg, px(10), Y0, px(10), Y0 - YH, { stroke: BLUE, 'stroke-dasharray': '3 3' }); T(svg, px(10) + 5, Y0 - YH + 12, 't³ ≤ 1024 → t ≤ 10', { 'font-size': 12, fill: BLUE });
      L(svg, px(32), Y0, px(32), Y0 - YH, { stroke: GOLD, 'stroke-dasharray': '3 3' }); T(svg, px(32) - 5, Y0 - YH + 12, 't² ≤ 1024 → t ≤ 32', { 'font-size': 12, fill: GOLD, 'text-anchor': 'end' });
      L(svg, 560, 200, 590, 200, { stroke: BLUE, 'stroke-width': 3 }); T(svg, 598, 204, 'tiled: 3D block t × t × t', { 'font-size': 12, fill: BLUE });
      L(svg, 560, 224, 590, 224, { stroke: GOLD, 'stroke-width': 3 }); T(svg, 598, 228, 'coarsened in z: block t × t, per plane', { 'font-size': 12, fill: GOLD });
      L(svg, 560, 248, 590, 248, { stroke: GRAY, 'stroke-width': 2, 'stroke-dasharray': '4 4' }); T(svg, 598, 252, 'dashed: too many threads to launch', { 'font-size': 12, class: 'lbl' });
      C(svg, px(8), py(tiled(8)), 5, { fill: BLUE }); T(svg, px(8) + 8, py(tiled(8)) + 16, '0.96', { 'font-size': 12, fill: BLUE });
      C(svg, px(32), py(coarse(32)), 5, { fill: GOLD }); T(svg, px(32) + 8, py(coarse(32)) + 16, '1.52', { 'font-size': 12, fill: GOLD });
      L(svg, px(t), Y0, px(t), Y0 - YH, { stroke: INK, 'stroke-width': 1.5 });
      C(svg, px(t), py(tiled(t)), 6, { fill: '#fff', stroke: BLUE, 'stroke-width': 2.5 }); C(svg, px(t), py(coarse(t)), 6, { fill: '#fff', stroke: GOLD, 'stroke-width': 2.5 });
      const ok3 = t ** 3 <= 1024, ok2 = t * t <= 1024;
      out.innerHTML = `t = <b>${t}</b> · 3D tile: ${fmt(t ** 3)} threads${ok3 ? '' : ' <span style="color:#B03A3A">(over 1024)</span>'}, ${fmt(t ** 3 * 4)} B shared, <b>${tiled(t).toFixed(2)} FLOP/B</b>` +
        `<br>coarsened: ${fmt(t * t)} threads${ok2 ? '' : ' <span style="color:#B03A3A">(over 1024)</span>'}, 3 planes = ${fmt(3 * t * t * 4)} B shared, <b>${coarse(t).toFixed(2)} FLOP/B</b> per plane`;
      syncButtons(root, 'p', t);
    };
    inp.addEventListener('input', draw);
    root.querySelectorAll('[data-p]').forEach(b => b.addEventListener('click', () => { inp.value = b.dataset.p; draw(); }));
    return { enter: draw, leave() {} };
  };

  // ------------------------------------------------------------ halo fraction, 2D against 3D
  ANIM.halo = (root) => {
    const svg = newSvg(root, '0 0 560 380', 'A 2D tile and a 3D tile drawn plane by plane, halo cells in light blue, and bars with the halo fraction of each');
    const q = (n) => root.querySelector(`[data-in=${n}]`), out = root.querySelector('[data-out]');
    const draw = () => {
      svg.innerHTML = '';
      const t = clampIn(q('t'), 3, 32, 8);
      const f2 = 1 - (t - 2) ** 2 / t ** 2, f3 = 1 - (t - 2) ** 3 / t ** 3;
      const ring = (r, c) => r === 0 || c === 0 || r === t - 1 || c === t - 1;
      // 2D tile
      T(svg, 20, 18, `2D tile ${t} × ${t}`, { 'font-size': 13, class: 'lbl' });
      const c2 = 150 / t;
      for (let r = 0; r < t; r++) for (let c = 0; c < t; c++) R(svg, 20 + c * c2, 28 + r * c2, c2 - (t > 20 ? .3 : 1), c2 - (t > 20 ? .3 : 1), { fill: ring(r, c) ? MBLUE : BLUE });
      // 3D tile, plane by plane
      T(svg, 200, 18, `3D tile ${t} × ${t} × ${t}, plane by plane`, { 'font-size': 13, class: 'lbl' });
      const cols = Math.ceil(Math.sqrt(t * 1.6)), rows = Math.ceil(t / cols), pw = Math.min(70, 340 / cols - 6, 210 / rows - 18), c3 = pw / t;
      for (let z = 0; z < t; z++) {
        const ox = 200 + (z % cols) * (pw + 6), oy = 28 + Math.floor(z / cols) * (pw + 16), face = z === 0 || z === t - 1;
        if (t <= 12) { for (let r = 0; r < t; r++) for (let c = 0; c < t; c++) R(svg, ox + c * c3, oy + r * c3, c3 - .6, c3 - .6, { fill: face || ring(r, c) ? MBLUE : BLUE }); }
        else { R(svg, ox, oy, pw, pw, { fill: MBLUE }); if (!face) R(svg, ox + c3, oy + c3, pw - 2 * c3, pw - 2 * c3, { fill: BLUE }); }
        if (t <= 16) T(svg, ox, oy + pw + 11, `z=${z}`, { 'font-size': 9, class: 'lbl' });
      }
      // bars
      const BY = 270;
      [['2D', f2], ['3D', f3]].forEach(([s, f], i) => {
        const y = BY + i * 44;
        T(svg, 20, y + 19, `${s} halo`, { 'font-size': 13, 'font-weight': 600 });
        R(svg, 110, y, 400, 26, { fill: '#f1f1f1' }); R(svg, 110, y, 400 * f, 26, { fill: MBLUE });
        T(svg, 116 + 400 * f, y + 18, `${(f * 100).toFixed(0)} %`, { 'font-size': 13 });
      });
      T(svg, 20, 368, 'dark blue: interior (output tile) · light blue: halo, loaded but not computed', { 'font-size': 11, class: 'lbl' });
      out.innerHTML = `2D: ${t * t} − ${(t - 2) ** 2} = ${t * t - (t - 2) ** 2} of ${t * t} → <b>${(f2 * 100).toFixed(1)} %</b>` +
        `<br>3D: ${fmt(t ** 3)} − ${fmt((t - 2) ** 3)} = ${fmt(t ** 3 - (t - 2) ** 3)} of ${fmt(t ** 3)} → <b>${(f3 * 100).toFixed(1)} %</b>` +
        `<br>3D block of ${fmt(t ** 3)} threads: ${t ** 3 <= 1024 ? 'fits' : '<span style="color:#B03A3A">over the 1024 limit</span>'}`;
      syncButtons(root, 'p', t);
    };
    root.querySelectorAll('input').forEach(i => i.addEventListener('input', draw));
    root.querySelectorAll('[data-p]').forEach(b => b.addEventListener('click', () => { q('t').value = b.dataset.p; draw(); }));
    return { enter: draw, leave() {} };
  };

  // ------------------------------------------------------------ warp access pattern: 8x8x8 against 32x8x4
  ANIM.coal = (root) => {
    let mode = 'cube';
    const svg = newSvg(root, '0 0 1000 360', 'The global-memory addresses loaded by one warp of the tiled kernel, for a cubic 8 by 8 by 8 block and for a flat 32 by 8 by 4 block, with the 32-byte sectors they touch');
    const N = 64, cw = 14, MX = 60, cap = root.querySelector('[data-cap]');
    const draw = (w) => {
      svg.innerHTML = '';
      const B = mode === 'cube' ? [8, 8, 8] : [32, 8, 4], OUT = B.map(b => b - 2);
      // warp w = threads 32w .. 32w+31 in x-fastest order; block index (1,1,1)
      const loads = [];
      for (let l = 0; l < 32; l++) {
        const id = 32 * w + l, tx = id % B[0], ty = Math.floor(id / B[0]) % B[1], tz = Math.floor(id / (B[0] * B[1]));
        loads.push({ i: OUT[2] + tz - 1, j: OUT[1] + ty - 1, k: OUT[0] + tx - 1, tx, ty, tz });
      }
      const rows = [...new Set(loads.map(o => `${o.i},${o.j}`))];
      T(svg, 20, 22, `block ${B.join(' × ')}, warp ${w}: threads ${32 * w}…${32 * w + 31}  →  ${rows.length} row${rows.length > 1 ? 's' : ''} of the grid`, { 'font-size': 14, 'font-weight': 600 });
      T(svg, MX, 46, 'each strip: one grid row (64 floats = 256 B); ticks: 32-byte sectors', { 'font-size': 12, class: 'lbl' });
      let sectors = 0;
      rows.forEach((key, ri) => {
        const y = 62 + ri * 58;
        const used = loads.filter(o => `${o.i},${o.j}` === key).map(o => o.k);
        const label = `grid row z = ${key.split(',')[0]}, y = ${key.split(',')[1]}: x = ${Math.min(...used)}…${Math.max(...used)}`;
        T(svg, MX, y + 2, label, { 'font-size': 11, class: 'lbl' });
        const sec = new Set(used.map(k => Math.floor(k / 8)));
        sectors += sec.size;
        for (let k = 0; k < N; k++) R(svg, MX + k * cw, y + 8, cw - 1, 22, { fill: used.includes(k) ? BLUE : sec.has(Math.floor(k / 8)) ? LGOLD : '#fff', stroke: '#d6d6d6', 'stroke-width': .5 });
        for (let s = 0; s <= 8; s++) L(svg, MX + s * 8 * cw - .5, y + 4, MX + s * 8 * cw - .5, y + 34, { stroke: INK, 'stroke-width': 1.2 });
        sec.forEach(s => R(svg, MX + s * 8 * cw - .5, y + 6, 8 * cw, 26, { fill: 'none', stroke: GOLD, 'stroke-width': 2.5 }));
      });
      const fetched = sectors * 32, eff = 128 / fetched;
      R(svg, MX, 300, 14, 14, { fill: BLUE }); T(svg, MX + 20, 312, 'loaded by the warp', { 'font-size': 12 });
      R(svg, MX + 190, 300, 14, 14, { fill: LGOLD, stroke: GOLD }); T(svg, MX + 210, 312, 'fetched but unused', { 'font-size': 12 });
      T(svg, MX, 344, `${sectors} sectors = ${fetched} B fetched for 128 B used: ${(eff * 100).toFixed(0)} % efficiency, ${rows.length} separate memory location${rows.length > 1 ? 's' : ''}`, { 'font-size': 14, 'font-weight': 600, fill: eff > .7 ? GREEN : RED });
      cap.innerHTML = mode === 'cube'
        ? (w === 0 ? 'An 8 × 8 × 8 block: a warp of 32 threads covers <b>4 rows of 8</b> consecutive x positions. Rows are N floats apart, so one load instruction hits 4 distant places.' : 'Every warp of the cube looks the same: 4 short rows, each 32 B long and starting at 6·bx − 1, so most rows straddle two sectors. About half of the fetched bytes are wasted.')
        : (w === 0 ? 'A 32 × 8 × 4 block: the warp is <b>one row of 32</b> consecutive floats, 128 B in one place. The row starts at 30·bx − 1, so 5 sectors instead of 4, still far better.' : 'Every warp is a full row. The catch: the output tile is only 30 × 6 × 2 = 360 points, and intensity drops to <b>0.85 FLOP/B</b> (0.96 for the cube). The tile is still too small: we need a different idea.');
    };
    const st = stepper(root, () => 4, draw, 2600);
    modeButtons(root, 'mode', (m) => { mode = m; st.reset(); });
    return st;
  };

  // ------------------------------------------------------------ thread coarsening: three planes marching along z
  ANIM.zmarch = (root) => {
    const svg = newSvg(root, '0 0 1000 370', 'Input planes of a column, three shared-memory buffers inPrev_s, inCurr_s and inNext_s, and output planes; the block marches along z one plane per iteration');
    const OUT = 4, IN = OUT + 2, PH = 34, PY = 50;
    const cap = root.querySelector('[data-cap]');
    const slab = (x, y, w, h, a) => P(svg, `M${x},${y} l${w},0 l22,-14 l${-w},0 z M${x},${y} l0,${h} l${w},0 l0,${-h}`, a);
    // steps: 0 init, then for each iteration 3 sub-steps (load, compute, shift), then summary
    const total = 1 + 3 * OUT + 1;
    return stepper(root, () => total, (k) => {
      svg.innerHTML = '';
      const it = k === 0 ? -1 : Math.min(OUT - 1, Math.floor((k - 1) / 3)), sub = k === 0 ? -1 : k === total - 1 ? 3 : (k - 1) % 3;
      // buffers content (input plane indices, relative: plane p = iStart - 1 + p)
      let prev = 0, curr = 1, next = null, loaded = 2, computed = 0;
      if (k > 0) {
        loaded = 2 + it + 1; next = it + 2; prev = it; curr = it + 1; computed = it + (sub >= 1 ? 1 : 0);
        if (sub >= 2) { prev = it + 1; curr = it + 2; next = it + 2; }
        if (k === total - 1) { computed = OUT; }
      }
      T(svg, 30, 22, 'input planes (global memory)', { 'font-size': 13, class: 'lbl' });
      for (let p = 0; p < IN; p++) {
        const y = PY + 14 + p * (PH + 8), isL = p < loaded;
        const role = p === prev && k < total - 1 ? 'prev' : p === curr && k < total - 1 ? 'curr' : p === next && k > 0 && sub === 0 ? 'next' : '';
        slab(30, y, 190, PH, { fill: role === 'curr' ? BLUE : role ? MBLUE : isL ? '#EEF3FB' : '#fff', stroke: role ? BLUE : '#bdbdbd', 'stroke-width': role ? 2 : 1 });
        T(svg, 236, y + 22, p === 0 ? 'z = iStart − 1' : `z = iStart + ${p - 1}`, { 'font-size': 12, class: 'lbl' });
      }
      T(svg, 400, 22, 'shared memory of the block', { 'font-size': 13, class: 'lbl' });
      const bufs = [['inPrev_s', prev], ['inCurr_s', curr], ['inNext_s', next]];
      bufs.forEach(([name, p], b) => {
        const y = 40 + b * 74, on = (b === 2 && sub === 0) || (sub === 2);
        R(svg, 400, y, 220, 54, { fill: b === 1 ? LBLUE : '#fff', stroke: on ? GOLD : '#9a9a9a', 'stroke-width': on ? 3 : 1.2, rx: 4 });
        T(svg, 412, y + 22, name, { 'font-size': 14, 'font-weight': 600 });
        T(svg, 412, y + 42, p == null ? 'empty' : p === 0 ? 'holds z = iStart − 1' : `holds z = iStart + ${p - 1}`, { 'font-size': 12, class: 'lbl' });
      });
      if (sub === 2 && k < total - 1) { const hA = arrowHead(svg, 'zm-a', GOLD); P(svg, 'M632,140 C680,140 680,80 632,70', { fill: 'none', stroke: GOLD, 'stroke-width': 2.5, 'marker-end': hA }); P(svg, 'M632,214 C680,214 680,154 632,144', { fill: 'none', stroke: GOLD, 'stroke-width': 2.5, 'marker-end': hA }); }
      T(svg, 740, 22, 'output planes', { 'font-size': 13, class: 'lbl' });
      for (let o = 0; o < OUT; o++) {
        const y = PY + 14 + (o + 1) * (PH + 8), cur = sub === 1 && o === it;
        slab(740, y, 160, PH, { fill: o < computed ? (cur ? GREEN : LGREEN) : '#fff', stroke: cur ? GREEN : '#bdbdbd', 'stroke-width': cur ? 2.5 : 1 });
        T(svg, 916, y + 22, `iStart + ${o}`, { 'font-size': 12, class: 'lbl' });
      }
      const ld = loaded * 1024, outs = computed * 900, ai = outs ? 13 * outs / (4 * ld + 4 * outs) : 0;
      T(svg, 400, 286, `loaded: ${loaded} planes × 32² = ${fmt(ld)} points`, { 'font-size': 13 });
      T(svg, 400, 308, `computed: ${computed} planes × 30² = ${fmt(outs)} outputs`, { 'font-size': 13 });
      T(svg, 400, 330, `intensity so far: ${ai ? ai.toFixed(2) : '—'} FLOP/B`, { 'font-size': 14, 'font-weight': 600, fill: BLUE });
      T(svg, 400, 352, 'shared memory: 3 × 32² × 4 B = 12 KB, whatever the depth', { 'font-size': 12, class: 'lbl' });
      cap.innerHTML = k === 0 ? 'A 2D block of t × t threads owns a <b>column</b> of the grid. Before the loop it loads the first two input planes: z = iStart − 1 into <code>inPrev_s</code>, z = iStart into <code>inCurr_s</code>.'
        : k === total - 1 ? `After ${OUT} planes. With 30 planes per block (t = 32) the block loads 32 planes and computes 30: <b>1.47 FLOP/B</b>; each extra step adds 1024 loads for 900 outputs, so a deep column tends to <b>1.52</b>.`
          : sub === 0 ? `Iteration ${it}: every thread loads one point of the next plane into <code>inNext_s</code>, then <code>__syncthreads()</code>. Only <b>one new plane</b> per output plane.`
            : sub === 1 ? `The inner 30 × 30 threads compute output plane iStart + ${it}: x and y neighbours from <code>inCurr_s</code>, z neighbours from <code>inPrev_s</code> and <code>inNext_s</code>.`
              : '<code>__syncthreads()</code>, then shift: <code>inPrev_s ← inCurr_s</code>, <code>inCurr_s ← inNext_s</code>. Each thread copies only its own slot.';
    }, 2000);
  };

  // ------------------------------------------------------------ register tiling: who reads which slot
  ANIM.regs = (root) => {
    const svg = newSvg(root, '0 0 1000 360', 'Three planes of the coarsening kernel; the x and y neighbours of a thread are read from slots written by other threads, the z neighbours only from its own slot');
    const n = 8, cs = 24, ty = 3, tx = 4, cap = root.querySelector('[data-cap]');
    const gx = [40, 280, 520], GY = 60;
    const S = [
      'The coarsening kernel keeps three planes in shared memory. Take thread (ty, tx) = (3, 4).',
      'Its x and y neighbours in <code>inCurr_s</code> were loaded by <b>four other threads</b>. Those slots are read by up to five threads: they must be in shared memory.',
      'But <code>inPrev_s[3][4]</code> and <code>inNext_s[3][4]</code> are read by this thread <b>only</b>: no other thread ever needs them, so sharing them buys nothing.',
      '<b>Register tiling</b>: keep the z neighbours in registers <code>inPrev</code>, <code>inNext</code> (and <code>inCurr</code>), and only one plane, <code>inCurr_s</code>, in shared memory.',
      'For 32 × 32 blocks: shared memory 12 KB → <b>4 KB</b>; 3 more registers per thread = <b>3072</b> per block. Global loads do not change, so neither does the intensity. Higher orders need more registers: a common register versus shared-memory trade-off.'
    ];
    return stepper(root, () => S.length, (k) => {
      svg.innerHTML = '';
      const reg = k >= 3;
      const names = ['inPrev_s', 'inCurr_s', 'inNext_s'];
      const hA = arrowHead(svg, 'rg-a', GOLD);
      names.forEach((nm, p) => {
        const x0 = gx[p], asReg = reg && p !== 1;
        T(svg, x0, GY - 14, asReg ? (p === 0 ? 'inPrev (registers)' : 'inNext (registers)') : nm + ' (shared)', { 'font-size': 13, 'font-weight': 600, fill: asReg ? GREEN : INK });
        for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
          const me = r === ty && c === tx, nb = p === 1 && Math.abs(r - ty) + Math.abs(c - tx) === 1;
          let f = p === 1 ? LBLUE : '#fff';
          if (k >= 1 && p === 1 && nb) f = GOLD;
          if (k >= 1 && me && p === 1) f = BLUE;
          if (k >= 2 && me && p !== 1) f = BLUE;
          if (asReg && !me) f = LGREEN;
          R(svg, x0 + c * cs, GY + r * cs, cs - (asReg ? 5 : 2), cs - (asReg ? 5 : 2), { fill: f, stroke: asReg ? GREEN : '#c9c9c9', rx: asReg ? 6 : 1 });
        }
        if (k >= 1 && p === 1) [[0, -1], [0, 1], [-1, 0], [1, 0]].forEach(([dr, dc]) => {
          const x1 = x0 + (tx + dc) * cs + cs / 2, y1 = GY + (ty + dr) * cs + cs / 2, x2 = x0 + tx * cs + cs / 2 - dc * 6, y2 = GY + ty * cs + cs / 2 - dr * 6;
          svg.appendChild(el('line', { x1, y1, x2: (x1 + x2) / 2, y2: (y1 + y2) / 2, stroke: INK, 'stroke-width': 1.5, 'marker-end': hA }));
        });
      });
      // memory bars
      const BX = 770;
      T(svg, BX, 46, 'per block of 32 × 32', { 'font-size': 13, class: 'lbl' });
      const sh = reg ? 4 : 12;
      T(svg, BX, 80, `shared memory: ${sh} KB`, { 'font-size': 13, 'font-weight': 600 });
      R(svg, BX, 88, 200, 20, { fill: '#f1f1f1' }); R(svg, BX, 88, 200 * sh / 12, 20, { fill: BLUE });
      T(svg, BX, 140, `extra registers: ${reg ? '3 072' : '0'}`, { 'font-size': 13, 'font-weight': 600 });
      R(svg, BX, 148, 200, 20, { fill: '#f1f1f1' }); R(svg, BX, 148, reg ? 200 : 0, 20, { fill: GREEN });
      T(svg, BX, 200, 'global loads per output:', { 'font-size': 13 }); T(svg, BX, 220, '1.21 in both kernels', { 'font-size': 13, 'font-weight': 600 });
      T(svg, 40, 300, k === 0 ? '' : k === 1 ? 'gold: written by other threads, read by thread (3,4)' : k === 2 ? 'blue: the only slots thread (3,4) reads in the prev and next planes' : 'green: per-thread registers, invisible to other threads', { 'font-size': 13 });
      T(svg, 40, 330, reg ? 'out = c0·inCurr + c1…c4·inCurr_s[ty±1][tx±1] + c5·inPrev + c6·inNext' : 'out = c0·inCurr_s[ty][tx] + c1…c4·inCurr_s[ty±1][tx±1] + c5·inPrev_s[ty][tx] + c6·inNext_s[ty][tx]', { 'font-size': 12, class: 'lbl' });
      cap.innerHTML = S[k];
    }, 2800);
  };

  // ------------------------------------------------------------ resources per block
  ANIM.res = (root) => {
    const q = (n) => root.querySelector(`[data-in=${n}]`), out = root.querySelector('[data-out]');
    const update = () => {
      const r = clampIn(q('r'), 1, 3, 1), t3 = clampIn(q('t3'), 3, 16, 8), t2 = clampIn(q('t2'), 4, 64, 32), D = clampIn(q('d'), 1, 512, 30);
      const Pn = 6 * r + 1, F = 2 * Pn - 1;
      const bad = (s) => `<span style="color:#B03A3A">${s}</span>`;
      const kb = (b) => b >= 1024 ? (b / 1024).toFixed(b % 1024 ? 1 : 0) + ' KB' : b + ' B';
      const thr = (n) => n > 1024 ? bad(fmt(n) + ' (over 1024)') : fmt(n);
      const shm = (b) => b > 48 * 1024 ? bad(kb(b) + ' (over 48 KB)') : kb(b);
      const o3 = Math.max(0, t3 - 2 * r), o2 = Math.max(0, t2 - 2 * r);
      const rows = [
        ['basic, 8 × 8 × 8', fmt(512), '—', 'baseline', fmt(512), F / (4 * Pn + 4)],
        [`tiled, ${t3} × ${t3} × ${t3}`, thr(t3 ** 3), shm(t3 ** 3 * 4), 'baseline', fmt(o3 ** 3), o3 ? F * o3 ** 3 / (4 * t3 ** 3 + 4 * o3 ** 3) : 0],
        [`coarsened, ${t2} × ${t2}`, thr(t2 * t2), shm((2 * r + 1) * t2 * t2 * 4) + ` (${2 * r + 1} planes)`, 'baseline', fmt(o2 * o2 * D), o2 ? F * o2 * o2 * D / (4 * t2 * t2 * (D + 2 * r) + 4 * o2 * o2 * D) : 0],
        ['+ register tiling', thr(t2 * t2), shm(t2 * t2 * 4) + ' (1 plane)', `+${2 * r + 1}/thread = ${fmt((2 * r + 1) * t2 * t2)}`, fmt(o2 * o2 * D), o2 ? F * o2 * o2 * D / (4 * t2 * t2 * (D + 2 * r) + 4 * o2 * o2 * D) : 0]
      ];
      let h = '<table><tr><th>kernel</th><th>threads</th><th>shared memory</th><th>extra registers</th><th>outputs / block</th><th>FLOP/B</th></tr>';
      rows.forEach(rw => { h += `<tr><td><b>${rw[0]}</b></td><td>${rw[1]}</td><td>${rw[2]}</td><td>${rw[3]}</td><td>${rw[4]}</td><td><b>${rw[5].toFixed(2)}</b></td></tr>`; });
      out.innerHTML = h + '</table>' +
        `<p class="small" style="margin:.6em 0 0;max-width:none;font-family:var(--body)">Order ${r}: ${Pn}-point stencil, ${F} FLOP per output, ideal ${(F / 8).toFixed(3)} FLOP/B. Halo ${r} thick, so tiles lose 2·${r} points per dimension. ` +
        `FLOP/B of internal blocks, loads and stores counted. Red: over the 1024 threads per block, or over the 48 KB of static shared memory per block.</p>`;
    };
    root.querySelectorAll('input').forEach(i => i.addEventListener('input', update));
    return { enter: update, leave() {} };
  };
