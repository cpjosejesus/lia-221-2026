  const arrowHead = (svg, id, color) => {
    const d = svg.querySelector('defs') || svg.insertBefore(el('defs'), svg.firstChild);
    const m = el('marker', { id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' });
    m.appendChild(el('path', { d: 'M0,0 L10,5 L0,10 z', fill: color })); d.appendChild(m);
    return `url(#${id})`;
  };
  const GRAY = '#9a9a9a', BLUE = '#2F6FD0', GOLD = '#C98A1B', RED = '#B03A3A', GREEN = '#2E7D4F', INK = '#1A1A1A';
  const LBLUE = '#C9DBF4', LGOLD = '#F6EAD2', LRED = '#F6DADA', LGREEN = '#D6EBDC', FAINT = '#F4F4F4';
  const syncButtons = (root, attr, val) => root.querySelectorAll(`[data-${attr}]`).forEach(b => b.classList.toggle('on', b.dataset[attr] === val));
  const range = (n) => [...Array(n).keys()];
  const sum = (a) => a.reduce((x, y) => x + y, 0);
  const lg = (n) => Math.round(Math.log2(n));
  const pow2 = (v, lo, hi) => Math.pow(2, Math.max(lg(lo), Math.min(lg(hi), Math.round(Math.log2(Math.max(1, +v || lo))))));
  const A = (o) => { for (const k in o) if (o[k] == null) delete o[k]; return o; };
  const tlist = (ts) => ts.length <= 6 ? ts.map(t => 't' + t).join(', ') : `t${ts[0]}, t${ts[1]}, …, t${ts[ts.length - 1]}`;

  // the in-place kernels on n elements with n/2 threads: one entry per loop iteration
  // simple: owner 2t, stride 1, 2, 4 ...; conv: owner t, stride n/2 ... 1; rev (exercise 3): owner n/2 + t, adds from the left
  const traceIts = (n, mode) => {
    const nt = n / 2, its = [];
    if (mode === 'simple') for (let s = 1; s <= nt; s *= 2) its.push({ s, ops: range(nt).filter(t => t % s === 0).map(t => ({ t, d: 2 * t, src: 2 * t + s })) });
    else if (mode === 'conv') for (let s = nt; s >= 1; s /= 2) its.push({ s, ops: range(s).map(t => ({ t, d: t, src: t + s })) });
    else for (let s = nt; s >= 1; s /= 2) its.push({ s, ops: range(nt).filter(t => t >= nt - s).map(t => ({ t, d: nt + t, src: nt + t - s })) });
    return its;
  };
  const ownerOf = (mode, t, nt) => mode === 'simple' ? 2 * t : mode === 'conv' ? t : nt + t;

  function traceAnim(root, o) {
    let mode = o.mode;
    const n = o.vals.length, nt = n / 2;
    const svg = newSvg(root, `0 0 1000 ${o.h}`, o.label);
    const cap = root.querySelector('[data-cap]');
    const rowsOf = (IT) => { const rs = [o.vals.slice()]; IT.forEach(it => { const p = rs[rs.length - 1], a = p.slice(); it.ops.forEach(q => { a[q.d] = p[q.d] + p[q.src]; }); rs.push(a); }); return rs; };
    const draw = (k) => {
      svg.innerHTML = '';
      const aB = arrowHead(svg, o.id + '-b', BLUE), aG = arrowHead(svg, o.id + '-g', '#bdbdbd');
      const IT = traceIts(n, mode), RS = rowsOf(IT), last = IT.length, fin = k > last;
      const X = (j) => o.x0 + j * o.cw, Y = (r) => o.y0 + r * o.rh, cw = o.cw - 4;
      const cur = k >= 1 && k <= last ? IT[k - 1] : null, act = cur ? cur.ops.map(q => q.t) : [];
      for (let t = 0; t < nt; t++) {
        const j = ownerOf(mode, t, nt), on = act.includes(t);
        T(svg, X(j) + cw / 2, o.y0 - 10, 't' + t, { ...mid, 'font-size': o.fs - 1, fill: on ? BLUE : k === 0 || fin ? INK : GRAY, 'font-weight': on ? 700 : 400 });
      }
      const res = mode === 'rev' ? n - 1 : 0;
      for (let r = 0; r <= Math.min(k, last); r++) {
        const y = Y(r), wr = r ? IT[r - 1].ops.map(q => q.d) : [];
        T(svg, o.x0 - 10, y + o.ch / 2 + 5, r ? `stride ${IT[r - 1].s}` : 'start', { 'font-size': 13, 'text-anchor': 'end', class: 'lbl' });
        for (let j = 0; j < n; j++) {
          const w = wr.includes(j), gold = fin && r === last && j === res;
          R(svg, X(j), y, cw, o.ch, { fill: gold ? GOLD : r === 0 ? LBLUE : w ? (r === k ? BLUE : '#6F9BDD') : FAINT, stroke: r === 0 || w ? 'none' : '#e2e2e2', rx: 2 });
          T(svg, X(j) + cw / 2, y + o.ch / 2 + 5, String(RS[r][j]), { ...mid, 'font-size': o.fs, fill: gold || (w && r) ? '#fff' : r === 0 ? INK : '#9a9a9a' });
        }
        if (r) IT[r - 1].ops.forEach(q => {
          const now = r === k;
          L(svg, X(q.d) + cw / 2, Y(r - 1) + o.ch + 1, X(q.d) + cw / 2, y - 3, { stroke: '#bdbdbd', 'stroke-width': 1.2, 'marker-end': aG });
          L(svg, X(q.src) + cw / 2, Y(r - 1) + o.ch + 1, X(q.d) + cw / 2 + (q.src > q.d ? 3 : -3), y - 3, { stroke: now ? BLUE : '#bdbdbd', 'stroke-width': now ? 2 : 1.2, 'marker-end': now ? aB : aG });
        });
      }
      if (o.warp) {
        const ws = o.warp, nw = nt / ws, PX = o.px;
        T(svg, PX, o.y0 - 10, `warps (toy size ${ws})`, { 'font-size': 13, class: 'lbl' });
        let wi = 0, useful = 0;
        for (let r = 1; r <= Math.min(k, last); r++) { const a = IT[r - 1].ops.map(q => q.t); for (let w = 0; w < nw; w++) if (a.some(t => Math.floor(t / ws) === w)) wi++; useful += a.length; }
        let curW = 0, idle = 0;
        for (let w = 0; w < nw; w++) {
          const y = o.y0 + 8 + w * 46, on = act.some(t => Math.floor(t / ws) === w);
          if (on) curW++;
          T(svg, PX, y + 22, `warp ${w}`, { 'font-size': 13 });
          for (let l = 0; l < ws; l++) {
            const t = w * ws + l, a = act.includes(t);
            if (on && !a) idle++;
            R(svg, PX + 70 + l * 44, y, 38, 32, { fill: a ? BLUE : on ? LRED : '#EDEDED', rx: 3 });
            T(svg, PX + 89 + l * 44, y + 21, 't' + t, { ...mid, 'font-size': 13, fill: a ? '#fff' : on ? RED : GRAY });
          }
        }
        const ly = o.y0 + 8 + nw * 46 + 24, lines = cur ? [
          `warps issued now: ${curW}`,
          `idle lanes in them: ${idle}`,
          `so far: ${useful} adds / ${wi * ws} slots`,
          `efficiency: ${Math.round(100 * useful / (wi * ws))} %`] : fin ? [`${useful} adds in total`, `${wi} warp issues × ${ws} lanes`, `= ${wi * ws} lane slots`, `efficiency: ${Math.round(100 * useful / (wi * ws))} %`] : ['blue: active lane', 'red: idle lane in an', 'issued warp (wasted)', 'grey: warp not issued'];
        lines.forEach((s, i) => T(svg, PX, ly + i * 24, s, { 'font-size': 14, 'font-weight': i === lines.length - 1 && (cur || fin) ? 700 : 400 }));
      }
      cap.innerHTML = o.caption(mode, k, IT, RS);
      syncButtons(root, 'mode', mode);
    };
    const st = stepper(root, () => traceIts(n, mode).length + 2, draw, o.ms || 2200);
    modeButtons(root, 'mode', (m) => { mode = m; st.reset(); });
    return st;
  }

  // ------------------------------------------------------------ max and sum reduction trees
  ANIM.trees = (root) => {
    let mode = 'max';
    const svg = newSvg(root, '0 0 1000 360', 'A reduction tree over eight inputs: four operations in the first step, two in the second, one in the third');
    const cap = root.querySelector('[data-cap]');
    const draw = (k) => {
      svg.innerHTML = '';
      const v0 = mode === 'max' ? [3, 1, 7, 0, 4, 1, 6, 3] : [4, 7, 2, 3, 8, 5, 9, 6];
      const op = mode === 'max' ? Math.max : (a, b) => a + b, sym = mode === 'max' ? 'max' : '+';
      const lv = [v0.map((v, i) => ({ v, x: 190 + i * 96 }))];
      for (let l = 1; l <= 3; l++) { const p = lv[l - 1]; lv.push(range(p.length / 2).map(j => ({ v: op(p[2 * j].v, p[2 * j + 1].v), x: (p[2 * j].x + p[2 * j + 1].x) / 2 }))); }
      const Y = (l) => 20 + l * 94, bw = 48, bh = 34;
      for (let l = 0; l <= Math.min(k, 3); l++) {
        T(svg, 20, Y(l) + 23, l ? `step ${l}` : 'input', { 'font-size': 14, class: 'lbl' });
        if (l) T(svg, 990, Y(l) - 20, `${lv[l].length} ${sym === '+' ? 'adder' : 'comparator'}${lv[l].length > 1 ? 's' : ''}`, { 'font-size': 14, 'text-anchor': 'end', fill: k === l ? BLUE : GRAY });
        lv[l].forEach((nd, j) => {
          if (l) {
            const a = lv[l - 1][2 * j], b = lv[l - 1][2 * j + 1], cy = Y(l) - 30;
            L(svg, a.x, Y(l - 1) + bh, nd.x - 10, cy - 12, { stroke: GRAY, 'stroke-width': 1.5 });
            L(svg, b.x, Y(l - 1) + bh, nd.x + 10, cy - 12, { stroke: GRAY, 'stroke-width': 1.5 });
            L(svg, nd.x, cy + 16, nd.x, Y(l), { stroke: GRAY, 'stroke-width': 1.5 });
            svg.appendChild(el('circle', { cx: nd.x, cy, r: 18, fill: k === l ? '#F2C230' : LGOLD, stroke: GOLD }));
            T(svg, nd.x, cy + 5, sym, { ...mid, 'font-size': 13 });
          }
          const top = l === 3 && k >= 3;
          R(svg, nd.x - bw / 2, Y(l), bw, bh, { fill: top ? BLUE : l === k ? '#6F9BDD' : LBLUE, rx: 3 });
          T(svg, nd.x, Y(l) + 23, String(nd.v), { ...mid, 'font-size': 16, 'font-weight': 600, fill: top || l === k ? '#fff' : INK });
        });
      }
      const seqP = mode === 'max' ? '((((((3 max 1) max 7) max 0) max 4) max 1) max 6) max 3' : '((((((4 + 7) + 2) + 3) + 8) + 5) + 9) + 6';
      const treeP = mode === 'max' ? '((3 max 1) max (7 max 0)) max ((4 max 1) max (6 max 3))' : '((4 + 7) + (2 + 3)) + ((8 + 5) + (9 + 6))';
      cap.innerHTML = [
        `Time flows downwards. ${mode === 'max' ? 'A max' : 'A sum'} reduction of 8 values; the sequential loop would need 7 steps, one operation each.`,
        `Step 1: <b>4</b> ${sym} operations in parallel on the 4 pairs of inputs.`,
        'Step 2: <b>2</b> operations on the 4 partial results.',
        `Step 3: <b>1</b> operation gives the result, <b>${lv[3][0].v}</b>. 7 operations in total (N − 1), but only log₂ 8 = <b>3</b> steps.`,
        `Same operations, different parentheses:\nsequential: ${seqP}\ntree:       ${treeP}\nAllowed because ${sym} is <b>associative</b>.`][k];
      syncButtons(root, 'mode', mode);
    };
    const st = stepper(root, () => 5, draw, 2000);
    modeButtons(root, 'mode', (m) => { mode = m; st.reset(); });
    return st;
  };

  // ------------------------------------------------------------ the order of float additions changes the result
  ANIM.forder = (root) => {
    let mode = 'seq';
    const svg = newSvg(root, '0 0 1000 320', 'Eight float values summed in four different orders; the rounded float32 results differ');
    const V = [1e8, -1e8, 1, 1, 1, 1, 1, 1].map(Math.fround);
    const fs = (v) => v === 1e8 ? '1e8' : v === -1e8 ? '−1e8' : String(v).replace('-', '−');
    const plan = (m) => {
      if (m === 'seq') return range(7).map(i => [{ d: 0, s: i + 1 }]);
      if (m === 'rev') return range(7).map(i => [{ d: 7, s: 6 - i }]);
      if (m === 'simple') return [1, 2, 4].map(s => range(8).filter(i => i % (2 * s) === 0).map(i => ({ d: i, s: i + s })));
      return [4, 2, 1].map(s => range(s).map(i => ({ d: i, s: i + s })));
    };
    const run = (m) => { const st = [V.slice()]; plan(m).forEach(ops => { const p = st[st.length - 1], a = p.slice(); ops.forEach(q => { a[q.d] = Math.fround(p[q.d] + p[q.s]); }); st.push(a); }); return st; };
    const NAMES = { seq: 'sequential', rev: 'reversed', simple: 'simple tree', conv: 'convergent tree' };
    const results = Object.keys(NAMES).map(m => { const st = run(m); return [m, st[st.length - 1][m === 'rev' ? 7 : 0]]; });
    const cap = root.querySelector('[data-cap]');
    const draw = (k) => {
      svg.innerHTML = '';
      const P = plan(mode), ST = run(mode), kk = Math.min(k, P.length), done = k > P.length;
      const a = ST[kk], ops = k >= 1 && k <= P.length ? P[k - 1] : [];
      const X = (i) => 100 + i * 100, Y = 40;
      const ah = arrowHead(svg, 'fo-a', BLUE);
      range(8).forEach(i => {
        T(svg, X(i) + 46, Y - 8, `[${i}]`, { ...mid, 'font-size': 13, class: 'lbl' });
        const w = ops.some(q => q.d === i), rd = ops.some(q => q.s === i);
        R(svg, X(i), Y, 92, 44, { fill: w ? BLUE : rd ? LGOLD : LBLUE, rx: 3 });
        T(svg, X(i) + 46, Y + 28, fs(a[i]), { ...mid, 'font-size': 15, fill: w ? '#fff' : INK, 'font-weight': 600 });
      });
      ops.forEach(q => {
        const x1 = X(q.s) + 46, x2 = X(q.d) + 46, dip = 30 + Math.abs(q.s - q.d) * 6;
        svg.appendChild(el('path', { d: `M${x1},${Y + 46} Q${(x1 + x2) / 2},${Y + 46 + dip} ${x2 + (x1 > x2 ? 6 : -6)},${Y + 48}`, fill: 'none', stroke: BLUE, 'stroke-width': 2, 'marker-end': ah }));
      });
      T(svg, 20, 172, k === 0 ? 'float32 values; exact sum = 6' : done ? `result: ${fs(ST[ST.length - 1][mode === 'rev' ? 7 : 0])}` : `step ${k} of ${P.length}:`, { 'font-size': 15, 'font-weight': 600 });
      ops.forEach((q, i) => {
        const x = ST[k - 1][q.d], y = ST[k - 1][q.s], ex = x + y, r = Math.fround(ex);
        T(svg, 150, 172 + i * 24, `${fs(x)} + ${fs(y)} = ${ex === r ? fs(r) : `${String(ex).replace('-', '−')} → rounds to ${fs(r)}`}`, { 'font-size': 14, fill: ex === r ? INK : RED });
      });
      L(svg, 20, 268, 980, 268, { stroke: '#e2e2e2' });
      results.forEach(([m, v], i) => {
        const on = m === mode && done;
        T(svg, 20 + i * 245, 296, `${NAMES[m]}: ${fs(v)}`, { 'font-size': 14, 'font-weight': on ? 700 : 400, fill: on ? BLUE : m === mode ? INK : GRAY });
      });
      const tip = 'Near 1e8, consecutive floats are 8 apart: adding 1 or 2 to ±1e8 is rounded away.';
      cap.innerHTML = k === 0 ? `Four orders for the same 8 numbers. Integers would give 6 in every order. ${tip}`
        : done ? ({ seq: 'Sequential: 1e8 − 1e8 = 0 first, then the six 1s are added exactly: <b>6</b>.', rev: 'Reversed: the 1s add up to 6, then 6 − 1e8 rounds to −99999992, and + 1e8 gives <b>8</b>.', simple: 'Simple-kernel tree (neighbours): the big values cancel in the first step, so the 1s survive: <b>6</b>.', conv: 'Convergent-kernel tree (pairs half an array apart): each 1 meets a ±1e8 and is rounded away: <b>0</b>.' }[mode] + '\nA <b>reordering</b> of the operands needs commutativity; floats are commutative but not associative, so every parallel order is a different rounding.')
          : tip;
      syncButtons(root, 'mode', mode);
    };
    const st = stepper(root, () => plan(mode).length + 2, draw, 1500);
    modeButtons(root, 'mode', (m) => { mode = m; st.reset(); });
    return st;
  };

  // ------------------------------------------------------------ work, span and parallelism
  ANIM.workspan = (root) => {
    const svg = newSvg(root, '0 0 500 270', 'Operations per step of a reduction tree: half of the remaining values each step');
    const q = (n) => root.querySelector(`[data-in=${n}]`), out = root.querySelector('[data-out]');
    const draw = () => {
      svg.innerHTML = '';
      const N = Math.max(2, Math.min(1 << 30, +q('N').value | 0)), P = Math.max(0, +q('P').value | 0);
      const ops = []; let m = N; while (m > 1) { ops.push(Math.floor(m / 2)); m = Math.ceil(m / 2); }
      const L0 = ops.length, mx = ops[0], X0 = 66, W = 414, H = 170, B = 222, bw = W / L0;
      const cpt = (v) => v >= 1e6 ? +(v / 1e6).toFixed(1) + ' M' : v >= 1e4 ? Math.round(v / 1e3) + ' k' : fmt(v);
      L(svg, X0, B, X0 + W, B, { stroke: INK }); L(svg, X0, B, X0, B - H - 6, { stroke: INK });
      T(svg, X0 - 6, B - H + 5, cpt(mx), { 'font-size': 13, 'text-anchor': 'end' }); T(svg, X0 - 6, B + 4, '0', { 'font-size': 13, 'text-anchor': 'end' });
      ops.forEach((o, i) => {
        const h = Math.max(1.5, H * o / mx), cut = P && o > P;
        R(svg, X0 + i * bw + 2, B - h, Math.max(1, bw - 4), h, { fill: cut ? GOLD : BLUE });
        if (L0 <= 12 || i === 0 || i === L0 - 1) { if (L0 <= 12) T(svg, X0 + i * bw + bw / 2, B + 18, String(i + 1), { ...mid, 'font-size': 13 }); }
        if (L0 <= 10) T(svg, X0 + i * bw + bw / 2, B - h - 6, cpt(o), { ...mid, 'font-size': 13 });
      });
      if (P && P < mx) { const y = B - H * P / mx; L(svg, X0, y, X0 + W, y, { stroke: RED, 'stroke-dasharray': '5 4', 'stroke-width': 1.5 }); T(svg, X0 + W, y - 6, `${fmt(P)} units`, { 'font-size': 13, 'text-anchor': 'end', fill: RED }); }
      T(svg, X0 + W, B + 40, L0 > 12 ? `${L0} steps →` : 'step →', { 'font-size': 13, 'text-anchor': 'end', class: 'lbl' });
      T(svg, X0 + W, 16, 'operations per step', { 'font-size': 13, class: 'lbl', 'text-anchor': 'end' });
      const steps = P ? sum(ops.map(o => Math.ceil(o / P))) : L0;
      out.innerHTML = `work: N − 1 = <b>${fmt(N - 1)}</b> operations (same as sequential)` +
        `<br>span: ⌈log₂ N⌉ = <b>${L0}</b> steps (sequential: ${fmt(N - 1)})` +
        `<br>peak parallelism: <b>${fmt(ops[0])}</b> in step 1` +
        `<br>average parallelism: ${fmt(N - 1)} / ${L0} = <b>${+((N - 1) / L0).toFixed(1)}</b>` +
        (P ? `<br>with ${fmt(P)} execution units: <b>${fmt(steps)}</b> steps (Σ ⌈ops / ${fmt(P)}⌉), speedup ${+((N - 1) / steps).toFixed(1)}×` : '<br>unlimited execution units');
      root.querySelectorAll('[data-p]').forEach(b => b.classList.toggle('on', b.dataset.p === [N, P].join(',')));
    };
    root.querySelectorAll('input').forEach(i => i.addEventListener('input', draw));
    root.querySelectorAll('[data-p]').forEach(b => b.addEventListener('click', () => { const v = b.dataset.p.split(','); q('N').value = v[0]; q('P').value = v[1]; draw(); }));
    return { enter: draw, leave() {} };
  };

  // ------------------------------------------------------------ a tournament is a max reduction
  ANIM.bracket = (root) => {
    let mode = 's4';
    const svg = newSvg(root, '0 0 1000 380', 'An eight-team knockout bracket: quarter finals, semi finals and final, scheduled on days with a limited number of stadiums');
    const TM = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], RT = [72, 85, 64, 91, 78, 69, 88, 74];
    const cap = root.querySelector('[data-cap]');
    const rounds = () => { const r = [range(8).map(i => ({ t: i, y: 40 + i * 40 }))]; for (let l = 1; l <= 3; l++) { const p = r[l - 1]; r.push(range(p.length / 2).map(j => { const a = p[2 * j], b = p[2 * j + 1]; return { t: RT[a.t] > RT[b.t] ? a.t : b.t, y: (a.y + b.y) / 2 }; })); } return r; };
    const day = (l, j) => mode === 's4' ? l : l === 1 ? (j < 3 ? 1 : 2) : l + 1;
    const days = () => mode === 's4' ? 3 : 4;
    const draw = (k) => {
      svg.innerHTML = '';
      const RD = rounds(), XC = [30, 290, 550, 810], bw = 150, bh = 30;
      ['teams', 'quarter-final winners', 'semi-final winners', 'champion'].forEach((s, i) => T(svg, XC[i], 22, s, { 'font-size': 13, class: 'lbl' }));
      RD.forEach((lvl, l) => lvl.forEach((nd, j) => {
        const played = l === 0 || day(l, j) <= k, now = l > 0 && day(l, j) === k;
        if (l) {
          const a = RD[l - 1][2 * j], b = RD[l - 1][2 * j + 1], xm = XC[l] - 40;
          [a, b].forEach(c => { L(svg, XC[l - 1] + bw, c.y + bh / 2, xm, c.y + bh / 2, { stroke: now ? GOLD : '#bdbdbd', 'stroke-width': now ? 2.5 : 1.5 }); });
          L(svg, xm, a.y + bh / 2, xm, b.y + bh / 2, { stroke: now ? GOLD : '#bdbdbd', 'stroke-width': now ? 2.5 : 1.5 });
          L(svg, xm, nd.y + bh / 2, XC[l], nd.y + bh / 2, { stroke: now ? GOLD : '#bdbdbd', 'stroke-width': now ? 2.5 : 1.5 });
          if (played) T(svg, xm - 6, nd.y + bh / 2 + 5, `day ${day(l, j)}`, { 'font-size': 13, 'text-anchor': 'end', fill: now ? GOLD : GRAY });
        }
        const lost = l < 3 && RD[l + 1] && k >= 1 && day(l + 1, Math.floor(j / 2)) <= k && RD[l + 1][Math.floor(j / 2)].t !== nd.t;
        R(svg, XC[l], nd.y, bw, bh, A({ fill: !played ? '#fff' : l === 3 ? GOLD : lost ? FAINT : now ? BLUE : LBLUE, stroke: played ? 'none' : '#d0d0d0', 'stroke-dasharray': played ? null : '4 3', rx: 3 }));
        if (played) T(svg, XC[l] + bw / 2, nd.y + 21, `team ${TM[nd.t]} (${RT[nd.t]})`, { ...mid, 'font-size': 14, fill: l === 3 || now ? '#fff' : lost ? GRAY : INK });
      }));
      const nD = days();
      const sched = mode === 's4' ? ['4 games in 4 stadiums', '2 games', '1 game'] : ['3 games in 3 stadiums', 'the 4th quarter final', '2 games', '1 game'];
      cap.innerHTML = k === 0 ? 'Eight teams; the rating in brackets decides every game. A knockout tournament is a <b>max reduction</b>: the operator returns the stronger team.'
        : k <= nD ? `Day ${k}: ${sched[k - 1]}. ` + (mode === 's4' ? ['All quarter finals are played in parallel: 4 venues, staff and audiences at once.', 'Half the venues are now idle.', 'One game, one venue: the champion is the maximum.'][k - 1] : ['Only 3 stadiums: one quarter final has to wait, as two of the real ones did when they shared a city.', 'The tree is the same; fewer resources stretch the first round over two days.', 'Semi finals.', 'Final.'][k - 1])
          : `Champion: team D. Span: <b>${nD}</b> days for ${8 - 1} games. ` + (mode === 's4' ? '1024 teams need 10 rounds if 512 games can run at once in round 1; 60 000 teams need 16.' : 'With too few resources the span grows: the same trade-off as execution units on a GPU.');
      syncButtons(root, 'mode', mode);
    };
    const st = stepper(root, () => days() + 2, draw, 1800);
    modeButtons(root, 'mode', (m) => { mode = m; st.reset(); });
    return st;
  };

  // ------------------------------------------------------------ simple vs convergent on 16 elements, with a warp view
  const V16 = [3, 1, 7, 0, 4, 1, 6, 3, 4, 7, 2, 3, 8, 5, 9, 6];
  ANIM.tree16 = (root) => traceAnim(root, {
    vals: V16, mode: 'simple', id: 't16', h: 400, x0: 96, cw: 38, ch: 30, y0: 38, rh: 80, fs: 14, warp: 4, px: 728,
    label: 'Sixteen elements reduced in place by eight threads; each row is the array after one loop iteration, and a warp panel counts active and idle lanes',
    caption: (mode, k, IT) => {
      const last = IT.length, it = IT[k - 1];
      if (k === 0) return mode === 'simple' ? 'Simple kernel: thread t <b>owns</b> input[2t] and is the only one that writes it (owner computes). 8 threads, 16 elements, toy warps of 4 lanes.'
        : 'Convergent kernel: thread t owns input[t]. The stride starts at blockDim.x = 8 and halves; only threads with t &lt; stride work.';
      if (k > last) return mode === 'simple' ? 'Thread 0 writes input[0] = 69 to *output. The active threads drift apart (0, 2, 4, 6 → 0, 4 → 0), so every warp keeps <b>idle lanes</b>: 15 adds used 28 lane slots.'
        : 'Same 15 adds, same result 69, but the active threads stay <b>packed at the front</b>: whole warps go idle and are not issued. 20 lane slots instead of 28.';
      const ts = it.ops.map(q => q.t);
      return `Iteration ${k}, stride ${it.s}: ${tlist(ts)} ${ts.length === 1 ? 'adds' : 'add'} input[i + ${it.s}] into input[i]` + (mode === 'simple' ? ` (threadIdx.x % ${it.s} == 0).` : ` (threadIdx.x &lt; ${it.s}).`) +
        '\n__syncthreads() ends the iteration: nobody reads a partial sum before it is written.';
    }
  });

  // ------------------------------------------------------------ execution resource efficiency, warp by warp
  ANIM.eff = (root) => {
    let kern = 'simple';
    const svg = newSvg(root, '0 0 500 320', 'A grid of warps by loop iterations, shaded by the number of active lanes in each warp');
    const q = (n) => root.querySelector(`[data-in=${n}]`), out = root.querySelector('[data-out]');
    const sim = (N, kk) => {
      const nt = N / 2, nw = Math.ceil(nt / 32), rows = [];
      const ss = kk === 'simple' ? range(lg(nt) + 1).map(i => 1 << i) : range(lg(nt) + 1).map(i => nt >> i);
      ss.forEach(s => { const a = Array(nw).fill(0); for (let t = 0; t < nt; t++) if (kk === 'simple' ? t % s === 0 : t < s) a[t >> 5]++; rows.push({ s, a }); });
      const issued = rows.map(r => r.a.filter(v => v).length);
      return { nt, nw, rows, issued, w: sum(issued), useful: sum(rows.map(r => sum(r.a))) };
    };
    const draw = () => {
      svg.innerHTML = '';
      const N = pow2(q('N').value, 4, 2048); q('N').value = N;
      const S = sim(N, kern), nr = S.rows.length;
      const cw = Math.min(46, 380 / S.nw), ch = Math.min(26, 236 / nr), X0 = 84, Y0 = 34;
      T(svg, X0, 16, `${kern === 'simple' ? 'simple' : 'convergent'} kernel, N = ${fmt(N)}: ${S.nt} threads, ${S.nw} warp${S.nw > 1 ? 's' : ''}`, { 'font-size': 13, class: 'lbl' });
      S.rows.forEach((r, i) => {
        const y = Y0 + i * ch;
        if (ch >= 14) T(svg, X0 - 6, y + ch / 2 + 5, `s = ${r.s}`, { 'font-size': 13, 'text-anchor': 'end' });
        r.a.forEach((v, w) => {
          R(svg, X0 + w * cw, y, cw - (cw > 6 ? 2 : 0.5), ch - 2, { fill: v ? (v === Math.min(32, S.nt) ? BLUE : '#7FA6E3') : '#fff', 'fill-opacity': v ? Math.max(.25, v / 32) : 1, stroke: v ? 'none' : '#e6e6e6' });
          if (v && cw >= 30 && ch >= 18) T(svg, X0 + w * cw + (cw - 2) / 2, y + ch / 2 + 4, String(v), { ...mid, 'font-size': 13, fill: v === Math.min(32, S.nt) ? '#fff' : INK });
        });
        T(svg, X0 + S.nw * cw + 8, y + ch / 2 + 5, ch >= 14 ? String(S.issued[i]) : '', { 'font-size': 13, fill: GOLD });
      });
      const yb = Y0 + nr * ch + 22;
      T(svg, X0, yb, 'cell = one warp in one iteration; number = active lanes', { 'font-size': 13, class: 'lbl' });
      T(svg, X0, yb + 20, 'white = warp not issued; gold = warps issued', { 'font-size': 13, class: 'lbl' });
      const row = (kk) => { const s = sim(N, kk); return `<tr class="${kk === kern ? 'hot' : ''}"><td>${kk === 'simple' ? 'simple' : 'convergent'}</td><td>${s.issued.join('+')} = ${s.w}</td><td>${fmt(s.w * 32)}</td><td>${fmt(s.useful)}</td><td>${Math.round(100 * s.useful / (s.w * 32))} %</td></tr>`; };
      out.innerHTML = `<table><tr><th>kernel</th><th>warps issued</th><th>× 32</th><th>adds</th><th>eff.</th></tr>${row('simple')}${row('convergent')}</table>` +
        `<p style="margin:.4em 0 0">An issued warp occupies all 32 lanes, however few are active. Efficiency = useful adds (N − 1 = ${fmt(N - 1)}) / lane slots consumed.</p>`;
      root.querySelectorAll('[data-p]').forEach(b => b.classList.toggle('on', +b.dataset.p === N));
      syncButtons(root, 'k', kern);
    };
    root.querySelectorAll('input').forEach(i => i.addEventListener('change', draw));
    root.querySelectorAll('[data-p]').forEach(b => b.addEventListener('click', () => { q('N').value = b.dataset.p; draw(); }));
    root.querySelectorAll('[data-k]').forEach(b => b.addEventListener('click', () => { kern = b.dataset.k; draw(); }));
    return { enter: draw, leave() {} };
  };

  // ------------------------------------------------------------ global memory requests, N = 256
  ANIM.memreq = (root) => {
    let mode = 'simple';
    const svg = newSvg(root, '0 0 1000 340', 'Four warps and the eight 128-byte segments of a 256-float array; ticks mark the addresses each warp touches in one iteration');
    const N = 256, NT = 128, SEGW = 32;
    const cap = root.querySelector('[data-cap]');
    const its = (m) => m === 'simple' ? range(8).map(i => 1 << i) : m === 'conv' ? range(8).map(i => 128 >> i) : [];
    const access = (m, s, w) => { // addresses of (load input[i], load input[i+s], store input[i]) for active threads of warp w
      const A = [], B = [];
      for (let t = w * 32; t < w * 32 + 32; t++) {
        if (m === 'simple' ? t % s === 0 : t < s) { const i = m === 'simple' ? 2 * t : t; A.push(i); B.push(i + s); }
      }
      return { A, B };
    };
    const segs = (xs) => new Set(xs.map(x => Math.floor(x / SEGW))).size;
    const steps = () => mode === 'shared' ? 4 : 10;
    const totals = { simple: 141, conv: 36, shared: 9 };
    const draw = (k) => {
      svg.innerHTML = '';
      const X0 = 120, sw = 80, Y0 = 52, rh = 54;
      range(8).forEach(j => T(svg, X0 + j * sw + 38, 36, `${j * 32}–${j * 32 + 31}`, { ...mid, 'font-size': 13, class: 'lbl' }));
      T(svg, X0, 16, 'input[ ] in 128-byte segments (32 floats)', { 'font-size': 13, class: 'lbl' });
      let run = 0, curReq = 0;
      const S = its(mode);
      const perIt = (s) => range(4).map(w => { const { A, B } = access(mode, s, w); return A.length ? segs(A) * 2 + segs(B) : 0; });
      if (mode !== 'shared') for (let i = 0; i < Math.min(k, S.length); i++) run += sum(perIt(S[i]));
      else { run = (k >= 1 ? 8 : 0) + (k >= 3 ? 1 : 0); }
      range(4).forEach(w => {
        const y = Y0 + w * rh;
        T(svg, 20, y + 26, `warp ${w}`, { 'font-size': 14 });
        let A = [], B = [], req = 0;
        if (mode !== 'shared' && k >= 1 && k <= S.length) { ({ A, B } = access(mode, S[k - 1], w)); req = A.length ? segs(A) * 2 + segs(B) : 0; }
        if (mode === 'shared' && k === 1) { A = range(32).map(l => w * 32 + l); B = A.map(x => x + NT); req = 2; }
        const touched = new Set([...A, ...B].map(x => Math.floor(x / SEGW)));
        range(8).forEach(j => R(svg, X0 + j * sw, y, sw - 4, 38, { fill: touched.has(j) ? LBLUE : FAINT, stroke: touched.has(j) ? BLUE : '#e2e2e2', 'stroke-width': touched.has(j) ? 1.5 : 1, rx: 2 }));
        const tick = (x, c, top) => { const j = Math.floor(x / SEGW), xx = X0 + j * sw + 4 + (x % SEGW) * ((sw - 12) / 31); L(svg, xx, top ? y + 3 : y + 20, xx, top ? y + 18 : y + 35, { stroke: c, 'stroke-width': 2 }); };
        A.forEach(x => tick(x, BLUE, true)); B.forEach(x => tick(x, GOLD, false));
        if (req) T(svg, 990, y + 26, `${req} request${req > 1 ? 's' : ''}`, { 'font-size': 14, 'text-anchor': 'end', fill: BLUE });
        else if (k >= 1 && (mode !== 'shared' ? k <= S.length : k === 1)) T(svg, 990, y + 26, 'idle', { 'font-size': 14, 'text-anchor': 'end', fill: GRAY });
        else if (mode !== 'shared' && k > S.length) T(svg, 990, y + 26, `${sum(S.map(s => perIt(s)[w]))} in all`, { 'font-size': 14, 'text-anchor': 'end', fill: GOLD });
        curReq += req;
      });
      const yb = Y0 + 4 * rh + 18;
      T(svg, X0, yb, 'blue tick: input[i] (load + store)   gold tick: input[i + stride] (load)', { 'font-size': 13, class: 'lbl' });
      T(svg, 20, yb + 30, `global memory requests so far: ${run}`, { 'font-size': 16, 'font-weight': 600 });
      const fin = mode === 'shared' ? k === 3 : k > S.length;
      if (fin) T(svg, 990, yb + 30, `simple 141 · convergent 36 · shared 9`, { 'font-size': 14, 'text-anchor': 'end', fill: GOLD });
      if (mode === 'shared') {
        cap.innerHTML = ['Shared-memory kernel: global memory is touched only by the first load and the final store.',
          'Each thread loads input[t] and input[t + 128]: per warp, 2 load instructions, each fully coalesced (one segment): 4 × 2 = <b>8</b> requests.',
          'The 7 iterations (stride 64 … 1) read and write input_s in <b>shared memory</b>: no global requests at all.',
          'Thread 0 stores *output: <b>1</b> request. Total N/32 + 1 = 8 + 1 = <b>9</b>, against 36 for the convergent kernel. The input array is not modified.'][k];
      } else if (k === 0) cap.innerHTML = `N = 256, 128 threads = 4 warps. Each iteration every active thread does 3 global accesses: load input[i], load input[i + stride], store input[i]. A warp instruction costs one request per <b>segment</b> it touches.`;
      else if (k <= S.length) {
        const s = S[k - 1], pr = perIt(s);
        cap.innerHTML = `Iteration ${k}, stride ${s}: ` + (mode === 'simple'
          ? (s <= 16 ? `active threads are ${2 * s} elements apart, so each instruction spans 2 segments: 2 + 2 + 2 = 6 requests per warp, using ${+(50 / s).toFixed(1)} % of the data fetched.` : `one active thread per active warp: 1 + 1 + 1 = 3 requests per active warp.`)
          : (s >= 32 ? `${s / 32} full warp${s > 32 ? 's' : ''}, consecutive addresses: 1 + 1 + 1 = 3 requests per active warp.` : `${s} active thread${s > 1 ? 's' : ''} in warp 0, all in one segment: 3 requests.`)) + ` This iteration: <b>${sum(pr)}</b>.`;
      } else cap.innerHTML = mode === 'simple' ? 'Total <b>141</b> = (4·5·2 + 4 + 2 + 1) · 3: five iterations where all 4 warps need 2 requests per instruction, then 4, 2 and 1 warps with one active thread, times 3 instructions. (Like the book, the final read of input[0] and the store to *output are not counted.)'
        : 'Total <b>36</b> = (4 + 2 + 1 + 5·1) · 3: every active warp touches one segment per instruction; the number of active warps is the same 12 that set the execution resources (384 = 12 · 32).';
      syncButtons(root, 'mode', mode);
    };
    const st = stepper(root, steps, draw, 1900);
    modeButtons(root, 'mode', (m) => { mode = m; st.reset(); });
    return st;
  };

  // ------------------------------------------------------------ shared memory: global vs shared accesses
  ANIM.shmem = (root) => {
    const svg = newSvg(root, '0 0 1000 370', 'Sixteen global input elements loaded and added pairwise into eight shared-memory slots, then reduced in shared memory');
    const cap = root.querySelector('[data-cap]');
    const draw = (k) => {
      svg.innerHTML = '';
      const aG = arrowHead(svg, 'sm-g', GREEN), aB = arrowHead(svg, 'sm-b', BLUE), aL = arrowHead(svg, 'sm-l', '#bdbdbd');
      const X = (j) => 100 + j * 38, cw = 34, ch = 30, YG = 40, Y = (r) => 118 + r * 64;
      T(svg, 100, 26, 'input[ ] in global memory (read only)', { 'font-size': 13, fill: BLUE });
      V16.forEach((v, j) => { const rd = k === 1; R(svg, X(j), YG, cw, ch, { fill: rd ? BLUE : LBLUE, rx: 2 }); T(svg, X(j) + cw / 2, YG + 20, String(v), { ...mid, 'font-size': 14, fill: rd ? '#fff' : INK }); });
      const rows = [range(8).map(t => V16[t] + V16[t + 8])];
      [4, 2, 1].forEach(s => { const p = rows[rows.length - 1], a = p.slice(); for (let t = 0; t < s; t++) a[t] = p[t] + p[t + s]; rows.push(a); });
      const lbl = ['t + 8 (load)', 'stride 4', 'stride 2', 'stride 1'];
      for (let r = 0; r <= Math.min(k - 1, 3); r++) {
        T(svg, 90, Y(r) + 20, lbl[r], { 'font-size': 13, 'text-anchor': 'end', class: 'lbl' });
        const act = r === 0 ? 8 : [4, 2, 1][r - 1], now = r === k - 1;
        for (let t = 0; t < 8; t++) {
          const w = t < act;
          R(svg, X(t), Y(r), cw, ch, { fill: w ? (now ? GREEN : '#7DB894') : FAINT, stroke: w ? 'none' : '#e2e2e2', rx: 2 });
          T(svg, X(t) + cw / 2, Y(r) + 20, String(rows[r][t]), { ...mid, 'font-size': 14, fill: w ? '#fff' : GRAY });
          if (w) {
            if (r === 0) { L(svg, X(t) + cw / 2, YG + ch + 2, X(t) + cw / 2, Y(0) - 3, { stroke: now ? BLUE : '#bdbdbd', 'stroke-width': now ? 1.6 : 1, 'marker-end': now ? aB : aL }); L(svg, X(t + 8) + cw / 2, YG + ch + 2, X(t) + cw / 2 + 3, Y(0) - 3, { stroke: now ? BLUE : '#bdbdbd', 'stroke-width': now ? 1.6 : 1, 'marker-end': now ? aB : aL }); }
            else { L(svg, X(t) + cw / 2, Y(r - 1) + ch + 1, X(t) + cw / 2, Y(r) - 3, { stroke: '#bdbdbd', 'stroke-width': 1.2, 'marker-end': aL }); L(svg, X(t + act) + cw / 2, Y(r - 1) + ch + 1, X(t) + cw / 2 + 3, Y(r) - 3, { stroke: now ? GREEN : '#bdbdbd', 'stroke-width': now ? 2 : 1.2, 'marker-end': now ? aG : aL }); }
          }
        }
      }
      if (k >= 1) T(svg, X(8) + 10, Y(0) + 20, '← input_s[ ] in shared memory', { 'font-size': 13, fill: GREEN });
      // counters
      const gr = k >= 1 ? 16 : 0, sw = (k >= 1 ? 8 : 0) + [0, 0, 4, 6, 7, 7][k], sr = [0, 0, 8, 12, 14, 15][k], gw = k >= 5 ? 1 : 0;
      const PX = 730;
      T(svg, PX, 120, 'this kernel', { 'font-size': 14, 'font-weight': 600 });
      [[`global reads   ${gr}`, BLUE], [`global writes  ${gw}`, BLUE], [`shared reads   ${sr}`, GREEN], [`shared writes  ${sw}`, GREEN]].forEach(([s, c], i) => T(svg, PX, 146 + i * 22, s, { 'font-size': 14, fill: c }));
      T(svg, PX, 252, 'convergent kernel', { 'font-size': 14, 'font-weight': 600 });
      T(svg, PX, 276, 'global: 31 reads, 16 writes', { 'font-size': 14, fill: GRAY });
      R(svg, PX, 300, 200, 36, { fill: k >= 5 ? BLUE : '#fff', stroke: BLUE, rx: 3 });
      T(svg, PX + 100, 324, k >= 5 ? '*output = 69' : '*output', { ...mid, 'font-size': 15, fill: k >= 5 ? '#fff' : BLUE });
      cap.innerHTML = ['16 elements, 8 threads (BLOCK_DIM = 8). Blue: global memory. Green: shared memory.',
        'input_s[t] = input[t] + input[t + BLOCK_DIM]: the first iteration happens while loading. 16 coalesced global reads, 8 shared writes.',
        'The loop starts at stride blockDim.x/2 = 4, with __syncthreads() at the top so that every input_s[t] is written before it is read.',
        'Stride 2: everything stays in shared memory, a few cycles away.',
        'Stride 1: input_s[0] holds the block sum.',
        'Thread 0 stores the result. Global accesses: 16 + 1 = <b>N + 1</b> (17 here) instead of 47, and the <b>input array is unchanged</b>.'][k];
    };
    return stepper(root, () => 6, draw, 2000);
  };

  // ------------------------------------------------------------ __shfl_down_sync and friends
  ANIM.shfl = (root) => {
    let mode = 'd8';
    const svg = newSvg(root, '0 0 1000 410', 'Lanes of a warp exchanging register values with shuffle instructions, step by step');
    const cap = root.querySelector('[data-cap]');
    const cfg = (m) => m === 'd32' ? { n: 32, ds: [16, 8, 4, 2, 1], w: 32, op: 'down' } : m === 'x8' ? { n: 8, ds: [4, 2, 1], w: 8, op: 'xor' } : m === 'w4' ? { n: 8, ds: [2, 1], w: 4, op: 'down' } : { n: 8, ds: [4, 2, 1], w: 8, op: 'down' };
    const srcOf = (c, i, d) => { if (c.op === 'xor') return i ^ d; const g = Math.floor(i / c.w) * c.w; return (i - g) + d < c.w ? i + d : i; };
    const useful = (c, i, d) => c.op === 'xor' || (i % c.w) < d;
    const draw = (k) => {
      svg.innerHTML = '';
      const c = cfg(mode), n = c.n, cw = 880 / n, X = (i) => 100 + i * cw, rh = n === 32 ? 54 : 74, Y = (r) => 46 + r * rh, bw = cw - (n === 32 ? 5 : 12), fs = n === 32 ? 13 : 16;
      const aB = arrowHead(svg, 'sh-b', BLUE), aL = arrowHead(svg, 'sh-l', '#c9c9c9');
      const v0 = n === 32 ? range(32).map(i => (i * 7 + 3) % 10) : [3, 1, 7, 0, 4, 1, 6, 3];
      const rows = [v0]; c.ds.forEach(d => { const p = rows[rows.length - 1]; rows.push(p.map((v, i) => v + p[srcOf(c, i, d)])); });
      T(svg, 90, 26, 'lane', { 'font-size': 13, 'text-anchor': 'end', class: 'lbl' });
      range(n).forEach(i => T(svg, X(i) + bw / 2, 26, String(i), { ...mid, 'font-size': 13, fill: c.w < n && i % c.w === 0 ? GOLD : GRAY }));
      if (c.w < n) range(n / c.w).forEach(g => { if (g) L(svg, X(g * c.w) - (cw - bw) / 2, 32, X(g * c.w) - (cw - bw) / 2, Y(Math.min(k, c.ds.length)) + 34, { stroke: GOLD, 'stroke-dasharray': '5 4', 'stroke-width': 1.5 }); });
      const last = c.ds.length;
      for (let r = 0; r <= Math.min(k, last); r++) {
        const d = r ? c.ds[r - 1] : 0;
        T(svg, 90, Y(r) + 21, r ? `${c.op === 'xor' ? 'mask' : 'delta'} ${d}` : 'start', { 'font-size': 13, 'text-anchor': 'end', class: 'lbl' });
        range(n).forEach(i => {
          const u = !r || useful(c, i, d), now = r === k;
          R(svg, X(i), Y(r), bw, 30, { fill: !r ? LBLUE : u ? (now ? BLUE : '#6F9BDD') : FAINT, stroke: !r || u ? 'none' : '#e2e2e2', rx: 2 });
          T(svg, X(i) + bw / 2, Y(r) + 20, String(rows[r][i]), { ...mid, 'font-size': fs, fill: r && u ? '#fff' : r ? '#a8a8a8' : INK });
          if (r && r === k) { const s = srcOf(c, i, d); if (s !== i) L(svg, X(s) + bw / 2, Y(r - 1) + 32, X(i) + bw / 2 + (s > i ? 2 : -2), Y(r) - 3, { stroke: u ? BLUE : '#c9c9c9', 'stroke-width': u ? 1.8 : 1, 'marker-end': u ? aB : aL }); }
        });
      }
      const call = (d) => c.op === 'xor' ? `v += __shfl_xor_sync(0xffffffff, v, ${d});` : `partialSum += __shfl_down_sync(0xffffffff, partialSum, ${d}${c.w < 32 && n === 8 && mode === 'w4' ? ', 4' : ''});`;
      const tot = sum(v0);
      cap.innerHTML = k === 0 ? ({ d8: 'A toy warp of 8 lanes (a real warp has 32: see the 32-lane mode). Each lane holds partialSum in a <b>register</b>.', d32: 'A full warp: 32 lanes, one register value each. warp_reduce needs log₂ 32 = 5 shuffles.', x8: '<code>__shfl_xor_sync</code>: lane i reads lane i XOR mask. A butterfly: in log₂ n steps <b>every</b> lane holds the total (an all-reduce).', w4: 'width = 4 splits the 8 lanes into 2 independent groups of 4; delta and the out-of-range test apply inside each group.' }[mode])
        : k <= last ? `<code>${call(c.ds[k - 1])}</code>\n` + (c.op === 'xor' ? `Lane i exchanges with lane i ^ ${c.ds[k - 1]}: every lane does useful work.` : `Lane i receives lane i + ${c.ds[k - 1]}'s value. Lanes with i + ${c.ds[k - 1]} beyond the ${c.w < n ? 'group' : 'warp'} get their <b>own</b> value back (grey): they compute garbage nobody reads. Only the first ${c.ds[k - 1]} lane${c.ds[k - 1] > 1 ? 's' : ''}${c.w < n ? ' of each group' : ''} matter.`)
          : (c.op === 'xor' ? `Every lane holds ${tot}. Three shuffles, no shared memory, no barrier.` : mode === 'w4' ? `Lane 0 holds ${sum(v0.slice(0, 4))} and lane 4 holds ${sum(v0.slice(4))}: one sum per group of 4.` : `Lane 0 holds the warp total, <b>${tot}</b>. ${last} shuffles replace ${last} rounds of shared-memory accesses and barriers. Only lane 0's result is meaningful.`);
      syncButtons(root, 'mode', mode);
    };
    const st = stepper(root, () => cfg(mode).ds.length + 2, draw, 2000);
    modeButtons(root, 'mode', (m) => { mode = m; st.reset(); });
    return st;
  };

  // ------------------------------------------------------------ shared kernel vs warp-level vs two-stage: barriers and shared accesses
  ANIM.stages = (root) => {
    const svg = newSvg(root, '0 0 500 300', 'Bars comparing barriers, shared-memory accesses and shuffles per block for three kernels');
    const q = (n) => root.querySelector(`[data-in=${n}]`), out = root.querySelector('[data-out]');
    const draw = () => {
      svg.innerHTML = '';
      const Tb = pow2(q('T').value, 64, 1024); q('T').value = Tb;
      const L2 = lg(Tb), nw = Tb / 32;
      const K = [
        { n: 'shared memory', b: L2, s: 4 * Tb - 2, sh: 0, c: '#8C8C8C' },
        { n: 'warp-level', b: L2 - 4, s: 4 * Tb - 64, sh: 5, c: BLUE },
        { n: 'two-stage', b: 1, s: 2 * nw, sh: 5 * (nw + 1), c: GREEN }];
      const groups = [['__syncthreads()', 'b'], ['shared-memory accesses', 's'], ['shuffle instructions (warp)', 'sh']];
      groups.forEach(([g, f], gi) => {
        const y0 = 18 + gi * 96, mx = Math.max(...K.map(k => k[f])) || 1;
        T(svg, 10, y0, g, { 'font-size': 13, class: 'lbl' });
        K.forEach((k, i) => {
          const y = y0 + 8 + i * 25, w = 300 * k[f] / mx;
          T(svg, 140, y + 16, k.n, { 'font-size': 13, 'text-anchor': 'end' });
          R(svg, 148, y, Math.max(2, w), 20, { fill: k.c, rx: 2 });
          T(svg, 154 + Math.max(2, w), y + 16, fmt(k[f]), { 'font-size': 13 });
        });
      });
      out.innerHTML = `<table><tr><th>per block</th><th>barriers</th><th>shared</th><th>useful shuffle lanes</th></tr>` +
        `<tr><td>shared mem.</td><td>${L2}</td><td>${fmt(4 * Tb - 2)}</td><td>—</td></tr>` +
        `<tr><td>warp-level</td><td>${L2 - 4}</td><td>${fmt(4 * Tb - 64)}</td><td>31/160, 1 warp</td></tr>` +
        `<tr class="hot"><td>two-stage</td><td>1</td><td>${2 * nw}</td><td>31/160, ${nw + 1} warps</td></tr></table>` +
        `<p style="margin:.4em 0 0">${Tb} threads, ${2 * Tb} elements. Shared accesses per thread. A warp_reduce has 5 shuffles × 32 lanes; only 16 + 8 + 4 + 2 + 1 = 31 results are used.</p>`;
      root.querySelectorAll('[data-p]').forEach(b => b.classList.toggle('on', +b.dataset.p === Tb));
    };
    root.querySelectorAll('input').forEach(i => i.addEventListener('change', draw));
    root.querySelectorAll('[data-p]').forEach(b => b.addEventListener('click', () => { q('T').value = b.dataset.p; draw(); }));
    return { enter: draw, leave() {} };
  };

  // ------------------------------------------------------------ multi-block reduction: segments and atomic adds in any order
  ANIM.multi = (root) => {
    let mode = 'atomic', order = [0, 1, 2, 3], runs = 0;
    const svg = newSvg(root, '0 0 1000 370', 'Four blocks each reduce one segment of the input, then add their partial sums to the output in an arbitrary order');
    const IN = range(32).map(i => (i * 5 + 2) % 9 + 1), PS = range(4).map(b => sum(IN.slice(b * 8, b * 8 + 8)));
    const cap = root.querySelector('[data-cap]');
    const shuffle = () => { order = [0, 1, 2, 3]; for (let i = 3; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; } runs++; };
    const steps = () => mode === 'atomic' ? 7 : 5;
    const draw = (k) => {
      if (k === 0) shuffle();
      svg.innerHTML = '';
      const ah = arrowHead(svg, 'mb-a', GOLD);
      const X = (i) => 84 + i * 26, BX = (b) => X(b * 8) + 104;
      T(svg, 84, 24, 'input (N = 32): segment b = 2 × blockDim.x elements, starting at 2·blockDim.x·blockIdx.x', { 'font-size': 13, class: 'lbl' });
      IN.forEach((v, i) => { R(svg, X(i), 34, 24, 28, { fill: [LBLUE, LGOLD, LGREEN, '#E4DAF2'][i >> 3], rx: 2 }); T(svg, X(i) + 12, 53, String(v), { ...mid, 'font-size': 13 }); });
      range(4).forEach(b => {
        L(svg, X(b * 8) + 2, 70, X(b * 8 + 8) - 4, 70, { stroke: INK, 'stroke-width': 1.5 });
        T(svg, BX(b), 90, `segment ${b} → block ${b}`, { ...mid, 'font-size': 13 });
        const done = k >= 1;
        R(svg, BX(b) - 80, 108, 160, 40, { fill: done ? BLUE : '#fff', stroke: BLUE, rx: 3 });
        T(svg, BX(b), 133, done ? `block ${b}: ${PS[b]}` : `block ${b}`, { ...mid, 'font-size': 14, fill: done ? '#fff' : BLUE });
      });
      const OX = 500, OY = 290;
      if (mode === 'atomic') {
        const arrived = Math.max(0, Math.min(4, k - 1));
        let acc = 0; order.slice(0, arrived).forEach(b => acc += PS[b]);
        range(4).forEach(b => {
          const pos = order.indexOf(b), on = pos < arrived, x1 = BX(b), y1 = 150, x2 = OX - 60 + b * 40, y2 = OY - 4;
          L(svg, x1, y1, x2, y2, A({ stroke: on ? GOLD : '#d0d0d0', 'stroke-width': on ? 2.5 : 1.2, 'stroke-dasharray': on ? null : '5 4', 'marker-end': on ? ah : null }));
          if (on) { const mx = (x1 + x2) / 2, my = (y1 + y2) / 2; svg.appendChild(el('circle', { cx: mx, cy: my, r: 14, fill: GOLD })); T(svg, mx, my + 5, String(pos + 1), { ...mid, 'font-size': 14, fill: '#fff', 'font-weight': 700 }); }
        });
        R(svg, OX - 110, OY, 220, 40, { fill: arrived === 4 ? GOLD : '#fff', stroke: GOLD, 'stroke-width': 2, rx: 3 });
        T(svg, OX, OY + 26, `*output = ${acc}`, { ...mid, 'font-size': 16, 'font-weight': 600, fill: arrived === 4 ? '#fff' : INK });
        T(svg, 20, 360, `run ${runs}: arrival order ${order.slice(0, arrived).map(b => 'block ' + b).join(', ') || '–'}`, { 'font-size': 14 });
      } else {
        const host = mode === 'host';
        range(4).forEach(b => { if (k >= 2) L(svg, BX(b), 150, 380 + b * 60, 200, { stroke: BLUE, 'stroke-width': 1.5 }); });
        T(svg, 344, 224, 'partial[ ]', { 'font-size': 13, 'text-anchor': 'end', fill: BLUE });
        range(4).forEach(b => { R(svg, 352 + b * 60, 202, 56, 32, { fill: k >= 2 ? LBLUE : '#fff', stroke: BLUE, rx: 2 }); T(svg, 380 + b * 60, 224, k >= 2 ? String(PS[b]) : '', { ...mid, 'font-size': 14 }); });
        if (k >= 3) { L(svg, 500, 236, 500, OY - 4, { stroke: GREEN, 'stroke-width': 2 }); T(svg, 512, 266, host ? 'cudaMemcpy, then a CPU loop' : 'second kernel: 1 block', { 'font-size': 14, fill: GREEN }); }
        R(svg, OX - 110, OY, 220, 40, { fill: k >= 3 ? GREEN : '#fff', stroke: GREEN, 'stroke-width': 2, rx: 3 });
        T(svg, OX, OY + 26, k >= 3 ? `*output = ${sum(PS)}` : '*output', { ...mid, 'font-size': 16, 'font-weight': 600, fill: k >= 3 ? '#fff' : INK });
      }
      const tot = sum(PS);
      cap.innerHTML = k === 0 ? (mode === 'atomic' ? '__syncthreads() only works inside a block, so blocks cannot wait for each other. Each block reduces its own <b>segment</b> independently (here 8 elements per block).' : 'Alternative without atomics: each block writes its partial sum to its own slot, partial[blockIdx.x].')
        : k === 1 ? 'Every block runs the two-stage reduction on its segment. Blocks finish in an order nobody controls.'
          : mode === 'atomic' ? (k <= 5 ? `Thread 0 of block ${order[k - 2]} adds ${PS[order[k - 2]]} with <code>atomic_ref&lt;float, thread_scope_device&gt;::fetch_add</code>. Atomics guarantee mutual exclusion, <b>not order</b>.` : `*output = ${tot} in every run, but the order changes from run to run (press Reset). For integers the sum is the same; for floats each order rounds differently, so the result can differ in the last bits. The operator must be associative <b>and</b> commutative.`)
            : k === 2 ? 'Each leader thread writes partial[blockIdx.x]: no conflicts, no atomics. The kernel ends; the kernel boundary is the grid-wide barrier.'
              : `${mode === 'host' ? 'The host copies partial[ ] back and adds it with a loop' : 'A second launch with one block reduces partial[ ]'}: <b>${tot}</b>, always in the same order, so float results are reproducible. Useful when atomics are unavailable or too slow.`;
      syncButtons(root, 'mode', mode);
    };
    const st = stepper(root, steps, draw, 1700);
    modeButtons(root, 'mode', (m) => { mode = m; st.reset(); });
    return st;
  };

  // ------------------------------------------------------------ thread coarsening: serialized blocks vs coarsened blocks
  ANIM.coarse = (root) => {
    const svg = newSvg(root, '0 0 500 320', 'Timelines of SMs running reduction blocks: fully utilized steps, underutilized tree steps and idle time, without and with coarsening');
    const q = (n) => root.querySelector(`[data-in=${n}]`), out = root.querySelector('[data-out]');
    const time = (B, C, S, L) => { const nb = Math.ceil(B / C); return { nb, per: 2 * C - 1 + L, full: 2 * C - 1, steps: Math.ceil(nb / S) * (2 * C - 1 + L) }; };
    const draw = () => {
      svg.innerHTML = '';
      const Tb = pow2(q('T').value, 2, 1024), B = Math.max(1, Math.min(100000, +q('B').value | 0)), C = Math.max(1, Math.min(64, +q('C').value | 0)), S = Math.max(1, Math.min(1000, +q('S').value | 0));
      q('T').value = Tb;
      const L = lg(Tb), U = time(B, 1, S, L), Cc = time(B, C, S, L), mxT = Math.max(U.steps, Cc.steps), rows = Math.min(S, 4);
      const cw = Math.min(40, 380 / mxT), X0 = 70;
      const panel = (y0, R0, title) => {
        T(svg, 10, y0, title, { 'font-size': 13, 'font-weight': 600 });
        for (let s = 0; s < rows; s++) {
          const y = y0 + 10 + s * 26;
          T(svg, X0 - 6, y + 16, `SM ${s}`, { 'font-size': 13, 'text-anchor': 'end', class: 'lbl' });
          const mine = Math.max(0, Math.ceil((R0.nb - s) / S));
          for (let t = 0; t < R0.steps; t++) {
            const blk = Math.floor(t / R0.per), ph = t % R0.per, busy = blk < mine;
            R(svg, X0 + t * cw, y, Math.max(.6, cw - (cw > 5 ? 1.5 : 0.3)), 20, { fill: !busy ? '#fff' : ph < R0.full ? BLUE : LBLUE, stroke: busy ? 'none' : '#e2e2e2' });
          }
          T(svg, X0 + R0.steps * cw + 6, y + 16, s === 0 ? `${R0.steps}` : '', { 'font-size': 13 });
        }
        if (S > rows) T(svg, X0, y0 + 10 + rows * 26 + 12, `… and ${S - rows} more SMs`, { 'font-size': 13, class: 'lbl' });
      };
      const h1 = 10 + rows * 26 + (S > rows ? 24 : 8);
      panel(18, U, `no coarsening: ${fmt(U.nb)} blocks`);
      panel(18 + h1 + 26, Cc, `COARSE_FACTOR ${C}: ${fmt(Cc.nb)} blocks`);
      const yl = 18 + 2 * (h1 + 26) + 4;
      svg.setAttribute('viewBox', `0 0 500 ${yl + 10}`);
      R(svg, X0, yl - 12, 14, 14, { fill: BLUE }); T(svg, X0 + 20, yl, 'all threads busy', { 'font-size': 13 });
      R(svg, X0 + 160, yl - 12, 14, 14, { fill: LBLUE }); T(svg, X0 + 180, yl, 'tree: idle threads, barriers', { 'font-size': 13 });
      const cands = range(64).map(i => i + 1).map(c => ({ c, ...time(B, c, S, L) }));
      const best = cands.reduce((a, b) => b.steps < a.steps ? b : a);
      const line = (R0, nm) => { const m = Math.ceil(R0.nb / S); return `${nm}: <b>${fmt(R0.steps)}</b> steps, ${fmt(m * R0.full)} with all threads busy, ${fmt(m * L)} underutilized`; };
      out.innerHTML = `${line(U, 'no coarsening')}<br>${line(Cc, `factor ${C}`)}` +
        `<br>${fmt(2 * Tb * B)} elements; per block: ${U.full} or ${Cc.full} busy steps, then ${L} tree steps; ${S} SM${S > 1 ? 's' : ''} (counts for SM 0)` +
        (Cc.nb < S ? `<div class="cfg-msg bad">Only ${fmt(Cc.nb)} blocks for ${S} SMs: ${S - Cc.nb} SMs stay idle. Too much coarsening.</div>` : '') +
        `<div class="cfg-msg ok">Best factor for this input: ${best.c} (${fmt(best.steps)} steps).</div>`;
      root.querySelectorAll('[data-p]').forEach(b => b.classList.toggle('on', b.dataset.p === [Tb, B, C, S].join(',')));
    };
    root.querySelectorAll('input').forEach(i => i.addEventListener('change', draw));
    root.querySelectorAll('[data-p]').forEach(b => b.addEventListener('click', () => { const v = b.dataset.p.split(','); ['T', 'B', 'C', 'S'].forEach((id, i) => q(id).value = v[i]); draw(); }));
    return { enter: draw, leave() {} };
  };

  // ------------------------------------------------------------ exercise 3: the reversed pattern (static picture)
  ANIM.ex3pic = (root) => {
    const svg = newSvg(root, '0 0 420 260', 'Sixteen cells; threads 0 to 7 own the last eight; each step adds from the left into the right end');
    const aB = arrowHead(svg, 'e3-b', '#555');
    const X = (j) => 10 + j * 25, Y = (r) => 40 + r * 48, cw = 22, ch = 20;
    range(8).forEach(t => T(svg, X(8 + t) + cw / 2, 26, 't' + t, { ...mid, 'font-size': 14 }));
    const its = traceIts(16, 'rev');
    for (let r = 0; r <= 4; r++) {
      const wr = r ? its[r - 1].ops.map(q => q.d) : range(16);
      range(16).forEach(j => R(svg, X(j), Y(r), cw, ch, { fill: wr.includes(j) ? LBLUE : '#fff', stroke: '#888', 'stroke-width': .8 }));
      if (r) its[r - 1].ops.forEach(q => L(svg, X(q.src) + cw / 2, Y(r - 1) + ch, X(q.d) + cw / 2, Y(r) - 2, { stroke: '#555', 'stroke-width': 1, 'marker-end': aB }));
    }
    return { enter() {}, leave() {} };
  };

  // ------------------------------------------------------------ exercise 6 traced
  ANIM.ex6 = (root) => traceAnim(root, {
    vals: [6, 2, 7, 4, 5, 8, 3, 1], mode: 'simple', id: 'e6', h: 342, x0: 190, cw: 84, ch: 38, y0: 42, rh: 84, fs: 17, warp: 0,
    label: 'The array 6 2 7 4 5 8 3 1 after each iteration of a reduction kernel', ms: 2000,
    caption: (mode, k, IT, RS) => {
      const nm = { simple: '(a) simple kernel', conv: '(b) convergent kernel', rev: 'exercise 3: reversed pattern' }[mode];
      if (k === 0) return `${nm}: 8 elements, 4 threads. Exact sum: 36.`;
      if (k > IT.length) { const r = RS[RS.length - 1]; return `${nm}: final array {${r.join(', ')}}. The sum ${mode === 'rev' ? 'is in input[7]' : 'is in input[0]'}: <b>36</b>. Blue cells were written in that iteration.`; }
      const it = IT[k - 1];
      return `Iteration ${k}, stride ${it.s}: ${it.ops.map(q => `input[${q.d}] += input[${q.src}]`).join(', ')}\n→ {${RS[k].join(', ')}}`;
    }
  });
