  // ------------------------------------------------------------ roofline
  ANIM.roofline = (root) => {
    let peak = 66.9, bw = 3.35;            // TFLOPS, TB/s
    const qf = root.querySelector('[data-in=flop]'), qb = root.querySelector('[data-in=bytes]'), cap = root.querySelector('[data-cap]'), legend = root.querySelector('[data-legend]');
    const pts = [
      { n: 'vector addition', I: 1 / 12, c: '#B03A3A' },
      { n: 'matmul, no tiling', I: 0.25, c: '#C98A1B' },
      { n: 'matmul, 16 × 16 tiles', I: 4, c: '#2F6FD0' },
      { n: 'matmul, 32 × 32 tiles', I: 8, c: '#2F6FD0' },
      { n: 'matmul ideal, N = 1024', I: 1024 / 6, c: '#2E7D4F' }
    ];
    const W = 700, H = 440, x0 = 70, x1 = 680, y0 = 16, y1 = 380, lx0 = -2, lx1 = 3, ly0 = -3, ly1 = 2;
    const X = (v) => x0 + (Math.log10(v) - lx0) / (lx1 - lx0) * (x1 - x0);
    const Y = (v) => y1 - (Math.log10(Math.max(v, 10 ** ly0)) - ly0) / (ly1 - ly0) * (y1 - y0);
    const fmtT = (t) => t >= 1 ? t.toFixed(1) + ' TFLOPS' : (t * 1000).toFixed(t * 1000 >= 10 ? 0 : 1) + ' GFLOPS';
    const fmtI = (I) => I < 1 ? I.toFixed(3) : I.toFixed(I >= 20 ? 0 : 1);
    let k = 0;
    function draw() {
      const svg = newSvg(root, `0 0 ${W} ${H}`, 'Roofline: attainable throughput against arithmetic intensity, with the memory-bandwidth slope and the compute roof');
      for (let e = lx0; e <= lx1; e++) { L(svg, X(10 ** e), y0, X(10 ** e), y1, { stroke: '#EEEEEE' }); T(svg, X(10 ** e), y1 + 20, String(10 ** e), { ...mid, 'font-size': 14, class: 'lbl' }); }
      for (let e = ly0; e <= ly1; e++) { L(svg, x0, Y(10 ** e), x1, Y(10 ** e), { stroke: '#EEEEEE' }); T(svg, x0 - 8, Y(10 ** e) + 5, String(10 ** e), { 'text-anchor': 'end', 'font-size': 14, class: 'lbl' }); }
      T(svg, (x0 + x1) / 2, H - 6, 'arithmetic intensity (FLOP per byte, log scale)', { ...mid, 'font-size': 14, class: 'lbl' });
      T(svg, 16, (y0 + y1) / 2, 'TFLOPS (log scale)', { ...mid, 'font-size': 14, class: 'lbl', transform: `rotate(-90 16 ${(y0 + y1) / 2})` });
      L(svg, x0, y0, x0, y1, { stroke: '#1A1A1A' }); L(svg, x0, y1, x1, y1, { stroke: '#1A1A1A' });
      const ridge = peak / bw, xs = Math.max(10 ** lx0, 10 ** ly0 / bw);
      svg.appendChild(el('polygon', { points: `${X(xs)},${y1} ${X(xs)},${Y(bw * xs)} ${X(ridge)},${Y(peak)} ${X(ridge)},${y1}`, fill: '#FBEDED' }));
      svg.appendChild(el('polygon', { points: `${X(ridge)},${y1} ${X(ridge)},${Y(peak)} ${X(10 ** lx1)},${Y(peak)} ${X(10 ** lx1)},${y1}`, fill: '#EAF4EC' }));
      svg.appendChild(el('polyline', { points: `${X(xs)},${Y(bw * xs)} ${X(ridge)},${Y(peak)} ${X(10 ** lx1)},${Y(peak)}`, fill: 'none', stroke: '#1A1A1A', 'stroke-width': 3.5 }));
      L(svg, X(ridge), Y(peak), X(ridge), y1, { stroke: '#6f6f6f', 'stroke-dasharray': '5 4' });
      T(svg, X(ridge) - 8, (Y(peak) + y1) / 2, `ridge ${ridge.toFixed(1)} FLOP/B`, { 'text-anchor': 'end', 'font-size': 14 });
      T(svg, X(10 ** lx1) - 6, Y(peak) - 10, `roof: ${fmtT(peak)}`, { 'text-anchor': 'end', 'font-size': 14 });
      T(svg, x0 + 10, y0 + 22, `slope = ${bw >= 1 ? bw + ' TB/s' : bw * 1000 + ' GB/s'} × intensity`, { 'font-size': 14 });
      T(svg, x0 + 10, y1 - 10, 'memory-bound', { 'font-size': 14, fill: '#B03A3A' });
      T(svg, X(10 ** lx1) - 6, y1 - 10, 'compute-bound', { 'text-anchor': 'end', 'font-size': 14, fill: '#2E7D4F' });
      const flop = Math.max(0, +qf.value || 0), bytes = Math.max(1, +qb.value || 1), uI = flop / bytes;
      const list = pts.slice(0, Math.min(k, pts.length)).concat(k > pts.length && uI > 0 ? [{ n: 'your kernel', I: uI, c: '#1A1A1A' }] : []);
      list.forEach((p, j) => {
        if (p.I < 10 ** lx0 || p.I > 10 ** lx1) return;
        const att = Math.min(peak, bw * p.I), cx = X(p.I), cy = Y(att), last = j === list.length - 1;
        svg.appendChild(el('circle', { cx, cy, r: last ? 13 : 11, fill: p.c, stroke: last ? '#1A1A1A' : '#fff', 'stroke-width': last ? 2.5 : 2 }));
        T(svg, cx, cy + 5, p.n === 'your kernel' ? 'Y' : String(j + 1), { ...mid, 'font-size': 13, fill: '#fff', 'font-weight': 700 });
      });
      legend.innerHTML = '<table><tr><th>#</th><th>kernel</th><th>FLOP/B</th><th>at most</th></tr>' +
        list.map((p, j) => { const a = Math.min(peak, bw * p.I); return `<tr${j === list.length - 1 ? ' style="font-weight:600"' : ''}><td style="color:${p.c}">${p.n === 'your kernel' ? 'Y' : j + 1}</td><td>${p.n}</td><td>${fmtI(p.I)}</td><td>${fmtT(a)} · ${p.I < ridge ? 'memory' : 'compute'}</td></tr>`; }).join('') + '</table>';
      const cur = list[list.length - 1];
      cap.innerHTML = k === 0 ? `Every kernel lies under two lines: the <b>slope</b> (bandwidth × intensity) and the <b>roof</b> (peak compute). They meet at the <b>ridge</b>: ${fmtT(peak)} / ${bw} TB/s = <b>${ridge.toFixed(1)} FLOP/B</b>.` :
        `<b>${cur.n}</b>: ${fmtI(cur.I)} FLOP/B, at most ${pct(Math.min(peak, bw * cur.I) / peak)} of the peak. ` + (cur.I < ridge ? 'Memory-bound: more intensity moves it up the slope.' : 'Compute-bound: only faster arithmetic helps.') +
        (cur.n === 'your kernel' ? ' A measured kernel sits <i>below</i> its point; the gap is what optimization can still win.' : '');
    }
    const st = stepper(root, () => pts.length + 2, (kk) => { k = kk; draw(); }, 2600);
    root.querySelectorAll('[data-dev]').forEach(b => b.addEventListener('click', () => {
      root.querySelectorAll('[data-dev]').forEach(x => x.classList.toggle('on', x === b));
      [peak, bw] = b.dataset.dev.split(',').map(Number); draw();
    }));
    [qf, qb].forEach(q => q.addEventListener('input', () => { st.leave(); k = pts.length + 1; draw(); }));
    return st;
  };

  // ------------------------------------------------------------ speed of light
  ANIM.sol = (root) => {
    const q = (n) => root.querySelector(`[data-in=${n}]`), out = root.querySelector('[data-out]');
    function update() {
      const n = Math.max(1, +q('n').value || 1), t = Math.max(0.001, +q('t').value || 1), bw = Math.max(0.001, +q('bw').value || 1);
      const bytes = 12 * n, rate = bytes / (t / 1000) / 1e12, frac = rate / bw, tmin = bytes / (bw * 1e12) * 1000;
      out.innerHTML = `data moved = 12 × ${fmt(n)} = <b>${(bytes / 1e9).toFixed(bytes >= 1e10 ? 0 : 2)} GB</b><br>` +
        `achieved bandwidth = ${(bytes / 1e9).toFixed(2)} GB / ${t} ms = <b>${rate.toFixed(2)} TB/s</b><br>` +
        `fraction of the peak = <b>${pct(frac)}</b>${frac > 1 ? ' (impossible: check the numbers)' : ''}<br>` +
        `fastest possible time = ${(bytes / 1e9).toFixed(2)} GB / ${bw} TB/s = <b>${tmin.toFixed(2)} ms</b>`;
      const svg = newSvg(root, '0 0 600 170', 'Achieved bandwidth compared with the peak');
      const w = 520, x = 60;
      T(svg, 0, 42, 'peak', { 'font-size': 14 }); R(svg, x, 22, w, 32, { fill: '#E6E6E6', rx: 3 }); T(svg, x + w - 6, 44, `${bw} TB/s`, { 'text-anchor': 'end', 'font-size': 13 });
      T(svg, 0, 102, 'kernel', { 'font-size': 14 }); R(svg, x, 82, Math.min(w, w * frac), 32, { fill: frac > 1 ? '#B03A3A' : '#2F6FD0', rx: 3 });
      T(svg, x + 8, 104, `${rate.toFixed(2)} TB/s · ${pct(frac)}`, { 'font-size': 13, fill: '#fff' });
      T(svg, x, 150, `room left for optimization: ${pct(Math.max(0, 1 - frac))}`, { 'font-size': 13, class: 'lbl' });
    }
    ['n', 't', 'bw'].forEach(n => q(n).addEventListener('input', update));
    return { enter: update, leave() {} };
  };

  // ------------------------------------------------------------ CUDA memory types
  ANIM.memtypes = (root) => {
    const svg = newSvg(root, '0 0 1000 420', 'CUDA memory model: registers per thread, shared memory per block, global and constant memory per grid, host access to global and constant memory');
    const groups = { reg: [], sh: [], gl: [], co: [], lo: [], host: [] };
    const add = (g, e) => { groups[g].push(e); return e; };
    R(svg, 6, 6, 150, 408, { class: 'frame', rx: 4 }); const hostBox = add('host', R(svg, 24, 160, 114, 90, { class: 'mem', rx: 3 }));
    T(svg, 81, 200, 'Host', { ...mid, 'font-size': 16, 'font-weight': 600 }); T(svg, 81, 222, '(CPU)', { ...mid, 'font-size': 13, class: 'lbl' });
    R(svg, 170, 6, 824, 408, { class: 'frame', rx: 4 }); T(svg, 186, 28, 'Device: the grid', { 'font-size': 15, 'font-weight': 600 });
    const threadPos = [];
    for (let b = 0; b < 2; b++) {
      const bx = 196 + b * 396;
      R(svg, bx, 40, 380, 200, { fill: 'none', stroke: '#6f6f6f', 'stroke-width': 1.5, rx: 4 });
      T(svg, bx + 12, 60, `Block (${b}, 0)`, { 'font-size': 14 });
      add('sh', R(svg, bx + 20, 70, 340, 36, { class: 'mem', rx: 3 })); T(svg, bx + 190, 93, 'Shared memory', { ...mid, 'font-size': 14 });
      for (let t = 0; t < 2; t++) {
        const tx = bx + 30 + t * 180;
        add('reg', R(svg, tx, 120, 140, 34, { class: 'mem', rx: 3 })); T(svg, tx + 70, 142, 'Registers', { ...mid, 'font-size': 13 });
        R(svg, tx, 178, 140, 40, { fill: '#C9DBF4', rx: 3 }); T(svg, tx + 70, 203, `Thread (${t}, 0)`, { ...mid, 'font-size': 13 });
        add('reg', L(svg, tx + 70, 154, tx + 70, 178, { class: 'wire' }));
        add('sh', L(svg, tx + 20, 106, tx + 20, 178, { class: 'wire' }));
        add('lo', R(svg, tx + 100, 222, 40, 12, { fill: 'none', stroke: 'none' }));
        threadPos.push(tx);
      }
    }
    const gl = add('gl', R(svg, 196, 280, 776, 50, { class: 'mem', rx: 3 })); T(svg, 584, 310, 'Global memory (DRAM)', { ...mid, 'font-size': 15 });
    const loc = add('lo', R(svg, 830, 286, 132, 38, { fill: '#fff', stroke: '#bdbdbd', 'stroke-dasharray': '4 3', rx: 3 })); T(svg, 896, 310, 'local memory', { ...mid, 'font-size': 12 });
    add('co', R(svg, 196, 350, 776, 50, { class: 'mem', rx: 3 })); T(svg, 584, 380, 'Constant memory (cached, 64 KB)', { ...mid, 'font-size': 15 });
    threadPos.forEach(tx => { add('gl', L(svg, tx + 110, 218, tx + 110, 280, { class: 'wire' })); add('co', L(svg, tx + 130, 218, tx + 130, 350, { class: 'wire', 'stroke-dasharray': '5 3' })); });
    add('host', L(svg, 138, 230, 196, 305, { class: 'wire' })); add('host', L(svg, 138, 240, 196, 375, { class: 'wire' }));
    const caps = [
      'Every memory has a <b>scope</b> (who can see it) and a <b>speed</b>. Step through them from fastest to slowest.',
      '<b>Registers</b>: private to one thread, read and written by it (R/W). The fastest storage; automatic scalar variables live here.',
      '<b>Shared memory</b>: one per block, on chip, R/W by all threads of that block. Block (0,0) cannot see block (1,0)\'s shared memory.',
      '<b>Global memory</b>: off-chip DRAM, R/W by every thread of every block, and by the host (cudaMemcpy). Large, slow, survives between kernels.',
      '<b>Constant memory</b>: written by the host, <b>read-only</b> for the device, cached so all threads can read it quickly. 64 KB in total.',
      '<b>Local memory</b>: per-thread, but stored in global memory (automatic arrays, spilled registers). Private like registers, slow like DRAM.'
    ];
    const key = [null, 'reg', 'sh', 'gl', 'co', 'lo'], cap = root.querySelector('[data-cap]');
    return stepper(root, () => caps.length, (k) => {
      Object.entries(groups).forEach(([g, list]) => list.forEach(e => e.classList.toggle('on', g === key[k] || (k === 3 && g === 'host') || (k === 4 && g === 'host'))));
      loc.setAttribute('stroke', k === 5 ? '#C98A1B' : '#bdbdbd'); loc.setAttribute('stroke-width', k === 5 ? 2.5 : 1);
      cap.innerHTML = caps[k];
    }, 2800);
  };

  // ------------------------------------------------------------ register vs memory operand
  ANIM.regs = (root) => {
    let mode = 'reg';
    const svg = newSvg(root, '0 0 1000 360', 'A processor executing fadd with operands in registers, or a load followed by fadd when an operand is in memory');
    const mem = R(svg, 220, 8, 560, 64, { class: 'mem', rx: 3 }); T(svg, 500, 34, 'Global memory (off chip)', { ...mid, 'font-size': 15 });
    const memVal = T(svg, 500, 58, 'address r4 + offset: 2.5', { ...mid, 'font-size': 13 });
    R(svg, 20, 92, 960, 260, { class: 'frame', rx: 5 }); T(svg, 500, 344, 'Processor', { ...mid, 'font-size': 14, 'font-weight': 600 });
    R(svg, 60, 112, 300, 120, { fill: '#F2F2F2', stroke: '#6f6f6f', rx: 3 }); T(svg, 210, 132, 'ALU', { ...mid, 'font-size': 15 });
    const aluT = T(svg, 210, 180, '', { ...mid, 'font-size': 16, 'font-weight': 600 });
    const rf = R(svg, 400, 112, 240, 120, { class: 'sp', rx: 3 }); T(svg, 520, 132, 'Register file', { ...mid, 'font-size': 14 });
    const regT = [0, 1, 2, 3].map(i => T(svg, 520, 156 + i * 20, '', { ...mid, 'font-size': 13 }));
    R(svg, 680, 112, 280, 120, { fill: '#fff', stroke: '#6f6f6f', rx: 3 }); T(svg, 820, 132, 'Control unit', { ...mid, 'font-size': 14 });
    const irT = T(svg, 820, 176, '', { ...mid, 'font-size': 15, 'font-weight': 600 });
    const cnt = T(svg, 820, 214, '', { ...mid, 'font-size': 13, class: 'lbl' });
    const wMem = L(svg, 520, 72, 520, 112, { class: 'wire' }), wAlu = L(svg, 400, 172, 360, 172, { class: 'wire blue' });
    const prog = {
      reg: [['fadd r1, r2, r3', 'decode: both operands are registers', 0], ['fadd r1, r2, r3', 'read r2 and r3, add', 0], ['fadd r1, r2, r3', 'write r1', 0]],
      mem: [['load r2, r4, offset', 'address = r4 + offset', 1], ['load r2, r4, offset', 'wait hundreds of cycles for DRAM', 1], ['fadd r1, r2, r3', 'now r2 holds the operand', 2], ['fadd r1, r2, r3', 'read r2 and r3, add, write r1', 2]]
    };
    const caps = {
      reg: ['<b>fadd r1, r2, r3</b>: the operands are named by register number, so the instruction itself carries them.',
        'The ALU reads r2 and r3 straight from the register file. No memory access at all.',
        '<b>One instruction</b>, no trip to memory. Register accesses also cost at least ten times less energy than global memory accesses.'],
      mem: ['If an operand is in global memory, it must first be brought into a register: <b>load r2, r4, offset</b> forms the address r4 + offset.',
        'The load waits for DRAM: hundreds of cycles, and it uses memory bandwidth that all threads compete for.',
        'Only then can <b>fadd r1, r2, r3</b> run. The processor fetches a limited number of instructions per cycle, so the extra load also costs issue slots.',
        '<b>Two instructions</b> and a slow, energy-hungry memory access. That is why data that is reused should be kept in registers or shared memory.']
    };
    const cap = root.querySelector('[data-cap]');
    const st = stepper(root, () => prog[mode].length, (k) => {
      const [ins, what, nIns] = prog[mode][k];
      irT.textContent = `IR: ${ins}`; aluT.textContent = what.startsWith('read') ? 'r2 + r3' : '';
      const loaded = mode === 'reg' || k >= 2;
      regT[0].textContent = `r1 = ${(mode === 'reg' ? k >= 2 : k >= 3) ? '3.5' : '?'}`;
      regT[1].textContent = `r2 = ${loaded ? '2.5' : '?'}`; regT[2].textContent = 'r3 = 1.0'; regT[3].textContent = mode === 'mem' ? 'r4 = base address' : '';
      wMem.classList.toggle('on', mode === 'mem' && (k === 0 || k === 1)); mem.classList.toggle('on', mode === 'mem' && k === 1);
      memVal.setAttribute('opacity', mode === 'mem' ? 1 : .3);
      wAlu.classList.toggle('on', what.startsWith('read')); rf.classList.toggle('on', what.startsWith('read') || what.startsWith('write'));
      cnt.textContent = mode === 'reg' ? 'instructions: 1 · memory accesses: 0' : `instructions: ${nIns} · memory accesses: 1`;
      cap.innerHTML = caps[mode][k];
    }, 2400);
    modeButtons(root, 'mode', (m) => { mode = m; st.reset(); });
    return st;
  };

  // ------------------------------------------------------------ shared vs registers
  ANIM.shared = (root) => {
    const svg = newSvg(root, '0 0 1000 342', 'Four processing units with private register files and one shared memory used by all threads of the block');
    R(svg, 6, 6, 988, 330, { class: 'frame', rx: 5 }); T(svg, 20, 30, 'One processing block of an SM', { 'font-size': 14, class: 'lbl' });
    const shm = R(svg, 30, 60, 180, 220, { class: 'mem', rx: 3 }); T(svg, 120, 88, 'Shared memory', { ...mid, 'font-size': 14 }); T(svg, 120, 108, '(per block)', { ...mid, 'font-size': 12, class: 'lbl' });
    const sv = T(svg, 120, 170, '', { ...mid, 'font-size': 18, 'font-weight': 600 });
    const units = [0, 1, 2, 3].map(i => {
      const x = 260 + i * 180;
      R(svg, x, 60, 160, 220, { fill: '#fff', stroke: '#6f6f6f', rx: 3 }); T(svg, x + 80, 84, `thread ${i}`, { ...mid, 'font-size': 14 });
      R(svg, x + 16, 100, 128, 40, { fill: '#F2F2F2', stroke: '#9a9a9a', rx: 2 }); T(svg, x + 80, 125, 'ALU', { ...mid, 'font-size': 13 });
      const rf = R(svg, x + 16, 156, 128, 104, { class: 'sp', rx: 3 }); T(svg, x + 80, 176, 'registers', { ...mid, 'font-size': 12 });
      const v = T(svg, x + 80, 218, 'r1 = ?', { ...mid, 'font-size': 15 });
      const w = svg.appendChild(el('polyline', { points: `120,280 120,${300 + i * 6} ${x + 80},${300 + i * 6} ${x + 80},280`, class: 'wire' }));
      return { rf, v, w };
    });
    const caps = [
      'Each thread has its own registers: thread 0 computes <b>r1 = 7</b>. The other threads cannot see that register.',
      'To share it, thread 0 <b>stores</b> r1 into shared memory: <code>tile[0] = r1;</code> (a store instruction, unlike a register operand).',
      'After a barrier, threads 1–3 <b>load</b> <code>tile[0]</code>: one value, written once, read by every thread of the block.',
      'Shared memory is on chip: far faster than global memory, but it needs load/store instructions, so it is slower than registers. It is a <b>scratchpad</b> the program manages itself. Since Hopper, blocks of a cluster can also reach each other\'s shared memory (<b>distributed shared memory</b>).'
    ];
    const cap = root.querySelector('[data-cap]');
    return stepper(root, () => 4, (k) => {
      units.forEach((u, i) => {
        u.v.textContent = i === 0 ? 'r1 = 7' : (k >= 2 ? 'r1 = 7' : 'r1 = ?');
        u.rf.classList.toggle('on', (i === 0 && k === 0) || (i > 0 && k === 2));
        u.v.setAttribute('fill', ((i === 0 && k === 0) || (i > 0 && k === 2)) ? '#fff' : '#1A1A1A');
        u.w.classList.toggle('on', (i === 0 && k === 1) || (i > 0 && k === 2));
      });
      sv.textContent = k >= 1 ? 'tile[0] = 7' : 'tile[0] = ?'; shm.classList.toggle('on', k >= 1 && k <= 2);
      cap.innerHTML = caps[k];
    }, 2600);
  };

  // ------------------------------------------------------------ copies calculator
  ANIM.copies = (root) => {
    const q = (n) => root.querySelector(`[data-in=${n}]`), out = root.querySelector('[data-out]');
    function update() {
      const b = Math.max(1, +q('b').value | 0), t = Math.max(1, Math.min(1024, +q('t').value | 0));
      out.innerHTML = `<table><tr><th>Variable</th><th>Copies</th></tr>` +
        `<tr><td>automatic scalar (register)</td><td><b>${fmt(b * t)}</b> = ${fmt(b)} × ${fmt(t)}</td></tr>` +
        `<tr><td>automatic array (local)</td><td><b>${fmt(b * t)}</b></td></tr>` +
        `<tr><td><code>__shared__</code></td><td><b>${fmt(b)}</b>, one per block</td></tr>` +
        `<tr><td><code>__device__</code>, <code>__constant__</code></td><td><b>1</b>, shared by all</td></tr></table>`;
    }
    ['b', 't'].forEach(n => q(n).addEventListener('input', update));
    return { enter: update, leave() {} };
  };

  // ------------------------------------------------------------ Figs 5.5 / 5.6: repeated accesses
  ANIM.access = (root) => {
    const svg = newSvg(root, '0 0 1000 420', 'Four threads of block (0,0) read rows of M and columns of N; every element is read twice');
    const S = 42, nX = 210, nY = 10, mX = 20, mY = 200, pX = 210, pY = 200;
    const grid = (x0, y0, nm) => { const c = []; for (let r = 0; r < 4; r++) for (let q = 0; q < 4; q++) {
      const rect = R(svg, x0 + q * S, y0 + r * S, S - 2, S - 2, { class: 'cell', rx: 2 });
      T(svg, x0 + q * S + S / 2 - 1, y0 + r * S + S / 2 + 4, `${nm}${r},${q}`, { ...mid, 'font-size': 12 });
      const badge = T(svg, x0 + q * S + S - 6, y0 + r * S + 12, '', { 'text-anchor': 'end', 'font-size': 10, fill: '#C98A1B', 'font-weight': 700 });
      c.push({ rect, badge, n: 0 }); } return c; };
    const N = grid(nX, nY, 'N'), M = grid(mX, mY, 'M'), P = grid(pX, pY, 'P');
    R(svg, pX - 2, pY - 2, 2 * S + 2, 2 * S + 2, { fill: 'none', stroke: '#1A1A1A', 'stroke-width': 3 });
    const tx0 = 410, colW = 120, rowH = 52;
    T(svg, tx0, 30, 'Access order of block (0, 0) →', { 'font-size': 14 });
    const thr = [[0, 0], [0, 1], [1, 0], [1, 1]];
    const cells = thr.map(([ty, tx], i) => {
      T(svg, tx0, 78 + i * rowH, `thread ${ty},${tx}`, { 'font-size': 13 });
      return [0, 1, 2, 3].map(k => { const r = R(svg, tx0 + 90 + k * colW, 56 + i * rowH, colW - 6, rowH - 8, { class: 'cell', rx: 2 }); const t = T(svg, tx0 + 90 + k * colW + (colW - 6) / 2, 82 + i * rowH, '', { ...mid, 'font-size': 13 }); return { r, t }; });
    });
    [0, 1, 2, 3].map(k => T(svg, tx0 + 90 + k * colW + (colW - 6) / 2, 50, `k = ${k}`, { ...mid, 'font-size': 12, class: 'lbl' }));
    const tot = T(svg, tx0, 290, '', { 'font-size': 14 });
    const tot2 = T(svg, tx0, 316, '', { 'font-size': 14 });
    const cap = root.querySelector('[data-cap]');
    return stepper(root, () => 6, (k) => {
      [...M, ...N].forEach(c => { c.n = 0; c.rect.setAttribute('class', 'cell'); c.badge.textContent = ''; });
      P.forEach((c, i) => c.rect.setAttribute('class', (Math.floor(i / 4) < 2 && i % 4 < 2) ? (k > 0 ? 'soft' : 'cell') : 'cell'));
      const upto = Math.min(k, 4);
      thr.forEach(([ty, tx], i) => cells[i].forEach((c, kk) => {
        const on = kk < upto; c.r.setAttribute('class', on ? (kk === upto - 1 && k <= 4 ? 'soft' : 'cell') : 'cell');
        c.t.textContent = on ? `M${ty},${kk} × N${kk},${tx}` : '';
        if (on) { M[ty * 4 + kk].n++; N[kk * 4 + tx].n++; }
      }));
      [...M, ...N].forEach(c => { if (c.n) { c.badge.textContent = '×' + c.n; c.rect.setAttribute('class', c.n > 1 ? 'soft2' : 'soft'); } });
      if (k >= 1 && k <= 4) thr.forEach(([ty, tx]) => { M[ty * 4 + k - 1].rect.setAttribute('class', 'hl'); N[(k - 1) * 4 + tx].rect.setAttribute('class', 'hl'); });
      const reads = 8 * upto, distinct = [...M, ...N].filter(c => c.n).length;
      tot.textContent = k ? `global reads so far: ${reads}` : ''; tot2.textContent = k ? `distinct elements: ${distinct}` : '';
      const msgs = ['Block (0, 0) has four threads; each computes one element of the outlined P tile with a dot product over k = 0…3.',
        'k = 0: thread (0,0) and thread (0,1) <b>both</b> read M0,0; threads (0,0) and (1,0) both read N0,0.',
        'k = 1: again every M and N element read here is read by two threads.',
        'k = 2: the four threads read 8 values, only 4 of them different.',
        'k = 3: done. Each thread made 4 M reads and 4 N reads.',
        '<b>32 global reads for 16 distinct values</b>: every element was read <b>twice</b> (orange ×2). If the threads loaded each value once and shared it, traffic would halve. With T × T tiles the saving is a factor of T.'];
      cap.innerHTML = msgs[k];
    }, 2000);
  };

  // ------------------------------------------------------------ traffic reduction calculator
  ANIM.reduce = (root) => {
    const q = (n) => root.querySelector(`[data-in=${n}]`), out = root.querySelector('[data-out]');
    function update() {
      const w = Math.max(1, +q('w').value | 0), t = Math.max(1, Math.min(64, +q('t').value | 0));
      const naive = 2 * w * w * w, phases = Math.ceil(w / t), tiled = 2 * phases * t * t * Math.ceil(w / t) ** 2;
      const exact = w % t === 0, I = 0.25 * t, att = Math.min(66.9, 3.35 * I);
      out.innerHTML = `global loads per thread: no tiling ${fmt(2 * w)}, tiled <b>${fmt(2 * phases)}</b> (${phases} phases × 2)<br>` +
        `all threads: ${fmt(naive)} → <b>${fmt(exact ? naive / t : tiled)}</b>${exact ? ` (÷ ${t})` : ' (Width not a multiple of the tile: padded tiles)'}<br>` +
        `intensity = 0.25 × ${t} = <b>${I} FLOP/B</b><br>` +
        `H100 limit = min(66.9, 3.35 × ${I}) = <b>${att.toFixed(1)} TFLOPS</b> (${pct(att / 66.9)})` +
        `<div class="cfg-msg ${I >= 20 ? 'ok' : 'bad'}">${I >= 20 ? 'Compute-bound on an H100.' : 'Still memory-bound on an H100 (threshold 20 FLOP/B).'}</div>`;
      const svg = newSvg(root, '0 0 600 260', 'Global loads with and without tiling');
      const x = 130, wmax = 440, ratio = exact ? 1 / t : tiled / naive;
      T(svg, 0, 50, 'no tiling', { 'font-size': 14 }); R(svg, x, 28, wmax, 36, { fill: '#C98A1B', rx: 3 });
      T(svg, x + 8, 52, fmt(naive) + ' loads', { 'font-size': 13, fill: '#fff' });
      T(svg, 0, 120, `tiles ${t} × ${t}`, { 'font-size': 14 }); R(svg, x, 98, Math.max(2, wmax * ratio), 36, { fill: '#2F6FD0', rx: 3 });
      T(svg, x + Math.max(2, wmax * ratio) + 8, 122, fmt(exact ? naive / t : tiled) + ' loads', { 'font-size': 13 });
      T(svg, 0, 190, 'shared memory per block', { 'font-size': 13, class: 'lbl' }); T(svg, x + 120, 190, `2 × ${t}² × 4 B = ${fmt(2 * t * t * 4)} B`, { 'font-size': 13 });
      T(svg, 0, 220, 'threads per block', { 'font-size': 13, class: 'lbl' }); T(svg, x + 120, 220, `${t}² = ${fmt(t * t)}${t * t > 1024 ? '  (over the 1024 limit!)' : ''}`, { 'font-size': 13, fill: t * t > 1024 ? '#B03A3A' : '#1A1A1A' });
    }
    ['w', 't'].forEach(n => q(n).addEventListener('input', update));
    root.querySelectorAll('[data-p]').forEach(b => b.addEventListener('click', () => { const [w, t] = b.dataset.p.split(','); q('w').value = w; q('t').value = t; update(); }));
    return { enter: update, leave() {} };
  };

  // ------------------------------------------------------------ Figs 5.7 / 5.8: phases
  ANIM.phases = (root) => {
    const Mv = [1, 2, 0, 1, 3, 1, 2, 0, 0, 1, 1, 2, 2, 0, 1, 3], Nv = [2, 1, 0, 1, 1, 0, 2, 1, 0, 3, 1, 2, 1, 2, 1, 0];
    let by = 0, bx = 0;
    const svg = newSvg(root, '0 0 1000 400', 'Block threads load a tile of M and of N into shared memory in each phase and accumulate Pvalue');
    const S = 36, nX = 200, nY = 6, mX = 20, mY = 160, pX = 200, pY = 160;
    const mk = (x0, y0, vals, nm) => { const out = []; for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
      const rect = R(svg, x0 + c * S, y0 + r * S, S - 2, S - 2, { class: 'cell', rx: 2 });
      const t = T(svg, x0 + c * S + S / 2 - 1, y0 + r * S + S / 2 + 5, vals ? String(vals[r * 4 + c]) : '', { ...mid, 'font-size': 14 });
      out.push({ rect, t }); }
      [2].forEach(g => { L(svg, x0 + g * S - 1, y0 - 2, x0 + g * S - 1, y0 + 4 * S, { stroke: '#1A1A1A', 'stroke-width': 2.5 }); L(svg, x0 - 2, y0 + g * S - 1, x0 + 4 * S, y0 + g * S - 1, { stroke: '#1A1A1A', 'stroke-width': 2.5 }); });
      T(svg, x0 - 6, y0 + 4 * S / 2, nm, { 'text-anchor': 'end', 'font-size': 15, 'font-weight': 700 });
      return out; };
    const N = mk(nX, nY, Nv, 'N'), M = mk(mX + 18, mY, Mv, 'M'), P = mk(pX, pY, null, 'P');
    const sx = 390;
    R(svg, sx - 8, 6, 170, 310, { fill: 'none', stroke: '#9a9a9a', 'stroke-dasharray': '5 4', rx: 4 });
    T(svg, sx + 77, 300, 'shared memory', { ...mid, 'font-size': 12, class: 'lbl' });
    const sh = (y0, nm) => { T(svg, sx, y0 - 6, nm, { 'font-size': 13, 'font-weight': 600 }); const o = []; for (let r = 0; r < 2; r++) for (let c = 0; c < 2; c++) { const rect = R(svg, sx + c * 52, y0 + r * 46, 48, 42, { class: 'cell', rx: 2 }); const t = T(svg, sx + c * 52 + 24, y0 + r * 46 + 27, '', { ...mid, 'font-size': 15 }); o.push({ rect, t }); } return o; };
    const Nds = sh(40, 'Nds'), Mds = sh(176, 'Mds');
    const barT = T(svg, sx + 77, 150, '', { ...mid, 'font-size': 13, fill: '#B03A3A', 'font-weight': 700 });
    const tX = 572, cw = [64, 124, 110, 124], hdr = ['thread', 'phase 0 loads', 'Pvalue', 'phase 1 loads'];
    let cx = tX; const colX = cw.map(w => { const x = cx; cx += w; return x; });
    hdr.forEach((h, i) => T(svg, colX[i] + 4, 24, h, { 'font-size': 12, class: 'lbl' }));
    const thr = [[0, 0], [0, 1], [1, 0], [1, 1]];
    const rows = thr.map(([ty, tx], i) => { const y = 34 + i * 70; T(svg, colX[0] + 4, y + 30, `(${ty}, ${tx})`, { 'font-size': 13 });
      const c = [1, 2, 3].map(j => { R(svg, colX[j], y, cw[j] - 4, 62, { class: 'cell', rx: 2 }); return [T(svg, colX[j] + 6, y + 20, '', { 'font-size': 11 }), T(svg, colX[j] + 6, y + 37, '', { 'font-size': 11 }), T(svg, colX[j] + 6, y + 54, '', { 'font-size': 11 })]; });
      return c; });
    const glob = T(svg, tX, 330, '', { 'font-size': 13 });
    const steps = ['start', 'load0', 'sync', 'comp0', 'sync2', 'load1', 'sync', 'comp1', 'store'];
    const cap = root.querySelector('[data-cap]');
    const st = stepper(root, () => steps.length, (k) => {
      const s = steps[k], ph = k >= 5 ? 1 : 0;
      const tileOnM = (r, c) => Math.floor(r / 2) === by && Math.floor(c / 2) === ph, tileOnN = (r, c) => Math.floor(r / 2) === ph && Math.floor(c / 2) === bx;
      const loading = s === 'load0' || s === 'load1';
      M.forEach((e, i) => e.rect.setAttribute('class', k > 0 && k < 8 && tileOnM(Math.floor(i / 4), i % 4) ? (loading ? 'hl' : 'soft') : 'cell'));
      N.forEach((e, i) => e.rect.setAttribute('class', k > 0 && k < 8 && tileOnN(Math.floor(i / 4), i % 4) ? (loading ? 'hl' : 'soft') : 'cell'));
      M.forEach((e, i) => e.t.setAttribute('fill', k > 0 && k < 8 && loading && tileOnM(Math.floor(i / 4), i % 4) ? '#fff' : '#1A1A1A'));
      N.forEach((e, i) => e.t.setAttribute('fill', k > 0 && k < 8 && loading && tileOnN(Math.floor(i / 4), i % 4) ? '#fff' : '#1A1A1A'));
      const pv = thr.map(([ty, tx]) => { const row = by * 2 + ty, col = bx * 2 + tx; let s0 = 0, s1 = 0;
        for (let kk = 0; kk < 2; kk++) s0 += Mv[row * 4 + kk] * Nv[kk * 4 + col];
        for (let kk = 2; kk < 4; kk++) s1 += Mv[row * 4 + kk] * Nv[kk * 4 + col]; return [s0, s0 + s1]; });
      const loadedPh = k >= 5 ? 1 : (k >= 1 ? 0 : -1);
      thr.forEach(([ty, tx], i) => {
        const row = by * 2 + ty, col = bx * 2 + tx, j = ty * 2 + tx;
        Mds[j].t.textContent = loadedPh >= 0 ? String(Mv[row * 4 + loadedPh * 2 + tx]) : '';
        Nds[j].t.textContent = loadedPh >= 0 ? String(Nv[(loadedPh * 2 + ty) * 4 + col]) : '';
        const p = P[row * 4 + col]; p.rect.setAttribute('class', k === 0 ? 'soft' : (k === 8 ? 'hl' : 'soft'));
        p.t.textContent = k >= 7 ? String(pv[i][1]) : k >= 3 ? String(pv[i][0]) : '0'; p.t.setAttribute('fill', k === 8 ? '#fff' : '#1A1A1A');
        const r = rows[i];
        r[0][0].textContent = k >= 1 ? `M${row},${tx} → Mds${ty},${tx}` : ''; r[0][1].textContent = k >= 1 ? `N${ty},${col} → Nds${ty},${tx}` : ''; r[0][2].textContent = '';
        r[1][0].textContent = k >= 3 ? `after ph 0: ${pv[i][0]}` : ''; r[1][1].textContent = k >= 7 ? `after ph 1: ${pv[i][1]}` : ''; r[1][2].textContent = k >= 8 ? `→ P${row},${col}` : '';
        r[2][0].textContent = k >= 5 ? `M${row},${2 + tx} → Mds${ty},${tx}` : ''; r[2][1].textContent = k >= 5 ? `N${2 + ty},${col} → Nds${ty},${tx}` : ''; r[2][2].textContent = '';
      });
      for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) if (!(Math.floor(r / 2) === by && Math.floor(c / 2) === bx)) { P[r * 4 + c].t.textContent = ''; P[r * 4 + c].rect.setAttribute('class', 'cell'); }
      [...Mds, ...Nds].forEach(e => e.rect.setAttribute('class', loading ? 'soft2' : (k >= 1 ? 'soft' : 'cell')));
      barT.textContent = s === 'sync' || s === 'sync2' ? '__syncthreads()' : '';
      const loads = k >= 5 ? 16 : k >= 1 ? 8 : 0;
      glob.textContent = `global loads by this block: ${loads}${k === 8 ? ' (no tiling: 32)' : ''}`;
      const t0 = thr[0], r0 = by * 2, c0 = bx * 2;
      const msgs = {
        start: `Block (${by}, ${bx}) computes the P tile at rows ${r0}–${r0 + 1}, columns ${c0}–${c0 + 1}. Every thread starts with Pvalue = 0 in a register.`,
        load0: `<b>Phase 0, load</b>: each of the 4 threads copies <b>one</b> M element and <b>one</b> N element into shared memory (Mds, Nds). 8 global loads in total.`,
        sync: `<b>__syncthreads()</b>: nobody may read Mds/Nds until all four threads have written their element.`,
        comp0: `<b>Phase 0, compute</b>: Pvalue += Mds[ty][0]·Nds[0][tx] + Mds[ty][1]·Nds[1][tx]. Each shared value is used by <b>two</b> threads.\nThread (0,0): ${Mv[r0 * 4]}·${Nv[c0]} + ${Mv[r0 * 4 + 1]}·${Nv[4 + c0]} = ${pv[0][0]}.`,
        sync2: `<b>__syncthreads()</b> again: everyone must finish reading the tiles before they are overwritten by phase 1.`,
        load1: `<b>Phase 1, load</b>: the same Mds and Nds are reused for the next tiles (columns 2–3 of M, rows 2–3 of N).`,
        comp1: `<b>Phase 1, compute</b>: Pvalue now holds the full dot product. Thread (0,0): ${pv[0][0]} + ${Mv[r0 * 4 + 2]}·${Nv[8 + c0]} + ${Mv[r0 * 4 + 3]}·${Nv[12 + c0]} = ${pv[0][1]}.`,
        store: `<b>Store</b>: each thread writes its Pvalue to P. The block made <b>16</b> global loads instead of 32: each element was loaded once and used twice.`
      };
      cap.innerHTML = msgs[s];
    }, 2400);
    root.querySelectorAll('[data-blk]').forEach(b => b.addEventListener('click', () => {
      root.querySelectorAll('[data-blk]').forEach(x => x.classList.toggle('on', x === b)); [by, bx] = b.dataset.blk.split(',').map(Number); st.reset();
    }));
    return st;
  };

  // ------------------------------------------------------------ Fig 5.10: index arithmetic
  ANIM.index = (root) => {
    const Wd = 8; let Tw = 2;
    const q = (n) => root.querySelector(`[data-in=${n}]`), cap = root.querySelector('[data-cap]');
    const clamp = () => { const g = Wd / Tw; ['bx', 'by'].forEach(n => { q(n).max = g - 1; if (+q(n).value > g - 1) q(n).value = g - 1; if (+q(n).value < 0) q(n).value = 0; }); ['tx', 'ty'].forEach(n => { q(n).max = Tw - 1; if (+q(n).value > Tw - 1) q(n).value = Tw - 1; if (+q(n).value < 0) q(n).value = 0; }); };
    function draw(ph) {
      clamp();
      const bx = +q('bx').value | 0, by = +q('by').value | 0, tx = +q('tx').value | 0, ty = +q('ty').value | 0;
      const Row = by * Tw + ty, Col = bx * Tw + tx, mc = ph * Tw + tx, nr = ph * Tw + ty;
      const svg = newSvg(root, '0 0 600 430', 'M, N and P with the tiles of the current phase and the elements loaded by one thread');
      const S = 22, nX = 300, nY = 8, mX = 60, mY = 220, pX = 300, pY = 220;
      const g = (x0, y0, nm, cls) => { for (let r = 0; r < Wd; r++) for (let c = 0; c < Wd; c++) R(svg, x0 + c * S, y0 + r * S, S - 1, S - 1, { class: cls(r, c), rx: 1 }); T(svg, x0 - 6, y0 + 14, nm, { 'text-anchor': 'end', 'font-size': 14, 'font-weight': 700 }); };
      const inTile = (r, c, tr, tc) => Math.floor(r / Tw) === tr && Math.floor(c / Tw) === tc;
      g(mX, mY, 'M', (r, c) => r === Row && c === mc ? 'hl' : inTile(r, c, by, ph) ? 'soft2' : r === Row ? 'soft' : 'cell');
      g(nX, nY, 'N', (r, c) => r === nr && c === Col ? 'hl' : inTile(r, c, ph, bx) ? 'soft2' : c === Col ? 'soft' : 'cell');
      g(pX, pY, 'P', (r, c) => r === Row && c === Col ? 'hl' : inTile(r, c, by, bx) ? 'soft' : 'cell');
      L(svg, mX, mY - 14, mX + (ph * Tw) * S, mY - 14, { stroke: '#1A1A1A', 'stroke-width': 2 });
      T(svg, mX, mY - 20, `ph·TILE_WIDTH = ${ph * Tw}`, { 'font-size': 12 });
      L(svg, nX + Wd * S + 10, nY, nX + Wd * S + 10, nY + ph * Tw * S, { stroke: '#1A1A1A', 'stroke-width': 2 });
      T(svg, nX + Wd * S + 16, nY + 12, `${ph * Tw}`, { 'font-size': 12 });
      T(svg, pX, pY + Wd * S + 22, `P[${Row}][${Col}]`, { 'font-size': 13, fill: '#2F6FD0', 'font-weight': 700 });
      cap.innerHTML = `Row = by·T + ty = ${by}·${Tw} + ${ty} = <b>${Row}</b>    Col = bx·T + tx = ${bx}·${Tw} + ${tx} = <b>${Col}</b>\n` +
        `phase ph = <b>${ph}</b> of ${Wd / Tw}:\n` +
        `  M[Row·Width + ph·T + tx] = M[${Row}·8 + ${ph * Tw} + ${tx}] = M[${Row * Wd + mc}] → <i>M[${Row}][${mc}]</i>\n` +
        `  N[(ph·T + ty)·Width + Col] = N[(${ph * Tw} + ${ty})·8 + ${Col}] = N[${nr * Wd + Col}] → <i>N[${nr}][${Col}]</i>\n` +
        `light orange: the tiles this block loads in this phase · blue: what this thread loads and computes`;
    }
    const st = stepper(root, () => Wd / Tw, (k) => draw(k), 1800);
    ['bx', 'by', 'tx', 'ty'].forEach(n => q(n).addEventListener('input', () => draw(st.k)));
    root.querySelectorAll('[data-tw]').forEach(b => b.addEventListener('click', () => {
      root.querySelectorAll('[data-tw]').forEach(x => x.classList.toggle('on', x === b)); Tw = +b.dataset.tw; st.reset();
    }));
    return st;
  };

  // ------------------------------------------------------------ why both barriers
  ANIM.hazards = (root) => {
    let mode = 'both';
    // shared tile of two slots; thread 0 fills slot 0, thread 1 (slower) fills slot 1; both need both slots
    const ev = {
      both: [
        ['phase 0: load Mds[0] = 3', 'phase 0: load Mds[1] … (slow)', [3, '?']],
        ['waits at line 18', 'load done: Mds[1] = 5', [3, 5]],
        ['compute: 3·a + 5·b ✓', 'compute: 3·c + 5·d (slow)', [3, 5]],
        ['waits at line 22', 'still computing, reads Mds[0] = 3 ✓', [3, 5]],
        ['phase 1: load Mds[0] = 8', 'phase 1: load Mds[1] = 2', [8, 2]]
      ],
      no18: [
        ['phase 0: load Mds[0] = 3', 'phase 0: load Mds[1] … (slow)', [3, '?']],
        ['compute at once: reads Mds[1] = ? ✗', 'load done: Mds[1] = 5', [3, 5]],
        ['result uses a stale or garbage value', 'compute: 3·c + 5·d', [3, 5]]
      ],
      no22: [
        ['phase 0: load Mds[0] = 3', 'phase 0: load Mds[1] … (slow)', [3, '?']],
        ['waits at line 18', 'load done: Mds[1] = 5', [3, 5]],
        ['compute: 3·a + 5·b ✓ (fast)', 'compute starts (slow)', [3, 5]],
        ['phase 1: overwrites Mds[0] = 8', 'still computing phase 0', [8, 5]],
        ['…', 'reads Mds[0] = 8 instead of 3 ✗', [8, 5]]
      ]
    };
    const caps = {
      both: 'With both barriers each phase is safe: line 18 makes the fast thread wait for the slow load (<b>read-after-write</b>); line 22 makes it wait before overwriting the tile (<b>write-after-read</b>).',
      no18: 'Without line 18 the fast thread reads Mds[1] before thread 1 has written it: a broken <b>read-after-write</b> (true) dependence.',
      no22: 'Without line 22 the fast thread starts phase 1 and overwrites Mds[0] while thread 1 still needs the phase 0 value: a broken <b>write-after-read</b> (false) dependence.'
    };
    const host = root.querySelector('[data-svg]'), cap = root.querySelector('[data-cap]');
    const st = stepper(root, () => ev[mode].length + 1, (k) => {
      const rows = ev[mode].slice(0, k);
      host.innerHTML = '<table style="font-size:.62em"><tr><th>time</th><th>thread 0 (fast)</th><th>thread 1 (slow)</th><th>Mds[0]</th><th>Mds[1]</th></tr>' +
        rows.map((r, i) => `<tr><td>${i + 1}</td>${[r[0], r[1]].map(x => `<td class="${x.includes('✗') ? 'bad' : x.includes('✓') ? 'ok' : ''}">${x}</td>`).join('')}<td>${r[2][0]}</td><td>${r[2][1]}</td></tr>`).join('') +
        '</table>';
      cap.innerHTML = k === 0 ? 'Two threads share a tile of two slots. Each loads one slot and needs both. Thread 1 is slower.' : (k === ev[mode].length ? caps[mode] : 'Step through the timeline.');
    }, 1800);
    modeButtons(root, 'mode', (m) => { mode = m; st.reset(); });
    return st;
  };

  // ------------------------------------------------------------ Figs 5.11 / 5.12: boundary checks
  ANIM.boundary = (root) => {
    let Wd = 3; const Tw = 2;
    const cap = root.querySelector('[data-cap]');
    const combos = () => { const g = Math.ceil(Wd / Tw), out = []; for (let by = 0; by < g; by++) for (let bx = 0; bx < g; bx++) for (let ph = 0; ph < g; ph++) out.push([by, bx, ph]); return out; };
    function draw(k) {
      const g = Math.ceil(Wd / Tw), n = g * Tw, [by, bx, ph] = combos()[k];
      const S = 30, svg = newSvg(root, '0 0 1000 400', 'Matrices padded to whole tiles; threads whose element does not exist load zero');
      const mX = 40, mY = 200, nX = 300, nY = 8, pX = 300, pY = 200;
      const draws = (x0, y0, nm, kind) => {
        for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
          const real = r < Wd && c < Wd;
          let cls = real ? 'cell' : 'faint';
          if (kind === 'M' && Math.floor(r / Tw) === by && Math.floor(c / Tw) === ph) cls = real ? 'soft' : 'idle';
          if (kind === 'N' && Math.floor(r / Tw) === ph && Math.floor(c / Tw) === bx) cls = real ? 'soft' : 'idle';
          if (kind === 'P' && Math.floor(r / Tw) === by && Math.floor(c / Tw) === bx) cls = real ? 'soft' : 'idle';
          R(svg, x0 + c * S, y0 + r * S, S - 2, S - 2, { class: cls, rx: 2 });
          if (real) T(svg, x0 + c * S + S / 2 - 1, y0 + r * S + S / 2 + 4, `${r},${c}`, { ...mid, 'font-size': 10 });
        }
        R(svg, x0 - 1, y0 - 1, Wd * S, Wd * S, { fill: 'none', stroke: '#1A1A1A', 'stroke-width': 2 });
        T(svg, x0 - 6, y0 + 14, nm, { 'text-anchor': 'end', 'font-size': 15, 'font-weight': 700 });
      };
      draws(mX, mY, 'M', 'M'); draws(nX, nY, 'N', 'N'); draws(pX, pY, 'P', 'P');
      const sx = 560;
      T(svg, sx, 30, `block (${by}, ${bx}), phase ${ph}`, { 'font-size': 16, 'font-weight': 600 });
      const lines = [], thr = [[0, 0], [0, 1], [1, 0], [1, 1]];
      thr.forEach(([ty, tx], i) => {
        const Row = by * Tw + ty, Col = bx * Tw + tx, mc = ph * Tw + tx, nr = ph * Tw + ty;
        const mOk = Row < Wd && mc < Wd, nOk = nr < Wd && Col < Wd, pOk = Row < Wd && Col < Wd;
        const y = 64 + i * 78;
        T(svg, sx, y, `thread (${ty}, ${tx})`, { 'font-size': 14, 'font-weight': 600 });
        T(svg, sx, y + 20, mOk ? `Mds ← M[${Row}][${mc}]` : `Mds ← 0.0f  (M[${Row}][${mc}] does not exist)`, { 'font-size': 13, fill: mOk ? '#1A1A1A' : '#B03A3A' });
        T(svg, sx, y + 38, nOk ? `Nds ← N[${nr}][${Col}]` : `Nds ← 0.0f  (N[${nr}][${Col}] does not exist)`, { 'font-size': 13, fill: nOk ? '#1A1A1A' : '#B03A3A' });
        T(svg, sx, y + 56, pOk ? `computes P[${Row}][${Col}]` : 'computes no P element (still loads, still syncs)', { 'font-size': 12, class: pOk ? '' : 'lbl' });
        if (!mOk) { R(svg, mX + mc * S + 4, mY + Row * S + 4, S - 10, S - 10, { fill: 'none', stroke: '#B03A3A', 'stroke-width': 2.5 }); lines.push(`M by thread (${ty},${tx})`); }
        else R(svg, mX + mc * S + 2, mY + Row * S + 2, S - 6, S - 6, { fill: 'none', stroke: '#2E7D4F', 'stroke-width': 2.5 });
        if (!nOk) { R(svg, nX + Col * S + 4, nY + nr * S + 4, S - 10, S - 10, { fill: 'none', stroke: '#B03A3A', 'stroke-width': 2.5 }); lines.push(`N by thread (${ty},${tx})`); }
        else R(svg, nX + Col * S + 2, nY + nr * S + 2, S - 6, S - 6, { fill: 'none', stroke: '#2E7D4F', 'stroke-width': 2.5 });
      });
      const wrongM = thr.map(([ty, tx]) => { const Row = by * Tw + ty, mc = ph * Tw + tx; return Row < Wd && mc >= Wd ? `M[${Row}][${mc}] would read M[${Row + 1}][${mc - Wd}] (next row)` : null; }).filter(Boolean);
      cap.innerHTML = `Width = ${Wd}, tiles 2 × 2: ${g} × ${g} blocks, ${g} phases each. Green: a real element is loaded. Red: the element does not exist, so the thread loads <b>0.0f</b>.\n` +
        (lines.length ? `Special treatment needed: ${lines.join(', ')}.` : 'In this block and phase every load is valid.') +
        (wrongM.length ? `\nWithout the check, ${wrongM[0]}: a wrong value. Past the last row the access would leave the array altogether.` : '');
    }
    const st = stepper(root, () => combos().length, (k) => draw(k), 2200);
    modeButtons(root, 'w', (v) => { Wd = +v; st.reset(); });
    return st;
  };

  // ------------------------------------------------------------ shared memory and occupancy
  ANIM.smem = (root) => {
    const q = (n) => root.querySelector(`[data-in=${n}]`), out = root.querySelector('[data-out]');
    function update() {
      const t = Math.max(1, Math.min(1024, +q('t').value | 0)), s = Math.max(0, +q('s').value || 0), smKB = Math.max(1, +q('sm').value || 1);
      const byThreads = Math.floor(2048 / t), bySmem = s > 0 ? Math.floor(smKB / s) : Infinity, byBlocks = 32;
      const blocks = Math.min(byThreads, bySmem, byBlocks), threads = blocks * t, occ = threads / 2048;
      const lim = [['threads', byThreads], ['shared memory', bySmem], ['block slots', byBlocks]].filter(([, v]) => v === blocks).map(([n]) => n);
      out.innerHTML = `shared memory per thread: ${s} KB / ${t} = <b>${(s * 1024 / t).toFixed(0)} B</b> (full occupancy needs ≤ ${(smKB * 1024 / 2048).toFixed(0)} B)<br>` +
        `blocks allowed: by threads ${byThreads}, by shared memory ${bySmem === Infinity ? '∞' : bySmem}, by block slots ${byBlocks}<br>` +
        `resident: <b>${blocks}</b> blocks = <b>${fmt(threads)}</b> threads, ${(blocks * s).toFixed(1)} of ${smKB} KB shared memory` +
        `<div class="cfg-big" style="margin-top:.3em">${pct(occ)} <small>occupancy</small></div>` +
        `<div class="cfg-msg ${occ >= 1 ? 'ok' : 'bad'}">${occ >= 1 ? 'Full occupancy.' : 'Limited by ' + lim.join(' and ') + '.'}</div>`;
      const svg = newSvg(root, '0 0 600 150', 'Shared memory of one SM filled by resident blocks');
      R(svg, 0, 20, 600, 50, { fill: '#EDEDED', rx: 3 });
      for (let b = 0; b < blocks && s > 0; b++) R(svg, 600 * b * s / smKB, 20, Math.max(1, 600 * s / smKB - 2), 50, { fill: b % 2 ? '#2F6FD0' : '#8FB3E8', rx: 2 });
      T(svg, 0, 14, `shared memory of one SM: ${smKB} KB`, { 'font-size': 13, class: 'lbl' });
      R(svg, 0, 96, 600, 30, { fill: '#EDEDED', rx: 3 }); R(svg, 0, 96, 600 * occ, 30, { fill: '#2E7D4F', rx: 3 });
      T(svg, 0, 90, `thread slots: ${fmt(threads)} of 2,048 used`, { 'font-size': 13, class: 'lbl' });
    }
    ['t', 's', 'sm'].forEach(n => q(n).addEventListener('input', update));
    root.querySelectorAll('[data-p]').forEach(b => b.addEventListener('click', () => { const [t, s] = b.dataset.p.split(','); q('t').value = t; q('s').value = s; q('sm').value = 228; update(); }));
    return { enter: update, leave() {} };
  };

