// Draws a diagram (boxes and arrows) in SVG, left to right: boxes go into columns by how far
// they are from the start of the process, arrows are curves. No libraries. Labels come from the
// model and are only ever set as text.

import { el, iconButton, toolbar } from './svg.js';

const BOX_W = 200;
const LINE = 16;
const GAP_X = 90;
const GAP_Y = 22;
const PAD = 28;
const LINE_CHARS = 26;

/** The label cut into at most three lines of about LINE_CHARS characters. */
export function wrap(label) {
  const lines = [];
  let line = '';
  for (const word of label.split(' ')) {
    if (line && `${line} ${word}`.length > LINE_CHARS) { lines.push(line); line = word; }
    else line = line ? `${line} ${word}` : word;
  }
  if (line) lines.push(line);
  if (lines.length > 3) {
    lines.length = 3;
    lines[2] = `${lines[2].slice(0, LINE_CHARS - 1)}…`;
  }
  return lines;
}

/**
 * Columns: the longest path from a box with nothing before it. An arrow that closes a loop
 * (found while walking) is kept for drawing but does not count for the columns.
 * Order inside a column: two sweeps that put a box near the middle of its neighbours.
 */
export function layout(diagram) {
  const ids = diagram.nodes.map((n) => n.id);
  const out = new Map(ids.map((id) => [id, []]));
  for (const edge of diagram.edges) out.get(edge.from)?.push(edge.to);

  const state = new Map(); // 1 = on the walk, 2 = done
  const back = new Set();
  const visit = (id) => {
    state.set(id, 1);
    for (const next of out.get(id)) {
      if (state.get(next) === 1) back.add(`${id}>${next}`);
      else if (!state.has(next)) visit(next);
    }
    state.set(id, 2);
  };
  for (const id of ids) if (!state.has(id)) visit(id);

  const forward = diagram.edges.filter((e) => !back.has(`${e.from}>${e.to}`));
  const level = new Map(ids.map((id) => [id, 0]));
  for (let pass = 0; pass < ids.length; pass += 1) {
    let changed = false;
    for (const edge of forward) {
      if (level.get(edge.to) < level.get(edge.from) + 1) { level.set(edge.to, level.get(edge.from) + 1); changed = true; }
    }
    if (!changed) break;
  }

  const columns = [];
  for (const id of ids) (columns[level.get(id)] ??= []).push(id);
  const slot = new Map();
  const place = () => columns.forEach((column) => column.forEach((id, i) => slot.set(id, i)));
  place();
  // The middle of the places of the neighbours of a box: the ones before it or the ones after.
  const middle = (id, side) => {
    const near = forward
      .filter((e) => (side === 'before' ? e.to === id : e.from === id))
      .map((e) => slot.get(side === 'before' ? e.from : e.to));
    return near.length ? near.reduce((x, y) => x + y, 0) / near.length : slot.get(id);
  };
  for (let sweep = 0; sweep < 2; sweep += 1) {
    for (let c = 1; c < columns.length; c += 1) {
      columns[c].sort((a, b) => middle(a, 'before') - middle(b, 'before'));
      place();
    }
    for (let c = columns.length - 2; c >= 0; c -= 1) {
      columns[c].sort((a, b) => middle(a, 'after') - middle(b, 'after'));
      place();
    }
  }

  const heights = new Map(diagram.nodes.map((n) => [n.id, wrap(n.label).length * LINE + 16]));
  const columnHeight = columns.map((column) => column.reduce((sum, id) => sum + heights.get(id) + GAP_Y, -GAP_Y));
  const tallest = Math.max(...columnHeight, 0);
  const boxes = new Map();
  columns.forEach((column, c) => {
    let y = PAD + (tallest - columnHeight[c]) / 2;
    for (const id of column) {
      boxes.set(id, { id, x: PAD + c * (BOX_W + GAP_X), y, w: BOX_W, h: heights.get(id) });
      y += heights.get(id) + GAP_Y;
    }
  });
  return {
    boxes,
    back,
    width: PAD * 2 + columns.length * BOX_W + Math.max(0, columns.length - 1) * GAP_X,
    height: PAD * 2 + tallest,
  };
}

/** Fills `container`; `onCite(citation)` opens the source for a box with a checked citation. */
export function renderDiagram(container, diagram, { onCite }) {
  const holder = document.createElement('div');
  holder.className = 'mindmap-canvas';
  const { boxes, back, width, height } = layout(diagram);
  // Opens fitted to the width of the panel (the panel must be visible to know it).
  let scale = Math.min(1, Math.max(0.4, ((container.clientWidth || 1000) - 32) / width));

  function draw() {
    const svg = el('svg', {
      width: width * scale, height: height * scale, viewBox: `0 0 ${width} ${height}`,
      role: 'img', 'aria-label': `Схема: ${diagram.title}`,
    });
    svg.append(el('defs', {}, el('marker', {
      id: 'arrow', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 8, markerHeight: 8, orient: 'auto-start-reverse',
    }, el('path', { d: 'M0,0 L10,5 L0,10 z', fill: 'var(--muted)' }))));
    for (const edge of diagram.edges) {
      const a = boxes.get(edge.from), b = boxes.get(edge.to);
      const loop = back.has(`${edge.from}>${edge.to}`);
      // An arrow that closes a loop leaves the top of its box and arches over to the top of the
      // target, so that it does not run along the forward arrows.
      const x1 = loop ? a.x + a.w / 2 : a.x + a.w, y1 = loop ? a.y : a.y + a.h / 2;
      const x2 = loop ? b.x + b.w / 2 : b.x, y2 = loop ? b.y : b.y + b.h / 2;
      const bend = (x2 - x1) / 2;
      const rise = Math.min(a.y, b.y) - 36;
      svg.append(el('path', {
        d: loop
          ? `M${x1},${y1} C${x1},${rise} ${x2},${rise} ${x2},${y2}`
          : `M${x1},${y1} C${x1 + bend},${y1} ${x2 - bend},${y2} ${x2},${y2}`,
        fill: 'none', stroke: 'var(--muted)', 'stroke-width': 1.5, 'marker-end': 'url(#arrow)',
        'stroke-dasharray': loop ? '5 4' : 'none',
      }));
      if (edge.label) {
        const text = el('text', { x: (x1 + x2) / 2, y: loop ? Math.min(a.y, b.y) - 24 : (y1 + y2) / 2 - 5, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--muted)' });
        text.textContent = edge.label;
        svg.append(text);
      }
    }
    for (const node of diagram.nodes) {
      const box = boxes.get(node.id);
      const group = el('g', { transform: `translate(${box.x},${box.y})` });
      group.classList.add('mm-node');
      group.append(el('rect', {
        width: box.w, height: box.h, rx: 10, fill: 'var(--bg)',
        stroke: node.cite ? 'var(--accent)' : 'var(--line)', 'stroke-width': node.cite ? 2 : 1.2,
      }));
      wrap(node.label).forEach((line, i) => {
        const text = el('text', { x: 12, y: 20 + i * LINE, 'font-size': 13, fill: 'var(--ink)' });
        text.textContent = line;
        group.append(text);
      });
      const tip = el('title');
      tip.textContent = node.cite ? `${node.label}\n«${node.cite.quote}» — ${node.cite.sourceTitle}` : node.label;
      group.append(tip);
      if (node.cite) {
        group.classList.add('has-cite');
        group.addEventListener('click', () => onCite(node.cite));
      }
      svg.append(group);
    }
    holder.replaceChildren(svg);
  }

  container.replaceChildren(
    toolbar(
      iconButton('Увеличить', '+', () => { scale = Math.min(2.5, scale + 0.2); draw(); }),
      iconButton('Уменьшить', '−', () => { scale = Math.max(0.4, scale - 0.2); draw(); }),
    ),
    holder,
  );
  draw();
}
