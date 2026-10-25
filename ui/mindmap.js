// Draws a mind map as a horizontal tree in SVG: the root on the left, branches to the right,
// curved links. No libraries. Titles come from the model and are only ever set as text.

import { el, iconButton, toolbar } from './svg.js';

const ROW = 36;           // vertical room of one leaf
const GAP = 56;           // horizontal gap between columns
const PAD = 24;
const CHAR = 7.4;         // average width of one character at 13px
const BRANCH_COLORS = ['#4f7cff', '#2fb67c', '#e0912f', '#c05bd6', '#e0566b', '#2fb3c9'];

function textWidth(title) {
  return Math.min(300, Math.max(60, title.length * CHAR + 28));
}

/** Positions: x by depth (columns as wide as their widest title), y by leaves, parents centred. */
export function layout(root, collapsed) {
  const widths = [];
  (function measure(node, depth) {
    widths[depth] = Math.max(widths[depth] ?? 0, textWidth(node.title));
    if (!collapsed.has(node)) node.children.forEach((child) => measure(child, depth + 1));
  })(root, 0);
  const columnX = [];
  let x = PAD;
  widths.forEach((width, depth) => { columnX[depth] = x; x += width + GAP; });

  const placed = [];
  const links = [];
  let nextRow = 0;
  function place(node, depth, branch) {
    const visible = collapsed.has(node) ? [] : node.children;
    const kids = visible.map((child, i) => place(child, depth + 1, depth === 0 ? i : branch));
    const y = kids.length
      ? (kids[0].y + kids[kids.length - 1].y) / 2
      : PAD + nextRow++ * ROW;
    const box = { node, depth, branch, x: columnX[depth], y, w: textWidth(node.title), hasKids: node.children.length > 0 };
    for (const kid of kids) links.push({ from: box, to: kid });
    placed.push(box);
    return box;
  }
  place(root, 0, 0);
  return {
    placed,
    links,
    width: x - GAP + PAD,
    height: PAD * 2 + Math.max(1, nextRow) * ROW - (ROW - 24),
  };
}

/**
 * Fills `container` with the map. `onCite(citation)` opens the place in the source for a node
 * that has a checked citation.
 */
export function renderMindMap(container, map, { onCite }) {
  const collapsed = new Set();
  let scale = 1;
  let fitted = false;
  const holder = document.createElement('div');
  holder.className = 'mindmap-canvas';

  function draw() {
    const { placed, links, width, height } = layout(map.root, collapsed);
    if (!fitted) {
      // Opens fitted to the width of the panel (the panel must be visible to know it).
      scale = Math.min(1, Math.max(0.4, ((container.clientWidth || 1000) - 32) / width));
      fitted = true;
    }
    const svg = el('svg', {
      width: width * scale, height: height * scale, viewBox: `0 0 ${width} ${height}`,
      role: 'img', 'aria-label': `Ментальная карта: ${map.root.title}`,
    });
    for (const { from, to } of links) {
      const x1 = from.x + from.w, y1 = from.y + 12, x2 = to.x, y2 = to.y + 12;
      const mid = (x1 + x2) / 2;
      svg.append(el('path', {
        d: `M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`,
        fill: 'none', stroke: BRANCH_COLORS[to.branch % BRANCH_COLORS.length], 'stroke-opacity': 0.7, 'stroke-width': 1.6,
      }));
    }
    for (const box of placed) {
      const { node } = box;
      const color = BRANCH_COLORS[box.branch % BRANCH_COLORS.length];
      const group = el('g', { transform: `translate(${box.x},${box.y})` });
      group.classList.add('mm-node');
      const body = el('rect', {
        width: box.w, height: 24, rx: 8,
        fill: box.depth === 0 ? 'var(--accent)' : 'var(--bg)',
        stroke: box.depth === 0 ? 'var(--accent)' : color, 'stroke-width': node.cite ? 1.8 : 1,
      });
      const label = el('text', {
        x: 12, y: 16.5, 'font-size': 13,
        fill: box.depth === 0 ? 'var(--accent-ink)' : 'var(--ink)',
      });
      const shown = node.title.length * CHAR + 28 > box.w
        ? `${node.title.slice(0, Math.floor((box.w - 28) / CHAR) - 1)}…` : node.title;
      label.textContent = shown;
      const tip = el('title');
      tip.textContent = node.cite ? `${node.title}\n«${node.cite.quote}» — ${node.cite.sourceTitle}` : node.title;
      group.append(body, label, tip);
      if (node.cite) {
        group.classList.add('has-cite');
        group.addEventListener('click', () => onCite(node.cite));
      }
      svg.append(group);
      if (box.hasKids) {
        const open = !collapsed.has(node);
        const toggle = el('g', { transform: `translate(${box.x + box.w + 11},${box.y + 12})` });
        toggle.classList.add('mm-toggle');
        toggle.append(
          el('circle', { r: 8, fill: 'var(--panel)', stroke: color, 'stroke-width': 1.2 }),
          Object.assign(el('text', { 'text-anchor': 'middle', y: 4, 'font-size': 12, fill: 'var(--ink)' }), { textContent: open ? '‹' : '›' }),
        );
        toggle.addEventListener('click', (event) => {
          event.stopPropagation();
          if (open) collapsed.add(node); else collapsed.delete(node);
          draw();
        });
        svg.append(toggle);
      }
    }
    holder.replaceChildren(svg);
  }

  const tools = toolbar(
    iconButton('Увеличить', '+', () => { scale = Math.min(2.5, scale + 0.2); draw(); }),
    iconButton('Уменьшить', '−', () => { scale = Math.max(0.4, scale - 0.2); draw(); }),
    iconButton('Свернуть до первого уровня', '⇤', () => {
      collapsed.clear();
      for (const child of map.root.children) if (child.children.length) collapsed.add(child);
      draw();
    }),
    iconButton('Развернуть всё', '⇥', () => { collapsed.clear(); draw(); }),
  );
  container.replaceChildren(tools, holder);
  draw();
}
