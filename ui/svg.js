// Small helpers shared by the pictures (mind map, diagram).

const NS = 'http://www.w3.org/2000/svg';

/** An SVG element with attributes and children. */
export function el(name, attrs = {}, ...children) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  for (const child of children) node.append(child);
  return node;
}

/** An icon button with an accessible name. */
export function iconButton(label, text, action) {
  const node = document.createElement('button');
  node.type = 'button';
  node.className = 'icon';
  node.setAttribute('aria-label', label);
  node.textContent = text;
  node.addEventListener('click', action);
  return node;
}

export function toolbar(...buttons) {
  const node = document.createElement('div');
  node.className = 'mindmap-tools';
  node.append(...buttons);
  return node;
}
