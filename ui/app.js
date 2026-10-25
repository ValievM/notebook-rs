// The window of notebook-rs. Plain JavaScript, no build step.
//
// Everything that comes from sources or from the model is untrusted text (web pages and PDFs
// can say anything). It is only ever put into the page as text nodes, never as HTML.

const api = await loadApi();

async function loadApi() {
  const tauri = window.__TAURI__;
  if (tauri) {
    return {
      invoke: tauri.core.invoke,
      pickFiles: async () => {
        const chosen = await tauri.dialog.open({
          multiple: true,
          filters: [{ name: 'Источники', extensions: ['txt', 'md', 'markdown', 'pdf'] }],
        });
        return chosen ? [].concat(chosen) : [];
      },
      onDrop: (handler) => tauri.webview.getCurrentWebview().onDragDropEvent((event) => handler(event.payload)),
    };
  }
  // Opened in a plain browser (to look at the layout): a fake engine with sample data.
  return (await import('./dev/mock.js')).api;
}

const $ = (id) => document.getElementById(id);

const state = {
  notebooks: [],
  current: null,
  sources: [],
  selected: new Set(),
  working: 0,
  ready: false,
};

/** Builds an element. Strings become text nodes; there is no way to pass HTML. */
function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === 'class') node.className = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function remember(key, value) {
  try { localStorage.setItem(key, String(value)); } catch { /* private window: fine */ }
}
function recall(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}

// ---------- messages to the person ----------

let toastTimer = null;
function toast(message, kind = 'info') {
  const node = $('toast');
  node.textContent = message;
  node.className = `toast ${kind}`;
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.hidden = true; }, kind === 'error' ? 9000 : 3500);
}

function banner(message, kind = 'info') {
  const node = $('banner');
  node.hidden = !message;
  node.textContent = message || '';
  node.className = `banner ${kind}`;
}

/** Marks that something is being done: buttons wait, and the work shows in the dialog. */
function working(delta) {
  state.working += delta;
  const busy = state.working > 0 || !state.ready;
  for (const id of ['send', 'summarize', 'compare', 'pick-files']) $(id).disabled = busy;
  if (!busy) renderSelection();
}

// ---------- notebooks ----------

async function loadNotebooks(selectId) {
  state.notebooks = await api.invoke('list_notebooks');
  renderNotebooks();
  const wanted = selectId ?? state.current ?? Number(recall('notebook'));
  const target = state.notebooks.find((n) => n.id === wanted) ?? state.notebooks[0];
  if (target && target.id !== state.current) await selectNotebook(target.id);
  else if (!target) { state.current = null; renderEmptyDialog(); renderSources(); }
}

function renderNotebooks() {
  const current = state.notebooks.find((n) => n.id === state.current);
  $('notebook-title').textContent = current ? current.name : 'Блокнот не выбран';
  const list = $('notebook-list');
  list.replaceChildren(...state.notebooks.map((notebook) =>
    h('li', {},
      h('button', {
        class: 'notebook-item',
        type: 'button',
        'aria-current': String(notebook.id === state.current),
        onclick: () => { closeNotebookMenu(); selectNotebook(notebook.id); },
      }, h('span', {}, notebook.name), h('span', { class: 'count' }, notebook.sources)))));
}

async function selectNotebook(id) {
  state.current = id;
  state.selected.clear();
  remember('notebook', id);
  closeViewer();
  renderNotebooks();
  $('messages').replaceChildren();
  renderEmptyDialog();
  await loadSources();
}

function closeNotebookMenu() {
  $('notebook-menu').hidden = true;
  $('notebook-button').setAttribute('aria-expanded', 'false');
}

$('notebook-button').addEventListener('click', () => {
  const open = $('notebook-menu').hidden;
  $('notebook-menu').hidden = !open;
  $('notebook-button').setAttribute('aria-expanded', String(open));
  if (open) $('new-notebook-name').focus();
});
document.addEventListener('click', (event) => {
  if (!event.target.closest('.notebook-switch')) closeNotebookMenu();
});

$('new-notebook').addEventListener('submit', async (event) => {
  event.preventDefault();
  const input = $('new-notebook-name');
  try {
    const created = await api.invoke('create_notebook', { name: input.value });
    input.value = '';
    closeNotebookMenu();
    await loadNotebooks(created.id);
  } catch (error) { toast(String(error), 'error'); }
});

// ---------- sources ----------

async function loadSources() {
  if (state.current === null) { state.sources = []; renderSources(); return; }
  try {
    state.sources = await api.invoke('list_sources', { notebookId: state.current });
  } catch (error) { toast(String(error), 'error'); state.sources = []; }
  const titles = new Set(state.sources.map((s) => s.title));
  for (const title of [...state.selected]) if (!titles.has(title)) state.selected.delete(title);
  renderSources();
}

function renderSources() {
  $('source-list').replaceChildren(...state.sources.map(sourceItem));
  renderSelection();
  renderEmptyDialog();
}

function sourceItem(source) {
  const checkbox = h('input', {
    type: 'checkbox',
    'aria-label': `Выбрать «${source.title}»`,
    onchange: (event) => {
      if (event.target.checked) state.selected.add(source.title);
      else state.selected.delete(source.title);
      renderSelection();
    },
  });
  checkbox.checked = state.selected.has(source.title);
  const remove = h('button', { class: 'source-delete', type: 'button', 'aria-label': `Удалить «${source.title}»` }, '×');
  let timer = null;
  remove.addEventListener('click', async () => {
    if (!remove.classList.contains('confirm')) {
      remove.classList.add('confirm');
      remove.textContent = 'Удалить?';
      timer = setTimeout(() => { remove.classList.remove('confirm'); remove.textContent = '×'; }, 3000);
      return;
    }
    clearTimeout(timer);
    try {
      await api.invoke('delete_source', { notebookId: state.current, title: source.title });
      await loadSources();
      await loadNotebooks();
    } catch (error) { toast(String(error), 'error'); }
  });
  return h('li', { class: 'source-item' },
    checkbox,
    h('button', { class: 'source-title', type: 'button', title: source.title, onclick: () => openSource(source.title) },
      source.title, h('small', {}, `${source.passages} фрагм.`)),
    remove);
}

function renderSelection() {
  const count = state.selected.size;
  const busy = state.working > 0 || !state.ready;
  const hasNotebook = state.current !== null && state.sources.length > 0;
  $('summarize').textContent = count === 1 ? 'Конспект источника' : 'Конспект блокнота';
  $('summarize').disabled = busy || !hasNotebook || count > 1;
  $('compare').disabled = busy || count !== 2;
  $('selection-hint').textContent =
    count === 0 ? 'Отметьте источники галочками: один для конспекта, два для сравнения.'
    : count === 1 ? 'Выбран один источник: конспект только по нему.'
    : count === 2 ? 'Выбрано два источника: их можно сравнить.'
    : 'Для конспекта оставьте один источник или снимите выбор, для сравнения нужно ровно два.';
}

function sourceStatus(message, isError = false) {
  const node = $('source-status');
  node.textContent = message;
  node.classList.toggle('error', isError);
}

async function addLocations(locations) {
  if (!state.ready) {
    sourceStatus('Поиск ещё готовится, подождите несколько секунд и повторите.', true);
    return;
  }
  if (state.working > 0) {
    sourceStatus('Подождите: ещё идёт предыдущая операция.', true);
    return;
  }
  if (state.current === null) {
    sourceStatus('Сначала выберите или создайте блокнот слева.', true);
    toast('Сначала выберите или создайте блокнот слева.', 'error');
    return;
  }
  working(1);
  const failures = [];
  try {
    for (const location of locations) {
      const name = location.split(/[\\/]/).pop();
      sourceStatus(`Добавляю «${name}»…`);
      try {
        const count = await api.invoke('add_source', { notebookId: state.current, location });
        toast(`«${name}»: добавлено, ${count} фрагм.`);
      } catch (error) {
        failures.push(`«${name}»: ${error}`);
        toast(String(error), 'error');
      }
    }
  } finally {
    // An error stays on screen until the next attempt: a toast is gone in seconds.
    sourceStatus(failures.join('\n'), failures.length > 0);
    working(-1);
  }
  await loadSources();
  await loadNotebooks();
}

$('pick-files').addEventListener('click', async () => {
  const files = await api.pickFiles();
  if (files.length) await addLocations(files);
});

$('add-url').addEventListener('submit', async (event) => {
  event.preventDefault();
  const input = $('url');
  const address = input.value.trim();
  if (!address) return;
  input.value = '';
  await addLocations([address]);
});

api.onDrop?.((payload) => {
  const drop = $('drop');
  if (payload.type === 'enter' || payload.type === 'over') drop.classList.add('over');
  else drop.classList.remove('over');
  if (payload.type === 'drop') {
    if (payload.paths?.length) addLocations(payload.paths);
    else sourceStatus('Окно получило перетаскивание, но без путей к файлам.', true);
  }
});

// ---------- dialog ----------

const STARTERS = ["О чём эти источники?", "Назови главные факты и числа", "Какие здесь сроки и условия?"];

function renderEmptyDialog() {
  const messages = $('messages');
  messages.querySelector('[data-empty]')?.remove();
  if (messages.children.length) return;
  const text = state.current === null
    ? ['Создайте блокнот', 'Блокнот это набор источников, по которым можно задавать вопросы.']
    : state.sources.length === 0
      ? ['Добавьте источники', 'Перетащите файлы или вставьте адрес страницы слева. Потом задавайте вопросы.']
      : ['Задайте вопрос', 'Ответ строится только по источникам блокнота, а каждая цитата проверена по тексту.'];
  const box = h('div', { class: 'empty', 'data-empty': '' }, h('strong', {}, text[0]), text[1]);
  if (state.current !== null && state.sources.length > 0) {
    // Starting points: they only fill the question box; the person sends it.
    box.append(h('div', { class: 'chips' }, ...STARTERS.map((starter) =>
      h('button', { class: 'chip', type: 'button', onclick: () => { $('question').value = starter; $('question').focus(); } }, starter))));
  }
  messages.append(box);
}

function addMessage(node) {
  const messages = $('messages');
  messages.querySelector('[data-empty]')?.remove();
  messages.append(node);
  messages.scrollTop = messages.scrollHeight;
  return node;
}

/** Runs one request to the model and puts its answer (or its error) into the dialog. */
async function runTask(kind, waiting, command, args) {
  const pending = addMessage(h('div', { class: 'message pending' }, waiting));
  working(1);
  try {
    const answer = await api.invoke(command, { notebookId: state.current, ...args });
    pending.replaceWith(answerMessage(kind, answer));
  } catch (error) {
    pending.replaceWith(h('div', { class: 'message error' }, String(error)));
  } finally {
    working(-1);
    $('messages').scrollTop = $('messages').scrollHeight;
  }
}

$('composer').addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = $('question').value.trim();
  if (!question || state.working > 0 || !state.ready) return;
  if (state.current === null) { toast('Сначала создайте блокнот.', 'error'); return; }
  $('question').value = '';
  addMessage(h('div', { class: 'message user' }, question));
  const deep = $('deep').checked;
  await runTask(deep ? 'Глубокий ответ' : 'Ответ', deep ? 'Планирую поиски и читаю источники…' : 'Ищу в источниках…', 'ask', { question, deep });
});

$('question').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) $('composer').requestSubmit();
});

$('summarize').addEventListener('click', async () => {
  const [source] = [...state.selected];
  const label = source ? `Конспект «${source}»` : 'Конспект блокнота';
  addMessage(h('div', { class: 'message user' }, label));
  await runTask('Конспект', 'Составляю конспект…', 'summarize', { source: source ?? null });
});

$('compare').addEventListener('click', async () => {
  const [first, second] = [...state.selected];
  addMessage(h('div', { class: 'message user' }, `Сравнить «${first}» и «${second}»`));
  await runTask('Сравнение', 'Сравниваю источники…', 'compare', { first, second });
});

// ---------- answers with citations ----------

function answerMessage(kind, answer) {
  const known = new Map();
  for (const citation of answer.citations) if (!known.has(citation.n)) known.set(citation.n, citation);
  const body = h('div', { class: 'message assistant' }, h('div', { class: 'kind' }, kind));
  renderRich(body, answer.text, known);
  if (answer.citations.length) {
    body.append(h('ul', { class: 'citations' }, answer.citations.map((citation) =>
      h('li', {}, h('button', { class: 'citation', type: 'button', onclick: () => openCitation(citation) },
        h('span', { class: 'cite-n' }, `[${citation.n}]`), ' ', h('b', {}, citation.sourceTitle),
        h('span', { class: 'quote' }, `«${citation.quote}»`))))));
  }
  if (answer.rejectedQuotes > 0) {
    body.append(h('p', { class: 'rejected' },
      `Отброшено цитат, которых нет в источниках: ${answer.rejectedQuotes}. Модель их придумала, они не показаны.`));
  }
  return body;
}

/** Lines become paragraphs, "- " lines become list items, "# " lines headings; [n] becomes a button. */
function renderRich(container, text, known) {
  let list = null;
  for (const raw of text.split('\n')) {
    const line = raw.trimEnd();
    if (!line.trim()) { list = null; continue; }
    const bullet = /^\s*[-*•]\s+(.*)$/.exec(line);
    if (bullet) {
      if (!list) { list = h('ul'); container.append(list); }
      list.append(h('li', {}, inline(bullet[1], known)));
      continue;
    }
    list = null;
    const heading = /^#{1,6}\s+(.*)$/.exec(line);
    container.append(heading ? h('h4', {}, inline(heading[1], known)) : h('p', {}, inline(line, known)));
  }
}

function inline(text, known) {
  const parts = [];
  let last = 0;
  for (const match of text.matchAll(/\[(\d+)\]/g)) {
    if (match.index > last) parts.push(text.slice(last, match.index));
    const citation = known.get(Number(match[1]));
    parts.push(citation
      ? h('button', {
        class: 'cite', type: 'button', title: `${citation.sourceTitle}: «${citation.quote}»`,
        'aria-label': `Цитата ${match[1]}: ${citation.sourceTitle}`,
        onclick: () => openCitation(citation),
      }, match[1])
      : match[0]);
    last = match.index + match[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

// ---------- source viewer ----------

function openCitation(citation) {
  return openSource(citation.sourceTitle, citation.ordinal, citation.quote);
}

async function openSource(title, ordinal = null, quote = null) {
  try {
    const passages = await api.invoke('source_view', { notebookId: state.current, title, ordinal, quote });
    $('viewer-title').textContent = title;
    $('viewer-body').replaceChildren(...passages.map((passage) =>
      h('section', { class: 'passage' },
        h('span', { class: 'ordinal' }, `фрагмент ${passage.ordinal + 1}`),
        passage.before,
        passage.hit === null ? null : h('mark', {}, passage.hit),
        passage.after)));
    $('viewer').hidden = false;
    $('viewer-body').querySelector('mark')?.scrollIntoView({ block: 'center' });
  } catch (error) { toast(String(error), 'error'); }
}

function closeViewer() { $('viewer').hidden = true; }
$('viewer-close').addEventListener('click', closeViewer);
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') { closeViewer(); closeMetrics(); closeSettings(); closeNotebookMenu(); }
});

// ---------- who answers ----------

async function refreshAccount() {
  try {
    const account = await api.invoke('account_status');
    state.account = account;
    $('account-line').textContent = account.signedIn
      ? 'Отвечает ChatGPT по вашей подписке'
      : 'Отвечает Codex. Вход через ChatGPT даст свою модель и лимиты.';
    $('sign-in-button').hidden = account.signedIn;
    $('sign-out-group').hidden = !account.signedIn;
    hideSignOutConfirm();
  } catch (error) {
    $('account-line').textContent = String(error);
  }
  await refreshModelChoice();
}

const EFFORT_LABELS = { minimal: 'минимальное', low: 'низкое', medium: 'среднее', high: 'высокое' };

function fillSelect(select, choices, current, placeholder) {
  select.replaceChildren();
  // The placeholder stands for "no choice made yet" and goes away once a choice exists.
  if (!current) select.append(new Option(placeholder, ''));
  for (const [value, label] of choices) select.append(new Option(label, value));
  select.value = current ?? '';
}

async function refreshModelChoice() {
  const box = $('model-choice');
  if (!state.account?.signedIn) { box.hidden = true; return; }
  try {
    const choice = await api.invoke('model_state');
    fillSelect($('model-select'), choice.models.map((m) => [m.slug, m.name]), choice.model, 'первая из списка аккаунта');
    fillSelect($('effort-select'), choice.efforts.map((e) => [e, EFFORT_LABELS[e] ?? e]), choice.effort, 'по умолчанию модели');
    box.hidden = false;
  } catch (error) {
    box.hidden = true;
    toast(String(error), 'error');
  }
}

async function choose(args) {
  try {
    await api.invoke('model_choose', args);
    toast('Выбор сохранён: он действует со следующего вопроса.');
  } catch (error) {
    toast(String(error), 'error');
  }
  await refreshModelChoice();
}

$('model-select').addEventListener('change', (event) => {
  if (event.target.value) choose({ model: event.target.value });
});
$('effort-select').addEventListener('change', (event) => {
  if (event.target.value) choose({ effort: event.target.value });
});

function hideSignOutConfirm() {
  $('sign-out-confirm').hidden = true;
  $('sign-out-button').hidden = false;
}

// Signing out takes two steps: the button only asks, and "Да, выйти" does it.
$('sign-out-button').addEventListener('click', () => {
  $('sign-out-button').hidden = true;
  $('sign-out-confirm').hidden = false;
  $('sign-out-no').focus();
});
$('sign-out-no').addEventListener('click', hideSignOutConfirm);
$('sign-out-yes').addEventListener('click', async () => {
  $('sign-out-yes').disabled = true;
  try {
    await api.invoke('account_sign_out');
    toast('Вход удалён: отвечает Codex.');
  } catch (error) {
    toast(String(error), 'error');
  } finally {
    $('sign-out-yes').disabled = false;
    await refreshAccount();
  }
});

$('sign-in-button').addEventListener('click', async () => {
  const button = $('sign-in-button');
  button.disabled = true;
  try {
    $('account-line').textContent = 'Жду подтверждения в браузере: войдите и разрешите использовать план ChatGPT (до 5 минут)…';
    await api.invoke('account_sign_in');
    toast('Вход выполнен: отвечает ChatGPT.');
  } catch (error) {
    toast(String(error), 'error');
  } finally {
    button.disabled = false;
    await refreshAccount();
  }
});

// ---------- theme ----------

const THEME_KEY = 'notebook-rs-theme';

// The choice is a per-window convenience: if the browser refuses storage, the system theme is used.
function savedTheme() {
  try { return localStorage.getItem(THEME_KEY) || 'system'; } catch { return 'system'; }
}

function applyTheme(theme) {
  if (theme === 'light' || theme === 'dark') document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
}

applyTheme(savedTheme());
$('theme-select').value = savedTheme();
$('theme-select').addEventListener('change', (event) => {
  applyTheme(event.target.value);
  try { localStorage.setItem(THEME_KEY, event.target.value); } catch { /* applied for now only */ }
});

// ---------- settings ----------

function closeSettings() { $('settings').hidden = true; hideSignOutConfirm(); }

async function openSettings() {
  closeViewer();
  closeMetrics();
  $('settings').hidden = false;
  await refreshAccount();
}

$('open-settings').addEventListener('click', openSettings);
$('settings-close').addEventListener('click', closeSettings);

// ---------- metrics ----------

const KIND_LABELS = {
  ask: 'вопрос целиком',
  ask_deep: 'глубокий вопрос',
  summarize: 'конспект',
  compare: 'сравнение',
  'llm.complete': 'вызов модели',
  'embed.query': 'вектор вопроса',
  'embed.passages': 'векторы фрагментов',
  'retrieve.hybrid': 'поиск (общий)',
  'retrieve.bm25': 'поиск по словам',
  'retrieve.vector': 'поиск по смыслу',
  'source.read': 'чтение источника',
  'source.chunk': 'нарезка',
  'source.add': 'добавление источника',
  'store.write': 'запись в базу',
  'engine.start': 'запуск движка',
  'engine.model_load': 'загрузка модели',
  'command.ask': 'окно: вопрос',
  'command.ask_deep': 'окно: глубокий вопрос',
  'command.add_source': 'окно: добавить источник',
  'command.summarize': 'окно: конспект',
  'command.compare': 'окно: сравнение',
  'command.delete_source': 'окно: удалить источник',
  'command.create_notebook': 'окно: новый блокнот',
  'command.source_view': 'окно: открыть источник',
};

function formatMs(ms) {
  if (ms === null || ms === undefined) return '—';
  if (ms < 1) return `${ms.toFixed(1)} мс`;
  if (ms < 1000) return `${Math.floor(ms)} мс`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)} с`;
  return `${Math.floor(ms / 60000)} мин ${Math.floor((ms % 60000) / 1000)} с`;
}

function formatMb(mb) {
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} ГБ` : `${Math.round(mb)} МБ`;
}

function formatAgo(ms) {
  if (ms < 60000) return 'только что';
  if (ms < 3600000) return `${Math.floor(ms / 60000)} мин назад`;
  if (ms < 86400000) return `${Math.floor(ms / 3600000)} ч назад`;
  return `${Math.floor(ms / 86400000)} дн назад`;
}

function renderMetrics(summary) {
  const body = $('metrics-body');
  const nothing = !summary.kinds.length && !summary.memory.length && !summary.failures.length;
  if (nothing) {
    body.replaceChildren(h('p', { class: 'hint' }, 'Пока ничего не записано за этот период.'));
    return;
  }
  const cell = (value, bad = false) => h('td', bad ? { class: 'bad' } : {}, value);
  const operations = h('table', { class: 'metrics' },
    h('thead', {}, h('tr', {}, ['Операция', 'Раз', 'Ошибок', 'Медиана', '95%', 'Макс.'].map((name) => h('th', {}, name)))),
    h('tbody', {}, summary.kinds.map((kind) => h('tr', {},
      h('td', {}, KIND_LABELS[kind.kind] ?? kind.kind, h('small', {}, kind.kind)),
      cell(kind.count), cell(kind.errors, kind.errors > 0),
      cell(formatMs(kind.p50Ms)), cell(formatMs(kind.p95Ms)), cell(formatMs(kind.maxMs))))));
  const sections = [h('h3', {}, 'Операции (больше всего времени сверху)'), operations];
  if (summary.memory.length) {
    sections.push(h('h3', {}, 'Память процесса'));
    sections.push(h('ul', { class: 'events' }, summary.memory.map((memory) => h('li', {},
      h('b', {}, memory.label), `: сейчас ${formatMb(memory.lastMb)}, максимум ${formatMb(memory.maxMb)}, замеров ${memory.samples}`))));
  }
  if (summary.failures.length) {
    sections.push(h('h3', {}, 'Последние ошибки'));
    sections.push(h('ul', { class: 'events' }, summary.failures.map((failure) => h('li', {},
      h('span', { class: 'when' }, `${formatAgo(summary.nowMs - failure.atMs)}  `),
      h('b', { class: 'bad' }, failure.kind),
      failure.subject ? `  «${failure.subject}»` : '',
      h('span', { class: 'detail' }, `${failure.error}: ${failure.detail}`)))));
  }
  if (summary.slowest.length) {
    sections.push(h('h3', {}, 'Самые медленные'));
    sections.push(h('ul', { class: 'events' }, summary.slowest.map((slow) => h('li', {},
      h('b', {}, formatMs(slow.ms)), `  ${slow.kind}`,
      slow.subject ? `  «${slow.subject}»` : '',
      h('span', { class: 'when' }, `  ${formatAgo(summary.nowMs - slow.atMs)}`)))));
  }
  body.replaceChildren(...sections);
}

async function refreshMetrics() {
  try {
    renderMetrics(await api.invoke('metrics_summary', { hours: Number($('metrics-hours').value) }));
  } catch (error) {
    $('metrics-body').replaceChildren(h('p', { class: 'hint error' }, String(error)));
  }
}

async function openMetrics() {
  closeViewer();
  closeSettings();
  $('metrics').hidden = false;
  await refreshMetrics();
}

function closeMetrics() { $('metrics').hidden = true; }

$('open-metrics').addEventListener('click', openMetrics);
$('metrics-close').addEventListener('click', closeMetrics);
$('metrics-refresh').addEventListener('click', refreshMetrics);
$('metrics-hours').addEventListener('change', refreshMetrics);

// ---------- start ----------

async function start() {
  refreshAccount();
  working(0); // buttons wait until the engine is ready
  renderEmptyDialog();
  for (;;) {
    const status = await api.invoke('status');
    if (status.error) { banner(status.error, 'error'); return; }
    if (status.ready) break;
    banner('Готовлю поиск: при первом запуске скачивается модель (около 120 МБ), это займёт пару минут.');
    await new Promise((resolve) => setTimeout(resolve, 700));
  }
  banner('');
  state.ready = true;
  working(0);
  await loadNotebooks();
}

start();
