// A fake engine with sample data, used only when the page is opened in a plain browser to look
// at the layout (the app itself always has `window.__TAURI__`, so this file is never loaded there).

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const data = {
  notebooks: [
    { id: 1, name: 'Аренда', sources: 3 },
    { id: 2, name: 'Диплом', sources: 0 },
  ],
  sources: [
    { title: 'договор.md', passages: 4 },
    { title: 'памятка.pdf', passages: 2 },
    { title: 'ru.wikipedia.org/wiki/Аренда', passages: 15 },
  ],
};

const passages = {
  'договор.md': [
    'Договор аренды > Срок\n\nСрок действия договора составляет два года с даты подписания.',
    'Договор аренды > Оплата\n\nАрендная плата вносится ежемесячно до десятого числа текущего месяца.',
  ],
  'памятка.pdf': ['Арендатор обязан освободить помещение за тридцать дней до окончания срока.'],
};

const quote = 'Срок действия договора составляет два года с даты подписания.';

const answer = {
  text: 'Договор действует два года [1]. Платить нужно ежемесячно до десятого числа [2].\n\n- Выезд: предупредить за тридцать дней [3]\n- Продление возможно по соглашению сторон',
  citations: [
    { n: 1, sourceTitle: 'договор.md', ordinal: 0, quote },
    { n: 2, sourceTitle: 'договор.md', ordinal: 1, quote: 'Арендная плата вносится ежемесячно до десятого числа' },
    { n: 3, sourceTitle: 'памятка.pdf', ordinal: 0, quote: 'освободить помещение за тридцать дней' },
  ],
  rejectedQuotes: 1,
};

export const api = {
  async invoke(command, args = {}) {
    await delay(command === 'ask' || command === 'summarize' || command === 'compare' ? 700 : 80);
    switch (command) {
      case 'status': return { ready: true, error: null };
      case 'list_notebooks': return structuredClone(data.notebooks);
      case 'create_notebook': {
        const created = { id: data.notebooks.length + 1, name: args.name.trim(), sources: 0 };
        data.notebooks.push(created);
        return created;
      }
      case 'list_sources': return args.notebookId === 1 ? structuredClone(data.sources) : [];
      case 'add_source': return 3;
      case 'delete_source':
        data.sources = data.sources.filter((s) => s.title !== args.title);
        return null;
      case 'metrics_summary': {
        const now = Date.now();
        return {
          nowMs: now,
          kinds: [
            { kind: 'llm.complete', count: 14, errors: 1, p50Ms: 8200, p95Ms: 14000, maxMs: 15100, totalMs: 120000 },
            { kind: 'command.ask', count: 9, errors: 0, p50Ms: 9100, p95Ms: 14800, maxMs: 15200, totalMs: 90000 },
            { kind: 'embed.passages', count: 27, errors: 0, p50Ms: 1900, p95Ms: 3400, maxMs: 3900, totalMs: 52000 },
            { kind: 'source.read', count: 6, errors: 2, p50Ms: 40, p95Ms: 620, maxMs: 700, totalMs: 1100 },
            { kind: 'embed.query', count: 9, errors: 0, p50Ms: 38, p95Ms: 61, maxMs: 70, totalMs: 330 },
            { kind: 'retrieve.bm25', count: 9, errors: 0, p50Ms: 0.8, p95Ms: 1.2, maxMs: 1.4, totalMs: 7 },
          ],
          memory: [
            { label: 'after_start', lastMb: 960, maxMb: 1100, samples: 4 },
            { label: 'after_add', lastMb: 1710, maxMb: 1830, samples: 6 },
          ],
          failures: [
            { atMs: now - 300000, kind: 'source.read', subject: '/Users/me/Загрузки/скан.pdf', error: 'source', detail: 'cannot read the source: скан.pdf: the PDF has no text layer (a scan needs OCR first)' },
            { atMs: now - 7200000, kind: 'llm.complete', subject: null, error: 'model_unavailable', detail: 'the model is unavailable: codex exited with status 1' },
          ],
          slowest: [
            { atMs: now - 600000, kind: 'llm.complete', subject: null, ms: 15100 },
            { atMs: now - 900000, kind: 'command.add_source', subject: 'книга.pdf', ms: 58500 },
          ],
        };
      }
      case 'ask': return structuredClone(answer);
      case 'summarize': return { ...structuredClone(answer), text: 'Договор аренды > Срок\n- Два года с даты подписания [1]\nДоговор аренды > Оплата\n- Ежемесячно до десятого числа [2]' };
      case 'compare': return { ...structuredClone(answer), text: 'Общее: оба источника говорят о выезде за тридцать дней [3].\nРазличия: договор задаёт срок и оплату [1][2].\nПротиворечия: нет.' };
      case 'source_view': {
        const texts = passages[args.title] ?? ['(образец текста источника)'];
        return texts.map((text, ordinal) => {
          const at = args.ordinal === ordinal && args.quote ? text.toLowerCase().indexOf(args.quote.toLowerCase()) : -1;
          return at < 0
            ? { ordinal, before: text, hit: null, after: '' }
            : { ordinal, before: text.slice(0, at), hit: text.slice(at, at + args.quote.length), after: text.slice(at + args.quote.length) };
        });
      }
      default: throw new Error(`mock: unknown command ${command}`);
    }
  },
  pickFiles: async () => [],
  onDrop: null,
};
