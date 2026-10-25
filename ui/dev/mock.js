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
