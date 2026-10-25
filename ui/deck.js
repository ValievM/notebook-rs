// A deck of flashcards: question, then answer with the place in the source that proves it.
// "Знаю" takes a card out of the round, "Ещё раз" puts it at the end. The round is only on the
// screen: nothing is saved about what the person knew.

function node(tag, className, ...children) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  element.append(...children);
  return element;
}

function button(label, action, className = '') {
  const element = node('button', className, label);
  element.type = 'button';
  element.addEventListener('click', action);
  return element;
}

/** Fills `container`; `onCite(citation)` opens the source of the card on screen. */
export function renderDeck(container, deck, { onCite }) {
  const total = deck.cards.length;
  let queue = deck.cards.map((_, i) => i);
  let shown = false;
  let known = 0;

  function draw() {
    if (total === 0) {
      container.replaceChildren(node('p', 'hint', 'Подтверждённых карточек нет: ни у одной не нашлась цитата в источнике.'));
      return;
    }
    if (queue.length === 0) {
      container.replaceChildren(node('div', 'deck-card',
        node('h3', '', 'Круг пройден'),
        node('p', '', `Знаете: ${known} из ${total}.`),
        button('Начать заново', () => { queue = deck.cards.map((_, i) => i); known = 0; shown = false; draw(); }, 'primary')));
      return;
    }
    const card = deck.cards[queue[0]];
    const parts = [
      node('p', 'hint', `Осталось в круге: ${queue.length} · знаете: ${known} из ${total}`),
      node('div', 'deck-card',
        node('div', 'deck-label', 'Вопрос'),
        node('p', 'deck-text', card.question)),
    ];
    const box = parts[1];
    if (!shown) {
      box.append(button('Показать ответ', () => { shown = true; draw(); }, 'primary'));
    } else {
      box.append(
        node('div', 'deck-label', 'Ответ'),
        node('p', 'deck-text', card.answer),
        button(`Источник: ${card.cite.sourceTitle}`, () => onCite(card.cite), 'quiet'),
        node('div', 'deck-buttons',
          button('Ещё раз', () => { queue.push(queue.shift()); shown = false; draw(); }),
          button('Знаю', () => { queue.shift(); known += 1; shown = false; draw(); }, 'primary')));
    }
    container.replaceChildren(...parts);
  }
  draw();
}
