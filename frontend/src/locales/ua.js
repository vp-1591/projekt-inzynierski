export const ua = {
  // Header
  appTitle: 'Детектор Дезінформації',
  appSubtitle: 'Аналіз контенту за допомогою ШІ',
  expertModeLabel: 'Режим експерта',
  langSwitchLabel: 'PL',

  // Input section
  placeholder: 'Вставте статтю для аналізу...',
  analyzeButton: 'Аналізувати текст',
  analyzingButton: 'Аналізується...',

  // Results
  noTechniques: 'Маніпулятивних технік не виявлено.',
  inputPrompt: 'Введіть текст для аналізу...',
  disclaimer: 'Пояснення згенеровані польською моделлю штучного інтелекту і відображатимуться польською мовою.',
  unknownTechnique: 'Нерозпізнана техніка (можлива галюцинація моделі)',

  // Expert panel (sidebar)
  expertPanelTitle: 'Панель експерта',
  datasetLabel: 'Датасет (JSONL)',
  trainingProgress: 'Прогрес навчання',
  evaluationProgress: 'Оцінка',
  metricCol: 'Метрика',
  baselineCol: 'Базова',
  newModelCol: 'Нова модель',
  promoteButton: 'Розгорнути модель',
  deployingButton: 'Розгортання...',
  deployedButton: 'Розгорнуто',
  deployErrorButton: 'Помилка!',
  uploadSuccess: 'Навчання успішно розпочато!',
  uploadError: 'Помилка: ',
  uploadConnError: 'Помилка з\'єднання: ',
  promoteWarning: 'Попередження: Нова модель має нижчий F1, ніж базова. Ви впевнені, що хочете її розгорнути?',
  promoteConnError: 'Помилка розгортання моделі: ',

  // Technique mapping
  techniques: {
    REFERENCE_ERROR:     { name: 'Помилка посилання',          description: 'Посилання на неіснуючі, ненадійні або невірно витлумачені джерела.' },
    WHATABOUTISM:        { name: 'Вотабаутизм',                description: 'Відволікання від аргументу шляхом вказівки на інші провини опонента.' },
    STRAWMAN:            { name: 'Солом\'яне опудало',         description: 'Атака на перекручену, спрощену версію аргументу опонента.' },
    EMOTIONAL_CONTENT:   { name: 'Емоційна мова',              description: 'Використання емоційно забарвлених слів для впливу на сприйняття читача.' },
    CHERRY_PICKING:      { name: 'Вибіркова подача (Cherry Picking)', description: 'Вибір лише тих фактів, які підтверджують заздалегідь визначену тезу.' },
    FALSE_CAUSE:         { name: 'Хибна причина',              description: 'Припущення причинно-наслідкового зв\'язку там, де його немає.' },
    MISLEADING_CLICKBAIT:{ name: 'Клікбейт / Маніпулятивний заголовок', description: 'Заголовок, що вводить в оману або не відповідає змісту статті.' },
    ANECDOTE:            { name: 'Анекдотичний доказ',         description: 'Аргументація на основі окремих непідтверджених历ій.' },
    LEADING_QUESTIONS:   { name: 'Навідні запитання',          description: 'Формулювання запитань таким чином, що нав\'язує конкретну відповідь.' },
    EXAGGERATION:        { name: 'Перебільшення',              description: 'Представлення фактів у перебільшеному вигляді для надання їм більшої ваги.' },
    QUOTE_MINING:        { name: 'Вирвана з контексту цитата', description: 'Використання справжніх цитат у спосіб, що спотворює їхній оригінальний зміст.' },
  }
};
