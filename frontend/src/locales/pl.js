export const pl = {
  // Header
  appTitle: 'Detektor Dezinformacji',
  appSubtitle: 'Analiza treści z wykorzystaniem AI',
  expertModeLabel: 'Tryb ekspercki',
  langSwitchLabel: 'UA',

  // Input section
  placeholder: 'Wklej artykuł do analizy...',
  analyzeButton: 'Analizuj Tekst',
  analyzingButton: 'Analizowanie...',

  // Results
  noTechniques: 'Nie wykryto technik manipulacji.',
  inputPrompt: 'Wprowadź tekst do analizy...',
  disclaimer: null, // No disclaimer needed in Polish mode
  unknownTechnique: 'Nierozpoznana technika (możliwa halucynacja modelu)',

  // Expert panel (sidebar)
  expertPanelTitle: 'Panel Ekspercki',
  datasetLabel: 'Dataset (JSONL)',
  trainingProgress: 'Postęp treningu',
  evaluationProgress: 'Ewaluacja',
  metricCol: 'Metric',
  baselineCol: 'Baseline',
  newModelCol: 'New Model',
  promoteButton: 'Wdróż model',
  deployingButton: 'Wdrażanie...',
  deployedButton: 'Wdrożono',
  deployErrorButton: 'Błąd!',
  uploadSuccess: 'Pomyślnie rozpoczęto trening!',
  uploadError: 'Błąd: ',
  uploadConnError: 'Błąd połączenia: ',
  promoteWarning: 'Ostrzeżenie: Nowy model ma niższe F1 score niż bazowy. Czy na pewno chcesz go wdrożyć?',
  promoteConnError: 'Błąd awansu modelu: ',

  // Technique mapping
  techniques: {
    REFERENCE_ERROR:     { name: 'Błąd źródłowy',               description: 'Powoływanie się na nieistniejące, niewiarygodne lub błędnie zinterpretowane źródła.' },
    WHATABOUTISM:        { name: 'Whataboutism',                 description: 'Odwracanie uwagi od argumentu poprzez wytykanie oponentowi innych przewinień.' },
    STRAWMAN:            { name: 'Chochoł (Słomiana kukła)',     description: 'Atakowanie zniekształconej, uproszczonej wersji argumentu przeciwnika.' },
    EMOTIONAL_CONTENT:   { name: 'Język emocjonalny',           description: 'Używanie słów nacechowanych emocjonalnie, by wpłynąć na ocenę odbiorcy.' },
    CHERRY_PICKING:      { name: 'Wybiórczość (Cherry Picking)', description: 'Wybieranie tylko tych faktów, które pasują do z góry założonej tezy.' },
    FALSE_CAUSE:         { name: 'Fałszywa przyczyna',          description: 'Sugerowanie związku przyczynowo-skutkowego tam, gdzie on nie występuje.' },
    MISLEADING_CLICKBAIT:{ name: 'Clickbait / Manipulacja tytułem', description: 'Tytuł wprowadzający w błąd lub niewspółmierny do treści artykułu.' },
    ANECDOTE:            { name: 'Dowód anegdotyczny',           description: 'Opieranie argumentacji na pojedynczych, niepotwierdzonych historiach.' },
    LEADING_QUESTIONS:   { name: 'Pytania sugerujące',          description: 'Formułowanie pytań w sposób, który narzuca konkretną odpowiedź.' },
    EXAGGERATION:        { name: 'Wyolbrzymienie',               description: 'Przedstawianie faktów w sposób przesadny, by nadać im większą wagę.' },
    QUOTE_MINING:        { name: 'Wyrywanie z kontekstu',        description: 'Używanie autentycznych cytatów w sposób wypaczający ich oryginalny sens.' },
  }
};
