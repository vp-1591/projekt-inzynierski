/**
 * Service to handle disinformation detection analysis using local Ollama instance.
 */

const OLLAMA_URL = 'http://localhost:11434/api/chat';
const MODEL_NAME = 'bielik-lora-mipd:latest';

// Mapping model tags to user-friendly Polish names and descriptions
const TECHNIQUE_MAPPING = {
  'REFERENCE_ERROR': {
    name: 'Błąd źródłowy',
    description: 'Powoływanie się na nieistniejące, niewiarygodne lub błędnie zinterpretowane źródła.'
  },
  'WHATABOUTISM': {
    name: 'Whataboutism',
    description: 'Odwracanie uwagi od argumentu poprzez wytykanie oponentowi innych przewinień.'
  },
  'STRAWMAN': {
    name: 'Chochoł (Słomiana kukła)',
    description: 'Atakowanie zniekształconej, uproszczonej wersji argumentu przeciwnika.'
  },
  'EMOTIONAL_CONTENT': {
    name: 'Język emocjonalny',
    description: 'Używanie słów nacechowanych emocjonalnie, by wpłynąć na ocenę odbiorcy.'
  },
  'CHERRY_PICKING': {
    name: 'Dowody anegdotyczne (Wybiórczość)',
    description: 'Wybieranie tylko tych faktów, które pasują do z góry założonej tezy.'
  },
  'FALSE_CAUSE': {
    name: 'Fałszywa przyczyna',
    description: 'Sugerowanie związku przyczynowo-skutkowego tam, gdzie on nie występuje.'
  },
  'MISLEADING_CLICKBAIT': {
    name: 'Clickbait / Manipulacja tytułem',
    description: 'Tytuł wprowadzający w błąd lub niewspółmierny do treści artykułu.'
  },
  'ANECDOTE': {
    name: 'Dowód anegdotyczny',
    description: 'Opieranie argumentacji na pojedynczych, niepotwierdzonych historiach.'
  },
  'LEADING_QUESTIONS': {
    name: 'Pytania sugerujące',
    description: 'Formułowanie pytań w sposób, który narzuca konkretną odpowiedź.'
  },
  'EXAGGERATION': {
    name: 'Wyolbrzymienie',
    description: 'Przedstawianie faktów w sposób przesadny, by nadać im większą wagę.'
  },
  'QUOTE_MINING': {
    name: 'Wyrywanie z kontekstu',
    description: 'Używanie autentycznych cytatów w sposób wypaczający ich oryginalny sens.'
  }
};

/**
 * Analyzes the provided text for disinformation techniques.
 * @param {string} text - The article text to analyze.
 * @returns {Promise<Array>} - A promise resolving to an array of detected techniques.
 */
export async function analyzeText(text) {
  try {
    const response = await fetch(OLLAMA_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: MODEL_NAME,
        messages: [
          {
            role: 'user',
            content: text
          }
        ],
        stream: false,
        format: 'json'
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.statusText}`);
    }

    const data = await response.json();
    const content = JSON.parse(data.message.content);
    
    // Model returns: { "discovered_techniques": ["TAG1", "TAG2"] }
    const tags = content.discovered_techniques || [];

    // Map tags to the format expected by the UI
    return tags.map((tag, index) => {
      const info = TECHNIQUE_MAPPING[tag] || { 
        name: tag, 
        description: 'Wykryto technikę manipulacji.' 
      };
      
      return {
        id: `${tag}-${index}-${Date.now()}`,
        technique_name: info.name,
        description: info.description
      };
    });

  } catch (error) {
    console.error("Analysis request failed:", error);
    throw error;
  }
}
