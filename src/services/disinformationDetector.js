/**
 * Service to handle disinformation detection analysis.
 * Currently uses a mock implementation to simulate AI processing.
 */

// Mock data to simulate varied results
const MOCK_TECHNIQUES = [
  {
    id: 'tech-1',
    technique_name: 'Ad Hominem',
    description: 'Atakowanie osoby wyrażającej pogląd zamiast merytorycznego odniesienia się do argumentu.',
    confidence_score: 0.92,
    raw_data: { original_label: 'ad_hominem', intensity: 'high' }
  },
  {
    id: 'tech-2',
    technique_name: 'Fałszywy Dylemat',
    description: 'Przedstawianie sytuacji jako wyboru tylko między dwiema opcjami, podczas gdy istnieje ich więcej.',
    confidence_score: 0.85,
    raw_data: { original_label: 'false_dilemma' }
  },
  {
    id: 'tech-3',
    technique_name: 'Język Emocjonalny',
    description: 'Używanie silnie nacechowanych słów w celu wywołania emocji i zaburzenia racjonalnej oceny.',
    confidence_score: 0.88,
    raw_data: { original_label: 'emotional_language', keywords: ['szokujące', 'skandal'] }
  },
  {
    id: 'tech-4',
    technique_name: 'Co słychać u...',
    description: 'Odwracanie uwagi od tematu poprzez oskarżanie oponenta o podobne lub gorsze przewinienia.',
    confidence_score: 0.75,
    raw_data: { original_label: 'whataboutism' }
  }
];

/**
 * Analyzes the provided text for disinformation techniques.
 * @param {string} text - The article text to analyze.
 * @returns {Promise<Array>} - A promise resolving to an array of detected techniques.
 */
export async function analyzeText(text) {
  return new Promise((resolve) => {
    // Simulate network delay (1.5 - 3 seconds)
    const delay = 1500 + Math.random() * 1500;
    
    setTimeout(() => {
      // Logic to return random subset of techniques for variety
      // In real implementation, this would call the Ollama endpoint
      
      const shuffled = [...MOCK_TECHNIQUES].sort(() => 0.5 - Math.random());
      const selected = shuffled.slice(0, Math.floor(Math.random() * 3) + 1); // Return 1 to 3 items
      
      resolve(selected);
    }, delay);
  });
}
