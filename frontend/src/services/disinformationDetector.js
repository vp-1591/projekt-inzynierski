/**
 * Service to handle disinformation detection analysis using local Ollama instance.
 * Returns raw data from the backend; mapping to localized strings is done in the UI layer.
 */

const BACKEND_URL = 'http://localhost:8000/analyze';

/**
 * Analyzes the provided text for disinformation techniques.
 * @param {string} text - The article text to analyze.
 * @returns {Promise<{ tags: string[], reasoning: string }>} - Raw tags and reasoning string.
 */
export async function analyzeText(text) {
  const response = await fetch(BACKEND_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    throw new Error(`Backend error: ${response.statusText}`);
  }

  const data = await response.json();

  return {
    // Use a strict array check — an LLM can hallucinate a string/object, which is truthy
    // and would bypass the `|| []` fallback, crashing .map() in the UI layer.
    tags: Array.isArray(data.discovered_techniques) ? data.discovered_techniques : [],
    // null signals to the UI that reasoning is absent; the UI will render a fallback.
    reasoning: data.reasoning || null,
  };
}
