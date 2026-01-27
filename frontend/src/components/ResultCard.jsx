import React from 'react'

export function ResultCard({ result }) {
  const techniqueNames = {
    'REFERENCE_ERROR': 'Błąd Źródłowy',
    'EMOTIONAL_LANGUAGE': 'Język Emocjonalny',
    'CHERRY_PICKING': 'Wybiórczość Danych',
    'EXAGGERATION': 'Wyolbrzymienie',
    'FALSE_DILEMMA': 'Fałszywy Dylemat',
    'AD_HOMINEM': 'Atak Osobisty',
  };

  const name = techniqueNames[result.name] || result.name;

  return (
    <div className="result-card">
      <div className="card-header">
        <h3>{name}</h3>
      </div>
      <div className="card-body">
        <p className="reasoning">{result.reasoning || "Brak szczegółowego uzasadnienia."}</p>
        {result.examples && result.examples.length > 0 && (
          <div className="examples">
            {result.examples.map((ex, i) => (
              <div key={i} className="example-item">"{ex}"</div>
            ))}
          </div>
        )}
      </div>

      <style>{`
        .result-card {
          background: var(--bg-secondary);
          border: 1px solid var(--border-subtle);
          border-radius: 8px;
          padding: 1.5rem;
          margin-bottom: 1rem;
          animation: slideUp 0.4s ease-out;
        }

        .card-header h3 {
          font-size: 1.1rem;
          font-weight: 500;
          margin-bottom: 0.75rem;
        }

        .reasoning {
          color: var(--text-secondary);
          font-size: 0.9rem;
          line-height: 1.5;
          margin-bottom: 0.5rem;
          font-style: italic;
        }

        .example-item {
          font-size: 0.85rem;
          color: var(--text-secondary);
          border-left: 1px solid var(--border-subtle);
          padding-left: 1rem;
          margin-top: 1rem;
          font-style: italic;
        }

        @keyframes slideUp {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  )
}
