import { useState } from 'react';

export function InputSection({ onAnalyze, isAnalyzing }) {
  const [text, setText] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (text.trim()) {
      onAnalyze(text);
    }
  };

  return (
    <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div style={{ position: 'relative' }}>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Wklej artykuł do analizy..."
          disabled={isAnalyzing}
          style={{
            width: '100%',
            minHeight: '300px',
            padding: '1.5rem',
            backgroundColor: 'var(--bg-secondary)',
            color: 'var(--text-primary)',
            border: '1px solid transparent',
            borderRadius: '16px',
            resize: 'vertical',
            outline: 'none',
            fontSize: '1rem',
            lineHeight: '1.6',
            transition: 'border-color var(--transition-fast), box-shadow var(--transition-fast)',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)'
          }}
          onFocus={(e) => {
            e.target.style.borderColor = 'var(--accent-blue)';
            e.target.style.boxShadow = '0 0 0 2px rgba(59, 130, 246, 0.1)';
          }}
          onBlur={(e) => {
            e.target.style.borderColor = 'transparent';
            e.target.style.boxShadow = '0 4px 6px -1px rgba(0, 0, 0, 0.1)';
          }}
        />
      </div>

      <button
        type="submit"
        disabled={!text.trim() || isAnalyzing}
        style={{
          alignSelf: 'center',
          padding: '1rem 3rem',
          backgroundColor: text.trim() ? 'var(--accent-blue)' : 'var(--bg-card)',
          color: 'white',
          borderRadius: '9999px',
          fontWeight: '600',
          fontSize: '1rem',
          transition: 'transform var(--transition-fast), opacity var(--transition-fast)',
          opacity: isAnalyzing ? 0.7 : 1,
          cursor: text.trim() && !isAnalyzing ? 'pointer' : 'not-allowed',
          transform: text.trim() && !isAnalyzing ? 'translateY(0)' : 'translateY(0)',
        }}
        onMouseEnter={(e) => {
          if (text.trim() && !isAnalyzing) e.target.style.transform = 'translateY(-2px)';
        }}
        onMouseLeave={(e) => {
          if (text.trim() && !isAnalyzing) e.target.style.transform = 'translateY(0)';
        }}
      >
        {isAnalyzing ? 'Analizowanie...' : 'Analizuj Tekst'}
      </button>
    </form>
  );
}
