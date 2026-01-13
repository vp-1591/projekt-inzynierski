export function ResultCard({ result }) {
  const { technique_name, description, confidence_score } = result;
  
  // Simple confidence color logic
  const getConfidenceLevel = (score) => {
    if (score >= 0.9) return { color: '#ef4444', label: 'Wysoka Pewność' }; // Red
    if (score >= 0.7) return { color: '#f59e0b', label: 'Średnia Pewność' }; // Amber
    return { color: '#3b82f6', label: 'Niska Pewność' }; // Blue
  };

  const confidence = confidence_score ? getConfidenceLevel(confidence_score) : null;

  return (
    <div style={{
      backgroundColor: 'var(--bg-card)',
      borderRadius: '12px',
      padding: '1.5rem',
      marginBottom: '1rem',
      border: '1px solid var(--border-subtle)',
      transition: 'transform var(--transition-fast)',
      animation: 'fadeIn 0.5s ease-out'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.5rem' }}>
        <h3 style={{ 
          fontSize: '1.25rem', 
          fontWeight: '600', 
          color: 'var(--text-primary)',
          letterSpacing: '-0.01em'
        }}>
          {technique_name}
        </h3>
        {confidence && (
          <span style={{
            fontSize: '0.75rem',
            padding: '0.25rem 0.75rem',
            borderRadius: '9999px',
            backgroundColor: `${confidence.color}20`, // 20% opacity
            color: confidence.color,
            fontWeight: '500',
            border: `1px solid ${confidence.color}40`
          }}>
            {confidence.label}
          </span>
        )}
      </div>
      
      <p style={{ 
        color: 'var(--text-secondary)', 
        fontSize: '0.95rem',
        lineHeight: '1.6' 
      }}>
        {description}
      </p>
    </div>
  );
}
