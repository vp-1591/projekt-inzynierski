export function ResultCard({ result }) {
  const { technique_name, description } = result;

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
