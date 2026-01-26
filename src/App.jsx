import { useState } from 'react'
import './index.css'
import { InputSection } from './components/InputSection'
import { ResultCard } from './components/ResultCard'
import { analyzeText } from './services/disinformationDetector'

function App() {
  const [results, setResults] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [error, setError] = useState(null)

  const handleAnalyze = async (text) => {
    setIsAnalyzing(true)
    setError(null)
    try {
      const data = await analyzeText(text)
      setResults(data)
    } catch (error) {
      console.error("Analysis failed", error)
      setError("Nie udało się połączyć z modelem AI. Upewnij się, że Ollama (bielik-local) jest uruchomiona.")
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleReset = () => {
    setResults(null)
    setError(null)
  }

  return (
    <div style={{ 
      maxWidth: '800px', 
      margin: '0 auto', 
      padding: '2rem 1rem',
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      gap: '3rem'
    }}>
      <header style={{ textAlign: 'center', marginTop: '2rem', transition: 'all 0.5s ease' }}>
        <h1 style={{ 
          fontSize: '2.5rem', 
          fontWeight: '300', 
          letterSpacing: '-0.02em',
          background: 'linear-gradient(to right, #fff, #a3a3a3)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          marginBottom: '0.5rem'
        }}>
          Detektor Dezinformacji
        </h1>
        <p style={{ color: 'var(--text-secondary)' }}>
          Analiza treści z wykorzystaniem AI
        </p>
      </header>
      
      <main style={{ flex: 1, width: '100%' }}>
        {error && (
          <div style={{ 
            padding: '1rem', 
            backgroundColor: '#ef444420', 
            color: '#ef4444', 
            borderRadius: '8px', 
            marginBottom: '2rem',
            border: '1px solid #ef444440',
            textAlign: 'center'
          }}>
            {error}
          </div>
        )}
        
        {!results ? (
          <div style={{ animation: 'fadeIn 0.5s ease-out' }}>
            <InputSection onAnalyze={handleAnalyze} isAnalyzing={isAnalyzing} />
          </div>
        ) : (
          <div style={{ animation: 'slideUp 0.5s ease-out' }}>
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center', 
              marginBottom: '2rem' 
            }}>
              <h2 style={{ fontSize: '1.5rem', fontWeight: '500' }}>Wyniki Analizy</h2>
              <button 
                onClick={handleReset}
                style={{
                  color: 'var(--text-secondary)',
                  fontSize: '0.9rem',
                  padding: '0.5rem 1rem',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: '8px',
                  transition: 'all 0.2s ease'
                }}
                onMouseEnter={(e) => {
                  e.target.style.borderColor = 'var(--text-primary)';
                  e.target.style.color = 'var(--text-primary)';
                }}
                onMouseLeave={(e) => {
                  e.target.style.borderColor = 'var(--border-subtle)';
                  e.target.style.color = 'var(--text-secondary)';
                }}
              >
                Nowa Analiza
              </button>
            </div>
            
            {results.length > 0 ? (
              results.map((item) => (
                <ResultCard key={item.id} result={item} />
              ))
            ) : (
              <p style={{ textAlign: 'center', color: 'var(--text-secondary)', marginTop: '2rem' }}>
                Nie wykryto znanych technik manipulacji.
              </p>
            )}
          </div>
        )}
      </main>

      <footer style={{ 
        textAlign: 'center', 
        fontSize: '0.875rem', 
        color: 'var(--text-secondary)',
        paddingBottom: '2rem'
      }}>
        Powered by MockAI Protocol
      </footer>
      
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  )
}

export default App
