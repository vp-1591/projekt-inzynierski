import { useState, useEffect, useRef } from 'react'
import './index.css'
import { InputSection } from './components/InputSection'
import { analyzeText } from './services/disinformationDetector'

function App() {
  const [results, setResults] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [error, setError] = useState(null)
  const [showExpertMode, setShowExpertMode] = useState(false)
  const [trainingStatus, setTrainingStatus] = useState({
    status: 'idle',
    training_progress: 0,
    evaluation_progress: 0,
    baseline_f1_non_empty: 0,
    baseline_exact_match: 0,
    new_f1_non_empty: 0,
    new_exact_match: 0
  });

  const pollInterval = useRef(null);

  useEffect(() => {
    if (showExpertMode) {
      pollInterval.current = setInterval(async () => {
        try {
          const response = await fetch('http://localhost:8000/training/status');
          const data = await response.json();
          setTrainingStatus(data);
        } catch (err) {
          console.error("Failed to poll status", err);
        }
      }, 2000);
    } else {
      if (pollInterval.current) clearInterval(pollInterval.current);
    }
    return () => {
      if (pollInterval.current) clearInterval(pollInterval.current);
    };
  }, [showExpertMode]);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/training/upload', {
        method: 'POST',
        body: formData,
      });
      if (response.ok) {
        alert("Pomyślnie rozpoczęto trening!");
      } else {
        const errorData = await response.json();
        alert("Błąd: " + (errorData.detail || "Nieznany błąd"));
      }
    } catch (err) {
      alert("Błąd połączenia: " + err.message);
    }
  };

  const handlePromote = async () => {
    if (trainingStatus.new_f1 < trainingStatus.baseline_f1) {
      if (!window.confirm("Ostrzeżenie: Nowy model ma niższe F1 score niż bazowy. Czy na pewno chcesz go wdrożyć?")) {
        return;
      }
    }
    
    try {
      const response = await fetch('http://localhost:8000/training/promote', { method: 'POST' });
      if (response.ok) {
        // Success feedback is handled by button state "Wdrażanie..." -> "Wdróż model" transition
      }
    } catch (err) {
      alert("Błąd awansu modelu: " + err.message);
    }
  };

  const handleAnalyze = async (text) => {
    setIsAnalyzing(true);
    setError(null);
    try {
      const data = await analyzeText(text);
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="app-container">
      <aside className={`expert-sidebar ${showExpertMode ? 'visible' : ''}`}>
        <div className="sidebar-header">
          <h2>Panel Ekspercki</h2>
        </div>
        
        <div className="sidebar-content">
          <div className="field-group">
            <label>Dataset (JSONL)</label>
            <div className="file-input-wrapper">
              <input type="file" onChange={handleFileUpload} />
            </div>
          </div>

          <div className="progress-section">
            <div className="progress-info">
              <span>Postęp treningu</span>
              <span>{trainingStatus.training_progress}%</span>
            </div>
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${trainingStatus.training_progress}%` }}
              ></div>
            </div>
          </div>

          <div className="progress-section">
            <div className="progress-info">
              <span>Ewaluacja</span>
              <span>{trainingStatus.evaluation_progress}%</span>
            </div>
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${trainingStatus.evaluation_progress}%` }}
              ></div>
            </div>
          </div>

          <div className="stats-table">
            <div className="stats-header">
              <span className="col-metric">Metric</span>
              <span className="col-val">Baseline</span>
              <span className="col-val">New Model</span>
            </div>
            
            <div className="stats-row">
              <span className="metric-label">F1 (non-empty)</span>
              <span className="stat-value">{trainingStatus.baseline_f1_non_empty.toFixed(4)}</span>
              <span className={`stat-value ${trainingStatus.new_f1_non_empty >= trainingStatus.baseline_f1_non_empty ? 'positive' : ''}`}>
                {trainingStatus.new_f1_non_empty.toFixed(4)}
              </span>
            </div>

            <div className="stats-row">
              <span className="metric-label">Exact Match (all docs)</span>
              <span className="stat-value">{trainingStatus.baseline_exact_match.toFixed(4)}</span>
              <span className={`stat-value ${trainingStatus.new_exact_match >= trainingStatus.baseline_exact_match ? 'positive' : ''}`}>
                {trainingStatus.new_exact_match.toFixed(4)}
              </span>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <button
              onClick={handlePromote}
              disabled={trainingStatus.status !== 'ready_to_promote'}
              className="promote-button"
            >
              {trainingStatus.status === 'deploying' ? 'Wdrażanie...' : 'Wdróż model'}
            </button>
            
            {/* Status Indicator Circle */}
            <div 
              title={`Status: ${trainingStatus.status}`}
              style={{
                width: '16px', 
                height: '16px', 
                borderRadius: '50%',
                backgroundColor: (() => {
                  switch (trainingStatus.status) {
                    case 'deploying': return '#fbbf24'; // Yellow
                    case 'deployment_success': return '#10b981'; // Green
                    case 'deployment_error': return '#ef4444'; // Red
                    case 'ready_to_promote': return '#3b82f6'; // Blue
                    default: return '#9ca3af'; // Gray
                  }
                })(),
                transition: 'background-color 0.3s ease'
              }}
            ></div>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="main-header">
          <div className="brand-text">
            <h1>Detektor Dezinformacji</h1>
            <p>Analiza treści z wykorzystaniem AI</p>
          </div>
          
          <div className="expert-toggle">
            <span>Tryb ekspercki</span>
            <button 
              onClick={() => setShowExpertMode(!showExpertMode)}
              className={`toggle-switch ${showExpertMode ? 'on' : 'off'}`}
            >
              <div className="handle" />
            </button>
          </div>
        </header>

        <div className="content-wrapper">
          <section className="analyze-section">
            <InputSection onAnalyze={handleAnalyze} isLoading={isAnalyzing} />
          </section>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}

          <section className="results-container">
            {results && (
              <div className="analysis-results">
                <div className="labels-list">
                  {results.techniques.map((tech, index) => (
                    <span 
                      key={index} 
                      className="tech-badge has-tooltip" 
                      data-title={tech.description}
                    >
                      {tech.name}
                    </span>
                  ))}
                </div>
                
                <div className="reasoning-block">
                  <p className="reasoning-text">{results.reasoning}</p>
                </div>
              </div>
            )}
            
            {!isAnalyzing && !results && !error && (
              <div className="placeholder">
                <p>Wprowadź tekst do analizy...</p>
              </div>
            )}
            
            {results && results.techniques.length === 0 && !isAnalyzing && (
              <div className="placeholder">
                <p>Nie wykryto technik manipulacji.</p>
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  )
}

export default App
