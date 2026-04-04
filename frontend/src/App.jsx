import { useState, useEffect, useMemo } from 'react'
import './index.css'
import { InputSection } from './components/InputSection'
import { analyzeText } from './services/disinformationDetector'
import { useLanguage } from './contexts/LanguageContext'

function App() {
  const { language, setLanguage, t } = useLanguage();

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

  useEffect(() => {
    let ws = null;
    if (showExpertMode) {
      ws = new WebSocket('ws://localhost:8000/ws/training/status');

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setTrainingStatus(data);
      };

      ws.onerror = (err) => {
        console.error("WebSocket error:", err);
      };

      ws.onclose = () => {
        console.log("WebSocket connection closed");
      };
    }

    return () => {
      if (ws) ws.close();
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
        alert(t.uploadSuccess);
      } else {
        const errorData = await response.json();
        alert(t.uploadError + (errorData.detail || '?'));
      }
    } catch (err) {
      alert(t.uploadConnError + err.message);
    }
  };

  const handlePromote = async () => {
    if (trainingStatus.new_f1_non_empty < trainingStatus.baseline_f1_non_empty) {
      if (!window.confirm(t.promoteWarning)) return;
    }

    try {
      const response = await fetch('http://localhost:8000/training/promote', { method: 'POST' });
      // fetch only throws on network failure; a 4xx/5xx must be checked explicitly.
      if (!response.ok) throw new Error(`Promotion failed: ${response.statusText}`);
    } catch (err) {
      alert(t.promoteConnError + err.message);
    }
  };

  const handleAnalyze = async (text) => {
    setIsAnalyzing(true);
    setError(null);
    try {
      const data = await analyzeText(text);
      // Store raw tags & reasoning. Localized mapping happens in a useMemo below,
      // so switching language re-runs the map without discarding the user's results.
      setResults({ rawTags: data.tags, reasoning: data.reasoning });
    } catch (err) {
      setError(err.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getPromoteButtonLabel = () => {
    switch (trainingStatus.status) {
      case 'deploying':          return t.deployingButton;
      case 'deployment_success': return t.deployedButton;
      case 'deployment_error':   return t.deployErrorButton;
      default:                   return t.promoteButton;
    }
  };

  // Derived from raw tags + current locale. Re-runs on language change so the UI
  // translates instantly without clearing state or re-fetching from the backend.
  const techniques = useMemo(() => {
    if (!results?.rawTags) return [];
    return results.rawTags.map(tag => {
      // Optional chaining guards against a locale file missing the techniques object.
      const info = t.techniques?.[tag] || { name: tag, description: t.unknownTechnique };
      return { name: info.name, description: info.description };
    });
  }, [results, t]);

  return (
    <div className="app-container">
      <aside className={`expert-sidebar ${showExpertMode ? 'visible' : ''}`}>
        <div className="sidebar-header">
          <h2>{t.expertPanelTitle}</h2>
        </div>

        <div className="sidebar-content">
          <div className="field-group">
            <label>{t.datasetLabel}</label>
            <div className="file-input-wrapper">
              <input type="file" onChange={handleFileUpload} />
            </div>
          </div>

          <div className="progress-section">
            <div className="progress-info">
              <span>{t.trainingProgress}</span>
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
              <span>{t.evaluationProgress}</span>
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
              <span className="col-metric">{t.metricCol}</span>
              <span className="col-val">{t.baselineCol}</span>
              <span className="col-val">{t.newModelCol}</span>
            </div>

            <div className="stats-row">
              <span className="metric-label">F1 (non-empty)</span>
              <span className="stat-value">{trainingStatus.baseline_f1_non_empty.toFixed(4)}</span>
              <span className={`stat-value ${trainingStatus.new_f1_non_empty > trainingStatus.baseline_f1_non_empty ? 'positive' : trainingStatus.new_f1_non_empty < trainingStatus.baseline_f1_non_empty ? 'negative' : ''}`}>
                {trainingStatus.new_f1_non_empty.toFixed(4)}
              </span>
            </div>

            <div className="stats-row">
              <span className="metric-label">Exact Match (all docs)</span>
              <span className="stat-value">{trainingStatus.baseline_exact_match.toFixed(4)}</span>
              <span className={`stat-value ${trainingStatus.new_exact_match > trainingStatus.baseline_exact_match ? 'positive' : trainingStatus.new_exact_match < trainingStatus.baseline_exact_match ? 'negative' : ''}`}>
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
              {getPromoteButtonLabel()}
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
                    case 'deploying':          return '#fbbf24';
                    case 'deployment_success': return '#10b981';
                    case 'deployment_error':   return '#ef4444';
                    case 'ready_to_promote':   return '#3b82f6';
                    default:                   return '#9ca3af';
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
            <h1>{t.appTitle}</h1>
            <p>{t.appSubtitle}</p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
            {/* Language Switcher */}
            <div className="lang-switcher">
              <button
                onClick={() => {
                  setLanguage(language === 'ua' ? 'pl' : 'ua');
                  // Raw tags stay in state; the 'techniques' useMemo re-runs on language change.
                }}
                className="lang-button"
                title={language === 'ua' ? 'Przełącz na polski' : 'Перемкнути на українську'}
              >
                {t.langSwitchLabel}
              </button>
            </div>

            {/* Expert Mode Toggle */}
            <div className="expert-toggle">
              <span>{t.expertModeLabel}</span>
              <button
                onClick={() => setShowExpertMode(!showExpertMode)}
                className={`toggle-switch ${showExpertMode ? 'on' : 'off'}`}
              >
                <div className="handle" />
              </button>
            </div>
          </div>
        </header>

        <div className="content-wrapper">
          <section className="analyze-section">
            <InputSection onAnalyze={handleAnalyze} isAnalyzing={isAnalyzing} />
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
                  {techniques.map((tech, index) => (
                    <span
                      key={index}
                      className="tech-badge has-tooltip"
                      data-title={tech.description}
                    >
                      {tech.name}
                    </span>
                  ))}
                </div>

                {language === 'ua' && (
                  <div className="disclaimer">
                    <p>{t.disclaimer}</p>
                  </div>
                )}

                {results.reasoning && (
                  <div className="reasoning-block">
                    <p className="reasoning-text">{results.reasoning}</p>
                  </div>
                )}
              </div>
            )}

            {!isAnalyzing && !results && !error && (
              <div className="placeholder">
                <p>{t.inputPrompt}</p>
              </div>
            )}

            {results && techniques.length === 0 && !isAnalyzing && (
              <div className="placeholder">
                <p>{t.noTechniques}</p>
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  )
}

export default App
