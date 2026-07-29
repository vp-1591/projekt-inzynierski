import { Component } from 'react';

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Unhandled error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary-fallback">
          <h2>Coś poszło nie tak</h2>
          <p>{this.state.error?.message || 'Nieoczekiwany błąd aplikacji'}</p>
          <button onClick={() => window.location.reload()}>Odśwież stronę</button>
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;