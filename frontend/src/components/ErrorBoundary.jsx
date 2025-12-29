import React from 'react'
import { AlertTriangle, RefreshCw, Home } from 'lucide-react'
import { useNavigate } from 'react-router-dom'

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null, errorInfo: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true }
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo)
    this.setState({
      error,
      errorInfo
    })
    
    // Log to error tracking service (e.g., Sentry) in production
    if (import.meta.env.PROD) {
      // TODO: Add error tracking service
      // logErrorToService(error, errorInfo)
    }
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null })
  }

  render() {
    if (this.state.hasError) {
      return (
        <ErrorFallback
          error={this.state.error}
          errorInfo={this.state.errorInfo}
          onReset={this.handleReset}
        />
      )
    }

    return this.props.children
  }
}

function ErrorFallback({ error, errorInfo, onReset }) {
  const navigate = useNavigate()

  return (
    <div className="min-h-screen animated-gradient flex items-center justify-center px-4">
      <div className="glass rounded-3xl p-8 max-w-2xl w-full text-center">
        <div className="flex justify-center mb-6">
          <div className="w-20 h-20 rounded-full bg-red-500/20 flex items-center justify-center">
            <AlertTriangle className="w-10 h-10 text-red-400" />
          </div>
        </div>
        
        <h1 className="text-3xl font-bold text-white mb-4">
          Something went wrong
        </h1>
        
        <p className="text-slate-400 mb-6">
          We're sorry, but something unexpected happened. Please try refreshing the page or return to the home page.
        </p>

        {import.meta.env.DEV && error && (
          <div className="mb-6 p-4 bg-slate-800/50 rounded-lg text-left">
            <p className="text-red-400 font-mono text-sm mb-2">
              {error.toString()}
            </p>
            {errorInfo && (
              <details className="text-xs text-slate-500">
                <summary className="cursor-pointer hover:text-slate-400 mb-2">
                  Stack Trace
                </summary>
                <pre className="overflow-auto max-h-40 text-slate-400">
                  {errorInfo.componentStack}
                </pre>
              </details>
            )}
          </div>
        )}

        <div className="flex gap-4 justify-center">
          <button
            onClick={onReset}
            className="px-6 py-3 bg-medical-500 hover:bg-medical-600 text-white rounded-xl transition-colors flex items-center gap-2"
          >
            <RefreshCw className="w-5 h-5" />
            Try Again
          </button>
          
          <button
            onClick={() => {
              onReset()
              navigate('/')
            }}
            className="px-6 py-3 bg-slate-700 hover:bg-slate-600 text-white rounded-xl transition-colors flex items-center gap-2"
          >
            <Home className="w-5 h-5" />
            Go Home
          </button>
        </div>
      </div>
    </div>
  )
}

export default ErrorBoundary







