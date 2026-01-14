import React from 'react'
import { AlertTriangle, RefreshCw, Home } from 'lucide-react'

/**
 * ErrorBoundary - Catches JavaScript errors in child components
 * 
 * This component prevents the entire app from crashing when a component
 * encounters an unhandled error. Instead, it shows a user-friendly error
 * message with recovery options.
 * 
 * @example
 * <ErrorBoundary fallback={<CustomErrorUI />}>
 *   <MyComponent />
 * </ErrorBoundary>
 */
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { 
      hasError: false, 
      error: null,
      errorInfo: null,
    }
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render shows the fallback UI
    return { hasError: true, error }
  }

  componentDidCatch(error, errorInfo) {
    // Log the error to console for debugging
    console.error('ErrorBoundary caught an error:', error, errorInfo)
    
    this.setState({
      error,
      errorInfo,
    })
    
    // You could also log to an error reporting service here
    // logErrorToService(error, errorInfo)
  }

  handleReload = () => {
    window.location.reload()
  }

  handleGoHome = () => {
    window.location.href = '/'
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null, errorInfo: null })
  }

  render() {
    if (this.state.hasError) {
      // Custom fallback UI if provided
      if (this.props.fallback) {
        return this.props.fallback
      }

      // Default error UI
      return (
        <div 
          className="min-h-[400px] flex items-center justify-center p-8"
          role="alert"
          aria-live="assertive"
        >
          <div className="max-w-md w-full glass rounded-3xl p-8 text-center">
            <div className="w-16 h-16 rounded-2xl bg-red-500/10 flex items-center justify-center mx-auto mb-6">
              <AlertTriangle className="w-8 h-8 text-red-400" aria-hidden="true" />
            </div>
            
            <h2 className="text-xl font-semibold text-white mb-2">
              Something went wrong
            </h2>
            
            <p className="text-slate-400 mb-6">
              We encountered an unexpected error. Please try again or return to the home page.
            </p>

            {/* Show error details in development */}
            {process.env.NODE_ENV === 'development' && this.state.error && (
              <details className="mb-6 text-left">
                <summary className="text-slate-500 text-sm cursor-pointer hover:text-slate-400 transition-colors">
                  Error details
                </summary>
                <pre className="mt-2 p-4 bg-slate-800/50 rounded-xl text-xs text-red-400 overflow-auto max-h-40">
                  {this.state.error.toString()}
                  {this.state.errorInfo?.componentStack}
                </pre>
              </details>
            )}

            <div className="flex flex-col sm:flex-row gap-3">
              <button
                onClick={this.handleRetry}
                className="flex-1 py-3 px-4 bg-medical-500 hover:bg-medical-400 text-white rounded-xl transition-colors flex items-center justify-center gap-2 focus:outline-none focus:ring-2 focus:ring-medical-300"
                aria-label="Try again"
              >
                <RefreshCw className="w-4 h-4" aria-hidden="true" />
                Try Again
              </button>
              
              <button
                onClick={this.handleGoHome}
                className="flex-1 py-3 px-4 bg-slate-700 hover:bg-slate-600 text-white rounded-xl transition-colors flex items-center justify-center gap-2 focus:outline-none focus:ring-2 focus:ring-slate-500"
                aria-label="Go to home page"
              >
                <Home className="w-4 h-4" aria-hidden="true" />
                Go Home
              </button>
            </div>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}

/**
 * RouteErrorBoundary - A simpler error boundary for route-level errors
 * Shows a full-page error state with navigation options
 */
class RouteErrorBoundary extends ErrorBoundary {
  render() {
    if (this.state.hasError) {
      return (
        <div 
          className="min-h-screen animated-gradient grid-bg flex items-center justify-center p-8"
          role="alert"
          aria-live="assertive"
        >
          <div className="max-w-lg w-full glass rounded-3xl p-8 text-center">
            <div className="w-20 h-20 rounded-2xl bg-red-500/10 flex items-center justify-center mx-auto mb-6">
              <AlertTriangle className="w-10 h-10 text-red-400" aria-hidden="true" />
            </div>
            
            <h1 className="text-2xl font-bold text-white mb-3">
              Oops! Something went wrong
            </h1>
            
            <p className="text-slate-400 mb-8 leading-relaxed">
              We're sorry, but this page encountered an error. 
              Our team has been notified and is working on a fix.
            </p>

            {/* Show error in development */}
            {process.env.NODE_ENV === 'development' && this.state.error && (
              <div className="mb-6 p-4 bg-slate-800/50 rounded-xl text-left">
                <p className="text-red-400 text-sm font-mono break-all">
                  {this.state.error.toString()}
                </p>
              </div>
            )}

            <div className="flex flex-col sm:flex-row gap-4">
              <button
                onClick={this.handleReload}
                className="flex-1 py-4 px-6 bg-gradient-to-r from-medical-500 to-medical-600 hover:from-medical-400 hover:to-medical-500 text-white font-semibold rounded-xl shadow-lg shadow-medical-500/25 transition-all flex items-center justify-center gap-2 focus:outline-none focus:ring-2 focus:ring-medical-300"
                aria-label="Reload this page"
              >
                <RefreshCw className="w-5 h-5" aria-hidden="true" />
                Reload Page
              </button>
              
              <button
                onClick={this.handleGoHome}
                className="flex-1 py-4 px-6 bg-slate-700 hover:bg-slate-600 text-white font-semibold rounded-xl transition-colors flex items-center justify-center gap-2 focus:outline-none focus:ring-2 focus:ring-slate-500"
                aria-label="Return to home page"
              >
                <Home className="w-5 h-5" aria-hidden="true" />
                Back to Home
              </button>
            </div>

            <p className="mt-6 text-slate-500 text-sm">
              If this problem persists, please contact support.
            </p>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}

/**
 * withErrorBoundary - HOC to wrap a component with an ErrorBoundary
 * 
 * @param {React.Component} Component - Component to wrap
 * @param {Object} props - Props to pass to ErrorBoundary
 * @returns {React.Component} Wrapped component
 * 
 * @example
 * const SafeComponent = withErrorBoundary(MyComponent)
 */
function withErrorBoundary(Component, errorBoundaryProps = {}) {
  return function WrappedComponent(props) {
    return (
      <ErrorBoundary {...errorBoundaryProps}>
        <Component {...props} />
      </ErrorBoundary>
    )
  }
}

export { ErrorBoundary, RouteErrorBoundary, withErrorBoundary }
export default ErrorBoundary
