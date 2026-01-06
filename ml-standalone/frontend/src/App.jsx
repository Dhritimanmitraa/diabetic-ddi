import { useState, useEffect } from 'react'
import DrugChecker from './components/DrugChecker'

function App() {
  const [health, setHealth] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch('http://localhost:8002/health')
        const data = await response.json()
        setHealth(data)
      } catch (error) {
        console.error('Health check failed:', error)
      } finally {
        setLoading(false)
      }
    }

    checkHealth()
    const interval = setInterval(checkHealth, 10000)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
            ML Drug Interaction Predictor
          </h1>
          <p className="text-gray-600">
            ML-powered drug interaction analysis with validation & caching
          </p>
        </header>

        {/* Status Indicators */}
        <div className="max-w-2xl mx-auto mb-6">
          <div className="bg-white rounded-lg shadow-md p-4">
            <h2 className="text-lg font-semibold mb-3">System Status</h2>

            {/* Main Status Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {/* ML Model Status */}
              <div className="flex items-center space-x-2 p-2 rounded-lg bg-purple-50">
                <div className={`w-3 h-3 rounded-full ${health?.ml_model_loaded ? 'bg-purple-500 animate-pulse' : 'bg-gray-400'}`}></div>
                <div className="text-sm">
                  <span className="block font-medium">ML Model</span>
                  <span className={health?.ml_model_loaded ? 'text-purple-600 text-xs' : 'text-gray-500 text-xs'}>
                    {health?.ml_model_loaded ? 'Loaded' : 'Not Loaded'}
                  </span>
                </div>
              </div>

              {/* Validation Status */}
              <div className="flex items-center space-x-2 p-2 rounded-lg bg-green-50">
                <div className={`w-3 h-3 rounded-full ${health?.validation_enabled ? 'bg-green-500' : 'bg-gray-400'}`}></div>
                <div className="text-sm">
                  <span className="block font-medium">Validation</span>
                  <span className={health?.validation_enabled ? 'text-green-600 text-xs' : 'text-gray-500 text-xs'}>
                    {health?.validation_enabled ? 'Enabled' : 'Disabled'}
                  </span>
                </div>
              </div>

              {/* Ollama Status */}
              <div className="flex items-center space-x-2 p-2 rounded-lg bg-blue-50">
                <div className={`w-3 h-3 rounded-full ${health?.ollama_connected ? 'bg-blue-500' : 'bg-yellow-500'}`}></div>
                <div className="text-sm">
                  <span className="block font-medium">Ollama LLM</span>
                  <span className={health?.ollama_connected ? 'text-blue-600 text-xs' : 'text-yellow-600 text-xs'}>
                    {health?.ollama_connected ? 'Connected' : 'Offline'}
                  </span>
                </div>
              </div>

              {/* Cache Status */}
              <div className="flex items-center space-x-2 p-2 rounded-lg bg-gray-50">
                <div className={`w-3 h-3 rounded-full ${health?.cached_predictions > 0 ? 'bg-gray-500' : 'bg-gray-300'}`}></div>
                <div className="text-sm">
                  <span className="block font-medium">Cache</span>
                  <span className="text-gray-600 text-xs">
                    {health?.cached_predictions || 0} items
                  </span>
                </div>
              </div>
            </div>

            {/* ML Model Info */}
            {health?.ml_model_info && health.ml_model_loaded && (
              <div className="mt-3 p-3 bg-purple-50 border border-purple-200 rounded-lg">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium text-purple-800">🤖 ML Model</span>
                  <span className="text-purple-600">v{health.ml_model_info.version || '1.0.0'}</span>
                </div>
                <div className="mt-2 grid grid-cols-3 gap-2 text-xs text-purple-700">
                  {health.ml_model_info.accuracy && (
                    <div className="bg-white p-1 rounded text-center">
                      <div className="font-bold text-lg text-purple-600">{(health.ml_model_info.accuracy * 100).toFixed(1)}%</div>
                      <div className="text-gray-500">Accuracy</div>
                    </div>
                  )}
                  {health.ml_model_info.model_type && (
                    <div className="bg-white p-1 rounded text-center">
                      <div className="font-bold text-purple-600">{health.ml_model_info.model_type}</div>
                      <div className="text-gray-500">Type</div>
                    </div>
                  )}
                  {health.ml_model_info.num_training_samples && (
                    <div className="bg-white p-1 rounded text-center">
                      <div className="font-bold text-purple-600">{(health.ml_model_info.num_training_samples / 1000).toFixed(0)}K</div>
                      <div className="text-gray-500">Samples</div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Validation Stats */}
            {health?.validation_stats && health.validation_enabled && (
              <div className="mt-3 p-3 bg-green-50 border border-green-200 rounded-lg">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium text-green-800">✅ Validation Service</span>
                  <span className="text-green-600">{health.validation_stats.ground_truth_count?.toLocaleString()} ground truth records</span>
                </div>
                {health.validation_stats.total_validated > 0 && (
                  <div className="mt-2 text-xs text-green-700">
                    Validated: {health.validation_stats.total_validated} predictions •
                    Accuracy: {(health.validation_stats.accuracy * 100).toFixed(1)}%
                  </div>
                )}
              </div>
            )}

            {/* LLM Model Info */}
            {health?.model_loaded && (
              <div className="mt-2 text-sm text-gray-600 flex items-center gap-2">
                <span>🧠 LLM:</span>
                <span className="font-semibold">{health.model_loaded}</span>
              </div>
            )}

            {/* Warning Messages */}
            {!health?.ml_model_loaded && (
              <div className="mt-3 p-2 bg-yellow-50 border border-yellow-200 rounded text-sm text-yellow-800">
                ⚠️ ML Model not loaded. Run <code className="bg-yellow-100 px-1 rounded">python train_model.py</code>
              </div>
            )}

            {!health?.ollama_connected && health?.ml_model_loaded && (
              <div className="mt-3 p-2 bg-blue-50 border border-blue-200 rounded text-sm text-blue-800">
                ℹ️ Ollama offline. ML predictions work, but without detailed explanations.
              </div>
            )}
          </div>
        </div>

        {/* Main Content */}
        <div className="max-w-2xl mx-auto">
          <DrugChecker disabled={!health?.ml_model_loaded && !health?.ollama_connected} />
        </div>

        {/* Footer */}
        <footer className="text-center mt-8 text-gray-600 text-sm">
          <p>Powered by XGBoost ML + Ollama LLM • Validated • Cached • Local & Private</p>
        </footer>
      </div>
    </div>
  )
}

export default App
