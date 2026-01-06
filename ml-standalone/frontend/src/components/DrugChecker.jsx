import { useState } from 'react'

function DrugChecker({ disabled = false }) {
  const [drug1, setDrug1] = useState('')
  const [drug2, setDrug2] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [suggestions1, setSuggestions1] = useState([])
  const [suggestions2, setSuggestions2] = useState([])

  const searchDrugs = async (query, setSuggestions) => {
    if (query.length < 2) {
      setSuggestions([])
      return
    }

    try {
      const response = await fetch(`http://localhost:8002/drugs/search?query=${encodeURIComponent(query)}&limit=5`)
      const data = await response.json()
      setSuggestions(data.results || [])
    } catch (err) {
      console.error('Drug search failed:', err)
    }
  }

  const handleDrug1Change = (e) => {
    const value = e.target.value
    setDrug1(value)
    searchDrugs(value, setSuggestions1)
  }

  const handleDrug2Change = (e) => {
    const value = e.target.value
    setDrug2(value)
    searchDrugs(value, setSuggestions2)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()

    if (!drug1.trim() || !drug2.trim()) {
      setError('Please enter both drug names')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch('http://localhost:8002/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          drug1: drug1.trim(),
          drug2: drug2.trim(),
          include_context: true,
          use_cache: true
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Prediction failed')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const getSeverityColor = (severity) => {
    const colors = {
      none: 'bg-green-100 text-green-800 border-green-300',
      mild: 'bg-yellow-100 text-yellow-800 border-yellow-300',
      moderate: 'bg-orange-100 text-orange-800 border-orange-300',
      severe: 'bg-red-100 text-red-800 border-red-300',
      contraindicated: 'bg-red-200 text-red-900 border-red-400',
      unknown: 'bg-gray-100 text-gray-800 border-gray-300'
    }
    return colors[severity?.toLowerCase()] || colors.unknown
  }

  const getSourceBadge = (source, isCached) => {
    if (isCached) {
      return { bg: 'bg-gray-100', text: 'text-gray-700', label: '📦 Cached' }
    }
    const badges = {
      ml: { bg: 'bg-purple-100', text: 'text-purple-800', label: '🤖 ML Model' },
      llm: { bg: 'bg-blue-100', text: 'text-blue-800', label: '🧠 LLM Only' },
      hybrid: { bg: 'bg-gradient-to-r from-purple-100 to-blue-100', text: 'text-purple-800', label: '🔬 ML + LLM Hybrid' }
    }
    return badges[source] || badges.llm
  }

  const getValidationBadge = (validation) => {
    if (!validation) return null

    if (!validation.is_validated) {
      return { bg: 'bg-gray-100', text: 'text-gray-600', icon: '❓', label: 'Unvalidated' }
    }

    if (validation.is_correct) {
      return { bg: 'bg-green-100', text: 'text-green-700', icon: '✅', label: 'Validated' }
    }

    return { bg: 'bg-orange-100', text: 'text-orange-700', icon: '⚠️', label: 'Conflicts with DB' }
  }

  const ConfidenceBar = ({ confidence, label, calibrated = false }) => (
    <div className="w-full">
      <div className="flex justify-between text-sm mb-1">
        <span className="font-medium text-gray-700">
          {label}
          {calibrated && <span className="text-xs text-purple-600 ml-1">(calibrated)</span>}
        </span>
        <span className="font-bold text-gray-900">{(confidence * 100).toFixed(1)}%</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
        <div
          className={`h-3 rounded-full transition-all duration-500 ${confidence > 0.8 ? 'bg-green-500' :
              confidence > 0.6 ? 'bg-yellow-500' :
                confidence > 0.4 ? 'bg-orange-500' : 'bg-red-500'
            }`}
          style={{ width: `${confidence * 100}%` }}
        />
      </div>
    </div>
  )

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-semibold mb-4">Check Drug Interaction</h2>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Drug 1
          </label>
          <div className="relative">
            <input
              type="text"
              value={drug1}
              onChange={handleDrug1Change}
              placeholder="Enter first drug name"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={disabled}
            />
            {suggestions1.length > 0 && (
              <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg">
                {suggestions1.map((suggestion, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => {
                      setDrug1(suggestion)
                      setSuggestions1([])
                    }}
                    className="w-full text-left px-4 py-2 hover:bg-blue-50"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Drug 2
          </label>
          <div className="relative">
            <input
              type="text"
              value={drug2}
              onChange={handleDrug2Change}
              placeholder="Enter second drug name"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={disabled}
            />
            {suggestions2.length > 0 && (
              <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg">
                {suggestions2.map((suggestion, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => {
                      setDrug2(suggestion)
                      setSuggestions2([])
                    }}
                    className="w-full text-left px-4 py-2 hover:bg-blue-50"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        <button
          type="submit"
          disabled={disabled || loading || !drug1.trim() || !drug2.trim()}
          className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 px-4 rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 disabled:from-gray-400 disabled:to-gray-400 disabled:cursor-not-allowed transition-all shadow-md hover:shadow-lg"
        >
          {loading ? (
            <span className="flex items-center justify-center">
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Analyzing...
            </span>
          ) : '🔬 Analyze with ML + AI'}
        </button>
      </form>

      {error && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-800">
          {error}
        </div>
      )}

      {result && (
        <div className="mt-6 space-y-4">
          <div className="border-t pt-4">
            {/* Header with badges */}
            <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
              <h3 className="text-lg font-semibold">Analysis Results</h3>
              <div className="flex gap-2 flex-wrap">
                {/* Cached badge */}
                {result.is_cached && (
                  <span className="px-2 py-1 rounded-full text-xs font-semibold bg-gray-100 text-gray-600">
                    📦 Cached
                  </span>
                )}
                {/* Source badge */}
                <span className={`px-2 py-1 rounded-full text-xs font-semibold ${getSourceBadge(result.prediction_source, false).bg} ${getSourceBadge(result.prediction_source, false).text}`}>
                  {getSourceBadge(result.prediction_source, false).label}
                </span>
                {/* Validation badge */}
                {result.validation && (
                  <span className={`px-2 py-1 rounded-full text-xs font-semibold ${getValidationBadge(result.validation).bg} ${getValidationBadge(result.validation).text}`}>
                    {getValidationBadge(result.validation).icon} {getValidationBadge(result.validation).label}
                  </span>
                )}
              </div>
            </div>

            <div className="space-y-4">
              {/* Interaction Status */}
              <div className={`p-4 rounded-lg border-2 ${result.prediction.has_interaction ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'}`}>
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-lg">
                    {result.prediction.has_interaction ? '⚠️ Interaction Detected' : '✅ No Significant Interaction'}
                  </span>
                  <span className={`px-4 py-2 rounded-full text-sm font-bold border ${getSeverityColor(result.prediction.severity)}`}>
                    {result.prediction.severity?.toUpperCase() || 'UNKNOWN'}
                  </span>
                </div>
              </div>

              {/* Validation Result */}
              {result.validation && (
                <div className={`p-4 rounded-lg border ${result.validation.is_validated
                    ? (result.validation.is_correct ? 'bg-green-50 border-green-200' : 'bg-orange-50 border-orange-200')
                    : 'bg-gray-50 border-gray-200'
                  }`}>
                  <div className="flex items-center gap-2 mb-2">
                    <span className="font-bold">
                      {result.validation.is_validated
                        ? (result.validation.is_correct ? '✅ Validated Against Database' : '⚠️ Prediction Conflicts with Database')
                        : '❓ No Database Record Found'}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600">{result.validation.message}</p>
                  {result.validation.ground_truth_severity && (
                    <p className="text-sm mt-1">
                      <span className="font-medium">Database severity:</span> {result.validation.ground_truth_severity}
                    </p>
                  )}
                  <div className="mt-2">
                    <ConfidenceBar
                      confidence={result.validation.calibrated_confidence}
                      label="Calibrated Confidence"
                      calibrated={true}
                    />
                  </div>
                </div>
              )}

              {/* ML Model Prediction */}
              {result.ml_prediction && (
                <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg">
                  <div className="flex items-center gap-2 mb-3">
                    <span className="text-purple-600 font-bold">🤖 ML Model Prediction</span>
                    {result.ml_prediction.model_version && (
                      <span className="text-xs bg-purple-200 text-purple-700 px-2 py-0.5 rounded">
                        v{result.ml_prediction.model_version}
                      </span>
                    )}
                  </div>
                  <ConfidenceBar
                    confidence={result.ml_prediction.confidence}
                    label="Model Confidence"
                  />
                  {result.ml_prediction.probabilities && (
                    <div className="mt-3 grid grid-cols-4 gap-2 text-xs">
                      {Object.entries(result.ml_prediction.probabilities).map(([severity, prob]) => (
                        <div key={severity} className="text-center p-1 bg-white rounded">
                          <div className="font-medium capitalize text-gray-700">{severity}</div>
                          <div className="text-purple-600 font-bold">{(prob * 100).toFixed(0)}%</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Explanation */}
              <div>
                <span className="font-medium block mb-1">📋 Explanation:</span>
                <p className="text-gray-700 bg-gray-50 p-3 rounded-lg">{result.prediction.explanation}</p>
              </div>

              {/* Mechanism */}
              {result.prediction.mechanism && result.prediction.mechanism !== 'Unknown' && (
                <div>
                  <span className="font-medium block mb-1">🔬 Mechanism:</span>
                  <p className="text-gray-700 bg-gray-50 p-3 rounded-lg">{result.prediction.mechanism}</p>
                </div>
              )}

              {/* Recommendations */}
              {result.prediction.recommendations && result.prediction.recommendations.length > 0 && (
                <div>
                  <span className="font-medium block mb-1">💡 Recommendations:</span>
                  <ul className="list-disc list-inside space-y-1 text-gray-700 bg-gray-50 p-3 rounded-lg">
                    {result.prediction.recommendations.map((rec, idx) => (
                      <li key={idx}>{rec}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* TWOSIDES Context */}
              {result.twosides_context && result.twosides_context.known_interaction && (
                <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                  <span className="font-medium block mb-1">📊 Database Context (TWOSIDES):</span>
                  <p className="text-sm text-gray-700">
                    {result.twosides_context.interaction_count} interaction(s) found in database
                    {result.twosides_context.side_effects.length > 0 && (
                      <span className="block mt-1">
                        Side effects: {result.twosides_context.side_effects.slice(0, 5).join(', ')}
                        {result.twosides_context.side_effects.length > 5 && '...'}
                      </span>
                    )}
                  </p>
                </div>
              )}

              {/* Footer */}
              <div className="text-xs text-gray-500 pt-2 border-t flex justify-between items-center flex-wrap gap-2">
                <span>
                  {result.llm_model && `LLM: ${result.llm_model}`}
                  {result.ml_prediction && result.llm_model && ' • '}
                  {result.ml_prediction && `ML: v${result.ml_prediction.model_version || '1.0'}`}
                  {result.is_cached && ' • 📦 From Cache'}
                </span>
                <span>⏱️ {result.processing_time_ms}ms</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default DrugChecker
