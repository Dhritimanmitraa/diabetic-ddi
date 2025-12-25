import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Brain, BarChart3, TrendingUp, Zap, Clock, 
  CheckCircle2, XCircle, RefreshCw, Award
} from 'lucide-react'

/**
 * ModelDashboard Component
 * 
 * Displays ML model performance metrics, optimization comparison results,
 * and feature importance visualization.
 */
function ModelDashboard() {
  const [modelInfo, setModelInfo] = useState(null)
  const [comparison, setComparison] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    
    try {
      // Fetch model info
      const infoRes = await fetch('http://localhost:8000/ml/model-info')
      const infoData = await infoRes.json()
      setModelInfo(infoData)

      // Fetch comparison data
      const compRes = await fetch('http://localhost:8000/ml/comparison')
      const compData = await compRes.json()
      setComparison(compData)
    } catch (err) {
      setError('Failed to load model data')
      console.error('Error fetching model data:', err)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="glass rounded-3xl p-8 max-w-4xl mx-auto">
        <div className="flex items-center justify-center py-12">
          <div className="spinner"></div>
          <span className="ml-4 text-slate-400">Loading model data...</span>
        </div>
      </div>
    )
  }

  if (error || modelInfo?.status === 'not_loaded') {
    return (
      <div className="glass rounded-3xl p-8 max-w-4xl mx-auto">
        <div className="text-center py-12">
          <Brain className="w-16 h-16 text-slate-600 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-white mb-2">
            ML Models Not Yet Trained
          </h3>
          <p className="text-slate-400 mb-6">
            Train the machine learning models to see predictions and metrics.
          </p>
          <code className="px-4 py-2 bg-slate-800 rounded-lg text-medical-400 text-sm">
            python -m scripts.train_models
          </code>
        </div>
      </div>
    )
  }

  const models = modelInfo?.models?.model_metrics || {}

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-purple-500/10 border border-purple-500/20 mb-4">
          <Brain className="w-5 h-5 text-purple-400" />
          <span className="text-purple-400 font-medium">ML Model Dashboard</span>
        </div>
        <h2 className="text-3xl font-display font-bold text-white mb-2">
          Model Performance & Optimization
        </h2>
        <p className="text-slate-400">
          Bayesian-optimized models for drug interaction prediction
        </p>
      </motion.div>

      {/* Model Performance Cards */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="grid grid-cols-1 md:grid-cols-3 gap-4"
      >
        {Object.entries(models).map(([modelName, metrics], index) => (
          <ModelCard 
            key={modelName} 
            name={modelName} 
            metrics={metrics}
            delay={0.1 + index * 0.1}
          />
        ))}
      </motion.div>

      {/* Optimization Comparison */}
      {comparison && comparison.status === 'loaded' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="glass rounded-3xl p-8"
        >
          <h3 className="text-xl font-semibold text-white mb-6 flex items-center gap-3">
            <TrendingUp className="w-6 h-6 text-purple-400" />
            Optimization Method Comparison
          </h3>

          {/* Summary Stats */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="p-4 bg-slate-800/30 rounded-xl">
              <p className="text-slate-400 text-sm mb-1">Bayesian Wins</p>
              <p className="text-2xl font-bold text-purple-400">
                {comparison.bayesian_wins}/{comparison.total_models_compared}
              </p>
            </div>
            <div className="p-4 bg-slate-800/30 rounded-xl">
              <p className="text-slate-400 text-sm mb-1">Avg. Trial Reduction</p>
              <p className="text-2xl font-bold text-green-400">
                {comparison.average_trial_reduction_percent?.toFixed(1)}%
              </p>
            </div>
            <div className="p-4 bg-slate-800/30 rounded-xl">
              <p className="text-slate-400 text-sm mb-1">Optimization Method</p>
              <p className="text-lg font-bold text-white">
                TPE (Bayesian)
              </p>
            </div>
          </div>

          {/* Detailed Comparison */}
          <div className="space-y-4">
            {comparison.detailed_comparisons?.map((comp, index) => (
              <ComparisonRow key={index} data={comp} />
            ))}
          </div>
        </motion.div>
      )}

      {/* Feature Importance */}
      {modelInfo?.feature_importance && Object.keys(modelInfo.feature_importance).length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="glass rounded-3xl p-8"
        >
          <h3 className="text-xl font-semibold text-white mb-6 flex items-center gap-3">
            <BarChart3 className="w-6 h-6 text-purple-400" />
            Feature Importance
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {Object.entries(modelInfo.feature_importance).slice(0, 2).map(([model, features]) => (
              <FeatureImportanceChart key={model} model={model} features={features} />
            ))}
          </div>
        </motion.div>
      )}

      {/* Refresh Button */}
      <div className="text-center">
        <button
          onClick={fetchData}
          className="px-6 py-3 bg-slate-800 hover:bg-slate-700 text-white rounded-xl transition-colors flex items-center gap-2 mx-auto"
        >
          <RefreshCw className="w-5 h-5" />
          Refresh Data
        </button>
      </div>
    </div>
  )
}

function ModelCard({ name, metrics, delay }) {
  const modelConfig = {
    random_forest: { icon: '🌲', color: 'green', label: 'Random Forest' },
    xgboost: { icon: '🚀', color: 'orange', label: 'XGBoost' },
    lightgbm: { icon: '⚡', color: 'yellow', label: 'LightGBM' },
  }

  const config = modelConfig[name] || { icon: '🤖', color: 'purple', label: name }
  const aucRoc = metrics?.auc_roc || 0
  const f1Score = metrics?.f1_score || 0

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      className="glass rounded-2xl p-6 card-hover"
    >
      <div className="flex items-center gap-3 mb-4">
        <span className="text-3xl">{config.icon}</span>
        <div>
          <h4 className="text-white font-semibold">{config.label}</h4>
          <p className="text-slate-500 text-sm">Classification Model</p>
        </div>
      </div>

      <div className="space-y-3">
        <MetricBar label="AUC-ROC" value={aucRoc} color="purple" />
        <MetricBar label="F1-Score" value={f1Score} color="medical" />
        <MetricBar label="Accuracy" value={metrics?.accuracy || 0} color="blue" />
      </div>

      {aucRoc >= 0.8 && (
        <div className="mt-4 flex items-center gap-2 text-green-400 text-sm">
          <Award className="w-4 h-4" />
          <span>High Performance</span>
        </div>
      )}
    </motion.div>
  )
}

function MetricBar({ label, value, color }) {
  const percent = Math.round(value * 100)
  const colorClasses = {
    purple: 'bg-purple-500',
    medical: 'bg-medical-500',
    blue: 'bg-blue-500',
    green: 'bg-green-500',
  }

  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="text-slate-400">{label}</span>
        <span className="text-white font-mono">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percent}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          className={`h-full rounded-full ${colorClasses[color] || colorClasses.purple}`}
        />
      </div>
    </div>
  )
}

function ComparisonRow({ data }) {
  const summary = data.comparison?.comparison_summary || {}
  const methods = summary.methods || {}

  return (
    <div className="p-4 bg-slate-800/30 rounded-xl">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-white font-medium capitalize">{data.model}</h4>
        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
          summary.winner === 'bayesian' 
            ? 'bg-purple-500/20 text-purple-400' 
            : 'bg-slate-600/20 text-slate-400'
        }`}>
          Winner: {summary.winner || 'N/A'}
        </span>
      </div>
      
      <div className="grid grid-cols-3 gap-4 text-sm">
        {['bayesian', 'random_search', 'grid_search'].map(method => {
          const methodData = methods[method] || {}
          return (
            <div key={method} className="text-center">
              <p className="text-slate-500 text-xs mb-1 capitalize">
                {method.replace('_', ' ')}
              </p>
              <p className="text-white font-mono">
                {(methodData.best_score * 100 || 0).toFixed(1)}%
              </p>
              <p className="text-slate-500 text-xs">
                {methodData.n_trials || 0} trials
              </p>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function FeatureImportanceChart({ model, features }) {
  // Sort features by importance
  const sortedFeatures = Object.entries(features)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)

  const maxImportance = Math.max(...sortedFeatures.map(f => f[1]))

  const modelLabels = {
    random_forest: 'Random Forest',
    xgboost: 'XGBoost',
    lightgbm: 'LightGBM',
  }

  return (
    <div>
      <h4 className="text-white font-medium mb-4">{modelLabels[model] || model}</h4>
      <div className="space-y-2">
        {sortedFeatures.map(([feature, importance]) => (
          <div key={feature} className="flex items-center gap-2">
            <span className="w-40 text-xs text-slate-400 truncate" title={feature}>
              {feature.replace(/_/g, ' ')}
            </span>
            <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-purple-500 rounded-full"
                style={{ width: `${(importance / maxImportance) * 100}%` }}
              />
            </div>
            <span className="w-12 text-xs text-slate-400 text-right">
              {(importance * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default ModelDashboard

