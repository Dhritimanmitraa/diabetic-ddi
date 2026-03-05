import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Brain, BarChart3, TrendingUp,
  RefreshCw, Award
} from 'lucide-react'
import usePageTitle from '../hooks/usePageTitle'
import { getMLModelInfo, getMLComparison } from '../services/api'

/**
 * ModelDashboard — ML model performance metrics and optimization comparison
 */
function ModelDashboard() {
  usePageTitle('ML Dashboard')
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
      const infoData = await getMLModelInfo()
      setModelInfo(infoData)

      const compData = await getMLComparison()
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
      <div className="glass rounded-2xl p-8 max-w-4xl mx-auto mt-24">
        <div className="flex items-center justify-center py-12">
          <div className="spinner"></div>
          <span className="ml-4 text-[var(--text-secondary)] text-sm">Loading model data...</span>
        </div>
      </div>
    )
  }

  if (error || modelInfo?.status === 'not_loaded') {
    return (
      <div className="glass rounded-2xl p-8 max-w-4xl mx-auto mt-24">
        <div className="text-center py-12">
          <Brain className="w-14 h-14 text-[var(--text-muted)] mx-auto mb-4" />
          <h3 className="text-lg font-display font-semibold text-[var(--text-primary)] mb-2">
            ML Models Not Yet Trained
          </h3>
          <p className="text-[var(--text-secondary)] text-sm mb-6">
            Train the machine learning models to see predictions and metrics.
          </p>
          <code className="px-4 py-2 bg-[var(--bg-elevated)] rounded-lg text-medical-400 text-sm border border-[var(--border)]">
            python -m scripts.train_models
          </code>
        </div>
      </div>
    )
  }

  const models = modelInfo?.models?.model_metrics || {}

  return (
    <div className="max-w-5xl mx-auto px-5 pt-24 pb-12 space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-8"
      >
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-purple-500/8 border border-purple-500/12 mb-4">
          <Brain className="w-4 h-4 text-purple-400" />
          <span className="text-purple-400 text-xs font-medium tracking-wide">ML Model Dashboard</span>
        </div>
        <h2 className="text-2xl sm:text-3xl font-display font-bold text-[var(--text-primary)] mb-2 tracking-tight">
          Model Performance & Optimization
        </h2>
        <p className="text-[var(--text-secondary)] text-sm">
          Bayesian-optimized models for drug interaction prediction
        </p>
      </motion.div>

      {/* Model Performance Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {Object.entries(models).map(([modelName, metrics], index) => (
          <ModelCard
            key={modelName}
            name={modelName}
            metrics={metrics}
            delay={0.05 + index * 0.08}
          />
        ))}
      </div>

      {/* Optimization Comparison */}
      {comparison && comparison.status === 'loaded' && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="glass rounded-2xl p-6 sm:p-8"
        >
          <h3 className="text-base font-display font-semibold text-[var(--text-primary)] mb-5 flex items-center gap-2.5">
            <TrendingUp className="w-5 h-5 text-purple-400" />
            Optimization Method Comparison
          </h3>

          {/* Summary Stats */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-5">
            <div className="p-4 bg-[var(--bg-elevated)]/50 rounded-xl border border-[var(--border)]">
              <p className="text-[var(--text-muted)] text-[10px] mb-1 uppercase tracking-wider font-medium">Bayesian Wins</p>
              <p className="text-xl font-bold text-purple-400">
                {comparison.bayesian_wins}/{comparison.total_models_compared}
              </p>
            </div>
            <div className="p-4 bg-[var(--bg-elevated)]/50 rounded-xl border border-[var(--border)]">
              <p className="text-[var(--text-muted)] text-[10px] mb-1 uppercase tracking-wider font-medium">Avg. Trial Reduction</p>
              <p className="text-xl font-bold text-green-400">
                {comparison.average_trial_reduction_percent?.toFixed(1)}%
              </p>
            </div>
            <div className="p-4 bg-[var(--bg-elevated)]/50 rounded-xl border border-[var(--border)]">
              <p className="text-[var(--text-muted)] text-[10px] mb-1 uppercase tracking-wider font-medium">Optimization Method</p>
              <p className="text-base font-bold text-[var(--text-primary)]">
                TPE (Bayesian)
              </p>
            </div>
          </div>

          {/* Detailed Comparison */}
          <div className="space-y-3">
            {comparison.detailed_comparisons?.map((comp, index) => (
              <ComparisonRow key={index} data={comp} />
            ))}
          </div>
        </motion.div>
      )}

      {/* Feature Importance */}
      {modelInfo?.feature_importance && Object.keys(modelInfo.feature_importance).length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="glass rounded-2xl p-6 sm:p-8"
        >
          <h3 className="text-base font-display font-semibold text-[var(--text-primary)] mb-5 flex items-center gap-2.5">
            <BarChart3 className="w-5 h-5 text-purple-400" />
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
          className="px-5 py-2.5 bg-[var(--bg-elevated)] hover:bg-[var(--bg-elevated)]/80 text-[var(--text-secondary)] hover:text-[var(--text-primary)] rounded-xl transition-colors flex items-center gap-2 mx-auto border border-[var(--border)] text-sm font-medium"
        >
          <RefreshCw className="w-4 h-4" />
          Refresh Data
        </button>
      </div>
    </div>
  )
}

function ModelCard({ name, metrics, delay }) {
  const modelConfig = {
    random_forest: { color: 'green', label: 'Random Forest' },
    xgboost: { color: 'orange', label: 'XGBoost' },
    lightgbm: { color: 'amber', label: 'LightGBM' },
  }

  const config = modelConfig[name] || { color: 'purple', label: name }
  const aucRoc = metrics?.auc_roc || 0
  const f1Score = metrics?.f1_score || 0

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.3 }}
      className="glass rounded-2xl p-5 card-hover"
    >
      <div className="flex items-center gap-3 mb-4">
        <div className="w-9 h-9 rounded-lg bg-purple-500/8 border border-purple-500/12 flex items-center justify-center">
          <Brain className="w-4.5 h-4.5 text-purple-400" />
        </div>
        <div>
          <h4 className="text-[var(--text-primary)] font-semibold text-sm">{config.label}</h4>
          <p className="text-[var(--text-muted)] text-[10px] uppercase tracking-wider">Classification</p>
        </div>
      </div>

      <div className="space-y-2.5">
        <MetricBar label="AUC-ROC" value={aucRoc} color="purple" />
        <MetricBar label="F1-Score" value={f1Score} color="medical" />
        <MetricBar label="Accuracy" value={metrics?.accuracy || 0} color="blue" />
      </div>

      {aucRoc >= 0.8 && (
        <div className="mt-3.5 flex items-center gap-1.5 text-green-400 text-xs font-medium">
          <Award className="w-3.5 h-3.5" />
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
      <div className="flex justify-between text-xs mb-1">
        <span className="text-[var(--text-muted)]">{label}</span>
        <span className="text-[var(--text-primary)] font-mono">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-1.5 bg-[var(--bg-elevated)] rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percent}%` }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
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
    <div className="p-4 bg-[var(--bg-elevated)]/50 rounded-xl border border-[var(--border)]">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-[var(--text-primary)] font-medium text-sm capitalize">{data.model}</h4>
        <span className={`px-2 py-0.5 rounded-md text-[10px] font-semibold uppercase tracking-wider border ${summary.winner === 'bayesian'
          ? 'bg-purple-500/10 text-purple-400 border-purple-500/15'
          : 'bg-[var(--bg-elevated)] text-[var(--text-muted)] border-[var(--border)]'
          }`}>
          Winner: {summary.winner || 'N/A'}
        </span>
      </div>

      <div className="grid grid-cols-3 gap-3 text-sm">
        {['bayesian', 'random_search', 'grid_search'].map(method => {
          const methodData = methods[method] || {}
          return (
            <div key={method} className="text-center">
              <p className="text-[var(--text-muted)] text-[10px] mb-1 capitalize tracking-wider">
                {method.replace('_', ' ')}
              </p>
              <p className="text-[var(--text-primary)] font-mono text-sm">
                {(methodData.best_score * 100 || 0).toFixed(1)}%
              </p>
              <p className="text-[var(--text-muted)] text-[10px]">
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
      <h4 className="text-[var(--text-primary)] font-medium text-sm mb-3.5">{modelLabels[model] || model}</h4>
      <div className="space-y-2">
        {sortedFeatures.map(([feature, importance]) => (
          <div key={feature} className="flex items-center gap-2">
            <span className="w-36 text-[10px] text-[var(--text-muted)] truncate" title={feature}>
              {feature.replace(/_/g, ' ')}
            </span>
            <div className="flex-1 h-1.5 bg-[var(--bg-elevated)] rounded-full overflow-hidden">
              <div
                className="h-full bg-purple-500 rounded-full transition-all"
                style={{ width: `${(importance / maxImportance) * 100}%` }}
              />
            </div>
            <span className="w-10 text-[10px] text-[var(--text-muted)] text-right font-mono">
              {(importance * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default ModelDashboard
