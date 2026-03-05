import { useState, memo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Brain, AlertTriangle, CheckCircle2,
  BarChart3, Zap, Info, ChevronDown
} from 'lucide-react'
import { getSeverityConfig, getProbabilityColor, getProbabilityBg } from '../utils/severity'

const MLPrediction = memo(function MLPrediction({ prediction, isLoading }) {
  const [showDetails, setShowDetails] = useState(false)

  if (isLoading) {
    return (
      <div className="glass rounded-2xl p-5 mt-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-purple-500/8 flex items-center justify-center">
            <Brain className="w-4.5 h-4.5 text-purple-400 animate-pulse" />
          </div>
          <div>
            <p className="text-[var(--text-primary)] font-medium text-sm">ML Analysis in Progress...</p>
            <p className="text-[var(--text-muted)] text-xs">Running prediction models</p>
          </div>
        </div>
      </div>
    )
  }

  if (!prediction || prediction.error) return null

  const {
    interaction_probability,
    predicted_interaction,
    severity_prediction,
    confidence,
    model_predictions,
  } = prediction

  const severityConfig = getSeverityConfig(severity_prediction || 'none')
  const probabilityPercent = Math.round(interaction_probability * 100)

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="glass rounded-2xl p-5 sm:p-6 mt-4"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-purple-500/8 border border-purple-500/12 flex items-center justify-center">
            <Brain className="w-5 h-5 text-purple-400" />
          </div>
          <div>
            <h3 className="text-[var(--text-primary)] font-semibold text-sm flex items-center gap-2">
              ML Prediction
              <span className="px-2 py-0.5 text-[10px] bg-purple-500/10 text-purple-400 rounded-md font-semibold uppercase tracking-wider">
                AI
              </span>
            </h3>
            <p className="text-[var(--text-muted)] text-xs">Bayesian-optimized ensemble</p>
          </div>
        </div>

        <button
          onClick={() => setShowDetails(!showDetails)}
          className="text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors text-xs flex items-center gap-1 px-2.5 py-1.5 rounded-lg hover:bg-[var(--bg-elevated)]"
        >
          <Info className="w-3.5 h-3.5" />
          {showDetails ? 'Hide' : 'Details'}
          <ChevronDown className={`w-3 h-3 transition-transform ${showDetails ? 'rotate-180' : ''}`} />
        </button>
      </div>

      {/* Main Prediction Display */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-5">
        {/* Probability */}
        <div className="p-4 bg-[var(--bg-elevated)]/50 rounded-xl border border-[var(--border)]">
          <p className="text-[var(--text-muted)] text-[10px] mb-2 uppercase tracking-wider font-medium">Interaction Probability</p>
          <span className={`text-2xl font-bold ${getProbabilityColor(interaction_probability)}`}>
            {probabilityPercent}%
          </span>
          <div className="mt-2.5 h-1.5 bg-[var(--bg-elevated)] rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${probabilityPercent}%` }}
              transition={{ duration: 0.6, ease: 'easeOut' }}
              className={`h-full rounded-full ${getProbabilityBg(interaction_probability)}`}
            />
          </div>
        </div>

        {/* Severity */}
        <div className="p-4 bg-[var(--bg-elevated)]/50 rounded-xl border border-[var(--border)]">
          <p className="text-[var(--text-muted)] text-[10px] mb-2 uppercase tracking-wider font-medium">Predicted Severity</p>
          <div className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg ${severityConfig.bgColor} ${severityConfig.borderColor} border`}>
            {predicted_interaction ? (
              <AlertTriangle className={`w-3.5 h-3.5 ${severityConfig.color}`} />
            ) : (
              <CheckCircle2 className={`w-3.5 h-3.5 ${severityConfig.color}`} />
            )}
            <span className={`font-semibold text-sm capitalize ${severityConfig.color}`}>
              {severity_prediction || 'None'}
            </span>
          </div>
        </div>

        {/* Confidence */}
        <div className="p-4 bg-[var(--bg-elevated)]/50 rounded-xl border border-[var(--border)]">
          <p className="text-[var(--text-muted)] text-[10px] mb-2 uppercase tracking-wider font-medium">Model Confidence</p>
          <span className="text-2xl font-bold text-[var(--text-primary)]">
            {Math.round(confidence * 100)}%
          </span>
          <p className="text-[var(--text-muted)] text-[10px] mt-1">
            Agreement between models
          </p>
        </div>
      </div>

      {/* Model Details */}
      <AnimatePresence>
        {showDetails && model_predictions && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="pt-4 border-t border-[var(--border)]">
              <h4 className="text-[var(--text-primary)] font-medium text-sm mb-3.5 flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-purple-400" />
                Individual Model Predictions
              </h4>

              <div className="space-y-2.5">
                {Object.entries(model_predictions).map(([model, prob]) => (
                  <ModelBar key={model} name={model} probability={prob} />
                ))}
              </div>

              <div className="mt-4 p-3 bg-purple-500/5 border border-purple-500/10 rounded-lg">
                <p className="text-purple-400 text-xs flex items-center gap-1.5">
                  <Zap className="w-3.5 h-3.5" />
                  Bayesian hyperparameter optimization (TPE)
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Disclaimer */}
      <div className="mt-4 pt-4 border-t border-[var(--border)]">
        <p className="text-[var(--text-muted)] text-[10px]">
          <span className="text-purple-400 font-medium">AI Prediction:</span> Machine learning prediction
          based on drug properties. Always verify with healthcare professionals.
        </p>
      </div>
    </motion.div>
  )
})

function ModelBar({ name, probability }) {
  const modelNames = {
    random_forest: 'Random Forest',
    xgboost: 'XGBoost',
    lightgbm: 'LightGBM',
  }

  const percent = Math.round(probability * 100)

  const barColor = probability >= 0.7 ? 'bg-red-500'
    : probability >= 0.4 ? 'bg-orange-500'
      : probability >= 0.2 ? 'bg-amber-500'
        : 'bg-green-500'

  return (
    <div className="flex items-center gap-3">
      <span className="w-28 text-xs text-[var(--text-secondary)]">{modelNames[name] || name}</span>
      <div className="flex-1 h-1.5 bg-[var(--bg-elevated)] rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percent}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
          className={`h-full rounded-full ${barColor}`}
        />
      </div>
      <span className="w-10 text-right text-xs font-mono text-[var(--text-primary)]">{percent}%</span>
    </div>
  )
}

export default MLPrediction
