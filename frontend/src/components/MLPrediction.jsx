import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Brain, Sparkles, AlertTriangle, CheckCircle2, 
  BarChart3, Zap, TrendingUp, Info
} from 'lucide-react'

/**
 * MLPrediction Component
 * 
 * Displays ML model predictions alongside database lookup results.
 * Shows probability scores from Random Forest, XGBoost, and LightGBM models.
 */
function MLPrediction({ prediction, isLoading }) {
  const [showDetails, setShowDetails] = useState(false)

  if (isLoading) {
    return (
      <div className="glass rounded-2xl p-6 mt-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-purple-500/10 flex items-center justify-center">
            <Brain className="w-5 h-5 text-purple-400 animate-pulse" />
          </div>
          <div>
            <p className="text-white font-medium">ML Analysis in Progress...</p>
            <p className="text-slate-400 text-sm">Running prediction models</p>
          </div>
        </div>
      </div>
    )
  }

  if (!prediction || prediction.error) {
    return null
  }

  const {
    interaction_probability,
    predicted_interaction,
    severity_prediction,
    confidence,
    model_predictions,
  } = prediction

  // Color based on probability
  const getProbabilityColor = (prob) => {
    if (prob >= 0.7) return 'text-red-400'
    if (prob >= 0.4) return 'text-orange-400'
    if (prob >= 0.2) return 'text-yellow-400'
    return 'text-green-400'
  }

  const getSeverityConfig = (severity) => {
    const configs = {
      contraindicated: { color: 'text-red-400', bg: 'bg-red-500/10', border: 'border-red-500/20' },
      major: { color: 'text-orange-400', bg: 'bg-orange-500/10', border: 'border-orange-500/20' },
      moderate: { color: 'text-yellow-400', bg: 'bg-yellow-500/10', border: 'border-yellow-500/20' },
      minor: { color: 'text-green-400', bg: 'bg-green-500/10', border: 'border-green-500/20' },
      none: { color: 'text-medical-400', bg: 'bg-medical-500/10', border: 'border-medical-500/20' },
    }
    return configs[severity] || configs.none
  }

  const severityConfig = getSeverityConfig(severity_prediction)
  const probabilityPercent = Math.round(interaction_probability * 100)

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass rounded-2xl p-6 mt-4"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-500/20 to-pink-500/20 border border-purple-500/20 flex items-center justify-center">
            <Brain className="w-6 h-6 text-purple-400" />
          </div>
          <div>
            <h3 className="text-white font-semibold flex items-center gap-2">
              ML Prediction
              <span className="px-2 py-0.5 text-xs bg-purple-500/20 text-purple-400 rounded-full">
                AI
              </span>
            </h3>
            <p className="text-slate-400 text-sm">Bayesian-optimized ensemble model</p>
          </div>
        </div>
        
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="text-slate-400 hover:text-white transition-colors text-sm flex items-center gap-1"
        >
          <Info className="w-4 h-4" />
          {showDetails ? 'Hide' : 'Details'}
        </button>
      </div>

      {/* Main Prediction Display */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {/* Probability */}
        <div className="p-4 bg-slate-800/30 rounded-xl">
          <p className="text-slate-400 text-sm mb-2">Interaction Probability</p>
          <div className="flex items-end gap-2">
            <span className={`text-3xl font-bold ${getProbabilityColor(interaction_probability)}`}>
              {probabilityPercent}%
            </span>
          </div>
          <div className="mt-2 h-2 bg-slate-700 rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${probabilityPercent}%` }}
              transition={{ duration: 0.8, ease: 'easeOut' }}
              className={`h-full rounded-full ${
                interaction_probability >= 0.7 ? 'bg-red-500' :
                interaction_probability >= 0.4 ? 'bg-orange-500' :
                interaction_probability >= 0.2 ? 'bg-yellow-500' : 'bg-green-500'
              }`}
            />
          </div>
        </div>

        {/* Severity Prediction */}
        <div className="p-4 bg-slate-800/30 rounded-xl">
          <p className="text-slate-400 text-sm mb-2">Predicted Severity</p>
          <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg ${severityConfig.bg} ${severityConfig.border} border`}>
            {predicted_interaction ? (
              <AlertTriangle className={`w-4 h-4 ${severityConfig.color}`} />
            ) : (
              <CheckCircle2 className={`w-4 h-4 ${severityConfig.color}`} />
            )}
            <span className={`font-medium capitalize ${severityConfig.color}`}>
              {severity_prediction || 'None'}
            </span>
          </div>
        </div>

        {/* Confidence */}
        <div className="p-4 bg-slate-800/30 rounded-xl">
          <p className="text-slate-400 text-sm mb-2">Model Confidence</p>
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-purple-400" />
            <span className="text-2xl font-bold text-white">
              {Math.round(confidence * 100)}%
            </span>
          </div>
          <p className="text-slate-500 text-xs mt-1">
            Agreement between models
          </p>
        </div>
      </div>

      {/* Model Details (Expandable) */}
      <AnimatePresence>
        {showDetails && model_predictions && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="pt-4 border-t border-slate-700/50">
              <h4 className="text-white font-medium mb-4 flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-purple-400" />
                Individual Model Predictions
              </h4>
              
              <div className="space-y-3">
                {Object.entries(model_predictions).map(([model, prob]) => (
                  <ModelBar 
                    key={model} 
                    name={model} 
                    probability={prob} 
                  />
                ))}
              </div>

              {/* Model Info */}
              <div className="mt-4 p-3 bg-purple-500/5 border border-purple-500/10 rounded-lg">
                <p className="text-purple-400 text-sm flex items-center gap-2">
                  <Zap className="w-4 h-4" />
                  Models trained with Bayesian hyperparameter optimization
                </p>
                <p className="text-slate-500 text-xs mt-1">
                  Using Tree-structured Parzen Estimator (TPE) for efficient search
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Disclaimer */}
      <div className="mt-4 pt-4 border-t border-slate-700/50">
        <p className="text-slate-500 text-xs">
          <span className="text-purple-400">AI Prediction:</span> This is a machine learning prediction 
          based on drug properties. Always verify with healthcare professionals.
        </p>
      </div>
    </motion.div>
  )
}

function ModelBar({ name, probability }) {
  const modelIcons = {
    random_forest: '',
    xgboost: '',
    lightgbm: '',
  }

  const modelNames = {
    random_forest: 'Random Forest',
    xgboost: 'XGBoost',
    lightgbm: 'LightGBM',
  }

  const percent = Math.round(probability * 100)

  return (
    <div className="flex items-center gap-3">
      <span className="w-6 text-center">{modelIcons[name] || '🤖'}</span>
      <span className="w-28 text-sm text-slate-400">{modelNames[name] || name}</span>
      <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percent}%` }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
          className={`h-full rounded-full ${
            probability >= 0.7 ? 'bg-red-500' :
            probability >= 0.4 ? 'bg-orange-500' :
            probability >= 0.2 ? 'bg-yellow-500' : 'bg-green-500'
          }`}
        />
      </div>
      <span className="w-12 text-right text-sm font-mono text-white">{percent}%</span>
    </div>
  )
}

export default MLPrediction

