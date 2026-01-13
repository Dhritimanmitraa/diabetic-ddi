import { useEffect } from 'react'
import { motion } from 'framer-motion'
import confetti from 'canvas-confetti'
import {
  Shield, ShieldAlert, ShieldX, AlertTriangle,
  CheckCircle2, XCircle, Info, ArrowRight,
  Pill, BookOpen, Stethoscope, Sparkles
} from 'lucide-react'

function ResultsDisplay({ results }) {
  if (!results) return null

  const { has_interaction, is_safe, interaction, safety_message, recommendations, drug1, drug2 } = results

  const getSeverityConfig = (severity) => {
    const configs = {
      minor: {
        icon: <Info className="w-8 h-8" />,
        color: 'text-green-400',
        bgColor: 'bg-green-500/10',
        borderColor: 'border-green-500/20',
        glowClass: 'glow-minor',
        label: 'Minor Interaction',
        emoji: '✅',
        riskPercent: 20,
      },
      moderate: {
        icon: <AlertTriangle className="w-8 h-8" />,
        color: 'text-yellow-400',
        bgColor: 'bg-yellow-500/10',
        borderColor: 'border-yellow-500/20',
        glowClass: 'glow-moderate',
        label: 'Moderate Interaction',
        emoji: '⚠️',
        riskPercent: 50,
      },
      major: {
        icon: <ShieldAlert className="w-8 h-8" />,
        color: 'text-orange-400',
        bgColor: 'bg-orange-500/10',
        borderColor: 'border-orange-500/20',
        glowClass: 'glow-major',
        label: 'Major Interaction',
        emoji: '🔶',
        riskPercent: 75,
      },
      contraindicated: {
        icon: <ShieldX className="w-8 h-8" />,
        color: 'text-red-400',
        bgColor: 'bg-red-500/10',
        borderColor: 'border-red-500/20',
        glowClass: 'glow-danger',
        label: 'Contraindicated',
        emoji: '🚫',
        riskPercent: 100,
      },
      safe: {
        icon: <Shield className="w-8 h-8" />,
        color: 'text-medical-400',
        bgColor: 'bg-medical-500/10',
        borderColor: 'border-medical-500/20',
        glowClass: 'glow-safe celebrate',
        label: 'Safe to Use',
        emoji: '💚',
        riskPercent: 0,
      },
    }
    return configs[severity] || configs.safe
  }

  const severity = interaction?.severity || (is_safe ? 'safe' : 'moderate')
  const config = getSeverityConfig(severity)

  // Celebration confetti for safe results!
  useEffect(() => {
    if (!has_interaction && is_safe) {
      // Burst of confetti for safe drugs!
      const duration = 2000
      const end = Date.now() + duration

      const colors = ['#14b89a', '#5feaca', '#22c55e', '#4ade80']

      const frame = () => {
        confetti({
          particleCount: 3,
          angle: 60,
          spread: 55,
          origin: { x: 0, y: 0.7 },
          colors: colors
        })
        confetti({
          particleCount: 3,
          angle: 120,
          spread: 55,
          origin: { x: 1, y: 0.7 },
          colors: colors
        })

        if (Date.now() < end) {
          requestAnimationFrame(frame)
        }
      }
      frame()
    }
  }, [has_interaction, is_safe])

  return (
    <section className="max-w-4xl mx-auto px-4 py-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass rounded-3xl p-8 relative overflow-hidden"
      >
        {/* Sparkle effect for safe results */}
        {!has_interaction && is_safe && (
          <div className="absolute inset-0 pointer-events-none">
            <Sparkles className="absolute top-4 right-4 w-6 h-6 text-medical-400 animate-pulse" />
            <Sparkles className="absolute top-12 right-16 w-4 h-4 text-medical-300 animate-pulse" style={{ animationDelay: '0.5s' }} />
            <Sparkles className="absolute bottom-8 left-8 w-5 h-5 text-medical-400 animate-pulse" style={{ animationDelay: '1s' }} />
          </div>
        )}

        {/* Header with status */}
        <div className="flex flex-col md:flex-row items-center gap-6 mb-8">
          {/* Status icon with glow */}
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: 'spring', delay: 0.2 }}
            className={`w-20 h-20 rounded-2xl ${config.bgColor} ${config.borderColor} border flex items-center justify-center ${config.color} ${config.glowClass}`}
          >
            {config.icon}
          </motion.div>

          {/* Status text with badge */}
          <div className="text-center md:text-left flex-1">
            <div className="flex items-center justify-center md:justify-start gap-3 mb-2">
              <h2 className={`font-display text-2xl font-bold ${config.color}`}>
                {config.label}
              </h2>
              <motion.span
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: 'spring', delay: 0.4 }}
                className={`px-3 py-1 rounded-full text-sm font-semibold ${config.bgColor} ${config.borderColor} border badge-pulse`}
              >
                {config.emoji} {severity?.toUpperCase()}
              </motion.span>
            </div>
            <p className="text-slate-400">{safety_message}</p>
          </div>
        </div>

        {/* Drug pair display */}
        <div className="flex items-center justify-center gap-4 mb-8 p-4 bg-slate-800/30 rounded-2xl">
          <DrugCard drug={drug1} />
          <div className={`p-2 rounded-full ${config.bgColor}`}>
            {has_interaction ? (
              <XCircle className={`w-6 h-6 ${config.color}`} />
            ) : (
              <CheckCircle2 className="w-6 h-6 text-medical-400" />
            )}
          </div>
          <DrugCard drug={drug2} />
        </div>

        {/* Interaction details */}
        {has_interaction && interaction && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="space-y-6 mb-8"
          >
            {/* Effect */}
            {interaction.effect && (
              <DetailCard
                icon={<AlertTriangle className="w-5 h-5" />}
                title="Effect"
                content={interaction.effect}
                color="text-yellow-400"
              />
            )}

            {/* Mechanism */}
            {interaction.mechanism && (
              <DetailCard
                icon={<BookOpen className="w-5 h-5" />}
                title="Mechanism"
                content={interaction.mechanism}
                color="text-blue-400"
              />
            )}

            {/* Management */}
            {interaction.management && (
              <DetailCard
                icon={<Stethoscope className="w-5 h-5" />}
                title="Management"
                content={interaction.management}
                color="text-medical-400"
              />
            )}
          </motion.div>
        )}

        {/* Recommendations */}
        {recommendations && recommendations.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="p-6 bg-slate-800/30 rounded-2xl"
          >
            <h3 className="font-display font-semibold text-white mb-4 flex items-center gap-2">
              <Info className="w-5 h-5 text-medical-400" />
              Recommendations
            </h3>
            <ul className="space-y-3">
              {recommendations.map((rec, index) => (
                <motion.li
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.5 + index * 0.1 }}
                  className="flex items-start gap-3 text-slate-300"
                >
                  <ArrowRight className="w-4 h-4 text-medical-400 mt-1 flex-shrink-0" />
                  <span>{rec}</span>
                </motion.li>
              ))}
            </ul>
          </motion.div>
        )}

        {/* Confidence score */}
        {interaction?.confidence_score && (
          <div className="mt-6 pt-6 border-t border-slate-700/50 flex items-center justify-between text-sm">
            <span className="text-slate-500">Data confidence</span>
            <div className="flex items-center gap-2">
              <div className="w-32 h-2 bg-slate-800 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${interaction.confidence_score * 100}%` }}
                  transition={{ duration: 0.8, ease: 'easeOut' }}
                  className="h-full bg-medical-400 rounded-full"
                />
              </div>
              <span className="text-slate-400">{Math.round(interaction.confidence_score * 100)}%</span>
            </div>
          </div>
        )}

        {/* Risk Level Gauge */}
        {has_interaction && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="mt-6 p-4 bg-slate-800/30 rounded-xl"
          >
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm text-slate-400 font-medium">Risk Level</span>
              <span className={`text-sm font-bold ${config.color}`}>{config.label}</span>
            </div>
            <div className="relative">
              <div className="risk-meter w-full" />
              <motion.div
                initial={{ left: '0%' }}
                animate={{ left: `${config.riskPercent}%` }}
                transition={{ duration: 0.8, ease: 'easeOut', delay: 0.3 }}
                className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2"
                style={{ left: `${config.riskPercent}%` }}
              >
                <div className={`w-4 h-4 rounded-full bg-white shadow-lg border-2 ${config.borderColor}`} />
              </motion.div>
            </div>
            <div className="flex justify-between mt-2 text-xs text-slate-500">
              <span>Safe</span>
              <span>Minor</span>
              <span>Moderate</span>
              <span>Major</span>
              <span>Danger</span>
            </div>
          </motion.div>
        )}
      </motion.div>
    </section>
  )
}

function DrugCard({ drug }) {
  return (
    <div className="flex-1 p-4 bg-slate-800/50 rounded-xl text-center">
      <div className="w-12 h-12 rounded-xl bg-medical-500/10 flex items-center justify-center mx-auto mb-3">
        <Pill className="w-6 h-6 text-medical-400" />
      </div>
      <p className="font-display font-semibold text-white mb-1">{drug?.name || 'Unknown'}</p>
      {drug?.generic_name && (
        <p className="text-slate-500 text-sm">{drug.generic_name}</p>
      )}
      {drug?.drug_class && (
        <span className="inline-block mt-2 px-2 py-0.5 bg-slate-700/50 rounded-full text-xs text-slate-400">
          {drug.drug_class}
        </span>
      )}
    </div>
  )
}

function DetailCard({ icon, title, content, color }) {
  return (
    <div className="p-4 bg-slate-800/30 rounded-xl">
      <h4 className={`font-medium ${color} mb-2 flex items-center gap-2`}>
        {icon}
        {title}
      </h4>
      <p className="text-slate-300">{content}</p>
    </div>
  )
}

export default ResultsDisplay

