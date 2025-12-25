import { motion } from 'framer-motion'
import { 
  Shield, ShieldAlert, ShieldX, AlertTriangle, 
  CheckCircle2, XCircle, Info, ArrowRight,
  Pill, BookOpen, Stethoscope
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
        label: 'Minor Interaction',
        emoji: '',
      },
      moderate: {
        icon: <AlertTriangle className="w-8 h-8" />,
        color: 'text-yellow-400',
        bgColor: 'bg-yellow-500/10',
        borderColor: 'border-yellow-500/20',
        label: 'Moderate Interaction',
        emoji: '',
      },
      major: {
        icon: <ShieldAlert className="w-8 h-8" />,
        color: 'text-orange-400',
        bgColor: 'bg-orange-500/10',
        borderColor: 'border-orange-500/20',
        label: 'Major Interaction',
        emoji: '',
      },
      contraindicated: {
        icon: <ShieldX className="w-8 h-8" />,
        color: 'text-red-400',
        bgColor: 'bg-red-500/10',
        borderColor: 'border-red-500/20',
        label: 'Contraindicated',
        emoji: '',
      },
      safe: {
        icon: <Shield className="w-8 h-8" />,
        color: 'text-medical-400',
        bgColor: 'bg-medical-500/10',
        borderColor: 'border-medical-500/20',
        label: 'Safe to Use',
        emoji: '',
      },
    }
    return configs[severity] || configs.safe
  }

  const severity = interaction?.severity || (is_safe ? 'safe' : 'moderate')
  const config = getSeverityConfig(severity)

  return (
    <section className="max-w-4xl mx-auto px-4 py-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass rounded-3xl p-8"
      >
        {/* Header with status */}
        <div className="flex flex-col md:flex-row items-center gap-6 mb-8">
          {/* Status icon */}
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: 'spring', delay: 0.2 }}
            className={`w-20 h-20 rounded-2xl ${config.bgColor} ${config.borderColor} border flex items-center justify-center ${config.color}`}
          >
            {config.icon}
          </motion.div>

          {/* Status text */}
          <div className="text-center md:text-left flex-1">
            <h2 className={`font-display text-2xl font-bold ${config.color} mb-2`}>
              {config.emoji} {config.label}
            </h2>
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
                <div
                  className="h-full bg-medical-400 rounded-full"
                  style={{ width: `${interaction.confidence_score * 100}%` }}
                />
              </div>
              <span className="text-slate-400">{Math.round(interaction.confidence_score * 100)}%</span>
            </div>
          </div>
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

