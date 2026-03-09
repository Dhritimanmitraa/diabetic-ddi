import { memo } from 'react'
import { motion } from 'framer-motion'
import {
  Shield, ShieldAlert, ShieldX, AlertTriangle,
  CheckCircle2, XCircle, Info, ArrowRight,
  Pill, BookOpen, Stethoscope
} from 'lucide-react'
import SideEffects from './SideEffects'
import { getSeverityConfig } from '../utils/severity'

const SEVERITY_ICONS = {
  minor: <Info className="w-5 h-5" />,
  moderate: <AlertTriangle className="w-5 h-5" />,
  major: <ShieldAlert className="w-5 h-5" />,
  contraindicated: <ShieldX className="w-5 h-5" />,
  safe: <Shield className="w-5 h-5" />,
}

const ResultsDisplay = memo(function ResultsDisplay({ results }) {
  const has_interaction = results?.has_interaction
  const is_safe = results?.is_safe

  if (!results) return null

  const { interaction, safety_message, recommendations, drug1, drug2 } = results
  const severity = interaction?.severity || (is_safe ? 'safe' : 'moderate')
  const config = getSeverityConfig(severity)
  const icon = SEVERITY_ICONS[severity] || SEVERITY_ICONS.safe

  return (
    <section className="max-w-3xl mx-auto px-5 py-8">
      <div className="rounded-2xl p-6 sm:p-8 relative overflow-hidden bg-[var(--bg-elevated)] border border-[var(--border)]">
        {/* Header */}
        <div className="flex flex-col sm:flex-row items-center gap-4 mb-6">
          <div className={`w-12 h-12 rounded-xl ${config.bgColor} ${config.borderColor} border flex items-center justify-center ${config.color}`}>
            {icon}
          </div>

          <div className="text-center sm:text-left flex-1">
            <div className="flex items-center justify-center sm:justify-start gap-3 mb-1.5">
              <h2 className={`font-display text-xl font-bold ${config.color}`}>
                {config.label}
              </h2>
              <span className={`px-2.5 py-0.5 rounded-md text-xs font-semibold ${config.bgColor} ${config.borderColor} border uppercase tracking-wider`}>
                {severity}
              </span>
            </div>
            <p className="text-[var(--text-secondary)] text-sm">{safety_message}</p>
          </div>
        </div>

        {/* Drug pair */}
        <div className="flex items-center justify-center gap-3 mb-7 p-4 bg-[var(--bg-elevated)] rounded-xl border border-[var(--border)]">
          <DrugCard drug={drug1} />
          <div className={`p-1.5 rounded-full ${config.bgColor}`}>
            {has_interaction ? (
              <XCircle className={`w-5 h-5 ${config.color}`} />
            ) : (
              <CheckCircle2 className="w-5 h-5 text-medical-400" />
            )}
          </div>
          <DrugCard drug={drug2} />
        </div>

        {/* Interaction details */}
        {has_interaction && interaction && (
          <div className="space-y-3 mb-6">
            {interaction.effect && (
              <DetailCard icon={<AlertTriangle className="w-4 h-4" />} title="Effect" content={interaction.effect} color="text-amber-400" />
            )}
            {interaction.mechanism && (
              <DetailCard icon={<BookOpen className="w-4 h-4" />} title="Mechanism" content={interaction.mechanism} color="text-blue-400" />
            )}
            {interaction.management && (
              <DetailCard icon={<Stethoscope className="w-4 h-4" />} title="Management" content={interaction.management} color="text-medical-400" />
            )}
          </div>
        )}

        {/* Recommendations */}
        {recommendations && recommendations.length > 0 && (
          <div className="p-5 bg-[var(--bg-primary)]/50 rounded-xl border border-[var(--border)] mb-6">
            <h3 className="font-display font-semibold text-sm text-[var(--text-primary)] mb-3 flex items-center gap-2">
              <Info className="w-4 h-4 text-medical-400" />
              Recommendations
            </h3>
            <ul className="space-y-2.5">
              {recommendations.map((rec, index) => (
                <li key={index} className="flex items-start gap-2.5 text-[var(--text-secondary)] text-sm">
                  <ArrowRight className="w-3.5 h-3.5 text-medical-400 mt-0.5 flex-shrink-0" />
                  <span>{rec}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Confidence score */}
        {interaction?.confidence_score && (
          <div className="flex items-center justify-between text-xs border-t border-[var(--border)] pt-5 mt-5">
            <span className="text-[var(--text-muted)]">Data confidence</span>
            <div className="flex items-center gap-2.5">
              <div className="w-28 h-1.5 bg-[var(--bg-elevated)] rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${interaction.confidence_score * 100}%` }}
                  transition={{ duration: 0.6, ease: 'easeOut' }}
                  className="h-full bg-medical-400 rounded-full"
                />
              </div>
              <span className="text-[var(--text-secondary)] font-medium">{Math.round(interaction.confidence_score * 100)}%</span>
            </div>
          </div>
        )}

        {/* Risk gauge */}
        {has_interaction && (
          <div className="mt-5 p-4 bg-[var(--bg-primary)]/50 rounded-xl border border-[var(--border)]">
            <div className="flex items-center justify-between mb-2.5">
              <span className="text-xs text-[var(--text-muted)] font-medium">Risk Level</span>
              <span className={`text-xs font-semibold ${config.color}`}>{config.label}</span>
            </div>
            <div className="relative">
              <div className="risk-meter w-full" />
              <div
                className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 transition-all duration-700"
                style={{ left: `${config.riskPercent}%` }}
              >
                <div className={`w-3 h-3 rounded-full bg-white shadow-md border-2 ${config.borderColor}`} />
              </div>
            </div>
            <div className="flex justify-between mt-1.5 text-[9px] text-[var(--text-muted)] uppercase tracking-wider">
              <span>Safe</span>
              <span>Minor</span>
              <span>Moderate</span>
              <span>Major</span>
              <span>Danger</span>
            </div>
          </div>
        )}
      </div>
    </section>
  )
})

function DrugCard({ drug }) {
  return (
    <div className="flex-1 p-4 bg-[var(--bg-elevated)] rounded-xl text-center">
      <div className="w-10 h-10 rounded-lg bg-medical-500/15 flex items-center justify-center mx-auto mb-2.5">
        <Pill className="w-5 h-5 text-medical-400" />
      </div>
      <p className="font-display font-bold text-base text-medical-300 mb-0.5">{drug?.name || 'Unknown'}</p>
      {drug?.generic_name && (
        <p className="text-[var(--text-muted)] text-xs">{drug.generic_name}</p>
      )}
      {drug?.drug_class && (
        <span className="inline-block mt-1.5 px-2 py-0.5 bg-[var(--bg-elevated)] rounded text-[10px] text-[var(--text-muted)]">
          {drug.drug_class}
        </span>
      )}
      {drug?.name && <SideEffects drugName={drug.name} />}
    </div>
  )
}

function DetailCard({ icon, title, content, color }) {
  return (
    <div className="p-4 bg-[var(--bg-elevated)]/50 rounded-xl border border-[var(--border)]">
      <h4 className={`font-semibold text-sm ${color} mb-1.5 flex items-center gap-2`}>
        {icon}
        {title}
      </h4>
      <p className="text-[var(--text-secondary)] text-sm leading-relaxed">{content}</p>
    </div>
  )
}

export default ResultsDisplay
