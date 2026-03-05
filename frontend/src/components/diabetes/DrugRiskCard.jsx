import { motion } from 'framer-motion'
import { getRiskBadgeConfig } from '../../utils/severity'

function RiskBadge({ level }) {
  const { bg, text, border, label } = getRiskBadgeConfig(level)
  return (
    <span className={`px-3 py-1 rounded-full text-sm font-medium ${bg} ${text} border ${border}`}>
      {label}
    </span>
  )
}

function MLBadge({ mlRisk, mlProb, source }) {
  if (!mlRisk && mlProb == null) return null
  const probText = mlProb != null ? `p=${Math.round(mlProb * 100)}%` : ''
  const src = source ? source.replace('_', ' ') : ''
  return (
    <div className="flex items-center gap-2 text-xs text-[var(--text-secondary)]">
      <span className="px-2 py-1 rounded-full bg-slate-700/70 border border-[var(--border)] text-slate-200">
        ML: {mlRisk || 'n/a'} {probText && `(${probText})`}
      </span>
      {src && (
        <span className={`px-2 py-1 rounded-full border ${source === 'rule_override' ? 'border-red-500/50 text-red-300 bg-red-500/10' : 'border-[var(--border)] bg-[var(--bg-elevated)]/70'}`}>
          {src}
        </span>
      )}
    </div>
  )
}

function LLMSkeleton() {
  return (
    <div className="mt-4 p-4 bg-purple-500/10 border border-purple-500/30 rounded-lg animate-pulse">
      <div className="flex items-center gap-2 mb-3">
        <div className="w-4 h-4 bg-purple-500/30 rounded" />
        <div className="h-4 w-24 bg-purple-500/20 rounded" />
        <div className="h-5 w-16 bg-purple-500/10 rounded-full" />
      </div>
      <div className="space-y-2">
        <div className="h-3 bg-slate-700/50 rounded w-full" />
        <div className="h-3 bg-slate-700/50 rounded w-4/5" />
        <div className="h-3 bg-slate-700/50 rounded w-3/5" />
      </div>
      <div className="flex gap-2 mt-3">
        <div className="h-6 w-20 bg-purple-500/10 rounded-full" />
        <div className="h-6 w-24 bg-purple-500/10 rounded-full" />
      </div>
    </div>
  )
}

export default function DrugRiskCard({ assessment }) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="p-4 rounded-xl bg-[var(--bg-elevated)]/50 border border-[var(--border)]/50"
    >
      <div className="flex items-center justify-between mb-3">
        <h4 className="font-semibold text-[var(--text-primary)] text-lg">{assessment.drug_name}</h4>
        <RiskBadge level={assessment.risk_level} />
      </div>

      {assessment.severity && (
        <div className="flex items-center gap-2 text-xs text-[var(--text-secondary)] mb-2">
          <span className="px-2 py-1 rounded-full bg-slate-700/70 border border-[var(--border)]">Severity: {assessment.severity}</span>
        </div>
      )}

      <div className="mb-3">
        <MLBadge mlRisk={assessment.ml_risk_level} mlProb={assessment.ml_probability} source={assessment.ml_decision_source} />
      </div>

      <div className="mb-3">
        <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all ${assessment.risk_score < 20 ? 'bg-emerald-500' : assessment.risk_score < 40 ? 'bg-amber-500' : assessment.risk_score < 60 ? 'bg-orange-500' : 'bg-red-500'}`}
            style={{ width: `${assessment.risk_score}%` }}
          />
        </div>
        <p className="text-xs text-[var(--text-muted)] mt-1">Risk Score: {assessment.risk_score}/100</p>
      </div>

      <p className="text-sm text-slate-300 mb-3">{assessment.recommendation}</p>

      {assessment.risk_factors?.length > 0 && (
        <div className="mb-3">
          <h5 className="text-xs font-medium text-[var(--text-secondary)] mb-1">Risk Factors:</h5>
          <ul className="text-sm text-slate-300 space-y-1">
            {assessment.risk_factors.map((factor, i) => (
              <li key={i} className="flex items-start gap-2"><span className="text-red-400">•</span>{factor}</li>
            ))}
          </ul>
        </div>
      )}

      {assessment.rule_references?.length > 0 && (
        <div className="mb-3">
          <h5 className="text-xs font-medium text-[var(--text-secondary)] mb-1">Why flagged:</h5>
          <ul className="text-xs text-[var(--text-secondary)] space-y-1">
            {assessment.rule_references.map((ref, i) => (
              <li key={i} className="flex items-start gap-2"><span className="text-slate-500">•</span>{ref}</li>
            ))}
          </ul>
        </div>
      )}

      {assessment.patient_factors?.length > 0 && (
        <div className="mb-3">
          <h5 className="text-xs font-medium text-[var(--text-secondary)] mb-1">Triggering factors:</h5>
          <div className="flex flex-wrap gap-1">
            {assessment.patient_factors.map((pf, i) => (
              <span key={i} className="text-xs bg-indigo-500/20 text-indigo-300 px-2 py-1 rounded">{pf}</span>
            ))}
          </div>
        </div>
      )}

      {assessment.evidence_sources?.length > 0 && (
        <div className="mb-3">
          <h5 className="text-xs font-medium text-[var(--text-secondary)] mb-1">Sources:</h5>
          <div className="flex flex-wrap gap-1">
            {assessment.evidence_sources.map((src, i) => (
              <span key={i} className="text-xs bg-slate-700/70 text-slate-200 px-2 py-1 rounded border border-[var(--border)]">{src}</span>
            ))}
          </div>
        </div>
      )}

      {assessment.monitoring?.length > 0 && (
        <div className="mb-3">
          <h5 className="text-xs font-medium text-[var(--text-secondary)] mb-1">Monitor:</h5>
          <div className="flex flex-wrap gap-1">
            {assessment.monitoring.map((item, i) => (
              <span key={i} className="text-xs bg-blue-500/20 text-blue-400 px-2 py-1 rounded">{item}</span>
            ))}
          </div>
        </div>
      )}

      {assessment.alternatives?.length > 0 && (
        <div>
          <h5 className="text-xs font-medium text-[var(--text-secondary)] mb-1">Safer Alternatives:</h5>
          <div className="flex flex-wrap gap-1">
            {assessment.alternatives.map((alt, i) => (
              <span key={i} className="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-1 rounded border border-emerald-500/40">
                {alt} <span className="text-[10px] text-emerald-200 ml-1">(safer)</span>
              </span>
            ))}
          </div>
        </div>
      )}

      {assessment.llm_analysis ? (
        <div className="mt-4 p-3 bg-purple-500/10 border border-purple-500/30 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <svg className="w-4 h-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
            <h5 className="text-sm font-semibold text-purple-300">LLM Analysis</h5>
            <span className="text-xs px-2 py-0.5 rounded-full bg-purple-500/20 text-purple-200 border border-purple-500/30">
              {assessment.llm_analysis.model_used || 'AI'}
            </span>
          </div>
          <div className="mb-2">
            <div className="flex items-center gap-2 mb-1">
              <RiskBadge level={assessment.llm_analysis.risk_level} />
              <span className="text-xs text-[var(--text-secondary)]">Risk Score: {assessment.llm_analysis.risk_score || 0}/100</span>
            </div>
          </div>
          {assessment.llm_analysis.reasoning && (
            <p className="text-sm text-slate-300 mb-3 leading-relaxed">{assessment.llm_analysis.reasoning}</p>
          )}
          {assessment.llm_analysis.key_concerns?.length > 0 && (
            <div className="mb-2">
              <h6 className="text-xs font-medium text-purple-300 mb-1">Key Concerns:</h6>
              <ul className="text-xs text-slate-300 space-y-1">
                {assessment.llm_analysis.key_concerns.map((concern, i) => (
                  <li key={i} className="flex items-start gap-2"><span className="text-purple-400 mt-0.5">•</span>{concern}</li>
                ))}
              </ul>
            </div>
          )}
          {assessment.llm_analysis.monitoring_needed?.length > 0 && (
            <div>
              <h6 className="text-xs font-medium text-purple-300 mb-1">Monitoring Recommendations:</h6>
              <div className="flex flex-wrap gap-1">
                {assessment.llm_analysis.monitoring_needed.map((monitor, i) => (
                  <span key={i} className="text-xs bg-purple-500/20 text-purple-200 px-2 py-1 rounded border border-purple-500/30">{monitor}</span>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        <LLMSkeleton />
      )}
    </motion.div>
  )
}

export { RiskBadge }
