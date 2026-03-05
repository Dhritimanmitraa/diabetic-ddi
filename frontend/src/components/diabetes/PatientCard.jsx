import { motion } from 'framer-motion'

function HealthGauge({ value, min, max, label, unit, goodRange, cautionRange }) {
  if (value == null) return null
  const percentage = Math.min(100, Math.max(0, ((value - min) / (max - min)) * 100))

  let color = 'bg-emerald-500'
  let textColor = 'text-emerald-400'
  if (goodRange && (value < goodRange[0] || value > goodRange[1])) {
    if (cautionRange && value >= cautionRange[0] && value <= cautionRange[1]) {
      color = 'bg-amber-500'
      textColor = 'text-amber-400'
    } else {
      color = 'bg-red-500'
      textColor = 'text-red-400'
    }
  }

  return (
    <div className="flex-1 min-w-[80px]">
      <div className="flex justify-between text-xs mb-1">
        <span className="text-[var(--text-secondary)]">{label}</span>
        <span className={`font-medium ${textColor}`}>{value}{unit}</span>
      </div>
      <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
        <div className={`h-full ${color} transition-all duration-500`} style={{ width: `${percentage}%` }} />
      </div>
    </div>
  )
}

export default function PatientCard({ patient, onSelect, isSelected }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      onClick={() => onSelect(patient)}
      className={`p-4 rounded-xl cursor-pointer transition-all duration-300 ${isSelected
        ? 'bg-medical-500/20 border-2 border-medical-500'
        : 'bg-[var(--bg-elevated)]/50 border border-[var(--border)]/50 hover:border-medical-500/50'
        }`}
    >
      <div className="flex items-center justify-between mb-3">
        <div>
          <h4 className="font-semibold text-[var(--text-primary)]">{patient.name || patient.patient_id}</h4>
          <p className="text-sm text-[var(--text-secondary)]">
            {patient.diabetes_type.replace('_', ' ').toUpperCase()} • {patient.years_with_diabetes || '?'} years
          </p>
        </div>
        <div className="text-right">
          <div className={`font-medium ${patient.hba1c && patient.hba1c < 7 ? 'text-emerald-400' :
            patient.hba1c && patient.hba1c < 8 ? 'text-amber-400' : 'text-red-400'
            }`}>HbA1c: {patient.hba1c || 'N/A'}%</div>
          <div className={`text-xs ${patient.egfr && patient.egfr >= 60 ? 'text-emerald-400' :
            patient.egfr && patient.egfr >= 30 ? 'text-amber-400' : 'text-red-400'
            }`}>eGFR: {patient.egfr || 'N/A'}</div>
        </div>
      </div>

      {(patient.hba1c || patient.egfr) && (
        <div className="flex gap-3 mb-2">
          <HealthGauge value={patient.hba1c} min={4} max={14} label="HbA1c" unit="%" goodRange={[4, 7]} cautionRange={[7, 8.5]} />
          <HealthGauge value={patient.egfr} min={0} max={120} label="eGFR" unit="" goodRange={[60, 120]} cautionRange={[30, 60]} />
        </div>
      )}

      {(patient.has_nephropathy || patient.has_cardiovascular || patient.has_neuropathy) && (
        <div className="flex gap-2 flex-wrap">
          {patient.has_nephropathy && <span className="text-xs bg-red-500/20 text-red-400 px-2 py-0.5 rounded">Nephropathy</span>}
          {patient.has_cardiovascular && <span className="text-xs bg-red-500/20 text-red-400 px-2 py-0.5 rounded">Cardiovascular</span>}
          {patient.has_neuropathy && <span className="text-xs bg-red-500/20 text-red-400 px-2 py-0.5 rounded">Neuropathy</span>}
        </div>
      )}
    </motion.div>
  )
}
