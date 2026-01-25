import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import toast from 'react-hot-toast'

/**
 * DosageCalculator Component
 * Calculates appropriate drug dosages based on patient characteristics
 */
function DosageCalculator({ patient, onClose }) {
    const [selectedDrug, setSelectedDrug] = useState('')
    const [result, setResult] = useState(null)

    // Common diabetes drugs with dosing guidelines based on eGFR
    const dosageGuidelines = {
        metformin: {
            name: 'Metformin',
            class: 'Biguanide',
            standardDose: '500-2000 mg/day',
            adjustments: [
                { egfrMin: 60, egfrMax: Infinity, dose: '500-2000 mg/day', note: 'Full dose approved' },
                { egfrMin: 45, egfrMax: 60, dose: '500-1000 mg/day', note: 'Reduce dose, monitor renal function' },
                { egfrMin: 30, egfrMax: 45, dose: 'Max 500 mg/day', note: 'Use with caution, increased monitoring' },
                { egfrMin: 0, egfrMax: 30, dose: 'CONTRAINDICATED', note: 'Do NOT use - risk of lactic acidosis' },
            ]
        },
        glipizide: {
            name: 'Glipizide',
            class: 'Sulfonylurea',
            standardDose: '2.5-20 mg/day',
            adjustments: [
                { egfrMin: 30, egfrMax: Infinity, dose: '2.5-20 mg/day', note: 'Start low, titrate carefully' },
                { egfrMin: 15, egfrMax: 30, dose: 'Max 2.5 mg/day', note: 'Increased hypoglycemia risk' },
                { egfrMin: 0, egfrMax: 15, dose: 'Avoid', note: 'Increased risk of severe hypoglycemia' },
            ]
        },
        glyburide: {
            name: 'Glyburide',
            class: 'Sulfonylurea',
            standardDose: '1.25-10 mg/day',
            adjustments: [
                { egfrMin: 60, egfrMax: Infinity, dose: '1.25-10 mg/day', note: 'Standard dosing' },
                { egfrMin: 0, egfrMax: 60, dose: 'AVOID', note: 'Active metabolites accumulate, use glipizide instead' },
            ]
        },
        sitagliptin: {
            name: 'Sitagliptin (Januvia)',
            class: 'DPP-4 Inhibitor',
            standardDose: '100 mg once daily',
            adjustments: [
                { egfrMin: 45, egfrMax: Infinity, dose: '100 mg once daily', note: 'Standard dose' },
                { egfrMin: 30, egfrMax: 45, dose: '50 mg once daily', note: 'Reduce dose by 50%' },
                { egfrMin: 0, egfrMax: 30, dose: '25 mg once daily', note: 'Reduce dose by 75%' },
            ]
        },
        empagliflozin: {
            name: 'Empagliflozin (Jardiance)',
            class: 'SGLT2 Inhibitor',
            standardDose: '10-25 mg once daily',
            adjustments: [
                { egfrMin: 45, egfrMax: Infinity, dose: '10-25 mg once daily', note: 'Full efficacy for glycemic control' },
                { egfrMin: 30, egfrMax: 45, dose: '10 mg once daily', note: 'Reduced glycemic efficacy, CV/renal benefits preserved' },
                { egfrMin: 20, egfrMax: 30, dose: '10 mg once daily', note: 'For cardiorenal protection only' },
                { egfrMin: 0, egfrMax: 20, dose: 'Avoid initiating', note: 'May continue if already on therapy' },
            ]
        },
        lisinopril: {
            name: 'Lisinopril',
            class: 'ACE Inhibitor',
            standardDose: '2.5-40 mg/day',
            adjustments: [
                { egfrMin: 30, egfrMax: Infinity, dose: '2.5-40 mg/day', note: 'Monitor potassium and creatinine' },
                { egfrMin: 10, egfrMax: 30, dose: 'Max 20 mg/day', note: 'Start low, titrate slowly' },
                { egfrMin: 0, egfrMax: 10, dose: 'Max 5 mg/day', note: 'Use with extreme caution, close monitoring' },
            ]
        },
        amlodipine: {
            name: 'Amlodipine',
            class: 'Calcium Channel Blocker',
            standardDose: '2.5-10 mg/day',
            adjustments: [
                { egfrMin: 0, egfrMax: Infinity, dose: '2.5-10 mg/day', note: 'No renal adjustment needed' },
            ]
        },
        atorvastatin: {
            name: 'Atorvastatin (Lipitor)',
            class: 'Statin',
            standardDose: '10-80 mg/day',
            adjustments: [
                { egfrMin: 0, egfrMax: Infinity, dose: '10-80 mg/day', note: 'No renal adjustment, beneficial for CV protection' },
            ]
        },
    }

    const calculateDosage = () => {
        if (!selectedDrug) {
            toast.error('Please select a drug')
            return
        }

        const drug = dosageGuidelines[selectedDrug]
        const patientEgfr = patient?.egfr || 90 // Default to normal if not specified
        const patientAge = patient?.age || 50

        // Find appropriate adjustment
        const adjustment = drug.adjustments.find(
            adj => patientEgfr >= adj.egfrMin && patientEgfr < adj.egfrMax
        ) || drug.adjustments[drug.adjustments.length - 1]

        // Determine risk level
        let riskLevel = 'safe'
        if (adjustment.dose.includes('CONTRAINDICATED') || adjustment.dose.includes('Avoid')) {
            riskLevel = 'contraindicated'
        } else if (adjustment.dose.includes('Max') || adjustment.note.includes('caution')) {
            riskLevel = 'caution'
        }

        // Age-based considerations
        let ageWarning = null
        if (patientAge >= 65) {
            ageWarning = 'Consider starting at lower dose in elderly patients and titrating slowly.'
        }

        setResult({
            drug: drug.name,
            drugClass: drug.class,
            standardDose: drug.standardDose,
            recommendedDose: adjustment.dose,
            note: adjustment.note,
            riskLevel,
            patientEgfr,
            patientAge,
            ageWarning,
        })
    }

    const riskColors = {
        safe: { bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/30' },
        caution: { bg: 'bg-amber-500/20', text: 'text-amber-400', border: 'border-amber-500/30' },
        contraindicated: { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/30' },
    }

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
            onClick={(e) => e.target === e.currentTarget && onClose()}
        >
            <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.9, opacity: 0 }}
                className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-2xl max-h-[90vh] overflow-auto shadow-2xl"
            >
                {/* Header */}
                <div className="p-4 border-b border-slate-700 flex items-center justify-between">
                    <div>
                        <h2 className="text-xl font-bold text-white flex items-center gap-2">
                            <span className="text-2xl"></span>
                            Dosage Calculator
                        </h2>
                        <p className="text-sm text-slate-400">
                            eGFR-based dosing adjustments for {patient?.name || 'patient'}
                        </p>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-white transition-colors"
                    >
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>

                {/* Patient Info */}
                <div className="p-4 bg-slate-800/50 border-b border-slate-700">
                    <div className="flex flex-wrap gap-4 text-sm">
                        <div>
                            <span className="text-slate-400">eGFR: </span>
                            <span className={`font-bold ${(patient?.egfr || 90) >= 60 ? 'text-emerald-400' :
                                (patient?.egfr || 90) >= 30 ? 'text-amber-400' : 'text-red-400'
                                }`}>
                                {patient?.egfr || 'Not specified'} mL/min/1.73m²
                            </span>
                        </div>
                        <div>
                            <span className="text-slate-400">Age: </span>
                            <span className="text-white font-medium">{patient?.age || 'Not specified'} years</span>
                        </div>
                        <div>
                            <span className="text-slate-400">Diabetes: </span>
                            <span className="text-white font-medium">
                                {(patient?.diabetes_type || 'type_2').replace('_', ' ').toUpperCase()}
                            </span>
                        </div>
                    </div>
                </div>

                {/* Drug Selection */}
                <div className="p-4">
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                        Select Medication
                    </label>
                    <select
                        value={selectedDrug}
                        onChange={(e) => {
                            setSelectedDrug(e.target.value)
                            setResult(null)
                        }}
                        className="w-full px-4 py-3 bg-slate-800 border border-slate-600 rounded-xl text-white focus:border-medical-500 focus:outline-none"
                    >
                        <option value="">-- Select a drug --</option>
                        {Object.entries(dosageGuidelines).map(([key, drug]) => (
                            <option key={key} value={key}>
                                {drug.name} ({drug.class})
                            </option>
                        ))}
                    </select>

                    <button
                        onClick={calculateDosage}
                        disabled={!selectedDrug}
                        className="w-full mt-4 py-3 bg-medical-500 text-white rounded-xl hover:bg-medical-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-medium"
                    >
                        Calculate Dose
                    </button>
                </div>

                {/* Result */}
                <AnimatePresence>
                    {result && (
                        <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            className="p-4 border-t border-slate-700"
                        >
                            <div className={`p-4 rounded-xl border ${riskColors[result.riskLevel].border} ${riskColors[result.riskLevel].bg}`}>
                                <div className="flex items-center justify-between mb-3">
                                    <h3 className="font-bold text-white text-lg">{result.drug}</h3>
                                    <span className={`px-3 py-1 rounded-full text-sm font-medium ${riskColors[result.riskLevel].text} ${riskColors[result.riskLevel].bg} border ${riskColors[result.riskLevel].border}`}>
                                        {result.riskLevel === 'safe' ? '✓ Safe' :
                                            result.riskLevel === 'caution' ? 'Caution' : 'Contraindicated'}
                                    </span>
                                </div>

                                <div className="space-y-3">
                                    <div className="flex justify-between">
                                        <span className="text-slate-400">Standard Dose:</span>
                                        <span className="text-slate-300">{result.standardDose}</span>
                                    </div>

                                    <div className="flex justify-between">
                                        <span className="text-slate-400">Recommended for this patient:</span>
                                        <span className={`font-bold ${riskColors[result.riskLevel].text}`}>
                                            {result.recommendedDose}
                                        </span>
                                    </div>

                                    <div className="flex justify-between">
                                        <span className="text-slate-400">Patient eGFR:</span>
                                        <span className="text-slate-300">{result.patientEgfr} mL/min/1.73m²</span>
                                    </div>
                                </div>

                                <div className="mt-4 p-3 bg-slate-900/50 rounded-lg">
                                    <p className={`text-sm ${riskColors[result.riskLevel].text}`}>
                                        {result.note}
                                    </p>
                                </div>

                                {result.ageWarning && (
                                    <div className="mt-3 p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                                        <p className="text-sm text-amber-400">
                                            {result.ageWarning}
                                        </p>
                                    </div>
                                )}
                            </div>

                            <p className="text-xs text-slate-500 mt-4 text-center">
                                This calculator is for reference only. Always consult prescribing information and clinical judgment.
                            </p>
                        </motion.div>
                    )}
                </AnimatePresence>
            </motion.div>
        </motion.div>
    )
}

export default DosageCalculator
