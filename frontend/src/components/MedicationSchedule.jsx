import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import toast from 'react-hot-toast'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/**
 * MedicationSchedule Component
 * Displays a visual daily medication schedule for a patient
 */
function MedicationSchedule({ patientId, medications = [] }) {
    const [schedule, setSchedule] = useState([])
    const [loading, setLoading] = useState(false)
    const [isExpanded, setIsExpanded] = useState(false)
    const [editingMed, setEditingMed] = useState(null)
    const [scheduleTimes, setScheduleTimes] = useState({})

    // Time slots for medication schedule
    const timeSlots = [
        { id: 'morning', label: 'Morning', time: '7:00 AM', icon: '🌅' },
        { id: 'afternoon', label: 'Afternoon', time: '12:00 PM', icon: '☀️' },
        { id: 'evening', label: 'Evening', time: '6:00 PM', icon: '🌆' },
        { id: 'bedtime', label: 'Bedtime', time: '10:00 PM', icon: '🌙' },
    ]

    // Initialize schedule from medications
    useEffect(() => {
        if (medications.length > 0) {
            const initialSchedule = {}
            medications.forEach(med => {
                // Parse frequency to suggest time slots
                const freq = (med.frequency || 'once daily').toLowerCase()
                let slots = ['morning']

                if (freq.includes('twice') || freq.includes('2x') || freq.includes('bid')) {
                    slots = ['morning', 'evening']
                } else if (freq.includes('three') || freq.includes('3x') || freq.includes('tid')) {
                    slots = ['morning', 'afternoon', 'evening']
                } else if (freq.includes('four') || freq.includes('4x') || freq.includes('qid')) {
                    slots = ['morning', 'afternoon', 'evening', 'bedtime']
                } else if (freq.includes('bedtime') || freq.includes('night') || freq.includes('hs')) {
                    slots = ['bedtime']
                } else if (freq.includes('morning')) {
                    slots = ['morning']
                }

                initialSchedule[med.drug_name] = {
                    slots,
                    withFood: freq.includes('food') || freq.includes('meal'),
                    notes: med.notes || ''
                }
            })
            setScheduleTimes(initialSchedule)
            buildScheduleView(initialSchedule)
        }
    }, [medications])

    const buildScheduleView = (scheduleData) => {
        const scheduleView = timeSlots.map(slot => ({
            ...slot,
            medications: medications.filter(med =>
                scheduleData[med.drug_name]?.slots?.includes(slot.id)
            ).map(med => ({
                ...med,
                withFood: scheduleData[med.drug_name]?.withFood || false
            }))
        }))
        setSchedule(scheduleView)
    }

    const toggleTimeSlot = (drugName, slotId) => {
        setScheduleTimes(prev => {
            const current = prev[drugName]?.slots || []
            const newSlots = current.includes(slotId)
                ? current.filter(s => s !== slotId)
                : [...current, slotId]

            const updated = {
                ...prev,
                [drugName]: {
                    ...(prev[drugName] || {}),
                    slots: newSlots
                }
            }
            buildScheduleView(updated)
            return updated
        })
    }

    const toggleWithFood = (drugName) => {
        setScheduleTimes(prev => {
            const updated = {
                ...prev,
                [drugName]: {
                    ...(prev[drugName] || {}),
                    withFood: !(prev[drugName]?.withFood || false)
                }
            }
            buildScheduleView(updated)
            return updated
        })
    }

    if (medications.length === 0) {
        return (
            <div className="p-4 bg-slate-800/50 rounded-xl border border-slate-700/50">
                <p className="text-sm text-slate-400 text-center">
                    No medications to schedule. Add medications to the patient profile first.
                </p>
            </div>
        )
    }

    return (
        <div className="mt-6">
            <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="flex items-center gap-3 text-lg font-semibold text-white mb-4 hover:text-medical-400 transition-colors"
            >
                <span className="text-2xl">📅</span>
                Medication Schedule
                <svg
                    className={`w-5 h-5 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
                <span className="text-xs px-2 py-1 bg-medical-500/20 text-medical-400 rounded-full">
                    {medications.length} medications
                </span>
            </button>

            <AnimatePresence>
                {isExpanded && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: 0.3 }}
                        className="overflow-hidden"
                    >
                        {/* Timeline View */}
                        <div className="space-y-4 mb-6">
                            {schedule.map((slot, idx) => (
                                <motion.div
                                    key={slot.id}
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: idx * 0.1 }}
                                    className={`p-4 rounded-xl border ${slot.medications.length > 0
                                            ? 'bg-slate-800/70 border-medical-500/30'
                                            : 'bg-slate-900/50 border-slate-700/30'
                                        }`}
                                >
                                    <div className="flex items-center gap-3 mb-3">
                                        <span className="text-2xl">{slot.icon}</span>
                                        <div>
                                            <h4 className="font-medium text-white">{slot.label}</h4>
                                            <p className="text-xs text-slate-400">{slot.time}</p>
                                        </div>
                                        <span className={`ml-auto text-xs px-2 py-1 rounded-full ${slot.medications.length > 0
                                                ? 'bg-medical-500/20 text-medical-400'
                                                : 'bg-slate-700 text-slate-500'
                                            }`}>
                                            {slot.medications.length} medications
                                        </span>
                                    </div>

                                    {slot.medications.length > 0 && (
                                        <div className="flex flex-wrap gap-2">
                                            {slot.medications.map((med, mIdx) => (
                                                <div
                                                    key={mIdx}
                                                    className="flex items-center gap-2 px-3 py-2 bg-slate-700/50 rounded-lg border border-slate-600/50"
                                                >
                                                    <span className="text-sm font-medium text-white">{med.drug_name}</span>
                                                    {med.dosage && (
                                                        <span className="text-xs text-slate-400">{med.dosage}</span>
                                                    )}
                                                    {med.withFood && (
                                                        <span className="text-xs px-1.5 py-0.5 bg-amber-500/20 text-amber-400 rounded">
                                                            🍽️ with food
                                                        </span>
                                                    )}
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </motion.div>
                            ))}
                        </div>

                        {/* Edit Schedule */}
                        <div className="p-4 bg-slate-800/50 rounded-xl border border-slate-700/50">
                            <h4 className="font-medium text-white mb-3 flex items-center gap-2">
                                <span>⚙️</span> Customize Schedule
                            </h4>
                            <div className="space-y-3">
                                {medications.map((med, idx) => (
                                    <div key={idx} className="p-3 bg-slate-900/50 rounded-lg">
                                        <div className="flex items-center justify-between mb-2">
                                            <span className="font-medium text-white">{med.drug_name}</span>
                                            <button
                                                onClick={() => toggleWithFood(med.drug_name)}
                                                className={`text-xs px-2 py-1 rounded transition-colors ${scheduleTimes[med.drug_name]?.withFood
                                                        ? 'bg-amber-500/30 text-amber-400 border border-amber-500/50'
                                                        : 'bg-slate-700 text-slate-400 border border-slate-600'
                                                    }`}
                                            >
                                                🍽️ With Food
                                            </button>
                                        </div>
                                        <div className="flex gap-2 flex-wrap">
                                            {timeSlots.map(slot => (
                                                <button
                                                    key={slot.id}
                                                    onClick={() => toggleTimeSlot(med.drug_name, slot.id)}
                                                    className={`px-3 py-1 rounded text-xs transition-colors ${scheduleTimes[med.drug_name]?.slots?.includes(slot.id)
                                                            ? 'bg-medical-500/30 text-medical-400 border border-medical-500/50'
                                                            : 'bg-slate-700 text-slate-400 border border-slate-600'
                                                        }`}
                                                >
                                                    {slot.icon} {slot.label}
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Timing Tips */}
                        <div className="mt-4 p-4 bg-blue-500/10 border border-blue-500/30 rounded-xl">
                            <h4 className="font-medium text-blue-400 mb-2 flex items-center gap-2">
                                <span>💡</span> Timing Tips for Diabetics
                            </h4>
                            <ul className="text-sm text-slate-300 space-y-1">
                                <li>• <strong>Metformin:</strong> Take with meals to reduce stomach upset</li>
                                <li>• <strong>Sulfonylureas:</strong> Take 30 minutes before meals</li>
                                <li>• <strong>SGLT2 inhibitors:</strong> Take in the morning to avoid nighttime urination</li>
                                <li>• <strong>Insulin:</strong> Timing depends on type (rapid, long-acting)</li>
                            </ul>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}

export default MedicationSchedule
