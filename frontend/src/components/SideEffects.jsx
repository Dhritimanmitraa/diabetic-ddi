import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

/**
 * SideEffects Component
 * Displays known side effects for a drug from the OffSIDES database
 */
function SideEffects({ drugName }) {
    const [sideEffects, setSideEffects] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [isExpanded, setIsExpanded] = useState(false)

    const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

    useEffect(() => {
        if (!drugName) return

        const fetchSideEffects = async () => {
            setLoading(true)
            setError(null)
            try {
                const response = await fetch(
                    `${API_BASE}/drugs/${encodeURIComponent(drugName)}/side-effects?limit=30`
                )
                if (!response.ok) throw new Error('Failed to fetch side effects')
                const data = await response.json()
                setSideEffects(data)
            } catch (err) {
                setError(err.message)
            } finally {
                setLoading(false)
            }
        }

        fetchSideEffects()
    }, [drugName, API_BASE])

    if (!drugName) return null

    const severityConfig = {
        severe: {
            label: 'Severe',
            color: 'bg-red-500/20 text-red-400 border-red-500/30',
            icon: '',
            bgColor: 'bg-red-950/20'
        },
        moderate: {
            label: 'Moderate',
            color: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
            icon: '',
            bgColor: 'bg-amber-950/20'
        },
        mild: {
            label: 'Mild',
            color: 'bg-green-500/20 text-green-400 border-green-500/30',
            icon: '',
            bgColor: 'bg-green-950/20'
        },
        unknown: {
            label: 'Other',
            color: 'bg-slate-500/20 text-slate-400 border-slate-500/30',
            icon: '',
            bgColor: 'bg-slate-900/40'
        }
    }

    return (
        <div className="mt-4">
            <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="flex items-center gap-2 text-sm text-medical-400 hover:text-medical-300 transition-colors"
            >
                <svg
                    className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-90' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
                <span className="font-medium">View Side Effects</span>
                {sideEffects && sideEffects.total_effects > 0 && (
                    <span className="px-2 py-0.5 text-xs bg-medical-500/20 text-medical-400 rounded-full">
                        {sideEffects.total_effects} known
                    </span>
                )}
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
                        <div className="mt-4 p-4 bg-slate-800/50 rounded-xl border border-slate-700/50">
                            <h4 className="text-sm font-medium text-slate-300 mb-3 flex items-center gap-2">
                                <span className="text-lg"></span>
                                Known Side Effects for {drugName}
                            </h4>

                            {loading && (
                                <div className="flex items-center justify-center py-4">
                                    <div className="animate-spin w-5 h-5 border-2 border-medical-500 border-t-transparent rounded-full" />
                                    <span className="ml-2 text-sm text-slate-400">Loading side effects...</span>
                                </div>
                            )}

                            {error && (
                                <div className="text-sm text-red-400 bg-red-500/10 p-3 rounded-lg">
                                    {error}
                                </div>
                            )}

                            {sideEffects && !loading && (
                                <>
                                    {sideEffects.total_effects === 0 ? (
                                        <p className="text-sm text-slate-400 italic">
                                            No side effects data available for this drug.
                                        </p>
                                    ) : (
                                        <div className="space-y-3">
                                            {Object.entries(sideEffects.effects_by_severity).map(([severity, effects]) => {
                                                if (effects.length === 0) return null
                                                const config = severityConfig[severity]
                                                return (
                                                    <div
                                                        key={severity}
                                                        className={`p-3 rounded-lg border ${config.color} ${config.bgColor}`}
                                                    >
                                                        <div className="flex items-center gap-2 mb-2">
                                                            <span>{config.icon}</span>
                                                            <span className="font-medium text-sm">{config.label}</span>
                                                            <span className="text-xs opacity-70">({effects.length})</span>
                                                        </div>
                                                        <div className="flex flex-wrap gap-1.5">
                                                            {effects.slice(0, 10).map((effect, idx) => (
                                                                <span
                                                                    key={idx}
                                                                    className="px-2 py-0.5 text-xs bg-black/20 rounded-full"
                                                                    title={effect}
                                                                >
                                                                    {effect.length > 30 ? effect.slice(0, 30) + '...' : effect}
                                                                </span>
                                                            ))}
                                                            {effects.length > 10 && (
                                                                <span className="px-2 py-0.5 text-xs opacity-60">
                                                                    +{effects.length - 10} more
                                                                </span>
                                                            )}
                                                        </div>
                                                    </div>
                                                )
                                            })}

                                            <p className="text-xs text-slate-500 mt-2">
                                                Data from OffSIDES adverse event database. Consult a healthcare provider for medical advice.
                                            </p>
                                        </div>
                                    )}
                                </>
                            )}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}

export default SideEffects
