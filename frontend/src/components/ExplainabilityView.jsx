import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Lightbulb,
    TrendingUp,
    TrendingDown,
    ChevronDown,
    ChevronUp,
    AlertTriangle,
    CheckCircle,
    Info,
    BarChart3,
    Brain
} from 'lucide-react';
import { getMLExplanation } from '../services/api';

/**
 * ExplainabilityView Component
 * 
 * Displays SHAP/LIME explanations for ML drug interaction predictions.
 * Shows feature importance, waterfall visualization, and natural language explanation.
 */
export default function ExplainabilityView({ drug1, drug2 }) {
    const [explanation, setExplanation] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [expanded, setExpanded] = useState(false);
    const [method, setMethod] = useState('auto');

    const fetchExplanation = useCallback(async () => {
        if (!drug1 || !drug2) {
            setError('Please select two drugs to explain');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const response = await getMLExplanation(drug1, drug2, method);
            setPrediction(response.prediction);
            setExplanation(response.explanation);
        } catch (err) {
            setError(err?.message || 'Failed to get explanation');
        } finally {
            setLoading(false);
        }
    }, [drug1, drug2, method]);

    // Fetch explanation when drugs change
    useEffect(() => {
        if (drug1 && drug2) {
            fetchExplanation();
        }
    }, [drug1, drug2, fetchExplanation]);

    const getSeverityColor = (severity) => {
        const colors = {
            contraindicated: 'text-red-500 bg-red-500/20',
            major: 'text-orange-500 bg-orange-500/20',
            moderate: 'text-yellow-500 bg-yellow-500/20',
            minor: 'text-blue-500 bg-blue-500/20',
            none: 'text-green-500 bg-green-500/20'
        };
        return colors[severity?.toLowerCase()] || colors.none;
    };

    const getSeverityIcon = (severity) => {
        if (severity === 'contraindicated' || severity === 'major') {
            return <AlertTriangle className="w-5 h-5" />;
        } else if (severity === 'moderate' || severity === 'minor') {
            return <Info className="w-5 h-5" />;
        }
        return <CheckCircle className="w-5 h-5" />;
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="bg-[var(--bg-elevated)] rounded-2xl p-6 shadow-xl border border-[var(--border)]"
        >
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-purple-500/20 rounded-lg">
                        <Brain className="w-6 h-6 text-purple-400" />
                    </div>
                    <div>
                        <h3 className="text-xl font-bold text-[var(--text-primary)]">AI Explanation</h3>
                        <p className="text-sm text-slate-400">
                            Why the model made this prediction
                        </p>
                    </div>
                </div>

                {/* Method Selector */}
                <select
                    value={method}
                    onChange={(e) => setMethod(e.target.value)}
                    className="bg-[var(--bg-elevated)] text-[var(--text-primary)] px-3 py-2 rounded-lg border border-[var(--border)] text-sm"
                >
                    <option value="auto">Auto (SHAP preferred)</option>
                    <option value="shap">SHAP</option>
                    <option value="lime">LIME</option>
                </select>
            </div>

            {/* Loading State */}
            {loading && (
                <div className="flex items-center justify-center py-12">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500"></div>
                    <span className="ml-4 text-slate-400">Analyzing prediction...</span>
                </div>
            )}

            {/* Error State */}
            {error && (
                <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 mb-4">
                    <p className="text-red-400">{error}</p>
                </div>
            )}

            {/* Explanation Content */}
            {explanation && !loading && (
                <div className="space-y-6">
                    {/* Prediction Summary */}
                    {prediction && (
                        <div className={`rounded-xl p-4 ${getSeverityColor(prediction.severity_prediction)}`}>
                            <div className="flex items-center gap-3">
                                {getSeverityIcon(prediction.severity_prediction)}
                                <div>
                                    <p className="font-semibold">
                                        {prediction.severity_prediction?.charAt(0).toUpperCase() +
                                            prediction.severity_prediction?.slice(1)} Risk
                                    </p>
                                    <p className="text-sm opacity-80">
                                        Probability: {(prediction.interaction_probability * 100).toFixed(1)}%
                                    </p>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Natural Language Explanation */}
                    <div className="bg-[var(--bg-surface)] rounded-xl p-5 border border-[var(--border)]">
                        <div className="flex items-start gap-3">
                            <Lightbulb className="w-5 h-5 text-yellow-400 mt-0.5 flex-shrink-0" />
                            <p className="text-slate-300 leading-relaxed">
                                {explanation.natural_language_explanation}
                            </p>
                        </div>
                    </div>

                    {/* Feature Importance Summary */}
                    <div className="bg-[var(--bg-surface)] rounded-xl p-5 border border-[var(--border)]">
                        <h4 className="text-lg font-semibold text-[var(--text-primary)] mb-4 flex items-center gap-2">
                            <BarChart3 className="w-5 h-5 text-blue-400" />
                            Feature Importance Breakdown
                        </h4>
                        <div className="space-y-3">
                            {Object.entries(explanation.feature_importance_summary || {})
                                .slice(0, 5)
                                .map(([group, importance]) => (
                                    <div key={group}>
                                        <div className="flex justify-between text-sm mb-1">
                                            <span className="text-slate-300">{group}</span>
                                            <span className="text-slate-400">{importance.toFixed(1)}%</span>
                                        </div>
                                        <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                                            <motion.div
                                                initial={{ width: 0 }}
                                                animate={{ width: `${importance}%` }}
                                                transition={{ duration: 0.5, delay: 0.1 }}
                                                className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"
                                            />
                                        </div>
                                    </div>
                                ))}
                        </div>
                    </div>

                    {/* Top Contributing Features */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {/* Risk Increasing Factors */}
                        <div className="bg-red-500/5 border border-red-500/20 rounded-xl p-4">
                            <h4 className="text-sm font-semibold text-red-400 mb-3 flex items-center gap-2">
                                <TrendingUp className="w-4 h-4" />
                                Factors Increasing Risk
                            </h4>
                            <ul className="space-y-2">
                                {(explanation.top_positive_features || []).slice(0, 3).map((feature, idx) => (
                                    <li key={idx} className="text-sm text-slate-300 flex items-center gap-2">
                                        <span className="w-1.5 h-1.5 rounded-full bg-red-400"></span>
                                        <span>{feature.group}</span>
                                        <span className="text-red-400 text-xs ml-auto">
                                            +{(feature.contribution * 100).toFixed(1)}%
                                        </span>
                                    </li>
                                ))}
                                {(!explanation.top_positive_features || explanation.top_positive_features.length === 0) && (
                                    <li className="text-sm text-slate-500 italic">No significant risk factors</li>
                                )}
                            </ul>
                        </div>

                        {/* Protective Factors */}
                        <div className="bg-green-500/5 border border-green-500/20 rounded-xl p-4">
                            <h4 className="text-sm font-semibold text-green-400 mb-3 flex items-center gap-2">
                                <TrendingDown className="w-4 h-4" />
                                Protective Factors
                            </h4>
                            <ul className="space-y-2">
                                {(explanation.top_negative_features || []).slice(0, 3).map((feature, idx) => (
                                    <li key={idx} className="text-sm text-slate-300 flex items-center gap-2">
                                        <span className="w-1.5 h-1.5 rounded-full bg-green-400"></span>
                                        <span>{feature.group}</span>
                                        <span className="text-green-400 text-xs ml-auto">
                                            -{(feature.contribution * 100).toFixed(1)}%
                                        </span>
                                    </li>
                                ))}
                                {(!explanation.top_negative_features || explanation.top_negative_features.length === 0) && (
                                    <li className="text-sm text-slate-500 italic">No significant protective factors</li>
                                )}
                            </ul>
                        </div>
                    </div>

                    {/* Expandable Waterfall Data */}
                    {explanation.waterfall_data?.shap_values && (
                        <div className="border-t border-[var(--border)] pt-4">
                            <button
                                onClick={() => setExpanded(!expanded)}
                                className="flex items-center gap-2 text-slate-400 hover:text-[var(--text-primary)] transition-colors"
                            >
                                {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                                <span className="text-sm">
                                    {expanded ? 'Hide' : 'Show'} technical details
                                </span>
                            </button>

                            <AnimatePresence>
                                {expanded && (
                                    <motion.div
                                        initial={{ height: 0, opacity: 0 }}
                                        animate={{ height: 'auto', opacity: 1 }}
                                        exit={{ height: 0, opacity: 0 }}
                                        className="mt-4 overflow-hidden"
                                    >
                                        <div className="bg-[var(--bg-elevated)] rounded-lg p-4 font-mono text-xs">
                                            <p className="text-slate-400 mb-2">
                                                Base value: {explanation.waterfall_data.base_value?.toFixed(4)}
                                            </p>
                                            <p className="text-slate-400 mb-2">
                                                Method: {explanation.explanation_method?.toUpperCase()}
                                            </p>
                                            <p className="text-slate-500">
                                                SHAP values array: [{explanation.waterfall_data.shap_values.length} features]
                                            </p>
                                        </div>
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </div>
                    )}

                    {/* Method Badge */}
                    <div className="flex items-center justify-end gap-2 text-xs text-slate-500">
                        <span>Explained using</span>
                        <span className="px-2 py-1 bg-[var(--bg-elevated)] rounded-md text-slate-400 uppercase">
                            {explanation.explanation_method}
                        </span>
                    </div>
                </div>
            )}

            {/* Initial State - Fetch Button */}
            {!explanation && !loading && !error && (
                <div className="text-center py-8">
                    <button
                        onClick={fetchExplanation}
                        disabled={!drug1 || !drug2}
                        className="px-6 py-3 bg-gradient-to-r from-purple-500 to-blue-500 text-white rounded-xl font-semibold disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg hover:shadow-purple-500/25 transition-all"
                    >
                        Explain This Prediction
                    </button>
                    {(!drug1 || !drug2) && (
                        <p className="text-slate-500 text-sm mt-2">
                            Select two drugs first
                        </p>
                    )}
                </div>
            )}
        </motion.div>
    );
}
