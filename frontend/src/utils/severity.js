/**
 * Shared severity configuration and styling utilities.
 * Used by ResultsDisplay, MLPrediction, AlternativesDisplay, DiabetesManager.
 */

/**
 * @typedef {Object} SeverityConfig
 * @property {string} color
 * @property {string} bgColor
 * @property {string} borderColor
 * @property {string} label
 * @property {number} riskPercent
 */

/** @type {Record<string, SeverityConfig>} */
export const SEVERITY_CONFIGS = {
    minor: {
        color: 'text-green-400',
        bgColor: 'bg-green-500/8',
        borderColor: 'border-green-500/15',
        label: 'Minor Interaction',
        riskPercent: 20,
    },
    moderate: {
        color: 'text-amber-400',
        bgColor: 'bg-amber-500/8',
        borderColor: 'border-amber-500/15',
        label: 'Moderate Interaction',
        riskPercent: 50,
    },
    major: {
        color: 'text-orange-400',
        bgColor: 'bg-orange-500/8',
        borderColor: 'border-orange-500/15',
        label: 'Major Interaction',
        riskPercent: 75,
    },
    contraindicated: {
        color: 'text-red-400',
        bgColor: 'bg-red-500/8',
        borderColor: 'border-red-500/15',
        label: 'Contraindicated',
        riskPercent: 100,
    },
    safe: {
        color: 'text-medical-400',
        bgColor: 'bg-medical-500/8',
        borderColor: 'border-medical-500/15',
        label: 'Safe to Use',
        riskPercent: 0,
    },
    none: {
        color: 'text-medical-400',
        bgColor: 'bg-medical-500/8',
        borderColor: 'border-medical-500/15',
        label: 'No Interaction',
        riskPercent: 0,
    },
}

/**
 * @param {string} severity
 * @returns {SeverityConfig}
 */
export function getSeverityConfig(severity) {
    return SEVERITY_CONFIGS[severity] || SEVERITY_CONFIGS.safe
}

/**
 * @param {number} prob
 * @returns {string}
 */
export function getProbabilityColor(prob) {
    if (prob >= 0.7) return 'text-red-400'
    if (prob >= 0.4) return 'text-orange-400'
    if (prob >= 0.2) return 'text-amber-400'
    return 'text-green-400'
}

/**
 * @param {number} prob
 * @returns {string}
 */
export function getProbabilityBg(prob) {
    if (prob >= 0.7) return 'bg-red-500'
    if (prob >= 0.4) return 'bg-orange-500'
    if (prob >= 0.2) return 'bg-amber-500'
    return 'bg-green-500'
}

/** @type {Record<string, {bg: string, text: string, border: string, label: string}>} */
export const RISK_BADGE_CONFIGS = {
    safe: { bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/30', label: 'Safe' },
    caution: { bg: 'bg-amber-500/20', text: 'text-amber-400', border: 'border-amber-500/30', label: 'Caution' },
    high_risk: { bg: 'bg-orange-500/20', text: 'text-orange-400', border: 'border-orange-500/30', label: 'High Risk' },
    contraindicated: { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/30', label: 'Contraindicated' },
    fatal: { bg: 'bg-red-900/40', text: 'text-red-300', border: 'border-red-700', label: 'Fatal Risk' },
}

/**
 * @param {string} level
 * @returns {{bg: string, text: string, border: string, label: string}}
 */
export function getRiskBadgeConfig(level) {
    return RISK_BADGE_CONFIGS[level] || RISK_BADGE_CONFIGS.caution
}
