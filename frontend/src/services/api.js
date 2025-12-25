/**
 * API Service for Drug Interaction Checker
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001'

/**
 * Make API request with error handling
 */
async function apiRequest(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`
  
  const defaultOptions = {
    headers: {
      'Content-Type': 'application/json',
    },
  }
  
  const response = await fetch(url, { ...defaultOptions, ...options })
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'An error occurred' }))
    throw new Error(error.detail || `HTTP error! status: ${response.status}`)
  }
  
  return response.json()
}

/**
 * Search for drugs by name
 * @param {string} query - Search query
 * @param {number} limit - Maximum results
 * @returns {Promise<Array>} List of matching drugs
 */
export async function searchDrugs(query, limit = 10) {
  return apiRequest(`/drugs/search?query=${encodeURIComponent(query)}&limit=${limit}`)
}

/**
 * Get drug by ID
 * @param {number} drugId - Drug ID
 * @returns {Promise<Object>} Drug details
 */
export async function getDrugById(drugId) {
  return apiRequest(`/drugs/${drugId}`)
}

/**
 * Get drug by name
 * @param {string} name - Drug name
 * @returns {Promise<Object>} Drug details
 */
export async function getDrugByName(name) {
  return apiRequest(`/drugs/name/${encodeURIComponent(name)}`)
}

/**
 * Check interaction between two drugs
 * @param {string} drug1 - First drug name
 * @param {string} drug2 - Second drug name
 * @returns {Promise<Object>} Interaction check result
 */
export async function checkInteraction(drug1, drug2) {
  return apiRequest('/interactions/check', {
    method: 'POST',
    body: JSON.stringify({
      drug1_name: drug1,
      drug2_name: drug2,
    }),
  })
}

/**
 * Get all interactions for a specific drug
 * @param {string} drugName - Drug name
 * @param {string} severity - Optional severity filter
 * @returns {Promise<Object>} List of interactions
 */
export async function getDrugInteractions(drugName, severity = null) {
  let url = `/interactions/drug/${encodeURIComponent(drugName)}`
  if (severity) {
    url += `?severity=${severity}`
  }
  return apiRequest(url)
}

/**
 * Get safe alternative drugs
 * @param {string} drug1 - First drug name
 * @param {string} drug2 - Second drug name
 * @returns {Promise<Object>} Alternative suggestions
 */
export async function getAlternatives(drug1, drug2) {
  return apiRequest('/alternatives', {
    method: 'POST',
    body: JSON.stringify({
      drug1_name: drug1,
      drug2_name: drug2,
    }),
  })
}

/**
 * Extract drug names from image using OCR
 * @param {string} imageBase64 - Base64 encoded image
 * @returns {Promise<Object>} OCR result with detected drugs
 */
export async function extractFromImage(imageBase64) {
  return apiRequest('/ocr/extract', {
    method: 'POST',
    body: JSON.stringify({
      image_base64: imageBase64,
    }),
  })
}

/**
 * Get database statistics
 * @returns {Promise<Object>} Database stats
 */
export async function getStats() {
  return apiRequest('/stats')
}

/**
 * Health check
 * @returns {Promise<Object>} Health status
 */
export async function healthCheck() {
  return apiRequest('/health')
}

// ============== ML API Endpoints ==============

/**
 * Get ML prediction for drug interaction
 * @param {string} drug1 - First drug name
 * @param {string} drug2 - Second drug name
 * @returns {Promise<Object>} ML prediction result
 */
export async function getMLPrediction(drug1, drug2) {
  return apiRequest('/ml/predict', {
    method: 'POST',
    body: JSON.stringify({
      drug1_name: drug1,
      drug2_name: drug2,
    }),
  })
}

/**
 * Get ML model information and metrics
 * @returns {Promise<Object>} Model info
 */
export async function getMLModelInfo() {
  return apiRequest('/ml/model-info')
}

/**
 * Get optimization method comparison results
 * @returns {Promise<Object>} Comparison results
 */
export async function getMLComparison() {
  return apiRequest('/ml/comparison')
}

/**
 * Get comparison history
 * @param {number} limit - Maximum results
 * @returns {Promise<Object>} Comparison history
 */
export async function getHistory(limit = 50) {
  return apiRequest(`/history?limit=${limit}`)
}

/**
 * Get comparison statistics
 * @returns {Promise<Object>} Stats
 */
export async function getHistoryStats() {
  return apiRequest('/history/stats')
}

export default {
  searchDrugs,
  getDrugById,
  getDrugByName,
  checkInteraction,
  getDrugInteractions,
  getAlternatives,
  extractFromImage,
  getStats,
  healthCheck,
  getMLPrediction,
  getMLModelInfo,
  getMLComparison,
  getHistory,
  getHistoryStats,
}

