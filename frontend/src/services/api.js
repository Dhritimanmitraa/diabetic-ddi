/**
 * API Service for Drug Interaction Checker
 */

import { getApiBaseUrl } from '../utils/platform'

const API_BASE_URL = getApiBaseUrl()


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

// ============== Prescription RAG Endpoints ==============

/**
 * Upload a prescription image or PDF for extraction
 * @param {File} file - The prescription file
 * @returns {Promise<Object>} Extraction result with medicines
 */
export async function uploadPrescription(file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE_URL}/prescription/upload`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Upload failed' }))
    throw new Error(error.detail || `HTTP error! status: ${response.status}`)
  }

  return response.json()
}

/**
 * Upload a prescription as base64 encoded image
 * @param {string} imageBase64 - Base64 encoded image
 * @param {string} filename - Original filename
 * @returns {Promise<Object>} Extraction result with medicines
 */
export async function uploadPrescriptionBase64(imageBase64, filename = 'prescription.jpg') {
  const formData = new FormData()
  formData.append('image_base64', imageBase64)
  formData.append('filename', filename)

  const response = await fetch(`${API_BASE_URL}/prescription/upload/base64`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Upload failed' }))
    throw new Error(error.detail || `HTTP error! status: ${response.status}`)
  }

  return response.json()
}

/**
 * Get prescription by ID
 * @param {number} prescriptionId - Prescription ID
 * @returns {Promise<Object>} Prescription details
 */
export async function getPrescription(prescriptionId) {
  return apiRequest(`/prescription/${prescriptionId}`)
}

/**
 * List all prescriptions with pagination
 * @param {number} limit - Maximum results
 * @param {number} offset - Offset for pagination
 * @returns {Promise<Object>} List of prescriptions
 */
export async function getPrescriptionHistory(limit = 20, offset = 0) {
  return apiRequest(`/prescription/history?limit=${limit}&offset=${offset}`)
}

/**
 * Delete a prescription
 * @param {number} prescriptionId - Prescription ID
 * @returns {Promise<Object>} Deletion confirmation
 */
export async function deletePrescription(prescriptionId) {
  return apiRequest(`/prescription/${prescriptionId}`, {
    method: 'DELETE',
  })
}

/**
 * Chat with a prescription using RAG
 * @param {number} prescriptionId - Prescription ID
 * @param {string} message - User's question
 * @returns {Promise<Object>} Chat response
 */
export async function chatWithPrescription(prescriptionId, message) {
  return apiRequest('/prescription/chat', {
    method: 'POST',
    body: JSON.stringify({
      prescription_id: prescriptionId,
      message: message,
    }),
  })
}

/**
 * Get chat history for a prescription
 * @param {number} prescriptionId - Prescription ID
 * @returns {Promise<Object>} Chat history
 */
export async function getPrescriptionChatHistory(prescriptionId) {
  return apiRequest(`/prescription/${prescriptionId}/chat-history`)
}

/**
 * Check prescription module health
 * @returns {Promise<Object>} Health status
 */
export async function getPrescriptionHealth() {
  return apiRequest('/prescription/health/status')
}

/**
 * Check drug interactions between prescription medicines
 * @param {Array<string>} drugNames - List of drug names to check
 * @returns {Promise<Object>} Interaction results
 */
export async function checkPrescriptionInteractions(drugNames) {
  return apiRequest('/prescription/check-interactions', {
    method: 'POST',
    body: JSON.stringify({ drug_names: drugNames }),
  })
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
  // Prescription RAG
  uploadPrescription,
  uploadPrescriptionBase64,
  getPrescription,
  getPrescriptionHistory,
  deletePrescription,
  chatWithPrescription,
  getPrescriptionChatHistory,
  getPrescriptionHealth,
  checkPrescriptionInteractions,
}

