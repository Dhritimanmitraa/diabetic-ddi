import { getApiBaseUrl } from '../utils/platform'
import {
  clearAuthSession,
  ensureAnonymousSession,
  getAccessToken,
  getAuthSession,
  loginUser,
  refreshAuthSession,
  registerUser,
  setAuthSession,
} from './session'

const API_BASE_URL = getApiBaseUrl()

const DEFAULT_TIMEOUT_MS = 15000
const RAG_TIMEOUT_MS = 90000
const ADMIN_API_KEY_STORAGE_KEY = 'drugguard_admin_api_key'

export function getAdminApiKey() {
  try {
    return localStorage.getItem(ADMIN_API_KEY_STORAGE_KEY) || ''
  } catch {
    return ''
  }
}

export function setAdminApiKey(value) {
  try {
    if (value) {
      localStorage.setItem(ADMIN_API_KEY_STORAGE_KEY, value)
    } else {
      localStorage.removeItem(ADMIN_API_KEY_STORAGE_KEY)
    }
  } catch {
    // Ignore storage failures in non-browser contexts.
  }
}

async function parseError(response) {
  const requestId = response.headers?.get?.('X-Request-ID')
  const fallback = `HTTP error! status: ${response.status}`
  try {
    const payload = await response.json()
    const detail = payload.detail || fallback
    return requestId ? `${detail} (request ${requestId})` : detail
  } catch {
    return requestId ? `${fallback} (request ${requestId})` : fallback
  }
}

function isFormData(body) {
  return typeof FormData !== 'undefined' && body instanceof FormData
}

async function apiRequest(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`
  const {
    signal: externalSignal,
    timeout = DEFAULT_TIMEOUT_MS,
    apiKey,
    auth = false,
    responseType = 'json',
    _retried = false,
    ...restOptions
  } = options

  if (auth && !getAccessToken()) {
    await ensureAnonymousSession(API_BASE_URL)
  }

  const body = restOptions.body
  const headers = {
    ...(isFormData(body) ? {} : { 'Content-Type': 'application/json' }),
    ...(restOptions.headers || {}),
  }

  if (apiKey) {
    headers['X-API-Key'] = apiKey
  }

  if (auth) {
    const token = getAccessToken()
    if (!token) {
      throw new Error('Authentication required')
    }
    headers.Authorization = `Bearer ${token}`
  }

  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeout)

  if (externalSignal) {
    externalSignal.addEventListener('abort', () => controller.abort(), { once: true })
  }

  try {
    const response = await fetch(url, {
      ...restOptions,
      headers,
      signal: controller.signal,
    })

    if (response.status === 401 && auth && !_retried && getAuthSession()?.refreshToken) {
      try {
        await refreshAuthSession(API_BASE_URL)
        return await apiRequest(endpoint, { ...options, _retried: true })
      } catch {
        clearAuthSession()
      }
    }

    if (!response.ok) {
      throw new Error(await parseError(response))
    }

    if (responseType === 'blob') {
      return response.blob()
    }

    if (responseType === 'text') {
      return response.text()
    }

    if (response.status === 204) {
      return null
    }

    return response.json()
  } catch (err) {
    if (err.name === 'AbortError' && !externalSignal?.aborted) {
      throw new Error('Request timed out. Please try again.')
    }
    throw err
  } finally {
    clearTimeout(timeoutId)
  }
}

export async function ensureAuthenticatedSession() {
  return ensureAnonymousSession(API_BASE_URL)
}

export async function registerAccount(payload) {
  return registerUser(API_BASE_URL, payload)
}

export async function loginAccount(payload) {
  return loginUser(API_BASE_URL, payload)
}

export async function refreshAccountToken() {
  return refreshAuthSession(API_BASE_URL)
}

export async function getCurrentAccount() {
  return apiRequest('/auth/me', { auth: true })
}

export function updateAuthSession(session) {
  setAuthSession(session)
}

export async function searchDrugs(query, limit = 10, options = {}) {
  return apiRequest(`/drugs/search?query=${encodeURIComponent(query)}&limit=${limit}`, options)
}

export async function listDrugs(limit = 50, offset = 0, options = {}) {
  return apiRequest(`/drugs?limit=${limit}&offset=${offset}`, options)
}

export async function getDrugById(drugId) {
  return apiRequest(`/drugs/${drugId}`)
}

export async function getDrugByName(name) {
  return apiRequest(`/drugs/name/${encodeURIComponent(name)}`)
}

export async function checkInteraction(drug1, drug2) {
  return apiRequest('/interactions/check', {
    method: 'POST',
    body: JSON.stringify({
      drug1_name: drug1,
      drug2_name: drug2,
    }),
  })
}

export async function getDrugInteractions(drugName, severity = null) {
  let url = `/interactions/drug/${encodeURIComponent(drugName)}`
  if (severity) {
    url += `?severity=${severity}`
  }
  return apiRequest(url)
}

export async function getAlternatives(drug1, drug2) {
  return apiRequest('/alternatives', {
    method: 'POST',
    body: JSON.stringify({
      drug1_name: drug1,
      drug2_name: drug2,
    }),
  })
}

export async function extractFromImage(imageBase64) {
  return apiRequest('/ocr/extract', {
    method: 'POST',
    body: JSON.stringify({
      image_base64: imageBase64,
    }),
  })
}

export async function getSideEffects(drugName, limit = 30) {
  return apiRequest(`/drugs/${encodeURIComponent(drugName)}/side-effects?limit=${limit}`)
}

export async function getStats() {
  return apiRequest('/stats')
}

export async function healthCheck() {
  return apiRequest('/health')
}

export async function getSystemStatus(apiKey = getAdminApiKey()) {
  return apiRequest('/admin/system-status', { apiKey })
}

export async function getMLPrediction(drug1, drug2) {
  return apiRequest('/ml/predict', {
    method: 'POST',
    body: JSON.stringify({
      drug1_name: drug1,
      drug2_name: drug2,
    }),
  })
}

export async function getMLModelInfo() {
  return apiRequest('/ml/model-info')
}

export async function getMLComparison() {
  return apiRequest('/ml/comparison')
}

export async function getHistory(limit = 50) {
  return apiRequest(`/history?limit=${limit}`)
}

export async function getHistoryStats() {
  return apiRequest('/history/stats')
}

export async function uploadPrescription(file) {
  const formData = new FormData()
  formData.append('file', file)

  return apiRequest('/prescription/upload', {
    method: 'POST',
    body: formData,
    timeout: RAG_TIMEOUT_MS,
    auth: true,
  })
}

export async function uploadPrescriptionBase64(imageBase64, filename = 'prescription.jpg') {
  const formData = new FormData()
  formData.append('image_base64', imageBase64)
  formData.append('filename', filename)

  return apiRequest('/prescription/upload/base64', {
    method: 'POST',
    body: formData,
    timeout: RAG_TIMEOUT_MS,
    auth: true,
  })
}

export async function getPrescription(prescriptionId) {
  return apiRequest(`/prescription/${prescriptionId}`, { auth: true })
}

export async function getPrescriptionHistory(limit = 20, offset = 0) {
  return apiRequest(`/prescription/history?limit=${limit}&offset=${offset}`, { auth: true })
}

export async function deletePrescription(prescriptionId) {
  return apiRequest(`/prescription/${prescriptionId}`, {
    method: 'DELETE',
    auth: true,
  })
}

export async function chatWithPrescription(prescriptionId, message) {
  return apiRequest('/prescription/chat', {
    method: 'POST',
    timeout: RAG_TIMEOUT_MS,
    auth: true,
    body: JSON.stringify({
      prescription_id: prescriptionId,
      message,
    }),
  })
}

export async function getPrescriptionChatHistory(prescriptionId) {
  return apiRequest(`/prescription/${prescriptionId}/chat-history`, { auth: true })
}

export async function getPrescriptionHealth() {
  return apiRequest('/prescription/health/status')
}

export async function checkPrescriptionInteractions(drugNames) {
  return apiRequest('/prescription/check-interactions', {
    method: 'POST',
    timeout: RAG_TIMEOUT_MS,
    body: JSON.stringify({ drug_names: drugNames }),
  })
}

export async function listDiabeticPatients() {
  return apiRequest('/diabetic/patients', { auth: true })
}

export async function createDiabeticPatient(payload) {
  return apiRequest('/diabetic/patients', {
    method: 'POST',
    body: JSON.stringify(payload),
    auth: true,
  })
}

export async function deleteDiabeticPatient(patientId) {
  return apiRequest(`/diabetic/patients/${patientId}`, {
    method: 'DELETE',
    auth: true,
  })
}

export async function getDiabeticPatientMedications(patientId) {
  return apiRequest(`/diabetic/patients/${patientId}/medications`, { auth: true })
}

export async function addDiabeticMedication(patientId, payload) {
  return apiRequest(`/diabetic/patients/${patientId}/medications`, {
    method: 'POST',
    body: JSON.stringify(payload),
    auth: true,
  })
}

export async function removeDiabeticMedication(patientId, medicationId) {
  return apiRequest(`/diabetic/patients/${patientId}/medications/${medicationId}`, {
    method: 'DELETE',
    auth: true,
  })
}

export async function getDiabeticModelInfo() {
  return apiRequest('/diabetic/model-info', { auth: true })
}

export async function checkDiabeticRisk(patientId, drugName) {
  return apiRequest('/diabetic/risk-check', {
    method: 'POST',
    auth: true,
    body: JSON.stringify({
      patient_id: patientId,
      drug_name: drugName,
    }),
  })
}

export async function checkDiabeticRiskLlm(patientId, drugName) {
  return apiRequest('/diabetic/risk-check/llm', {
    method: 'POST',
    timeout: RAG_TIMEOUT_MS,
    auth: true,
    body: JSON.stringify({
      patient_id: patientId,
      drug_name: drugName,
    }),
  })
}

export async function checkDiabeticMedicationList(patientId) {
  return apiRequest('/diabetic/medication-list-check', {
    method: 'POST',
    auth: true,
    body: JSON.stringify({ patient_id: patientId }),
  })
}

export async function getDiabeticReport(patientId) {
  return apiRequest(`/diabetic/report/${patientId}`, { auth: true })
}

export async function downloadDiabeticReportPdf(patientId) {
  return apiRequest(`/diabetic/report/${patientId}/pdf`, {
    auth: true,
    responseType: 'blob',
    timeout: RAG_TIMEOUT_MS,
  })
}

export async function analyzeDiabeticReport(file, autoCreatePatient = true) {
  const formData = new FormData()
  formData.append('file', file)
  return apiRequest(`/diabetic/analyze-report?auto_create_patient=${autoCreatePatient}`, {
    method: 'POST',
    body: formData,
    auth: true,
    timeout: RAG_TIMEOUT_MS,
  })
}

export default {
  searchDrugs,
  listDrugs,
  getDrugById,
  getDrugByName,
  checkInteraction,
  getDrugInteractions,
  getAlternatives,
  extractFromImage,
  getSideEffects,
  getStats,
  healthCheck,
  getSystemStatus,
  getMLPrediction,
  getMLModelInfo,
  getMLComparison,
  getHistory,
  getHistoryStats,
  uploadPrescription,
  uploadPrescriptionBase64,
  getPrescription,
  getPrescriptionHistory,
  deletePrescription,
  chatWithPrescription,
  getPrescriptionChatHistory,
  getPrescriptionHealth,
  checkPrescriptionInteractions,
  listDiabeticPatients,
  createDiabeticPatient,
  deleteDiabeticPatient,
  getDiabeticPatientMedications,
  addDiabeticMedication,
  removeDiabeticMedication,
  getDiabeticModelInfo,
  checkDiabeticRisk,
  checkDiabeticRiskLlm,
  checkDiabeticMedicationList,
  getDiabeticReport,
  downloadDiabeticReportPdf,
  analyzeDiabeticReport,
  ensureAuthenticatedSession,
  registerAccount,
  loginAccount,
  refreshAccountToken,
  getCurrentAccount,
  updateAuthSession,
}
