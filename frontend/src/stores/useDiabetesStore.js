import { create } from 'zustand'
import {
  addDiabeticMedication,
  analyzeDiabeticReport,
  checkDiabeticMedicationList,
  checkDiabeticRisk,
  checkDiabeticRiskLlm,
  createDiabeticPatient,
  deleteDiabeticPatient,
  downloadDiabeticReportPdf,
  getDiabeticModelInfo,
  getDiabeticPatientMedications,
  getDiabeticReport,
  listDiabeticPatients,
  listDrugs,
  removeDiabeticMedication,
  searchDrugs,
} from '../services/api'

const useDiabetesStore = create((set, get) => ({
  patients: [],
  selectedPatient: null,
  medications: [],
  checkResult: null,
  report: null,
  reportAnalysis: null,
  loading: false,
  modelInfo: null,
  allDrugs: [],
  drugsLoading: false,

  initialize: async () => {
    await Promise.all([get().loadPatients(), get().loadModelInfo()])
  },

  setSelectedPatient: (patient) => set({ selectedPatient: patient }),
  setCheckResult: (checkResult) => set({ checkResult }),
  setReport: (report) => set({ report }),
  setReportAnalysis: (reportAnalysis) => set({ reportAnalysis }),

  clearCheckResult: () => set({ checkResult: null }),

  loadPatients: async () => {
    const patients = await listDiabeticPatients()
    set({ patients })
    return patients
  },

  loadMedications: async (patientId) => {
    const medications = await getDiabeticPatientMedications(patientId)
    set({ medications })
    return medications
  },

  selectPatient: async (patient) => {
    set({ selectedPatient: patient, checkResult: null, report: null })
    if (patient?.patient_id) {
      await get().loadMedications(patient.patient_id)
    } else {
      set({ medications: [] })
    }
  },

  loadModelInfo: async () => {
    const modelInfo = await getDiabeticModelInfo()
    set({ modelInfo })
    return modelInfo
  },

  loadAllDrugs: async (search = '', page = 0) => {
    set({ drugsLoading: true })
    try {
      const limit = 50
      const offset = page * limit
      const allDrugs = search.trim()
        ? await searchDrugs(search.trim(), limit)
        : await listDrugs(limit, offset)
      set({ allDrugs: Array.isArray(allDrugs) ? allDrugs : (allDrugs.drugs || []) })
      return allDrugs
    } finally {
      set({ drugsLoading: false })
    }
  },

  createPatientRecord: async (payload) => {
    set({ loading: true })
    try {
      const patient = await createDiabeticPatient(payload)
      await get().loadPatients()
      return patient
    } finally {
      set({ loading: false })
    }
  },

  deletePatientRecord: async (patientId) => {
    set({ loading: true })
    try {
      await deleteDiabeticPatient(patientId)
      const patients = await get().loadPatients()
      if (get().selectedPatient?.patient_id === patientId) {
        set({ selectedPatient: null, medications: [], checkResult: null, report: null })
      }
      return patients
    } finally {
      set({ loading: false })
    }
  },

  runRiskCheck: async (patientId, drugName) => {
    set({ loading: true })
    try {
      const checkResult = await checkDiabeticRisk(patientId, drugName)
      set({ checkResult })
      return checkResult
    } finally {
      set({ loading: false })
    }
  },

  assessMedicationRisks: async (patientId, medicines) => {
    const checks = await Promise.all(
      (medicines || []).map(async (medicine) => {
        try {
          const result = await checkDiabeticRisk(patientId, medicine.name)
          return {
            medicine: medicine.name,
            ...result,
          }
        } catch {
          return null
        }
      }),
    )
    return checks.filter(Boolean)
  },

  loadLlmRiskAnalysis: async (patientId, drugName) => {
    const llmResult = await checkDiabeticRiskLlm(patientId, drugName)
    set((state) => ({
      checkResult: state.checkResult
        ? { ...state.checkResult, llm_analysis: llmResult.llm_analysis }
        : llmResult,
    }))
    return llmResult
  },

  checkMedicationList: async (patientId) => {
    set({ loading: true })
    try {
      const checkResult = await checkDiabeticMedicationList(patientId)
      set({ checkResult })
      return checkResult
    } finally {
      set({ loading: false })
    }
  },

  loadReport: async (patientId) => {
    set({ loading: true })
    try {
      const report = await getDiabeticReport(patientId)
      set({ report })
      return report
    } finally {
      set({ loading: false })
    }
  },

  downloadReportPdf: async (patientId) => {
    return downloadDiabeticReportPdf(patientId)
  },

  analyzeReportUpload: async (file, autoCreatePatient = true) => {
    set({ loading: true })
    try {
      const reportAnalysis = await analyzeDiabeticReport(file, autoCreatePatient)
      set({ reportAnalysis })
      if (reportAnalysis?.patient_created) {
        await get().loadPatients()
      }
      return reportAnalysis
    } finally {
      set({ loading: false })
    }
  },

  addMedicationToPatient: async (patientId, payload) => {
    await addDiabeticMedication(patientId, payload)
    return get().loadMedications(patientId)
  },

  removeMedicationFromPatient: async (patientId, medicationId) => {
    await removeDiabeticMedication(patientId, medicationId)
    return get().loadMedications(patientId)
  },
}))

export default useDiabetesStore
