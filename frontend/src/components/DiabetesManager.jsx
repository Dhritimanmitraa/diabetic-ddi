import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ArrowLeft, X, Plus, Check } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import toast from 'react-hot-toast'
import usePageTitle from '../hooks/usePageTitle'
import MedicationSchedule from './MedicationSchedule'
import DosageCalculator from './DosageCalculator'
import PatientCard from './diabetes/PatientCard'
import DrugRiskCard, { RiskBadge } from './diabetes/DrugRiskCard'
import { searchDrugs as searchDrugCatalog } from '../services/api'
import useDiabetesStore from '../stores/useDiabetesStore'

export default function DiabetesManager() {
  usePageTitle('Diabetes DDI')
  const navigate = useNavigate()
  const patients = useDiabetesStore((state) => state.patients)
  const selectedPatient = useDiabetesStore((state) => state.selectedPatient)
  const medications = useDiabetesStore((state) => state.medications)
  const checkResult = useDiabetesStore((state) => state.checkResult)
  const report = useDiabetesStore((state) => state.report)
  const loading = useDiabetesStore((state) => state.loading)
  const modelInfo = useDiabetesStore((state) => state.modelInfo)
  const allDrugs = useDiabetesStore((state) => state.allDrugs)
  const drugsLoading = useDiabetesStore((state) => state.drugsLoading)
  const storeReportAnalysis = useDiabetesStore((state) => state.reportAnalysis)
  const setSelectedPatient = useDiabetesStore((state) => state.setSelectedPatient)
  const setCheckResult = useDiabetesStore((state) => state.setCheckResult)
  const loadPatients = useDiabetesStore((state) => state.loadPatients)
  const loadMedications = useDiabetesStore((state) => state.loadMedications)
  const loadModelInfo = useDiabetesStore((state) => state.loadModelInfo)
  const loadAllDrugs = useDiabetesStore((state) => state.loadAllDrugs)
  const createPatientRecord = useDiabetesStore((state) => state.createPatientRecord)
  const deletePatientRecord = useDiabetesStore((state) => state.deletePatientRecord)
  const runRiskCheck = useDiabetesStore((state) => state.runRiskCheck)
  const loadLlmRiskAnalysis = useDiabetesStore((state) => state.loadLlmRiskAnalysis)
  const checkMedicationList = useDiabetesStore((state) => state.checkMedicationList)
  const loadReport = useDiabetesStore((state) => state.loadReport)
  const downloadReportPdf = useDiabetesStore((state) => state.downloadReportPdf)
  const addMedicationToPatient = useDiabetesStore((state) => state.addMedicationToPatient)
  const analyzeReportUpload = useDiabetesStore((state) => state.analyzeReportUpload)
  const [activeSection, setActiveSection] = useState('patients') // patients, check, report

  // New patient form
  const [showNewPatient, setShowNewPatient] = useState(false)
  const [newPatient, setNewPatient] = useState({
    patient_id: '',
    name: '',
    age: '',
    diabetes_type: 'type_2',
    hba1c: '',
    egfr: '',
    potassium: '',
    has_nephropathy: false,
    has_cardiovascular: false,
    has_neuropathy: false,
  })

  // Drug check
  const [drugToCheck, setDrugToCheck] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [searchLoading, setSearchLoading] = useState(false)
  const [searchError, setSearchError] = useState(null)

  // Browse all drugs
  const [showDrugBrowser, setShowDrugBrowser] = useState(false)
  const [showDosageCalculator, setShowDosageCalculator] = useState(false)
  const [drugBrowserPage, setDrugBrowserPage] = useState(0)
  const [drugBrowserSearch, setDrugBrowserSearch] = useState('')

  // Recent drug searches (persisted in localStorage)
  const [recentDrugs, setRecentDrugs] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('recentDrugs') || '[]')
    } catch { return [] }
  })

  // Sample patients loading state
  const [loadingSamples, setLoadingSamples] = useState(false)

  // Report upload states
  const [showReportUpload, setShowReportUpload] = useState(false)
  const [uploadingReport, setUploadingReport] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const reportAnalysis = storeReportAnalysis

  // Upload and analyze lab report with Gemini Vision
  const uploadLabReport = async (file) => {
    if (!file) return

    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg', 'application/pdf']
    if (!validTypes.includes(file.type)) {
      toast.error('Please upload a JPEG, PNG, or PDF file')
      return
    }

    if (file.size > 10 * 1024 * 1024) {
      toast.error('File too large. Max size: 10MB')
      return
    }

    setUploadingReport(true)
    toast.loading('Analyzing report with AI...', { id: 'report-upload' })

    try {
      const data = await analyzeReportUpload(file, true)
      if (data?.patient_created) {
        toast.success(`Patient "${data.extracted_values.patient.name}" created from report!`, { id: 'report-upload' })
      } else {
        toast.success('Report analyzed successfully!', { id: 'report-upload' })
      }
    } catch (err) {
      toast.error(err?.message || 'Error analyzing report', { id: 'report-upload' })
      console.error(err)
    } finally {
      setUploadingReport(false)
    }
  }

  // Handle drag and drop
  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) uploadLabReport(file)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = () => setDragOver(false)

  // Handle file input
  const handleFileSelect = (e) => {
    const file = e.target.files[0]
    if (file) uploadLabReport(file)
  }

  // Load sample patients from real lab reports
  const loadSamplePatients = async () => {
    setLoadingSamples(true)
    toast.loading('Loading sample patients from lab reports...', { id: 'loading-samples' })
    try {
      // Sample patients data extracted from real lab reports - ALL PARAMETERS INCLUDED
      const samplePatients = [
        {
          patient_id: "PAT001",
          name: "Ch. Jagadesh Kumar",
          age: 41,
          gender: "M",
          diabetes_type: "prediabetes",
          labs: {
            hba1c: 6.2,
            mean_blood_glucose: 131.8,  // Mean BG from HbA1c report
            egfr: 95.6,
            creatinine: 1.0
          },
          complications: { has_nephropathy: false, has_cardiovascular: false, has_neuropathy: false }
        },
        {
          patient_id: "PAT002",
          name: "D Demullu",
          age: 64,
          gender: "M",
          diabetes_type: "prediabetes",
          labs: {
            fasting_glucose: 102,
            postprandial_glucose: 124,  // PPBS from report
            creatinine: 1.3,
            // Full Lipid Profile
            total_cholesterol: 146,
            triglycerides: 136,
            hdl_cholesterol: 45,
            ldl_cholesterol: 74,
            vldl_cholesterol: 27
          },
          complications: { has_nephropathy: false, has_hyperlipidemia: false }
        },
        {
          patient_id: "PAT003",
          name: "Ch. Yaryyamma",
          age: 61,
          gender: "F",
          diabetes_type: "type_2",
          years_with_diabetes: 5,
          labs: { hba1c: 7.2, fasting_glucose: 140, egfr: 85, creatinine: 1.1 },
          complications: { has_hypertension: true }
        },
        {
          patient_id: "PAT004",
          name: "Test - Well Controlled",
          age: 55,
          gender: "M",
          diabetes_type: "type_2",
          years_with_diabetes: 8,
          labs: {
            hba1c: 6.8,
            fasting_glucose: 110,
            postprandial_glucose: 135,
            egfr: 92,
            creatinine: 0.9,
            potassium: 4.2,
            alt: 25,
            ast: 22,
            total_cholesterol: 180,
            triglycerides: 120,
            hdl_cholesterol: 50,
            ldl_cholesterol: 100,
            vldl_cholesterol: 24
          },
          complications: { has_hyperlipidemia: true }
        },
        {
          patient_id: "PAT005",
          name: "Test - High Risk",
          age: 68,
          gender: "M",
          diabetes_type: "type_2",
          years_with_diabetes: 15,
          labs: {
            hba1c: 8.5,
            fasting_glucose: 180,
            postprandial_glucose: 250,
            mean_blood_glucose: 200,
            egfr: 35,
            creatinine: 2.5,
            potassium: 5.2,
            alt: 45,
            ast: 40,
            total_cholesterol: 260,
            triglycerides: 220,
            hdl_cholesterol: 35,
            ldl_cholesterol: 170,
            vldl_cholesterol: 44
          },
          complications: { has_nephropathy: true, has_retinopathy: true, has_neuropathy: true, has_cardiovascular: true, has_hypertension: true, has_hyperlipidemia: true, has_obesity: true }
        }
      ]

      let successCount = 0
      for (const patient of samplePatients) {
        try {
          await deletePatientRecord(patient.patient_id).catch(() => {})
          await createPatientRecord(patient)
          successCount++
        } catch (e) {
          console.error(`Failed to create ${patient.patient_id}:`, e)
        }
      }

      toast.success(`Loaded ${successCount}/${samplePatients.length} sample patients!`, { id: 'loading-samples' })
      await loadPatients()
    } catch (err) {
      toast.error('Failed to load sample patients', { id: 'loading-samples' })
      console.error(err)
    } finally {
      setLoadingSamples(false)
    }
  }

  // Add drug to recent history
  const addToRecentDrugs = (drugName) => {
    const updated = [drugName, ...recentDrugs.filter(d => d.toLowerCase() !== drugName.toLowerCase())].slice(0, 8)
    setRecentDrugs(updated)
    localStorage.setItem('recentDrugs', JSON.stringify(updated))
  }

  // Fetch patients on mount
  useEffect(() => {
    void loadPatients().catch((err) => {
      console.error('Error fetching patients:', err)
    })
    void loadModelInfo().catch((err) => {
      console.error('Error fetching model info:', err)
    })
  }, [loadModelInfo, loadPatients])

  // Fetch medications when patient selected
  useEffect(() => {
    if (selectedPatient) {
      void loadMedications(selectedPatient.patient_id).catch((err) => {
        console.error('Error fetching medications:', err)
      })
    }
  }, [loadMedications, selectedPatient])

  // Search drugs from the real DB
  const searchDrugs = async (query) => {
    setSearchQuery(query)
    setSearchError(null)
    if (!query || query.trim().length < 2) {
      setSearchResults([])
      return
    }
    setSearchLoading(true)
    try {
      const data = await searchDrugCatalog(query.trim(), 12)
      setSearchResults(data || [])
    } catch (err) {
      setSearchError('Search failed')
      setSearchResults([])
    } finally {
      setSearchLoading(false)
    }
  }

  const selectDrugFromSearch = (drug) => {
    const name = drug?.name || drug?.generic_name || ''
    setDrugToCheck(name)
    setSearchQuery(name)
    setSearchResults([])
    setCheckResult(null)
  }

  // Fetch all drugs from database for browsing
  const fetchAllDrugs = async (search = '', page = 0) => {
    try {
      await loadAllDrugs(search, page)
    } catch (err) {
      console.error('Error fetching drugs:', err)
    }
  }

  // Open drug browser
  const openDrugBrowser = () => {
    setShowDrugBrowser(true)
    fetchAllDrugs('', 0)
  }

  // Check drug from browser
  const checkDrugFromBrowser = async (drugName) => {
    setDrugToCheck(drugName)
    setSearchQuery(drugName)
    setShowDrugBrowser(false)
    // Auto-check the drug
    if (selectedPatient) {
      try {
        const data = await runRiskCheck(selectedPatient.patient_id, drugName)
        setCheckResult(data)
        setActiveSection('check')
      } catch (err) {
        toast.error('Error checking drug risk')
      }
    }
  }

  const createPatient = async (e) => {
    e.preventDefault()
    try {
      const payload = {
        patient_id: newPatient.patient_id,
        name: newPatient.name,
        age: newPatient.age ? parseInt(newPatient.age) : null,
        diabetes_type: newPatient.diabetes_type,
        labs: {
          hba1c: newPatient.hba1c ? parseFloat(newPatient.hba1c) : null,
          egfr: newPatient.egfr ? parseFloat(newPatient.egfr) : null,
          potassium: newPatient.potassium ? parseFloat(newPatient.potassium) : null,
        },
        complications: {
          has_nephropathy: newPatient.has_nephropathy,
          has_cardiovascular: newPatient.has_cardiovascular,
          has_neuropathy: newPatient.has_neuropathy,
        }
      }
      await createPatientRecord(payload)
      toast.success('Patient created successfully!')
      setShowNewPatient(false)
      setNewPatient({ patient_id: '', name: '', age: '', diabetes_type: 'type_2', hba1c: '', egfr: '', potassium: '', has_nephropathy: false, has_cardiovascular: false, has_neuropathy: false })
    } catch (err) {
      toast.error(err?.message || 'Error creating patient')
    }
  }

  const checkDrugRisk = async () => {
    if (!selectedPatient || !drugToCheck.trim()) {
      toast.error('Select a patient and enter a drug name')
      return
    }
    try {
      const data = await runRiskCheck(selectedPatient.patient_id, drugToCheck.trim())
      setCheckResult(data)
      setActiveSection('check')
      addToRecentDrugs(drugToCheck.trim())

      loadLlmRiskAnalysis(selectedPatient.patient_id, drugToCheck.trim())
        .then(() => {
          toast.success('LLM analysis complete', { duration: 2000 })
        })
        .catch((err) => {
          console.error('LLM analysis failed:', err)
        })
    } catch (err) {
      toast.error(err?.message || 'Error checking drug risk')
    }
  }

  const checkAllMedications = async () => {
    if (!selectedPatient) {
      toast.error('Select a patient first')
      return
    }
    try {
      const data = await checkMedicationList(selectedPatient.patient_id)
      setCheckResult(data)
      setActiveSection('check')
    } catch (err) {
      toast.error(err?.message || 'Error checking medications')
    }
  }

  const generateReport = async () => {
    if (!selectedPatient) {
      toast.error('Select a patient first')
      return
    }
    try {
      await loadReport(selectedPatient.patient_id)
      setActiveSection('report')
    } catch (err) {
      toast.error(err?.message || 'Error generating report')
    }
  }

  const downloadPDF = async () => {
    if (!selectedPatient) {
      toast.error('Select a patient first')
      return
    }
    toast.loading('Generating PDF report...', { id: 'pdf-loading' })
    try {
      const blob = await downloadReportPdf(selectedPatient.patient_id)
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `DrugGuard_Report_${selectedPatient.patient_id}_${new Date().toISOString().split('T')[0]}.pdf`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
      toast.success('PDF downloaded successfully!', { id: 'pdf-loading' })
    } catch (err) {
      toast.error(err?.message || 'Error downloading PDF', { id: 'pdf-loading' })
    }
  }

  const addMedication = async (drugName) => {
    if (!selectedPatient || !drugName.trim()) return
    try {
      await addMedicationToPatient(selectedPatient.patient_id, { drug_name: drugName.trim() })
      toast.success('Medication added')
    } catch (err) {
      toast.error(err?.message || 'Error adding medication')
    }
  }

  return (
    <div className="min-h-screen pt-20 pb-12 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl font-bold text-[var(--text-primary)] mb-2">
            <span className="text-medical-400">Diabetic</span> Patient DDI Checker
          </h1>
          <p className="text-[var(--text-secondary)]">
            Specialized drug interaction analysis for diabetic patients
          </p>
          {modelInfo && (
            <div className="mt-3 flex justify-center gap-2 text-xs text-slate-300">
              <span className={`px-2 py-1 rounded-full border ${modelInfo.loaded ? 'border-emerald-500/50 text-emerald-300 bg-emerald-500/10' : 'border-amber-500/50 text-amber-300 bg-amber-500/10'}`}>
                Model: {modelInfo.loaded ? 'Loaded' : 'Not loaded'}
              </span>
              {modelInfo.model_version && (
                <span className="px-2 py-1 rounded-full border border-[var(--border)] bg-[var(--bg-elevated)]/70">
                  v{modelInfo.model_version}
                </span>
              )}
            </div>
          )}
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Panel - Patients */}
          <div className="lg:col-span-1">
            <div className="glass-card p-6 rounded-2xl">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-[var(--text-primary)]">Patients</h2>
                <div className="flex gap-2">
                  {/* Upload Report Button */}
                  <button
                    onClick={() => setShowReportUpload(!showReportUpload)}
                    disabled={uploadingReport}
                    className="px-3 py-2 rounded-lg bg-purple-500/20 text-purple-400 hover:bg-purple-500/30 transition-colors text-xs font-medium disabled:opacity-50 flex items-center gap-1"
                    title="Upload lab report for AI analysis"
                  >
                    {uploadingReport ? (
                      <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                    ) : (
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                    )}
                    Upload Report
                  </button>
                  <button
                    onClick={loadSamplePatients}
                    disabled={loadingSamples}
                    className="px-3 py-2 rounded-lg bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 transition-colors text-xs font-medium disabled:opacity-50 flex items-center gap-1"
                    title="Load 5 sample patients from real lab reports"
                  >
                    {loadingSamples ? (
                      <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                    ) : (
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                      </svg>
                    )}
                    Samples
                  </button>
                  <button
                    onClick={() => setShowNewPatient(!showNewPatient)}
                    className="p-2 rounded-lg bg-medical-500/20 text-medical-400 hover:bg-medical-500/30 transition-colors"
                    title="Add new patient"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                    </svg>
                  </button>
                </div>
              </div>

              {/* New Patient Form */}
              <AnimatePresence>
                {showNewPatient && (
                  <motion.form
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    onSubmit={createPatient}
                    className="mb-4 p-4 bg-[var(--bg-elevated)]/50 rounded-xl border border-[var(--border)]/50 space-y-3"
                  >
                    <input
                      type="text"
                      placeholder="Patient ID *"
                      value={newPatient.patient_id}
                      onChange={(e) => setNewPatient({ ...newPatient, patient_id: e.target.value })}
                      className="w-full px-3 py-2 bg-[var(--bg-elevated)]/50 border border-[var(--border)] rounded-lg text-[var(--text-primary)] text-sm"
                      required
                    />
                    <input
                      type="text"
                      placeholder="Name"
                      value={newPatient.name}
                      onChange={(e) => setNewPatient({ ...newPatient, name: e.target.value })}
                      className="w-full px-3 py-2 bg-[var(--bg-elevated)]/50 border border-[var(--border)] rounded-lg text-[var(--text-primary)] text-sm"
                    />
                    <div className="grid grid-cols-2 gap-2">
                      <input
                        type="number"
                        placeholder="Age"
                        value={newPatient.age}
                        onChange={(e) => setNewPatient({ ...newPatient, age: e.target.value })}
                        className="px-3 py-2 bg-[var(--bg-elevated)]/50 border border-[var(--border)] rounded-lg text-[var(--text-primary)] text-sm"
                      />
                      <select
                        value={newPatient.diabetes_type}
                        onChange={(e) => setNewPatient({ ...newPatient, diabetes_type: e.target.value })}
                        className="px-3 py-2 bg-[var(--bg-elevated)]/50 border border-[var(--border)] rounded-lg text-[var(--text-primary)] text-sm"
                      >
                        <option value="type_1">Type 1</option>
                        <option value="type_2">Type 2</option>
                        <option value="gestational">Gestational</option>
                        <option value="prediabetes">Prediabetes</option>
                      </select>
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                      <input
                        type="number"
                        step="0.1"
                        placeholder="HbA1c %"
                        value={newPatient.hba1c}
                        onChange={(e) => setNewPatient({ ...newPatient, hba1c: e.target.value })}
                        className="px-3 py-2 bg-[var(--bg-elevated)]/50 border border-[var(--border)] rounded-lg text-[var(--text-primary)] text-sm"
                      />
                      <input
                        type="number"
                        placeholder="eGFR"
                        value={newPatient.egfr}
                        onChange={(e) => setNewPatient({ ...newPatient, egfr: e.target.value })}
                        className="px-3 py-2 bg-[var(--bg-elevated)]/50 border border-[var(--border)] rounded-lg text-[var(--text-primary)] text-sm"
                      />
                      <input
                        type="number"
                        step="0.1"
                        placeholder="K+ mEq/L"
                        value={newPatient.potassium}
                        onChange={(e) => setNewPatient({ ...newPatient, potassium: e.target.value })}
                        className="px-3 py-2 bg-[var(--bg-elevated)]/50 border border-[var(--border)] rounded-lg text-[var(--text-primary)] text-sm"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-xs text-[var(--text-secondary)]">Complications:</label>
                      <div className="flex flex-wrap gap-2">
                        {['nephropathy', 'cardiovascular', 'neuropathy'].map(comp => (
                          <label key={comp} className="flex items-center gap-1 text-sm text-slate-300">
                            <input
                              type="checkbox"
                              checked={newPatient[`has_${comp}`]}
                              onChange={(e) => setNewPatient({ ...newPatient, [`has_${comp}`]: e.target.checked })}
                              className="rounded bg-[var(--bg-elevated)] border-[var(--border)]"
                            />
                            {comp.charAt(0).toUpperCase() + comp.slice(1)}
                          </label>
                        ))}
                      </div>
                    </div>
                    <button
                      type="submit"
                      disabled={loading}
                      className="w-full py-2 bg-medical-500 text-white rounded-lg hover:bg-medical-600 transition-colors disabled:opacity-50"
                    >
                      {loading ? 'Creating...' : 'Create Patient'}
                    </button>
                  </motion.form>
                )}
              </AnimatePresence>

              {/* Report Upload Section */}
              <AnimatePresence>
                {showReportUpload && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mb-4"
                  >
                    <div className="p-4 bg-gradient-to-br from-purple-900/20 to-[var(--bg-elevated)] rounded-xl border border-purple-500/30">
                      <h3 className="text-sm font-medium text-purple-400 mb-3 flex items-center gap-2">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                        </svg>
                        AI Report Analysis (Gemini Vision)
                      </h3>

                      {/* Drop Zone */}
                      <div
                        onDrop={handleDrop}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        className={`relative border-2 border-dashed rounded-lg p-6 text-center transition-all cursor-pointer ${dragOver
                          ? 'border-purple-400 bg-purple-500/20'
                          : 'border-[var(--border)] hover:border-purple-500/50 bg-[var(--bg-elevated)]/50'
                          }`}
                      >
                        <input
                          type="file"
                          accept="image/jpeg,image/png,application/pdf"
                          onChange={handleFileSelect}
                          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                        />
                        {uploadingReport ? (
                          <div className="flex flex-col items-center">
                            <svg className="w-8 h-8 animate-spin text-purple-400 mb-2" fill="none" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                            </svg>
                            <p className="text-purple-400 text-sm">Analyzing with Gemini...</p>
                          </div>
                        ) : (
                          <>
                            <svg className="w-10 h-10 text-[var(--text-muted)] mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                            </svg>
                            <p className="text-[var(--text-secondary)] text-sm">Drop lab report here or click to upload</p>
                            <p className="text-[var(--text-muted)] text-xs mt-1">JPEG, PNG, or PDF • Max 10MB</p>
                          </>
                        )}
                      </div>

                      {/* Extracted Results */}
                      {reportAnalysis && (
                        <div className="mt-4 space-y-3">
                          {/* Patient Info */}
                          {reportAnalysis.extracted_values?.patient?.name && (
                            <div className="p-3 bg-[var(--bg-elevated)]/50 rounded-lg">
                              <div className="flex items-center justify-between">
                                <span className="text-[var(--text-primary)] font-medium">
                                  {reportAnalysis.extracted_values.patient.name}
                                </span>
                                <span className="text-xs px-2 py-0.5 bg-emerald-500/20 text-emerald-400 rounded">
                                  {reportAnalysis.patient_created ? 'Patient Created' : 'Extracted'}
                                </span>
                              </div>
                              <p className="text-[var(--text-secondary)] text-xs mt-1">
                                {reportAnalysis.extracted_values.patient.age && `Age: ${reportAnalysis.extracted_values.patient.age}`}
                                {reportAnalysis.extracted_values.patient.gender && ` • ${reportAnalysis.extracted_values.patient.gender}`}
                              </p>
                            </div>
                          )}

                          {/* Lab Values Grid */}
                          <div className="grid grid-cols-2 gap-2 text-xs">
                            {reportAnalysis.extracted_values?.glucose?.hba1c && (
                              <div className="p-2 bg-[var(--bg-elevated)]/50 rounded">
                                <span className="text-[var(--text-secondary)]">HbA1c:</span>
                                <span className="text-[var(--text-primary)] ml-1">{reportAnalysis.extracted_values.glucose.hba1c}%</span>
                              </div>
                            )}
                            {reportAnalysis.extracted_values?.glucose?.fasting_glucose && (
                              <div className="p-2 bg-[var(--bg-elevated)]/50 rounded">
                                <span className="text-[var(--text-secondary)]">FBS:</span>
                                <span className="text-[var(--text-primary)] ml-1">{reportAnalysis.extracted_values.glucose.fasting_glucose}</span>
                              </div>
                            )}
                            {reportAnalysis.extracted_values?.kidney?.egfr && (
                              <div className="p-2 bg-[var(--bg-elevated)]/50 rounded">
                                <span className="text-[var(--text-secondary)]">eGFR:</span>
                                <span className="text-[var(--text-primary)] ml-1">{reportAnalysis.extracted_values.kidney.egfr}</span>
                              </div>
                            )}
                            {reportAnalysis.extracted_values?.kidney?.creatinine && (
                              <div className="p-2 bg-[var(--bg-elevated)]/50 rounded">
                                <span className="text-[var(--text-secondary)]">Creat:</span>
                                <span className="text-[var(--text-primary)] ml-1">{reportAnalysis.extracted_values.kidney.creatinine}</span>
                              </div>
                            )}
                            {reportAnalysis.extracted_values?.lipid?.total_cholesterol && (
                              <div className="p-2 bg-[var(--bg-elevated)]/50 rounded">
                                <span className="text-[var(--text-secondary)]">Chol:</span>
                                <span className="text-[var(--text-primary)] ml-1">{reportAnalysis.extracted_values.lipid.total_cholesterol}</span>
                              </div>
                            )}
                            {reportAnalysis.extracted_values?.lipid?.triglycerides && (
                              <div className="p-2 bg-[var(--bg-elevated)]/50 rounded">
                                <span className="text-[var(--text-secondary)]">TG:</span>
                                <span className="text-[var(--text-primary)] ml-1">{reportAnalysis.extracted_values.lipid.triglycerides}</span>
                              </div>
                            )}
                          </div>

                          {/* AI Health Summary */}
                          {reportAnalysis.health_summary?.overall_status && (
                            <div className={`p-3 rounded-lg ${reportAnalysis.health_summary.overall_status === 'good' ? 'bg-emerald-500/10 border border-emerald-500/30' :
                              reportAnalysis.health_summary.overall_status === 'moderate' ? 'bg-amber-500/10 border border-amber-500/30' :
                                'bg-red-500/10 border border-red-500/30'
                              }`}>
                              <div className="flex items-center gap-2 mb-2">
                                <span className={`text-sm font-medium ${reportAnalysis.health_summary.overall_status === 'good' ? 'text-emerald-400' :
                                  reportAnalysis.health_summary.overall_status === 'moderate' ? 'text-amber-400' :
                                    'text-red-400'
                                  }`}>
                                  AI Assessment: {reportAnalysis.health_summary.overall_status.toUpperCase()}
                                </span>
                              </div>
                              {reportAnalysis.health_summary.key_findings?.length > 0 && (
                                <ul className="text-xs text-slate-300 space-y-1">
                                  {reportAnalysis.health_summary.key_findings.slice(0, 3).map((f, i) => (
                                    <li key={i}>• {f}</li>
                                  ))}
                                </ul>
                              )}
                            </div>
                          )}

                          {/* Use Patient Button */}
                          {reportAnalysis.patient_id && (
                            <button
                              onClick={() => {
                                const patient = patients.find(p => p.patient_id === reportAnalysis.patient_id)
                                if (patient) {
                                  setSelectedPatient(patient)
                                  setShowReportUpload(false)
                                  toast.success('Patient selected! Now check drug risks.')
                                }
                              }}
                              className="w-full py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors text-sm"
                            >
                              Use This Patient for DDI Check
                            </button>
                          )}
                        </div>
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Patient List */}
              <div className="space-y-3 max-h-[500px] overflow-y-auto">
                {patients.length === 0 ? (
                  <p className="text-[var(--text-muted)] text-center py-4">No patients yet. Create one above.</p>
                ) : (
                  patients.map(patient => (
                    <PatientCard
                      key={patient.id}
                      patient={patient}
                      onSelect={setSelectedPatient}
                      isSelected={selectedPatient?.id === patient.id}
                    />
                  ))
                )}
              </div>
            </div>
          </div>

          {/* Right Panel - Drug Check & Results */}
          <div className="lg:col-span-2">
            {!selectedPatient ? (
              <div className="glass-card p-12 rounded-2xl text-center">
                <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-medical-500/10 flex items-center justify-center">
                  <ArrowLeft className="w-8 h-8 text-medical-400" />
                </div>
                <h3 className="text-xl font-semibold text-[var(--text-primary)] mb-2">Select a Patient</h3>
                <p className="text-[var(--text-secondary)]">Choose or create a patient to check drug risks</p>
              </div>
            ) : (
              <div className="space-y-6">
                {/* Selected Patient Info */}
                <div className="glass-card p-6 rounded-2xl">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h2 className="text-2xl font-bold text-[var(--text-primary)]">{selectedPatient.name || selectedPatient.patient_id}</h2>
                      <p className="text-[var(--text-secondary)]">
                        {selectedPatient.diabetes_type.replace('_', ' ')} diabetes •
                        {selectedPatient.age ? ` ${selectedPatient.age} years old` : ''} •
                        HbA1c: {selectedPatient.hba1c || 'N/A'}% •
                        eGFR: {selectedPatient.egfr || 'N/A'} •
                        Kidney: {selectedPatient.kidney_stage || 'unknown'}
                      </p>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex flex-wrap gap-3 mb-4">
                    <div className="flex-1 flex flex-col gap-2">
                      <div className="flex gap-2">
                        <input
                          type="text"
                          placeholder="Search drug from DB..."
                          value={searchQuery}
                          onChange={(e) => {
                            setDrugToCheck(e.target.value)
                            searchDrugs(e.target.value)
                          }}
                          onKeyDown={(e) => e.key === 'Enter' && checkDrugRisk()}
                          className="flex-1 px-4 py-2 bg-[var(--bg-elevated)] border border-[var(--border)] rounded-xl text-medical-300 placeholder:text-[var(--text-muted)] font-medium"
                        />
                        <button
                          onClick={checkDrugRisk}
                          disabled={loading || !drugToCheck.trim()}
                          className="px-6 py-2 bg-medical-500 text-white rounded-xl hover:bg-medical-600 transition-colors disabled:opacity-50"
                        >
                          Check Drug
                        </button>
                      </div>
                      {/* Autocomplete dropdown */}
                      {searchQuery.trim().length >= 2 && (
                        <div className="relative">
                          <div className="absolute z-20 mt-1 w-full bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl shadow-lg max-h-56 overflow-y-auto">
                            {searchLoading && (
                              <div className="px-3 py-2 text-sm text-[var(--text-secondary)]">Searching...</div>
                            )}
                            {searchError && (
                              <div className="px-3 py-2 text-sm text-red-400">{searchError}</div>
                            )}
                            {!searchLoading && !searchError && searchResults.length === 0 && (
                              <div className="px-3 py-2 text-sm text-slate-500">No matches</div>
                            )}
                            {searchResults.map((d) => (
                              <button
                                key={d.id}
                                type="button"
                                onClick={() => selectDrugFromSearch(d)}
                                className="w-full text-left px-3 py-2 hover:bg-[var(--bg-elevated)] text-sm transition-colors flex justify-between gap-2"
                              >
                                <span className="text-medical-300 font-medium">{d.name}</span>
                                {d.generic_name && (
                                  <span className="text-xs text-[var(--text-secondary)]">{d.generic_name}</span>
                                )}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                    <button
                      onClick={openDrugBrowser}
                      disabled={loading}
                      className="px-4 py-2 bg-purple-500/20 text-purple-400 border border-purple-500/30 rounded-xl hover:bg-purple-500/30 transition-colors disabled:opacity-50"
                    >
                      Browse All Drugs
                    </button>
                    <button
                      onClick={() => navigate(`/patient-prescription?patient=${selectedPatient?.patient_id}`)}
                      disabled={loading}
                      className="px-4 py-2 bg-indigo-500/20 text-indigo-400 border border-indigo-500/30 rounded-xl hover:bg-indigo-500/30 transition-colors disabled:opacity-50 flex items-center gap-2"
                      title="Scan prescription for this patient"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                      </svg>
                      Scan Rx
                    </button>
                    <button
                      onClick={checkAllMedications}
                      disabled={loading || medications.length === 0}
                      title={medications.length === 0 ? 'Add medications to patient first using + Add button below' : 'Check all patient medications'}
                      className="px-4 py-2 bg-amber-500/20 text-amber-400 border border-amber-500/30 rounded-xl hover:bg-amber-500/30 transition-colors disabled:opacity-50"
                    >
                      Check Patient Meds ({medications.length})
                    </button>
                    <button
                      onClick={generateReport}
                      disabled={loading}
                      className="px-4 py-2 bg-blue-500/20 text-blue-400 border border-blue-500/30 rounded-xl hover:bg-blue-500/30 transition-colors disabled:opacity-50"
                    >
                      Full Report
                    </button>
                    <button
                      onClick={downloadPDF}
                      disabled={loading}
                      className="px-4 py-2 bg-purple-500/20 text-purple-400 border border-purple-500/30 rounded-xl hover:bg-purple-500/30 transition-colors disabled:opacity-50 flex items-center gap-2"
                      title="Download PDF Report"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      PDF
                    </button>
                    <button
                      onClick={() => setShowDosageCalculator(true)}
                      disabled={loading}
                      className="px-4 py-2 bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 rounded-xl hover:bg-emerald-500/30 transition-colors disabled:opacity-50 flex items-center gap-2"
                      title="Dosage Calculator"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                      </svg>
                      Dose Calc
                    </button>
                  </div>

                  {/* Quick chips for common risk classes */}
                  <div className="flex flex-wrap gap-2 mb-4">
                    {[
                      { label: 'Steroids', value: 'prednisone' },
                      { label: 'Sulfonylurea', value: 'glyburide' },
                      { label: 'TZD', value: 'pioglitazone' },
                      { label: 'NSAID', value: 'ibuprofen' },
                      { label: 'SGLT2i', value: 'empagliflozin' },
                    ].map((chip) => (
                      <button
                        key={chip.value}
                        onClick={() => {
                          setDrugToCheck(chip.value)
                          setSearchQuery(chip.value)
                          searchDrugs(chip.value)
                        }}
                        className="px-3 py-1 rounded-full text-xs bg-slate-700/60 text-slate-200 border border-[var(--border)] hover:border-medical-400 transition-colors"
                      >
                        {chip.label}
                      </button>
                    ))}
                  </div>

                  {/* Recent Drug Searches */}
                  {recentDrugs.length > 0 && (
                    <div className="mb-4">
                      <h4 className="text-xs font-medium text-[var(--text-muted)] mb-2 flex items-center gap-1">
                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        Recent Searches
                      </h4>
                      <div className="flex flex-wrap gap-2">
                        {recentDrugs.map((drug, i) => (
                          <button
                            key={i}
                            onClick={() => {
                              setDrugToCheck(drug)
                              setSearchQuery(drug)
                              setSearchResults([])
                            }}
                            className="px-3 py-1 rounded-full text-xs bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 hover:border-indigo-400 transition-colors flex items-center gap-1"
                          >
                            <span>{drug}</span>
                          </button>
                        ))}
                        <button
                          onClick={() => {
                            setRecentDrugs([])
                            localStorage.removeItem('recentDrugs')
                          }}
                          className="px-2 py-1 rounded-full text-xs text-[var(--text-muted)] hover:text-slate-300 transition-colors"
                          title="Clear history"
                        >
                          <X className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Current Medications */}
                  <div>
                    <h4 className="text-sm font-medium text-[var(--text-secondary)] mb-2">Current Medications ({medications.length})</h4>
                    <div className="flex flex-wrap gap-2">
                      {medications.map(med => (
                        <span key={med.id} className="px-3 py-1 bg-medical-500/15 text-medical-300 border border-medical-500/20 rounded-full text-sm font-medium">
                          {med.drug_name}
                        </span>
                      ))}
                      <AddMedicationInline onAdd={addMedication} />
                    </div>
                  </div>
                </div>

                {/* Results */}
                <AnimatePresence mode="wait">
                  {loading && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="glass-card p-12 rounded-2xl text-center"
                    >
                      <div className="spinner mx-auto mb-4"></div>
                      <p className="text-[var(--text-secondary)]">Analyzing drug safety...</p>
                    </motion.div>
                  )}

                  {/* Single Drug Check Result */}
                  {!loading && checkResult && activeSection === 'check' && !checkResult.assessments && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="glass-card p-6 rounded-2xl"
                    >
                      <h3 className="text-xl font-bold text-[var(--text-primary)] mb-4">Drug Risk Assessment</h3>
                      <DrugRiskCard assessment={checkResult} />
                    </motion.div>
                  )}

                  {/* Multiple Drugs Check Result */}
                  {!loading && checkResult && activeSection === 'check' && checkResult.assessments && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="glass-card p-6 rounded-2xl"
                    >
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-xl font-bold text-[var(--text-primary)]">Medication List Assessment</h3>
                        <RiskBadge level={checkResult.overall_risk_level} />
                      </div>

                      {/* Summary */}
                      <div className="grid grid-cols-5 gap-2 mb-4">
                        {[
                          { label: 'Safe', count: checkResult.safe_count, bg: 'bg-emerald-500/10', text: 'text-emerald-400' },
                          { label: 'Caution', count: checkResult.caution_count, bg: 'bg-amber-500/10', text: 'text-amber-400' },
                          { label: 'High Risk', count: checkResult.high_risk_count, bg: 'bg-orange-500/10', text: 'text-orange-400' },
                          { label: 'Contraind.', count: checkResult.contraindicated_count, bg: 'bg-red-500/10', text: 'text-red-400' },
                          { label: 'Fatal', count: checkResult.fatal_count, bg: 'bg-red-900/20', text: 'text-red-300' },
                        ].map(item => (
                          <div key={item.label} className={`p-3 rounded-lg ${item.bg} text-center`}>
                            <div className={`text-2xl font-bold ${item.text}`}>{item.count}</div>
                            <div className="text-xs text-[var(--text-secondary)]">{item.label}</div>
                          </div>
                        ))}
                      </div>

                      {/* Critical Alerts */}
                      {checkResult.critical_alerts?.length > 0 && (
                        <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-xl">
                          <h4 className="text-sm font-bold text-red-400 mb-2">Critical Alerts</h4>
                          {checkResult.critical_alerts.map((alert, i) => (
                            <p key={i} className="text-sm text-red-300">{alert}</p>
                          ))}
                        </div>
                      )}

                      {/* Recommendations */}
                      {checkResult.recommendations?.length > 0 && (
                        <div className="mb-4">
                          <h4 className="text-sm font-medium text-[var(--text-secondary)] mb-2">Recommendations</h4>
                          {checkResult.recommendations.map((rec, i) => (
                            <p key={i} className="text-sm text-slate-300 mb-1">{rec}</p>
                          ))}
                        </div>
                      )}

                      {/* Individual Assessments */}
                      <div className="space-y-3">
                        {checkResult.assessments.map((assessment, i) => (
                          <DrugRiskCard key={i} assessment={assessment} />
                        ))}
                      </div>
                    </motion.div>
                  )}

                  {/* Full Report */}
                  {!loading && report && activeSection === 'report' && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="glass-card p-6 rounded-2xl"
                    >
                      <div className="flex items-center justify-between mb-6">
                        <h3 className="text-xl font-bold text-[var(--text-primary)]">Full DDI Report</h3>
                        <div className="text-right">
                          <div className={`text-3xl font-bold ${report.overall_safety_score > 70 ? 'text-emerald-400' :
                            report.overall_safety_score > 40 ? 'text-amber-400' : 'text-red-400'
                            }`}>
                            {report.overall_safety_score}%
                          </div>
                          <div className="text-xs text-[var(--text-secondary)]">Safety Score</div>
                        </div>
                      </div>

                      {report.action_required && (
                        <div className="mb-4 p-4 bg-red-500/10 border border-red-500/30 rounded-xl">
                          <div className="flex items-center gap-2 text-red-400 font-bold mb-2">
                            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                            </svg>
                            ACTION REQUIRED
                          </div>
                          <p className="text-red-300 text-sm">{report.summary}</p>
                        </div>
                      )}

                      {/* Fatal Risks */}
                      {report.fatal_risks?.length > 0 && (
                        <div className="mb-4">
                          <h4 className="text-sm font-bold text-red-400 mb-2">Fatal Risks</h4>
                          <div className="space-y-2">
                            {report.fatal_risks.map((risk, i) => (
                              <div key={i} className="p-3 bg-red-900/30 border border-red-700 rounded-lg">
                                <span className="font-medium text-red-300">{risk.drug}</span>
                                <p className="text-sm text-red-400">{risk.reason}</p>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Monitoring Plan */}
                      {report.monitoring_plan?.length > 0 && (
                        <div className="mb-4">
                          <h4 className="text-sm font-medium text-[var(--text-secondary)] mb-2">Monitoring Plan</h4>
                          <div className="flex flex-wrap gap-2">
                            {report.monitoring_plan.map((item, i) => (
                              <span key={i} className="px-3 py-1 bg-blue-500/20 text-blue-400 rounded-full text-sm">{item}</span>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Medication Schedule */}
                      <MedicationSchedule
                        patientId={selectedPatient?.patient_id}
                        medications={medications}
                      />

                      <p className="text-sm text-[var(--text-muted)] mt-4">
                        Report generated: {new Date(report.report_generated_at).toLocaleString()}
                      </p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Drug Browser Modal */}
      <AnimatePresence>
        {showDrugBrowser && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
            onClick={(e) => e.target === e.currentTarget && setShowDrugBrowser(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-2xl w-full max-w-4xl max-h-[80vh] overflow-hidden shadow-2xl"
            >
              {/* Header */}
              <div className="p-4 border-b border-[var(--border)] flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-bold text-[var(--text-primary)]">Browse All Drugs</h2>
                  <p className="text-sm text-[var(--text-secondary)]">Click any drug to check risk for {selectedPatient?.name || 'patient'}</p>
                </div>
                <button
                  onClick={() => setShowDrugBrowser(false)}
                  className="p-2 hover:bg-[var(--bg-elevated)] rounded-lg text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              {/* Search */}
              <div className="p-4 border-b border-[var(--border)]">
                <input
                  type="text"
                  placeholder="Search drugs by name..."
                  value={drugBrowserSearch}
                  onChange={(e) => {
                    setDrugBrowserSearch(e.target.value)
                    fetchAllDrugs(e.target.value, 0)
                  }}
                  className="w-full px-4 py-3 bg-[var(--bg-elevated)] border border-[var(--border)] rounded-xl text-medical-300 placeholder-[var(--text-muted)] focus:border-medical-500 focus:outline-none font-medium"
                />
              </div>

              {/* Drug List */}
              <div className="p-4 overflow-y-auto max-h-[50vh]">
                {drugsLoading ? (
                  <div className="text-center py-8">
                    <div className="spinner mx-auto mb-2"></div>
                    <p className="text-[var(--text-secondary)]">Loading drugs...</p>
                  </div>
                ) : allDrugs.length === 0 ? (
                  <div className="text-center py-8">
                    <p className="text-[var(--text-secondary)]">No drugs found. Try a different search.</p>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                    {allDrugs.map((drug) => (
                      <button
                        key={drug.id}
                        onClick={() => checkDrugFromBrowser(drug.name || drug.generic_name)}
                        className="p-3 bg-[var(--bg-elevated)]/50 hover:bg-[var(--bg-elevated)]/80 border border-[var(--border)] hover:border-medical-500/50 rounded-xl text-left transition-all group"
                      >
                        <div className="font-semibold text-medical-300 group-hover:text-medical-200 transition-colors">
                          {drug.name}
                        </div>
                        {drug.generic_name && drug.generic_name !== drug.name && (
                          <div className="text-xs text-[var(--text-secondary)]">{drug.generic_name}</div>
                        )}
                        {drug.drug_class && (
                          <div className="text-xs text-[var(--text-muted)] mt-1">{drug.drug_class}</div>
                        )}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {/* Pagination */}
              <div className="p-4 border-t border-[var(--border)] flex items-center justify-between">
                <button
                  onClick={() => {
                    const newPage = Math.max(0, drugBrowserPage - 1)
                    setDrugBrowserPage(newPage)
                    fetchAllDrugs(drugBrowserSearch, newPage)
                  }}
                  disabled={drugBrowserPage === 0}
                  className="px-4 py-2 bg-[var(--bg-elevated)] text-slate-300 rounded-lg hover:bg-[var(--bg-elevated)] disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  ← Previous
                </button>
                <span className="text-[var(--text-secondary)]">
                  Page {drugBrowserPage + 1} • Showing {allDrugs.length} drugs
                </span>
                <button
                  onClick={() => {
                    const newPage = drugBrowserPage + 1
                    setDrugBrowserPage(newPage)
                    fetchAllDrugs(drugBrowserSearch, newPage)
                  }}
                  disabled={allDrugs.length < 50}
                  className="px-4 py-2 bg-[var(--bg-elevated)] text-slate-300 rounded-lg hover:bg-[var(--bg-elevated)] disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Next →
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Dosage Calculator Modal */}
      <AnimatePresence>
        {showDosageCalculator && selectedPatient && (
          <DosageCalculator
            patient={selectedPatient}
            onClose={() => setShowDosageCalculator(false)}
          />
        )}
      </AnimatePresence>
    </div>
  )
}

// Inline medication add component — replaces native prompt()
function AddMedicationInline({ onAdd }) {
  const [isAdding, setIsAdding] = useState(false)
  const [name, setName] = useState('')
  const inputRef = useRef(null)

  useEffect(() => {
    if (isAdding && inputRef.current) {
      inputRef.current.focus()
    }
  }, [isAdding])

  const handleSubmit = () => {
    if (name.trim()) {
      onAdd(name.trim())
      setName('')
      setIsAdding(false)
    }
  }

  if (!isAdding) {
    return (
      <button
        onClick={() => setIsAdding(true)}
        className="px-3 py-1 bg-medical-500/20 text-medical-400 rounded-full text-sm hover:bg-medical-500/30 transition-colors flex items-center gap-1"
      >
        <Plus className="w-3.5 h-3.5" /> Add
      </button>
    )
  }

  return (
    <div className="flex items-center gap-1.5">
      <input
        ref={inputRef}
        type="text"
        value={name}
        onChange={(e) => setName(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') handleSubmit()
          if (e.key === 'Escape') { setIsAdding(false); setName('') }
        }}
        placeholder="Medication name..."
        className="px-3 py-1 bg-[var(--bg-elevated)] border border-[var(--border)] rounded-full text-sm text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:border-medical-500 focus:outline-none w-40"
      />
      <button
        onClick={handleSubmit}
        disabled={!name.trim()}
        className="p-1 rounded-full bg-medical-500/20 text-medical-400 hover:bg-medical-500/30 disabled:opacity-40 transition-colors"
      >
        <Check className="w-3.5 h-3.5" />
      </button>
      <button
        onClick={() => { setIsAdding(false); setName('') }}
        className="p-1 rounded-full bg-slate-700/50 text-[var(--text-muted)] hover:bg-slate-700 transition-colors"
      >
        <X className="w-3.5 h-3.5" />
      </button>
    </div>
  )
}
