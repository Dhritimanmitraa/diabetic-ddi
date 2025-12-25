import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import toast from 'react-hot-toast'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001'

// Risk level badge component
const RiskBadge = ({ level }) => {
  const config = {
    safe: { bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/30', label: 'Safe' },
    caution: { bg: 'bg-amber-500/20', text: 'text-amber-400', border: 'border-amber-500/30', label: 'Caution' },
    high_risk: { bg: 'bg-orange-500/20', text: 'text-orange-400', border: 'border-orange-500/30', label: 'High Risk' },
    contraindicated: { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/30', label: 'Contraindicated' },
    fatal: { bg: 'bg-red-900/40', text: 'text-red-300', border: 'border-red-700', label: 'Fatal Risk' },
  }
  const { bg, text, border, label } = config[level] || config.caution
  return (
    <span className={`px-3 py-1 rounded-full text-sm font-medium ${bg} ${text} border ${border}`}>
      {label}
    </span>
  )
}

const MLBadge = ({ mlRisk, mlProb, source }) => {
  if (!mlRisk && mlProb == null) return null
  const probText = mlProb != null ? `p=${Math.round(mlProb * 100)}%` : ''
  const src = source ? source.replace('_', ' ') : ''
  return (
    <div className="flex items-center gap-2 text-xs text-slate-400">
      <span className="px-2 py-1 rounded-full bg-slate-700/70 border border-slate-600 text-slate-200">
        ML: {mlRisk || 'n/a'} {probText && `(${probText})`}
      </span>
      {src && (
        <span
          className={`px-2 py-1 rounded-full border ${
            source === 'rule_override'
              ? 'border-red-500/50 text-red-300 bg-red-500/10'
              : 'border-slate-700 bg-slate-800/70'
          }`}
        >
          {src}
        </span>
      )}
    </div>
  )
}

// Patient card component
const PatientCard = ({ patient, onSelect, isSelected }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    onClick={() => onSelect(patient)}
    className={`p-4 rounded-xl cursor-pointer transition-all duration-300 ${
      isSelected 
        ? 'bg-medical-500/20 border-2 border-medical-500' 
        : 'bg-slate-800/50 border border-slate-700/50 hover:border-medical-500/50'
    }`}
  >
    <div className="flex items-center justify-between">
      <div>
        <h4 className="font-semibold text-white">{patient.name || patient.patient_id}</h4>
        <p className="text-sm text-slate-400">
          {patient.diabetes_type.replace('_', ' ').toUpperCase()} • {patient.years_with_diabetes || '?'} years
        </p>
      </div>
      <div className="text-right">
        <div className="text-medical-400 font-medium">HbA1c: {patient.hba1c || 'N/A'}%</div>
        <div className="text-xs text-slate-500">eGFR: {patient.egfr || 'N/A'}</div>
      </div>
    </div>
    {(patient.has_nephropathy || patient.has_cardiovascular || patient.has_neuropathy) && (
      <div className="mt-2 flex gap-2">
        {patient.has_nephropathy && <span className="text-xs bg-red-500/20 text-red-400 px-2 py-1 rounded">Nephropathy</span>}
        {patient.has_cardiovascular && <span className="text-xs bg-red-500/20 text-red-400 px-2 py-1 rounded">Cardiovascular</span>}
        {patient.has_neuropathy && <span className="text-xs bg-red-500/20 text-red-400 px-2 py-1 rounded">Neuropathy</span>}
      </div>
    )}
  </motion.div>
)

// Drug risk assessment card
const DrugRiskCard = ({ assessment }) => (
  <motion.div
    initial={{ opacity: 0, scale: 0.95 }}
    animate={{ opacity: 1, scale: 1 }}
    className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/50"
  >
    <div className="flex items-center justify-between mb-3">
      <h4 className="font-semibold text-white text-lg">{assessment.drug_name}</h4>
      <RiskBadge level={assessment.risk_level} />
    </div>

    {assessment.severity && (
      <div className="flex items-center gap-2 text-xs text-slate-400 mb-2">
        <span className="px-2 py-1 rounded-full bg-slate-700/70 border border-slate-600">Severity: {assessment.severity}</span>
      </div>
    )}

    <div className="mb-3">
      <MLBadge mlRisk={assessment.ml_risk_level} mlProb={assessment.ml_probability} source={assessment.ml_decision_source} />
    </div>
    
    <div className="mb-3">
      <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
        <div 
          className={`h-full transition-all ${
            assessment.risk_score < 20 ? 'bg-emerald-500' :
            assessment.risk_score < 40 ? 'bg-amber-500' :
            assessment.risk_score < 60 ? 'bg-orange-500' : 'bg-red-500'
          }`}
          style={{ width: `${assessment.risk_score}%` }}
        />
      </div>
      <p className="text-xs text-slate-500 mt-1">Risk Score: {assessment.risk_score}/100</p>
    </div>

    <p className="text-sm text-slate-300 mb-3">{assessment.recommendation}</p>

    {assessment.risk_factors?.length > 0 && (
      <div className="mb-3">
        <h5 className="text-xs font-medium text-slate-400 mb-1">Risk Factors:</h5>
        <ul className="text-sm text-slate-300 space-y-1">
          {assessment.risk_factors.map((factor, i) => (
            <li key={i} className="flex items-start gap-2">
              <span className="text-red-400">•</span>
              {factor}
            </li>
          ))}
        </ul>
      </div>
    )}

    {assessment.rule_references?.length > 0 && (
      <div className="mb-3">
        <h5 className="text-xs font-medium text-slate-400 mb-1">Why flagged:</h5>
        <ul className="text-xs text-slate-400 space-y-1">
          {assessment.rule_references.map((ref, i) => (
            <li key={i} className="flex items-start gap-2">
              <span className="text-slate-500">•</span>
              {ref}
            </li>
          ))}
        </ul>
      </div>
    )}

    {assessment.patient_factors?.length > 0 && (
      <div className="mb-3">
        <h5 className="text-xs font-medium text-slate-400 mb-1">Triggering factors:</h5>
        <div className="flex flex-wrap gap-1">
          {assessment.patient_factors.map((pf, i) => (
            <span key={i} className="text-xs bg-indigo-500/20 text-indigo-300 px-2 py-1 rounded">{pf}</span>
          ))}
        </div>
      </div>
    )}

    {assessment.evidence_sources?.length > 0 && (
      <div className="mb-3">
        <h5 className="text-xs font-medium text-slate-400 mb-1">Sources:</h5>
        <div className="flex flex-wrap gap-1">
          {assessment.evidence_sources.map((src, i) => (
            <span key={i} className="text-xs bg-slate-700/70 text-slate-200 px-2 py-1 rounded border border-slate-600">{src}</span>
          ))}
        </div>
      </div>
    )}

    {assessment.monitoring?.length > 0 && (
      <div className="mb-3">
        <h5 className="text-xs font-medium text-slate-400 mb-1">Monitor:</h5>
        <div className="flex flex-wrap gap-1">
          {assessment.monitoring.map((item, i) => (
            <span key={i} className="text-xs bg-blue-500/20 text-blue-400 px-2 py-1 rounded">{item}</span>
          ))}
        </div>
      </div>
    )}

    {assessment.alternatives?.length > 0 && (
      <div>
        <h5 className="text-xs font-medium text-slate-400 mb-1">Safer Alternatives:</h5>
        <div className="flex flex-wrap gap-1">
          {assessment.alternatives.map((alt, i) => (
            <span key={i} className="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-1 rounded border border-emerald-500/40">
              {alt} <span className="text-[10px] text-emerald-200 ml-1">(safer)</span>
            </span>
          ))}
        </div>
      </div>
    )}
  </motion.div>
)

export default function DiabetesManager() {
  const [patients, setPatients] = useState([])
  const [selectedPatient, setSelectedPatient] = useState(null)
  const [medications, setMedications] = useState([])
  const [checkResult, setCheckResult] = useState(null)
  const [report, setReport] = useState(null)
  const [loading, setLoading] = useState(false)
  const [activeSection, setActiveSection] = useState('patients') // patients, check, report
  const [modelInfo, setModelInfo] = useState(null)
  
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
  const [allDrugs, setAllDrugs] = useState([])
  const [drugsLoading, setDrugsLoading] = useState(false)
  const [drugBrowserPage, setDrugBrowserPage] = useState(0)
  const [drugBrowserSearch, setDrugBrowserSearch] = useState('')

  // Fetch patients on mount
  useEffect(() => {
    fetchPatients()
    fetchModelInfo()
  }, [])

  // Fetch medications when patient selected
  useEffect(() => {
    if (selectedPatient) {
      fetchMedications(selectedPatient.patient_id)
    }
  }, [selectedPatient])

  const fetchPatients = async () => {
    try {
      const res = await fetch(`${API_URL}/diabetic/patients`)
      if (res.ok) {
        const data = await res.json()
        setPatients(data)
      }
    } catch (err) {
      console.error('Error fetching patients:', err)
    }
  }

  const fetchMedications = async (patientId) => {
    try {
      const res = await fetch(`${API_URL}/diabetic/patients/${patientId}/medications`)
      if (res.ok) {
        const data = await res.json()
        setMedications(data)
      }
    } catch (err) {
      console.error('Error fetching medications:', err)
    }
  }

  const fetchModelInfo = async () => {
    try {
      const res = await fetch(`${API_URL}/diabetic/model-info`)
      if (res.ok) {
        const data = await res.json()
        setModelInfo(data)
      }
    } catch (err) {
      console.error('Error fetching model info:', err)
    }
  }

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
      const res = await fetch(`${API_URL}/drugs/search?query=${encodeURIComponent(query.trim())}&limit=12`)
      if (res.ok) {
        const data = await res.json()
        setSearchResults(data || [])
      } else {
        setSearchError('Search failed')
        setSearchResults([])
      }
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
    setDrugsLoading(true)
    try {
      const limit = 50
      const offset = page * limit
      let url = `${API_URL}/drugs?limit=${limit}&offset=${offset}`
      if (search.trim()) {
        url = `${API_URL}/drugs/search?query=${encodeURIComponent(search.trim())}&limit=${limit}`
      }
      const res = await fetch(url)
      if (res.ok) {
        const data = await res.json()
        setAllDrugs(Array.isArray(data) ? data : (data.drugs || []))
      }
    } catch (err) {
      console.error('Error fetching drugs:', err)
    } finally {
      setDrugsLoading(false)
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
      setLoading(true)
      try {
        const res = await fetch(`${API_URL}/diabetic/risk-check`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            patient_id: selectedPatient.patient_id,
            drug_name: drugName
          })
        })
        if (res.ok) {
          const data = await res.json()
          setCheckResult(data)
          setActiveSection('check')
        }
      } catch (err) {
        toast.error('Error checking drug risk')
      } finally {
        setLoading(false)
      }
    }
  }

  const createPatient = async (e) => {
    e.preventDefault()
    setLoading(true)
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
      const res = await fetch(`${API_URL}/diabetic/patients`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      if (res.ok) {
        toast.success('Patient created successfully!')
        setShowNewPatient(false)
        setNewPatient({ patient_id: '', name: '', age: '', diabetes_type: 'type_2', hba1c: '', egfr: '', potassium: '', has_nephropathy: false, has_cardiovascular: false, has_neuropathy: false })
        fetchPatients()
      } else {
        const err = await res.json()
        toast.error(err.detail || 'Failed to create patient')
      }
    } catch (err) {
      toast.error('Error creating patient')
    } finally {
      setLoading(false)
    }
  }

  const checkDrugRisk = async () => {
    if (!selectedPatient || !drugToCheck.trim()) {
      toast.error('Select a patient and enter a drug name')
      return
    }
    setLoading(true)
    try {
      const res = await fetch(`${API_URL}/diabetic/risk-check`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_id: selectedPatient.patient_id,
          drug_name: drugToCheck.trim()
        })
      })
      if (res.ok) {
        const data = await res.json()
        setCheckResult(data)
        setActiveSection('check')
      } else {
        const err = await res.json()
        toast.error(err.detail || 'Failed to check drug')
      }
    } catch (err) {
      toast.error('Error checking drug risk')
    } finally {
      setLoading(false)
    }
  }

  const checkAllMedications = async () => {
    if (!selectedPatient) {
      toast.error('Select a patient first')
      return
    }
    setLoading(true)
    try {
      const res = await fetch(`${API_URL}/diabetic/medication-list-check`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patient_id: selectedPatient.patient_id })
      })
      if (res.ok) {
        const data = await res.json()
        setCheckResult(data)
        setActiveSection('check')
      } else {
        const err = await res.json()
        toast.error(err.detail || 'Failed to check medications')
      }
    } catch (err) {
      toast.error('Error checking medications')
    } finally {
      setLoading(false)
    }
  }

  const generateReport = async () => {
    if (!selectedPatient) {
      toast.error('Select a patient first')
      return
    }
    setLoading(true)
    try {
      const res = await fetch(`${API_URL}/diabetic/report/${selectedPatient.patient_id}`)
      if (res.ok) {
        const data = await res.json()
        setReport(data)
        setActiveSection('report')
      } else {
        const err = await res.json()
        toast.error(err.detail || 'Failed to generate report')
      }
    } catch (err) {
      toast.error('Error generating report')
    } finally {
      setLoading(false)
    }
  }

  const addMedication = async (drugName) => {
    if (!selectedPatient || !drugName.trim()) return
    try {
      const res = await fetch(`${API_URL}/diabetic/patients/${selectedPatient.patient_id}/medications`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ drug_name: drugName.trim() })
      })
      if (res.ok) {
        toast.success('Medication added')
        fetchMedications(selectedPatient.patient_id)
      }
    } catch (err) {
      toast.error('Error adding medication')
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
          <h1 className="text-4xl font-bold text-white mb-2">
            <span className="text-medical-400">Diabetic</span> Patient DDI Checker
          </h1>
          <p className="text-slate-400">
            Specialized drug interaction analysis for diabetic patients
          </p>
          {modelInfo && (
            <div className="mt-3 flex justify-center gap-2 text-xs text-slate-300">
              <span className={`px-2 py-1 rounded-full border ${modelInfo.loaded ? 'border-emerald-500/50 text-emerald-300 bg-emerald-500/10' : 'border-amber-500/50 text-amber-300 bg-amber-500/10'}`}>
                Model: {modelInfo.loaded ? 'Loaded' : 'Not loaded'}
              </span>
              {modelInfo.model_version && (
                <span className="px-2 py-1 rounded-full border border-slate-600 bg-slate-800/70">
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
                <h2 className="text-xl font-semibold text-white">Patients</h2>
                <button
                  onClick={() => setShowNewPatient(!showNewPatient)}
                  className="p-2 rounded-lg bg-medical-500/20 text-medical-400 hover:bg-medical-500/30 transition-colors"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                  </svg>
                </button>
              </div>

              {/* New Patient Form */}
              <AnimatePresence>
                {showNewPatient && (
                  <motion.form
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    onSubmit={createPatient}
                    className="mb-4 p-4 bg-slate-800/50 rounded-xl border border-slate-700/50 space-y-3"
                  >
                    <input
                      type="text"
                      placeholder="Patient ID *"
                      value={newPatient.patient_id}
                      onChange={(e) => setNewPatient({...newPatient, patient_id: e.target.value})}
                      className="w-full px-3 py-2 bg-slate-900/50 border border-slate-600 rounded-lg text-white text-sm"
                      required
                    />
                    <input
                      type="text"
                      placeholder="Name"
                      value={newPatient.name}
                      onChange={(e) => setNewPatient({...newPatient, name: e.target.value})}
                      className="w-full px-3 py-2 bg-slate-900/50 border border-slate-600 rounded-lg text-white text-sm"
                    />
                    <div className="grid grid-cols-2 gap-2">
                      <input
                        type="number"
                        placeholder="Age"
                        value={newPatient.age}
                        onChange={(e) => setNewPatient({...newPatient, age: e.target.value})}
                        className="px-3 py-2 bg-slate-900/50 border border-slate-600 rounded-lg text-white text-sm"
                      />
                      <select
                        value={newPatient.diabetes_type}
                        onChange={(e) => setNewPatient({...newPatient, diabetes_type: e.target.value})}
                        className="px-3 py-2 bg-slate-900/50 border border-slate-600 rounded-lg text-white text-sm"
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
                        onChange={(e) => setNewPatient({...newPatient, hba1c: e.target.value})}
                        className="px-3 py-2 bg-slate-900/50 border border-slate-600 rounded-lg text-white text-sm"
                      />
                      <input
                        type="number"
                        placeholder="eGFR"
                        value={newPatient.egfr}
                        onChange={(e) => setNewPatient({...newPatient, egfr: e.target.value})}
                        className="px-3 py-2 bg-slate-900/50 border border-slate-600 rounded-lg text-white text-sm"
                      />
                      <input
                        type="number"
                        step="0.1"
                        placeholder="K+ mEq/L"
                        value={newPatient.potassium}
                        onChange={(e) => setNewPatient({...newPatient, potassium: e.target.value})}
                        className="px-3 py-2 bg-slate-900/50 border border-slate-600 rounded-lg text-white text-sm"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-xs text-slate-400">Complications:</label>
                      <div className="flex flex-wrap gap-2">
                        {['nephropathy', 'cardiovascular', 'neuropathy'].map(comp => (
                          <label key={comp} className="flex items-center gap-1 text-sm text-slate-300">
                            <input
                              type="checkbox"
                              checked={newPatient[`has_${comp}`]}
                              onChange={(e) => setNewPatient({...newPatient, [`has_${comp}`]: e.target.checked})}
                              className="rounded bg-slate-800 border-slate-600"
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

              {/* Patient List */}
              <div className="space-y-3 max-h-[500px] overflow-y-auto">
                {patients.length === 0 ? (
                  <p className="text-slate-500 text-center py-4">No patients yet. Create one above.</p>
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
                <div className="text-6xl mb-4">👈</div>
                <h3 className="text-xl font-semibold text-white mb-2">Select a Patient</h3>
                <p className="text-slate-400">Choose or create a patient to check drug risks</p>
              </div>
            ) : (
              <div className="space-y-6">
                {/* Selected Patient Info */}
                <div className="glass-card p-6 rounded-2xl">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h2 className="text-2xl font-bold text-white">{selectedPatient.name || selectedPatient.patient_id}</h2>
                      <p className="text-slate-400">
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
                          className="flex-1 px-4 py-2 bg-slate-900/50 border border-slate-600 rounded-xl text-white"
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
                          <div className="absolute z-20 mt-1 w-full bg-slate-900/95 border border-slate-700 rounded-xl shadow-lg max-h-56 overflow-y-auto">
                            {searchLoading && (
                              <div className="px-3 py-2 text-sm text-slate-400">Searching...</div>
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
                                className="w-full text-left px-3 py-2 hover:bg-slate-800 text-sm text-white transition-colors flex justify-between gap-2"
                              >
                                <span>{d.name}</span>
                                {d.generic_name && (
                                  <span className="text-xs text-slate-400">{d.generic_name}</span>
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
                        className="px-3 py-1 rounded-full text-xs bg-slate-700/60 text-slate-200 border border-slate-600 hover:border-medical-400 transition-colors"
                      >
                        {chip.label}
                      </button>
                    ))}
                  </div>

                  {/* Current Medications */}
                  <div>
                    <h4 className="text-sm font-medium text-slate-400 mb-2">Current Medications ({medications.length})</h4>
                    <div className="flex flex-wrap gap-2">
                      {medications.map(med => (
                        <span key={med.id} className="px-3 py-1 bg-slate-700/50 text-slate-300 rounded-full text-sm">
                          {med.drug_name}
                        </span>
                      ))}
                      <button
                        onClick={() => {
                          const name = prompt('Enter medication name:')
                          if (name) addMedication(name)
                        }}
                        className="px-3 py-1 bg-medical-500/20 text-medical-400 rounded-full text-sm hover:bg-medical-500/30"
                      >
                        + Add
                      </button>
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
                      <p className="text-slate-400">Analyzing drug safety...</p>
                    </motion.div>
                  )}

                  {/* Single Drug Check Result */}
                  {!loading && checkResult && activeSection === 'check' && !checkResult.assessments && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="glass-card p-6 rounded-2xl"
                    >
                      <h3 className="text-xl font-bold text-white mb-4">Drug Risk Assessment</h3>
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
                        <h3 className="text-xl font-bold text-white">Medication List Assessment</h3>
                        <RiskBadge level={checkResult.overall_risk_level} />
                      </div>
                      
                      {/* Summary */}
                      <div className="grid grid-cols-5 gap-2 mb-4">
                        {[
                          { label: 'Safe', count: checkResult.safe_count, color: 'emerald' },
                          { label: 'Caution', count: checkResult.caution_count, color: 'amber' },
                          { label: 'High Risk', count: checkResult.high_risk_count, color: 'orange' },
                          { label: 'Contraind.', count: checkResult.contraindicated_count, color: 'red' },
                          { label: 'Fatal', count: checkResult.fatal_count, color: 'red' },
                        ].map(item => (
                          <div key={item.label} className={`p-3 rounded-lg bg-${item.color}-500/10 text-center`}>
                            <div className={`text-2xl font-bold text-${item.color}-400`}>{item.count}</div>
                            <div className="text-xs text-slate-400">{item.label}</div>
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
                          <h4 className="text-sm font-medium text-slate-400 mb-2">Recommendations</h4>
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
                        <h3 className="text-xl font-bold text-white">Full DDI Report</h3>
                        <div className="text-right">
                          <div className={`text-3xl font-bold ${
                            report.overall_safety_score > 70 ? 'text-emerald-400' :
                            report.overall_safety_score > 40 ? 'text-amber-400' : 'text-red-400'
                          }`}>
                            {report.overall_safety_score}%
                          </div>
                          <div className="text-xs text-slate-400">Safety Score</div>
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
                          <h4 className="text-sm font-medium text-slate-400 mb-2">📋 Monitoring Plan</h4>
                          <div className="flex flex-wrap gap-2">
                            {report.monitoring_plan.map((item, i) => (
                              <span key={i} className="px-3 py-1 bg-blue-500/20 text-blue-400 rounded-full text-sm">{item}</span>
                            ))}
                          </div>
                        </div>
                      )}

                      <p className="text-sm text-slate-500">
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
              className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-4xl max-h-[80vh] overflow-hidden shadow-2xl"
            >
              {/* Header */}
              <div className="p-4 border-b border-slate-700 flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-bold text-white">Browse All Drugs</h2>
                  <p className="text-sm text-slate-400">Click any drug to check risk for {selectedPatient?.name || 'patient'}</p>
                </div>
                <button
                  onClick={() => setShowDrugBrowser(false)}
                  className="p-2 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-white transition-colors"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              {/* Search */}
              <div className="p-4 border-b border-slate-700">
                <input
                  type="text"
                  placeholder="Search drugs by name..."
                  value={drugBrowserSearch}
                  onChange={(e) => {
                    setDrugBrowserSearch(e.target.value)
                    fetchAllDrugs(e.target.value, 0)
                  }}
                  className="w-full px-4 py-3 bg-slate-800 border border-slate-600 rounded-xl text-white placeholder-slate-500 focus:border-medical-500 focus:outline-none"
                />
              </div>

              {/* Drug List */}
              <div className="p-4 overflow-y-auto max-h-[50vh]">
                {drugsLoading ? (
                  <div className="text-center py-8">
                    <div className="spinner mx-auto mb-2"></div>
                    <p className="text-slate-400">Loading drugs...</p>
                  </div>
                ) : allDrugs.length === 0 ? (
                  <div className="text-center py-8">
                    <p className="text-slate-400">No drugs found. Try a different search.</p>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                    {allDrugs.map((drug) => (
                      <button
                        key={drug.id}
                        onClick={() => checkDrugFromBrowser(drug.name || drug.generic_name)}
                        className="p-3 bg-slate-800/50 hover:bg-slate-700/70 border border-slate-700 hover:border-medical-500/50 rounded-xl text-left transition-all group"
                      >
                        <div className="font-medium text-white group-hover:text-medical-400 transition-colors">
                          {drug.name}
                        </div>
                        {drug.generic_name && drug.generic_name !== drug.name && (
                          <div className="text-xs text-slate-400">{drug.generic_name}</div>
                        )}
                        {drug.drug_class && (
                          <div className="text-xs text-slate-500 mt-1">{drug.drug_class}</div>
                        )}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {/* Pagination */}
              <div className="p-4 border-t border-slate-700 flex items-center justify-between">
                <button
                  onClick={() => {
                    const newPage = Math.max(0, drugBrowserPage - 1)
                    setDrugBrowserPage(newPage)
                    fetchAllDrugs(drugBrowserSearch, newPage)
                  }}
                  disabled={drugBrowserPage === 0}
                  className="px-4 py-2 bg-slate-800 text-slate-300 rounded-lg hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  ← Previous
                </button>
                <span className="text-slate-400">
                  Page {drugBrowserPage + 1} • Showing {allDrugs.length} drugs
                </span>
                <button
                  onClick={() => {
                    const newPage = drugBrowserPage + 1
                    setDrugBrowserPage(newPage)
                    fetchAllDrugs(drugBrowserSearch, newPage)
                  }}
                  disabled={allDrugs.length < 50}
                  className="px-4 py-2 bg-slate-800 text-slate-300 rounded-lg hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Next →
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

