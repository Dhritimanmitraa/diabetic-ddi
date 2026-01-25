import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import toast from 'react-hot-toast'
import {
    Upload, Pill,
    AlertTriangle, Shield, Camera, User, Loader2, XCircle, SwitchCamera
} from 'lucide-react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/**
 * PatientPrescriptionScanner Component
 * Integrates prescription scanning with diabetic patient DDI checking
 * Allows linking prescriptions to patients for personalized risk analysis
 */
function PatientPrescriptionScanner({ initialPatientId = null }) {
    // Patient state
    const [patients, setPatients] = useState([])
    const [selectedPatient, setSelectedPatient] = useState(null)
    const [patientsLoading, setPatientsLoading] = useState(false)

    // Prescription state
    const [prescription, setPrescription] = useState(null)
    const [isUploading, setIsUploading] = useState(false)
    const [riskAssessments, setRiskAssessments] = useState([])
    const [isCheckingRisks, setIsCheckingRisks] = useState(false)

    // UI state
    const [activeView, setActiveView] = useState('select') // select, upload, results

    // Camera state
    const [showCamera, setShowCamera] = useState(false)
    const [cameraStream, setCameraStream] = useState(null)
    const [facingMode, setFacingMode] = useState('environment')

    const fileInputRef = useRef(null)
    const videoRef = useRef(null)
    const canvasRef = useRef(null)

    // Fetch patients on mount
    useEffect(() => {
        fetchPatients()
    }, [])

    // Select initial patient if provided
    useEffect(() => {
        if (initialPatientId && patients.length > 0) {
            const patient = patients.find(p => p.patient_id === initialPatientId)
            if (patient) {
                setSelectedPatient(patient)
                setActiveView('upload')
            }
        }
    }, [initialPatientId, patients])

    const fetchPatients = async () => {
        setPatientsLoading(true)
        try {
            const res = await fetch(`${API_URL}/diabetic/patients`)
            if (res.ok) {
                const data = await res.json()
                setPatients(data || [])
            }
        } catch (err) {
            console.error('Failed to fetch patients:', err)
        } finally {
            setPatientsLoading(false)
        }
    }

    const handlePatientSelect = (patient) => {
        setSelectedPatient(patient)
        setActiveView('upload')
        setPrescription(null)
        setRiskAssessments([])
    }

    // Camera functions
    const startCamera = useCallback(async () => {
        try {
            if (cameraStream) {
                cameraStream.getTracks().forEach(track => track.stop())
            }

            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: facingMode,
                    width: { ideal: 1920 },
                    height: { ideal: 1080 }
                }
            })

            setCameraStream(stream)
            setShowCamera(true)

            setTimeout(() => {
                if (videoRef.current) {
                    videoRef.current.srcObject = stream
                }
            }, 100)
        } catch (err) {
            console.error('Camera error:', err)
            if (err.name === 'NotAllowedError') {
                toast.error('Camera permission denied. Please allow camera access.')
            } else if (err.name === 'NotFoundError') {
                toast.error('No camera found on this device')
            } else {
                toast.error('Could not access camera: ' + err.message)
            }
        }
    }, [facingMode, cameraStream])

    const stopCamera = useCallback(() => {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop())
            setCameraStream(null)
        }
        setShowCamera(false)
    }, [cameraStream])

    const switchCamera = useCallback(() => {
        setFacingMode(prev => prev === 'environment' ? 'user' : 'environment')
    }, [])

    // Restart camera when facing mode changes
    useEffect(() => {
        if (showCamera && cameraStream) {
            startCamera()
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [facingMode])

    // Cleanup camera on unmount
    useEffect(() => {
        return () => {
            if (cameraStream) {
                cameraStream.getTracks().forEach(track => track.stop())
            }
        }
    }, [cameraStream])

    const capturePhoto = useCallback(async () => {
        if (!videoRef.current || !canvasRef.current) return

        const video = videoRef.current
        const canvas = canvasRef.current

        canvas.width = video.videoWidth
        canvas.height = video.videoHeight

        const ctx = canvas.getContext('2d')
        ctx.drawImage(video, 0, 0)

        canvas.toBlob(async (blob) => {
            if (!blob) {
                toast.error('Failed to capture image')
                return
            }

            const file = new File([blob], 'prescription_capture.jpg', { type: 'image/jpeg' })
            stopCamera()

            // Process the captured image
            await processFile(file)
        }, 'image/jpeg', 0.9)
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [stopCamera])

    const processFile = async (file) => {
        setIsUploading(true)
        setPrescription(null)
        setRiskAssessments([])

        try {
            const formData = new FormData()
            formData.append('file', file)

            const res = await fetch(`${API_URL}/prescription/upload`, {
                method: 'POST',
                body: formData
            })

            if (res.ok) {
                const result = await res.json()
                if (result.status === 'completed') {
                    setPrescription(result)
                    setActiveView('results')
                    toast.success(`Extracted ${result.medicines?.length || 0} medicine(s)!`)

                    if (selectedPatient && result.medicines?.length > 0) {
                        checkPatientRisks(result.medicines)
                    }
                } else {
                    toast.error(result.message || 'Extraction failed')
                }
            } else {
                const err = await res.json()
                toast.error(err.detail || 'Upload failed')
            }
        } catch (err) {
            console.error('Upload error:', err)
            toast.error('Failed to process prescription')
        } finally {
            setIsUploading(false)
        }
    }

    const handleFileSelect = async (event) => {
        const file = event.target.files?.[0]
        if (!file) return

        // Validate file
        const allowedTypes = ['image/jpeg', 'image/png', 'image/webp', 'application/pdf']
        if (!allowedTypes.includes(file.type)) {
            toast.error('Please upload an image (JPEG, PNG) or PDF')
            return
        }

        if (file.size > 10 * 1024 * 1024) {
            toast.error('File too large. Maximum size is 10MB')
            return
        }

        await processFile(file)
    }

    const checkPatientRisks = async (medicines) => {
        if (!selectedPatient) return

        setIsCheckingRisks(true)
        const assessments = []

        try {
            for (const med of medicines) {
                try {
                    const res = await fetch(`${API_URL}/diabetic/risk-check`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            patient_id: selectedPatient.patient_id,
                            drug_name: med.name
                        })
                    })

                    if (res.ok) {
                        const data = await res.json()
                        assessments.push({
                            medicine: med.name,
                            ...data
                        })
                    }
                } catch (err) {
                    console.error(`Failed to check ${med.name}:`, err)
                }
            }

            setRiskAssessments(assessments)

            // Show summary toast
            const highRiskCount = assessments.filter(
                a => ['high_risk', 'contraindicated', 'fatal'].includes(a.risk_level)
            ).length

            if (highRiskCount > 0) {
                toast.error(`${highRiskCount} medicine(s) have HIGH RISK for this patient!`, { duration: 5000 })
            } else {
                toast.success('Risk assessment complete', { duration: 3000 })
            }
        } finally {
            setIsCheckingRisks(false)
        }
    }

    const addMedicinesToPatient = async () => {
        if (!selectedPatient || !prescription?.medicines?.length) return

        let added = 0
        for (const med of prescription.medicines) {
            try {
                const res = await fetch(`${API_URL}/diabetic/patients/${selectedPatient.patient_id}/medications`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        drug_name: med.name,
                        dosage: med.dosage || null,
                        frequency: med.frequency || null
                    })
                })
                if (res.ok) added++
            } catch (err) {
                console.error(`Failed to add ${med.name}:`, err)
            }
        }

        if (added > 0) {
            toast.success(`Added ${added} medicine(s) to patient profile!`)
        }
    }

    const getRiskColor = (level) => {
        const colors = {
            safe: { bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/30' },
            caution: { bg: 'bg-amber-500/20', text: 'text-amber-400', border: 'border-amber-500/30' },
            high_risk: { bg: 'bg-orange-500/20', text: 'text-orange-400', border: 'border-orange-500/30' },
            contraindicated: { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/30' },
            fatal: { bg: 'bg-red-900/30', text: 'text-red-300', border: 'border-red-700' },
        }
        return colors[level] || colors.caution
    }

    return (
        <div className="max-w-4xl mx-auto p-4">
            {/* Header */}
            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-center mb-8"
            >
                <h2 className="text-3xl font-bold text-white mb-2">
                    <span className="text-medical-400">Patient</span> Prescription Scanner
                </h2>
                <p className="text-slate-400">
                    Scan prescriptions and get personalized drug risk analysis for diabetic patients
                </p>
            </motion.div>

            {/* Progress Steps */}
            <div className="flex items-center justify-center gap-4 mb-8">
                {[
                    { id: 'select', label: '1. Select Patient', icon: User },
                    { id: 'upload', label: '2. Scan Prescription', icon: Camera },
                    { id: 'results', label: '3. Risk Analysis', icon: Shield }
                ].map((step, idx) => {
                    const isActive = activeView === step.id
                    const isPast = ['select', 'upload', 'results'].indexOf(activeView) > idx
                    const Icon = step.icon

                    return (
                        <div key={step.id} className="flex items-center">
                            <button
                                onClick={() => {
                                    if (step.id === 'select') setActiveView('select')
                                    else if (step.id === 'upload' && selectedPatient) setActiveView('upload')
                                    else if (step.id === 'results' && prescription) setActiveView('results')
                                }}
                                disabled={
                                    (step.id === 'upload' && !selectedPatient) ||
                                    (step.id === 'results' && !prescription)
                                }
                                className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-all ${isActive
                                    ? 'bg-medical-500 text-white'
                                    : isPast
                                        ? 'bg-medical-500/20 text-medical-400'
                                        : 'bg-slate-800/50 text-slate-500'
                                    } ${step.id !== 'select' && !selectedPatient ? 'opacity-50 cursor-not-allowed' : ''}`}
                            >
                                <Icon className="w-4 h-4" />
                                <span className="text-sm font-medium hidden sm:inline">{step.label}</span>
                            </button>
                            {idx < 2 && (
                                <div className={`w-8 h-0.5 mx-2 ${isPast ? 'bg-medical-500' : 'bg-slate-700'}`} />
                            )}
                        </div>
                    )
                })}
            </div>

            <AnimatePresence mode="wait">
                {/* Step 1: Select Patient */}
                {activeView === 'select' && (
                    <motion.div
                        key="select"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="glass rounded-2xl p-6"
                    >
                        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                            <User className="w-5 h-5 text-medical-400" />
                            Select Diabetic Patient
                        </h3>

                        {patientsLoading ? (
                            <div className="text-center py-8">
                                <Loader2 className="w-8 h-8 animate-spin text-medical-400 mx-auto" />
                            </div>
                        ) : patients.length === 0 ? (
                            <div className="text-center py-8 text-slate-400">
                                <p>No patients found. Create a patient in the Diabetes Manager first.</p>
                            </div>
                        ) : (
                            <div className="grid gap-3 max-h-[400px] overflow-y-auto">
                                {patients.map(patient => (
                                    <button
                                        key={patient.id}
                                        onClick={() => handlePatientSelect(patient)}
                                        className={`p-4 rounded-xl text-left transition-all border ${selectedPatient?.id === patient.id
                                            ? 'bg-medical-500/20 border-medical-500'
                                            : 'bg-slate-800/50 border-slate-700/50 hover:border-medical-500/50'
                                            }`}
                                    >
                                        <div className="flex items-center justify-between">
                                            <div>
                                                <p className="font-medium text-white">{patient.name || patient.patient_id}</p>
                                                <p className="text-sm text-slate-400">
                                                    {(patient.diabetes_type || 'type_2').replace('_', ' ').toUpperCase()}
                                                    {patient.age && ` • ${patient.age} years`}
                                                </p>
                                            </div>
                                            <div className="text-right text-sm">
                                                {patient.egfr && (
                                                    <span className={`px-2 py-1 rounded-full ${patient.egfr >= 60 ? 'bg-emerald-500/20 text-emerald-400' :
                                                        patient.egfr >= 30 ? 'bg-amber-500/20 text-amber-400' :
                                                            'bg-red-500/20 text-red-400'
                                                        }`}>
                                                        eGFR: {patient.egfr}
                                                    </span>
                                                )}
                                            </div>
                                        </div>
                                    </button>
                                ))}
                            </div>
                        )}
                    </motion.div>
                )}

                {/* Step 2: Upload Prescription */}
                {activeView === 'upload' && (
                    <motion.div
                        key="upload"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="space-y-4"
                    >
                        {/* Selected Patient Info */}
                        {selectedPatient && (
                            <div className="glass rounded-xl p-4 flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                    <div className="w-10 h-10 rounded-full bg-medical-500/20 flex items-center justify-center">
                                        <User className="w-5 h-5 text-medical-400" />
                                    </div>
                                    <div>
                                        <p className="font-medium text-white">{selectedPatient.name || selectedPatient.patient_id}</p>
                                        <p className="text-xs text-slate-400">
                                            {selectedPatient.diabetes_type?.replace('_', ' ').toUpperCase()}
                                            {selectedPatient.egfr && ` • eGFR: ${selectedPatient.egfr}`}
                                        </p>
                                    </div>
                                </div>
                                <button
                                    onClick={() => setActiveView('select')}
                                    className="text-xs text-medical-400 hover:text-medical-300"
                                >
                                    Change
                                </button>
                            </div>
                        )}

                        {/* Camera View */}
                        {showCamera ? (
                            <div className="glass rounded-2xl overflow-hidden">
                                <div className="relative bg-black aspect-[4/3]">
                                    <video
                                        ref={videoRef}
                                        autoPlay
                                        playsInline
                                        muted
                                        className="w-full h-full object-cover"
                                    />
                                    <canvas ref={canvasRef} className="hidden" />

                                    {/* Camera Controls */}
                                    <div className="absolute top-4 right-4 flex gap-2">
                                        <button
                                            onClick={switchCamera}
                                            className="p-3 bg-black/50 hover:bg-black/70 text-white rounded-full transition-colors"
                                            title="Switch Camera"
                                        >
                                            <SwitchCamera className="w-5 h-5" />
                                        </button>
                                        <button
                                            onClick={stopCamera}
                                            className="p-3 bg-black/50 hover:bg-red-500/70 text-white rounded-full transition-colors"
                                            title="Close Camera"
                                        >
                                            <XCircle className="w-5 h-5" />
                                        </button>
                                    </div>

                                    <div className="absolute bottom-4 left-0 right-0 text-center">
                                        <p className="text-white/70 text-sm mb-2">Position prescription in frame</p>
                                    </div>
                                </div>

                                {/* Capture Button */}
                                <div className="p-6 flex justify-center">
                                    <button
                                        onClick={capturePhoto}
                                        className="w-20 h-20 rounded-full bg-white flex items-center justify-center hover:scale-105 transition-transform shadow-lg"
                                    >
                                        <div className="w-16 h-16 rounded-full border-4 border-medical-500 flex items-center justify-center">
                                            <Camera className="w-8 h-8 text-medical-500" />
                                        </div>
                                    </button>
                                </div>
                            </div>
                        ) : (
                            <>
                                {/* Upload Options */}
                                <div className="grid grid-cols-2 gap-4">
                                    {/* Camera Button */}
                                    <button
                                        onClick={startCamera}
                                        disabled={isUploading}
                                        className="glass rounded-2xl p-8 text-center hover:bg-slate-800/50 transition-all border-2 border-transparent hover:border-medical-500/30 disabled:opacity-50"
                                    >
                                        <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-medical-500/10 flex items-center justify-center">
                                            <Camera className="w-8 h-8 text-medical-400" />
                                        </div>
                                        <h3 className="text-lg font-semibold text-white mb-1">
                                            Take Photo
                                        </h3>
                                        <p className="text-slate-500 text-sm">
                                            Use camera to capture
                                        </p>
                                    </button>

                                    {/* File Upload Button */}
                                    <div
                                        onClick={() => !isUploading && fileInputRef.current?.click()}
                                        className={`glass rounded-2xl p-8 text-center cursor-pointer border-2 border-transparent transition-all ${isUploading
                                            ? 'opacity-50 cursor-not-allowed'
                                            : 'hover:bg-slate-800/50 hover:border-purple-500/30'
                                            }`}
                                    >
                                        <input
                                            ref={fileInputRef}
                                            type="file"
                                            accept="image/jpeg,image/png,image/webp,application/pdf"
                                            onChange={handleFileSelect}
                                            className="hidden"
                                        />
                                        <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-purple-500/10 flex items-center justify-center">
                                            <Upload className="w-8 h-8 text-purple-400" />
                                        </div>
                                        <h3 className="text-lg font-semibold text-white mb-1">
                                            Upload File
                                        </h3>
                                        <p className="text-slate-500 text-sm">
                                            JPEG, PNG, or PDF
                                        </p>
                                    </div>
                                </div>

                                {/* Processing State */}
                                {isUploading && (
                                    <div className="glass rounded-2xl p-8 text-center">
                                        <Loader2 className="w-12 h-12 text-medical-400 animate-spin mx-auto mb-4" />
                                        <p className="text-medical-400 font-medium">Processing prescription...</p>
                                        <p className="text-slate-500 text-sm mt-1">Extracting medicines with AI</p>
                                    </div>
                                )}

                                {/* Tips */}
                                <div className="p-4 bg-amber-500/5 border border-amber-500/10 rounded-xl">
                                    <p className="text-amber-400 text-sm font-medium mb-2">Tips for best results:</p>
                                    <ul className="text-slate-400 text-sm space-y-1">
                                        <li>• Ensure good lighting and clear focus</li>
                                        <li>• Include all medicines in the frame</li>
                                        <li>• Handwritten prescriptions work too!</li>
                                    </ul>
                                </div>
                            </>
                        )}
                    </motion.div>
                )}

                {/* Step 3: Results */}
                {activeView === 'results' && prescription && (
                    <motion.div
                        key="results"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="space-y-6"
                    >
                        {/* Patient Context */}
                        <div className="glass rounded-xl p-4 flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <Shield className="w-5 h-5 text-medical-400" />
                                <span className="text-white font-medium">
                                    Analyzing for: {selectedPatient?.name || selectedPatient?.patient_id}
                                </span>
                            </div>
                            {isCheckingRisks && (
                                <span className="flex items-center gap-2 text-sm text-slate-400">
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                    Checking patient-specific risks...
                                </span>
                            )}
                        </div>

                        {/* Medicines with Risk Assessments */}
                        <div className="glass rounded-2xl p-6">
                            <div className="flex items-center justify-between mb-4">
                                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                                    <Pill className="w-5 h-5 text-medical-400" />
                                    Extracted Medicines ({prescription.medicines?.length || 0})
                                </h3>
                                <button
                                    onClick={addMedicinesToPatient}
                                    className="text-xs px-3 py-1.5 bg-medical-500/20 text-medical-400 rounded-lg hover:bg-medical-500/30 transition-colors"
                                >
                                    + Add All to Patient
                                </button>
                            </div>

                            <div className="space-y-4">
                                {prescription.medicines?.map((med, idx) => {
                                    const assessment = riskAssessments.find(a => a.medicine === med.name)
                                    const riskColor = assessment ? getRiskColor(assessment.risk_level) : null

                                    return (
                                        <div key={idx} className="p-4 bg-slate-800/50 rounded-xl border border-slate-700/50">
                                            <div className="flex items-start justify-between mb-2">
                                                <div>
                                                    <h4 className="font-semibold text-white">{med.name}</h4>
                                                    <p className="text-sm text-slate-400">
                                                        {med.dosage && <span>{med.dosage}</span>}
                                                        {med.frequency && <span> • {med.frequency}</span>}
                                                    </p>
                                                </div>

                                                {assessment && (
                                                    <span className={`px-3 py-1 rounded-full text-xs font-medium capitalize ${riskColor.bg} ${riskColor.text} border ${riskColor.border}`}>
                                                        {assessment.risk_level?.replace('_', ' ')}
                                                    </span>
                                                )}

                                                {isCheckingRisks && !assessment && (
                                                    <Loader2 className="w-4 h-4 animate-spin text-slate-400" />
                                                )}
                                            </div>

                                            {/* Risk Factors */}
                                            {assessment?.risk_factors?.length > 0 && (
                                                <div className="mt-3 pt-3 border-t border-slate-700/50">
                                                    <p className="text-xs text-slate-500 mb-2">Risk Factors:</p>
                                                    <ul className="space-y-1">
                                                        {assessment.risk_factors.slice(0, 3).map((factor, i) => (
                                                            <li key={i} className="text-sm text-slate-300 flex items-start gap-2">
                                                                <AlertTriangle className={`w-3 h-3 mt-1 flex-shrink-0 ${riskColor?.text || 'text-slate-400'}`} />
                                                                {factor}
                                                            </li>
                                                        ))}
                                                    </ul>
                                                </div>
                                            )}

                                            {/* Recommendations */}
                                            {assessment?.recommendations?.length > 0 && (
                                                <div className="mt-2">
                                                    <p className="text-xs text-slate-500 mb-1">Recommendation:</p>
                                                    <p className="text-sm text-slate-300">{assessment.recommendations[0]}</p>
                                                </div>
                                            )}
                                        </div>
                                    )
                                })}
                            </div>
                        </div>

                        {/* Summary */}
                        {riskAssessments.length > 0 && (
                            <div className="glass rounded-xl p-4">
                                <h4 className="text-sm font-medium text-slate-400 mb-3">Summary</h4>
                                <div className="flex flex-wrap gap-3">
                                    {['safe', 'caution', 'high_risk', 'contraindicated', 'fatal'].map(level => {
                                        const count = riskAssessments.filter(a => a.risk_level === level).length
                                        if (count === 0) return null
                                        const color = getRiskColor(level)
                                        return (
                                            <div key={level} className={`px-3 py-2 rounded-lg ${color.bg} ${color.border} border`}>
                                                <span className={`text-lg font-bold ${color.text}`}>{count}</span>
                                                <span className={`text-xs ml-1 capitalize ${color.text}`}>{level.replace('_', ' ')}</span>
                                            </div>
                                        )
                                    })}
                                </div>
                            </div>
                        )}

                        {/* Actions */}
                        <div className="flex gap-3">
                            <button
                                onClick={() => {
                                    setPrescription(null)
                                    setRiskAssessments([])
                                    setActiveView('upload')
                                }}
                                className="flex-1 py-3 bg-slate-700 text-white rounded-xl hover:bg-slate-600 transition-colors"
                            >
                                Scan Another
                            </button>
                            <button
                                onClick={() => window.location.href = '/diabetes'}
                                className="flex-1 py-3 bg-medical-500 text-white rounded-xl hover:bg-medical-600 transition-colors"
                            >
                                View in Diabetes Manager
                            </button>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}

export default PatientPrescriptionScanner
