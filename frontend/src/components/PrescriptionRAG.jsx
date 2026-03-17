import { useState, useRef, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Upload, FileText, Pill, AlertCircle,
  MessageSquare, Send, Trash2, ChevronDown, ChevronUp,
  Sun, Moon, Sunset, Loader2, Check,
  Mic, MicOff, AlertTriangle, Shield, Camera, SwitchCamera, XCircle
} from 'lucide-react'
import toast from 'react-hot-toast'
import usePageTitle from '../hooks/usePageTitle'
import usePrescriptionStore from '../stores/usePrescriptionStore'

function PrescriptionRAG() {
  usePageTitle('Prescription Scanner')
  const activeSection = usePrescriptionStore((state) => state.activeSection)
  const setActiveSection = usePrescriptionStore((state) => state.setActiveSection)
  const isUploading = usePrescriptionStore((state) => state.isUploading)
  const prescription = usePrescriptionStore((state) => state.prescription)
  const history = usePrescriptionStore((state) => state.history)
  const chatMessages = usePrescriptionStore((state) => state.chatMessages)
  const isChatLoading = usePrescriptionStore((state) => state.isChatLoading)
  const drugWarnings = usePrescriptionStore((state) => state.drugWarnings)
  const isCheckingWarnings = usePrescriptionStore((state) => state.isCheckingWarnings)
  const loadHistory = usePrescriptionStore((state) => state.loadHistory)
  const uploadFile = usePrescriptionStore((state) => state.uploadFile)
  const sendChat = usePrescriptionStore((state) => state.sendChatMessage)
  const removePrescription = usePrescriptionStore((state) => state.removePrescription)
  const selectPrescriptionFromHistory = usePrescriptionStore((state) => state.selectPrescriptionFromHistory)

  const [chatInput, setChatInput] = useState('')
  const [expandedMedicine, setExpandedMedicine] = useState(null)

  // Voice input state
  const [isListening, setIsListening] = useState(false)
  const [voiceSupported, setVoiceSupported] = useState(false)
  const recognitionRef = useRef(null)

  // Uploaded image preview state
  const [uploadedImageUrl, setUploadedImageUrl] = useState(null)

  // Camera capture state
  const [showCamera, setShowCamera] = useState(false)
  const [cameraStream, setCameraStream] = useState(null)
  const [facingMode, setFacingMode] = useState('environment') // 'user' or 'environment'
  const videoRef = useRef(null)
  const canvasRef = useRef(null)

  const fileInputRef = useRef(null)
  const chatEndRef = useRef(null)

  // Initialize speech recognition
  useEffect(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
      recognitionRef.current = new SpeechRecognition()
      recognitionRef.current.continuous = false
      recognitionRef.current.interimResults = true
      recognitionRef.current.lang = 'en-US'

      recognitionRef.current.onresult = (event) => {
        const transcript = Array.from(event.results)
          .map(result => result[0].transcript)
          .join('')
        setChatInput(transcript)
      }

      recognitionRef.current.onend = () => {
        setIsListening(false)
      }

      recognitionRef.current.onerror = (event) => {
        console.error('Speech recognition error:', event.error)
        setIsListening(false)
        if (event.error === 'not-allowed') {
          toast.error('Microphone access denied')
        }
      }

      setVoiceSupported(true)
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.abort()
      }
    }
  }, [])

  // Load history on mount
  useEffect(() => {
    loadHistory().catch((err) => {
      console.error('Failed to load history:', err)
      toast.error('Failed to load prescription history')
    })
  }, [loadHistory])

  // Scroll chat to bottom only when USER sends a message (not on every update)
  // Disabled auto-scroll to prevent page jumping
  // useEffect(() => {
  //   chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  // }, [chatMessages])

  // Voice input toggle
  const toggleVoiceInput = useCallback(() => {
    if (!recognitionRef.current) return

    if (isListening) {
      recognitionRef.current.stop()
      setIsListening(false)
    } else {
      try {
        recognitionRef.current.start()
        setIsListening(true)
        toast.success('Listening... Speak now', { duration: 2000 })
      } catch (err) {
        console.error('Failed to start speech recognition:', err)
        toast.error('Could not start voice input')
      }
    }
  }, [isListening])

  // Camera functions
  const startCamera = useCallback(async () => {
    try {
      // Stop any existing stream first
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

      // Connect stream to video element after state updates
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

    // Set canvas size to video size
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    // Draw video frame to canvas
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0)

    // Convert to blob
    canvas.toBlob(async (blob) => {
      if (!blob) {
        toast.error('Failed to capture image')
        return
      }

      // Create file from blob
      const file = new File([blob], 'prescription_capture.jpg', { type: 'image/jpeg' })

      // Stop camera
      stopCamera()

      // Process the file like a normal upload
      // Create preview URL for the captured image
      const previewUrl = URL.createObjectURL(blob)
      setUploadedImageUrl(previewUrl)

      try {
        const result = await uploadFile(file)

        if (result.status === 'completed') {
          toast.success(`Extracted ${result.medicines?.length || 0} medicine(s)!`)
        } else {
          toast.error(result.message || 'Extraction failed')
        }
      } catch (err) {
        console.error('Upload error:', err)
        toast.error(err.message || 'Failed to process prescription')
      }
    }, 'image/jpeg', 0.9)
  }, [stopCamera, uploadFile])

  const handleFileSelect = async (event) => {
    const file = event.target.files?.[0]
    if (!file) return

    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/png', 'image/webp', 'application/pdf']
    if (!allowedTypes.includes(file.type)) {
      toast.error('Please upload an image (JPEG, PNG) or PDF')
      return
    }

    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
      toast.error('File too large. Maximum size is 10MB')
      return
    }

    // Create preview URL for the uploaded image
    if (file.type.startsWith('image/')) {
      const previewUrl = URL.createObjectURL(file)
      setUploadedImageUrl(previewUrl)
    } else {
      setUploadedImageUrl(null) // PDF doesn't have image preview
    }

    try {
      const result = await uploadFile(file)

      if (result.status === 'completed') {
        toast.success(`Extracted ${result.medicines?.length || 0} medicine(s)!`)
      } else {
        toast.error(result.message || 'Extraction failed')
      }
    } catch (err) {
      console.error('Upload error:', err)
      toast.error(err.message || 'Failed to upload prescription')
    }
  }

  const handleDrop = (event) => {
    event.preventDefault()
    const file = event.dataTransfer.files?.[0]
    if (file) {
      const input = fileInputRef.current
      const dt = new DataTransfer()
      dt.items.add(file)
      input.files = dt.files
      handleFileSelect({ target: { files: dt.files } })
    }
  }

  const handleDragOver = (event) => {
    event.preventDefault()
  }

  const sendChatMessage = useCallback(async (message) => {
    if (!message?.trim() || !prescription?.id) return

    setChatInput('')

    try {
      await sendChat(message)
    } catch (err) {
      console.error('Chat error:', err)
      toast.error('Failed to get response')
    }
  }, [prescription?.id, sendChat])

  const handleSendChat = () => sendChatMessage(chatInput)

  const handleQuickSend = useCallback((question) => {
    if (isChatLoading) return
    sendChatMessage(question)
  }, [isChatLoading, sendChatMessage])

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendChat()
    }
  }

  const handleDeletePrescription = async (id) => {
    toast((t) => (
      <div className="flex items-center gap-3">
        <span className="text-sm">Delete this prescription?</span>
        <button
          onClick={async () => {
            toast.dismiss(t.id)
            try {
              await removePrescription(id)
              toast.success('Prescription deleted')
            } catch (err) {
              toast.error('Failed to delete')
            }
          }}
          className="px-3 py-1 bg-red-500 text-white text-xs font-medium rounded-lg hover:bg-red-600 transition-colors"
        >
          Delete
        </button>
        <button
          onClick={() => toast.dismiss(t.id)}
          className="px-3 py-1 bg-slate-600 text-white text-xs font-medium rounded-lg hover:bg-slate-500 transition-colors"
        >
          Cancel
        </button>
      </div>
    ), { duration: 10000 })
  }

  const handleSelectFromHistory = async (item) => {
    try {
      await selectPrescriptionFromHistory(item)
    } catch (e) {
      console.error('Failed to load prescription details:', e)
      toast.error('Failed to load prescription details')
    }
  }

  return (
    <div className="min-h-screen pt-20 pb-12">
      <div className="max-w-6xl mx-auto px-4">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl font-sans font-bold text-[var(--text-primary)] mb-3">
            Prescription <span className="gradient-text">Scanner</span>
          </h1>
          <p className="text-[var(--text-secondary)] max-w-2xl mx-auto">
            Upload your prescription image or PDF. AI will extract medicine details
            and answer your questions about dosage, timing, and more.
          </p>
        </motion.div>

        {/* Tab Navigation */}
        <div className="flex justify-center gap-2 mb-8" role="tablist" aria-label="Prescription sections">
          {['upload', 'result', 'history'].map((tab, idx) => {
            const isDisabled = tab === 'result' && !prescription
            return (
              <button
                key={tab}
                role="tab"
                id={`rx-tab-${tab}`}
                aria-selected={activeSection === tab}
                aria-controls={`rx-panel-${tab}`}
                tabIndex={activeSection === tab ? 0 : -1}
                onClick={() => setActiveSection(tab)}
                onKeyDown={(e) => {
                  const tabs = ['upload', 'result', 'history']
                  if (e.key === 'ArrowRight') { e.preventDefault(); setActiveSection(tabs[(idx + 1) % 3]) }
                  if (e.key === 'ArrowLeft') { e.preventDefault(); setActiveSection(tabs[(idx + 2) % 3]) }
                }}
                disabled={isDisabled}
                className={`px-5 py-2.5 rounded-xl font-medium capitalize transition-all ${activeSection === tab
                  ? 'bg-medical-500 text-white shadow-lg shadow-medical-500/25'
                  : isDisabled
                    ? 'bg-slate-800/30 text-slate-600 cursor-not-allowed'
                    : 'bg-[var(--bg-elevated)]/50 text-[var(--text-secondary)] hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)]'
                  }`}
              >
                {tab === 'upload' && <Upload className="w-4 h-4 inline mr-2" />}
                {tab === 'result' && <Pill className="w-4 h-4 inline mr-2" />}
                {tab === 'history' && <FileText className="w-4 h-4 inline mr-2" />}
                {tab}
              </button>
            )
          })}
        </div>

        <AnimatePresence mode="wait">
          {/* Upload Section */}
          {activeSection === 'upload' && (
            <motion.div
              key="upload"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="max-w-2xl mx-auto"
            >
              {/* Camera View */}
              {showCamera ? (
                <div className="glass rounded-2xl overflow-hidden">
                  {/* Camera Preview */}
                  <div className="relative bg-black aspect-[4/3]">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      className="w-full h-full object-cover"
                    />
                    <canvas ref={canvasRef} className="hidden" />

                    {/* Camera Controls Overlay */}
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

                    {/* Capture Hint */}
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
                  <div className="grid grid-cols-2 gap-4 mb-4">
                    {/* Camera Button */}
                    <button
                      onClick={startCamera}
                      disabled={isUploading}
                      className="glass rounded-2xl p-8 text-center hover:bg-[var(--bg-elevated)]/50 transition-all border-2 border-transparent hover:border-medical-500/30 disabled:opacity-50"
                    >
                      <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-medical-500/10 flex items-center justify-center">
                        <Camera className="w-8 h-8 text-medical-400" />
                      </div>
                      <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-1">
                        Take Photo
                      </h3>
                      <p className="text-[var(--text-muted)] text-sm">
                        Use camera to capture
                      </p>
                    </button>

                    {/* File Upload Button */}
                    <div
                      onClick={() => !isUploading && fileInputRef.current?.click()}
                      onDrop={handleDrop}
                      onDragOver={handleDragOver}
                      className={`glass rounded-2xl p-8 text-center cursor-pointer border-2 border-transparent transition-all ${isUploading
                        ? 'opacity-50 cursor-not-allowed'
                        : 'hover:bg-[var(--bg-elevated)]/50 hover:border-medical-500/30'
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
                      <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-1">
                        Upload File
                      </h3>
                      <p className="text-[var(--text-muted)] text-sm">
                        JPEG, PNG, or PDF
                      </p>
                    </div>
                  </div>

                  {/* Processing State */}
                  {isUploading && (
                    <div className="glass rounded-2xl p-8 text-center">
                      <Loader2 className="w-12 h-12 text-medical-400 animate-spin mx-auto mb-4" />
                      <p className="text-medical-400 font-medium">Processing prescription...</p>
                      <p className="text-[var(--text-muted)] text-sm mt-1">Extracting medicine details with AI</p>
                    </div>
                  )}
                </>
              )}

              {/* Tips */}
              {!showCamera && (
                <div className="mt-6 p-4 bg-amber-500/5 border border-amber-500/10 rounded-xl">
                  <p className="text-amber-400 text-sm font-medium mb-2">Tips for best results:</p>
                  <ul className="text-[var(--text-secondary)] text-sm space-y-1">
                    <li>• Ensure good lighting and clear focus</li>
                    <li>• Include all medicines in the frame</li>
                    <li>• Handwritten prescriptions work too!</li>
                  </ul>
                </div>
              )}
            </motion.div>
          )}

          {/* Result Section */}
          {activeSection === 'result' && prescription && (
            <motion.div
              key="result"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="grid lg:grid-cols-2 gap-6"
            >
              {/* Medicines Panel */}
              <div className="glass rounded-2xl p-6">
                {/* Uploaded Prescription Image Preview */}
                {uploadedImageUrl && (
                  <div className="mb-4">
                    <div className="rounded-xl overflow-hidden border border-[var(--border)]/50 bg-slate-800/30">
                      <img
                        src={uploadedImageUrl}
                        alt="Uploaded prescription"
                        className="w-full h-auto max-h-48 object-contain"
                      />
                    </div>
                  </div>
                )}

                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-semibold text-[var(--text-primary)] flex items-center gap-2">
                    <Pill className="w-5 h-5 text-medical-400" />
                    Medicines ({prescription.medicines?.length || 0})
                  </h2>
                  <span className="text-xs px-2 py-1 rounded-full bg-medical-500/20 text-medical-400">
                    {prescription.vision_model_used}
                  </span>
                </div>

                {prescription.medicines?.length > 0 ? (
                  <div className="space-y-3 max-h-[500px] overflow-y-auto pr-2">
                    {prescription.medicines.map((med, idx) => (
                      <MedicineCard
                        key={idx}
                        medicine={med}
                        expanded={expandedMedicine === idx}
                        onToggle={() => setExpandedMedicine(expandedMedicine === idx ? null : idx)}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8 text-slate-500">
                    <AlertCircle className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>No medicines extracted</p>
                  </div>
                )}

                {prescription.extraction_confidence && (
                  <div className="mt-4 pt-4 border-t border-[var(--border)]/50">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-[var(--text-secondary)]">Extraction Confidence</span>
                      <span className={`font-medium ${prescription.extraction_confidence > 0.7 ? 'text-green-400' :
                        prescription.extraction_confidence > 0.4 ? 'text-amber-400' : 'text-red-400'
                        }`}>
                        {Math.round(prescription.extraction_confidence * 100)}%
                      </span>
                    </div>
                  </div>
                )}

                {/* Drug Warnings Panel */}
                {(drugWarnings || isCheckingWarnings) && (
                  <div className="mt-4 pt-4 border-t border-[var(--border)]/50">
                    <div className="flex items-center gap-2 mb-3">
                      <Shield className="w-4 h-4 text-amber-400" />
                      <span className="text-sm font-medium text-[var(--text-primary)]">Drug Interactions</span>
                    </div>

                    {isCheckingWarnings ? (
                      <div className="flex items-center gap-2 text-sm text-[var(--text-secondary)]">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Checking interactions...
                      </div>
                    ) : drugWarnings?.interactions?.length > 0 ? (
                      <div className="space-y-2">
                        {drugWarnings.interactions.slice(0, 3).map((interaction, idx) => (
                          <div
                            key={idx}
                            className={`p-3 rounded-lg text-sm ${interaction.severity === 'contraindicated' ? 'bg-red-500/10 border border-red-500/30' :
                              interaction.severity === 'major' ? 'bg-orange-500/10 border border-orange-500/30' :
                                interaction.severity === 'moderate' ? 'bg-amber-500/10 border border-amber-500/30' :
                                  'bg-slate-700/30 border border-[var(--border)]/30'
                              }`}
                          >
                            <div className="flex items-start gap-2">
                              <AlertTriangle className={`w-4 h-4 flex-shrink-0 mt-0.5 ${interaction.severity === 'contraindicated' ? 'text-red-400' :
                                interaction.severity === 'major' ? 'text-orange-400' :
                                  interaction.severity === 'moderate' ? 'text-amber-400' :
                                    'text-[var(--text-secondary)]'
                                }`} />
                              <div>
                                <p className="font-medium text-[var(--text-primary)]">
                                  {interaction.drug1} + {interaction.drug2}
                                </p>
                                <p className={`text-xs capitalize ${interaction.severity === 'contraindicated' ? 'text-red-400' :
                                  interaction.severity === 'major' ? 'text-orange-400' :
                                    interaction.severity === 'moderate' ? 'text-amber-400' :
                                      'text-[var(--text-secondary)]'
                                  }`}>
                                  {interaction.severity} interaction
                                </p>
                                {interaction.description && (
                                  <p className="text-[var(--text-secondary)] mt-1 text-xs line-clamp-2">
                                    {interaction.description}
                                  </p>
                                )}
                              </div>
                            </div>
                          </div>
                        ))}
                        {drugWarnings.interactions.length > 3 && (
                          <p className="text-xs text-[var(--text-muted)] text-center">
                            +{drugWarnings.interactions.length - 3} more interactions
                          </p>
                        )}
                      </div>
                    ) : (
                      <div className="flex items-center gap-2 text-sm text-green-400">
                        <Check className="w-4 h-4" />
                        No known interactions found
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Chat Panel */}
              <div className="glass rounded-2xl p-6 flex flex-col">
                <div className="flex items-center gap-2 mb-4">
                  <MessageSquare className="w-5 h-5 text-purple-400" />
                  <h2 className="text-xl font-semibold text-[var(--text-primary)]">Ask Questions</h2>
                </div>

                {/* Chat Messages */}
                <div className="flex-1 min-h-[300px] max-h-[400px] overflow-y-auto space-y-3 mb-4 pr-2">
                  {chatMessages.length === 0 ? (
                    <div className="text-center py-8 text-slate-500">
                      <MessageSquare className="w-10 h-10 mx-auto mb-3 opacity-50" />
                      <p className="mb-3">Ask about your prescription</p>
                      <div className="flex flex-wrap justify-center gap-2">
                        {[
                          'When should I take these?',
                          'What are the side effects?',
                          'Can I take these together?',
                          'Is this safe for diabetics?',
                          'Morning vs night medicines?',
                          'What is each medicine for?'
                        ].map((q, i) => (
                          <button
                            key={i}
                            onClick={() => handleQuickSend(q)}
                            disabled={isChatLoading}
                            className="text-xs px-3 py-1.5 rounded-full bg-[var(--bg-elevated)]/50 text-[var(--text-secondary)] hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                          >
                            {q}
                          </button>
                        ))}
                      </div>
                    </div>
                  ) : (
                    chatMessages.map((msg, idx) => (
                      <div
                        key={idx}
                        className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                      >
                        <div className={`max-w-[85%] px-4 py-2.5 rounded-2xl ${msg.role === 'user'
                          ? 'bg-medical-500 text-white rounded-br-md'
                          : msg.error
                            ? 'bg-red-500/10 text-red-300 rounded-bl-md'
                            : 'bg-slate-800 text-slate-200 rounded-bl-md'
                          }`}>
                          <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                          {msg.model_used && (
                            <p className="text-xs opacity-50 mt-1">{msg.model_used}</p>
                          )}
                        </div>
                      </div>
                    ))
                  )}
                  {isChatLoading && (
                    <div className="flex justify-start">
                      <div className="bg-slate-800 px-4 py-3 rounded-2xl rounded-bl-md">
                        <Loader2 className="w-5 h-5 text-[var(--text-secondary)] animate-spin" />
                      </div>
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>

                {/* Chat Input */}
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder={isListening ? "Listening..." : "Ask about your prescription..."}
                    className={`flex-1 px-4 py-3 bg-[var(--bg-elevated)] border rounded-xl text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none transition-all ${isListening
                      ? 'border-red-500/50 animate-pulse'
                      : 'border-[var(--border)] focus:border-medical-500/50'
                      }`}
                    disabled={isChatLoading}
                  />
                  {voiceSupported && (
                    <button
                      onClick={toggleVoiceInput}
                      disabled={isChatLoading}
                      className={`px-4 py-3 rounded-xl transition-all ${isListening
                        ? 'bg-red-500 hover:bg-red-400 text-white animate-pulse'
                        : 'bg-slate-700 hover:bg-slate-600 text-slate-300'
                        }`}
                      title={isListening ? "Stop listening" : "Voice input"}
                    >
                      {isListening ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
                    </button>
                  )}
                  <button
                    onClick={handleSendChat}
                    disabled={!chatInput.trim() || isChatLoading}
                    className="px-4 py-3 bg-medical-500 hover:bg-medical-400 disabled:bg-slate-700 disabled:text-[var(--text-muted)] text-white rounded-xl transition-colors"
                  >
                    <Send className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </motion.div>
          )}

          {/* History Section */}
          {activeSection === 'history' && (
            <motion.div
              key="history"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="max-w-3xl mx-auto"
            >
              <div className="glass rounded-2xl p-6">
                <h2 className="text-xl font-semibold text-[var(--text-primary)] mb-4 flex items-center gap-2">
                  <FileText className="w-5 h-5 text-[var(--text-secondary)]" />
                  Recent Prescriptions
                </h2>

                {history.length > 0 ? (
                  <div className="space-y-3">
                    {history.map((item) => (
                      <div
                        key={item.id}
                        className="flex items-center justify-between p-4 bg-slate-800/30 rounded-xl hover:bg-[var(--bg-elevated)]/50 transition-colors"
                      >
                        <div
                          className="flex-1 cursor-pointer"
                          onClick={() => handleSelectFromHistory(item)}
                        >
                          <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-lg bg-medical-500/10 flex items-center justify-center">
                              <FileText className="w-5 h-5 text-medical-400" />
                            </div>
                            <div>
                              <p className="text-[var(--text-primary)] font-medium">
                                {item.filename || `Prescription #${item.id}`}
                              </p>
                              <p className="text-[var(--text-muted)] text-sm">
                                {item.medicines?.length || 0} medicines • {new Date(item.created_at).toLocaleDateString()}
                              </p>
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className={`text-xs px-2 py-1 rounded-full ${item.status === 'completed'
                            ? 'bg-green-500/20 text-green-400'
                            : item.status === 'failed'
                              ? 'bg-red-500/20 text-red-400'
                              : 'bg-amber-500/20 text-amber-400'
                            }`}>
                            {item.status}
                          </span>
                          <button
                            onClick={() => handleDeletePrescription(item.id)}
                            className="p-2 text-[var(--text-muted)] hover:text-red-400 transition-colors"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12 text-slate-500">
                    <FileText className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>No prescriptions uploaded yet</p>
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

// Medicine Card Component
function MedicineCard({ medicine, expanded, onToggle }) {
  return (
    <div className="bg-slate-800/30 rounded-xl overflow-hidden">
      <button
        onClick={onToggle}
        className="w-full p-4 flex items-center justify-between text-left hover:bg-[var(--bg-elevated)]/50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-medical-500/10 flex items-center justify-center">
            <Pill className="w-5 h-5 text-medical-400" />
          </div>
          <div>
            <p className="text-[var(--text-primary)] font-medium">{medicine.name}</p>
            {medicine.dosage && (
              <p className="text-[var(--text-muted)] text-sm">{medicine.dosage}</p>
            )}
          </div>
        </div>
        <div className="flex items-center gap-3">
          {/* Timing indicators */}
          <div className="flex gap-1">
            {medicine.morning && (
              <span title="Morning" className="w-6 h-6 rounded-full bg-amber-500/20 flex items-center justify-center">
                <Sun className="w-3 h-3 text-amber-400" />
              </span>
            )}
            {medicine.afternoon && (
              <span title="Afternoon" className="w-6 h-6 rounded-full bg-orange-500/20 flex items-center justify-center">
                <Sunset className="w-3 h-3 text-orange-400" />
              </span>
            )}
            {medicine.night && (
              <span title="Night" className="w-6 h-6 rounded-full bg-indigo-500/20 flex items-center justify-center">
                <Moon className="w-3 h-3 text-indigo-400" />
              </span>
            )}
          </div>
          {expanded ? (
            <ChevronUp className="w-5 h-5 text-slate-500" />
          ) : (
            <ChevronDown className="w-5 h-5 text-slate-500" />
          )}
        </div>
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 space-y-2 text-sm">
              {medicine.generic_name && (
                <div className="flex justify-between">
                  <span className="text-slate-500">Generic</span>
                  <span className="text-slate-300">{medicine.generic_name}</span>
                </div>
              )}
              {medicine.quantity && (
                <div className="flex justify-between">
                  <span className="text-slate-500">Quantity</span>
                  <span className="text-slate-300">{medicine.quantity}</span>
                </div>
              )}
              {medicine.frequency && (
                <div className="flex justify-between">
                  <span className="text-slate-500">Frequency</span>
                  <span className="text-slate-300">{medicine.frequency}</span>
                </div>
              )}
              {medicine.duration && (
                <div className="flex justify-between">
                  <span className="text-slate-500">Duration</span>
                  <span className="text-slate-300">{medicine.duration}</span>
                </div>
              )}
              {medicine.instructions && (
                <div className="pt-2 border-t border-[var(--border)]/50">
                  <span className="text-[var(--text-muted)] block mb-1">Instructions</span>
                  <span className="text-slate-300">{medicine.instructions}</span>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default PrescriptionRAG
