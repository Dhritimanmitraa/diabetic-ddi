import { useState, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Webcam from 'react-webcam'
import { Camera, Upload, X, Loader2, RefreshCw, Check } from 'lucide-react'
import toast from 'react-hot-toast'
import { extractFromImage, checkInteraction, getAlternatives } from '../services/api'

function CameraCapture({ setResults, setAlternatives, setIsLoading }) {
  const [showCamera, setShowCamera] = useState(false)
  const [capturedImage, setCapturedImage] = useState(null)
  const [detectedDrugs, setDetectedDrugs] = useState([])
  const [selectedDrugs, setSelectedDrugs] = useState([])
  const [isProcessing, setIsProcessing] = useState(false)
  const webcamRef = useRef(null)
  const fileInputRef = useRef(null)

  const videoConstraints = {
    width: 1280,
    height: 720,
    facingMode: 'environment', // Use back camera on mobile
  }

  const capture = useCallback(() => {
    const imageSrc = webcamRef.current?.getScreenshot()
    if (imageSrc) {
      setCapturedImage(imageSrc)
      setShowCamera(false)
      processImage(imageSrc)
    }
  }, [webcamRef])

  const handleFileUpload = (event) => {
    const file = event.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onloadend = () => {
        const base64 = reader.result
        setCapturedImage(base64)
        processImage(base64)
      }
      reader.readAsDataURL(file)
    }
  }

  const processImage = async (imageBase64) => {
    setIsProcessing(true)
    setDetectedDrugs([])
    setSelectedDrugs([])
    
    try {
      const result = await extractFromImage(imageBase64)
      
      if (result.detected_drugs && result.detected_drugs.length > 0) {
        setDetectedDrugs(result.detected_drugs)
        toast.success(`Found ${result.detected_drugs.length} drug(s)!`)
        
        // Auto-select first two drugs if available
        if (result.detected_drugs.length >= 2) {
          setSelectedDrugs([result.detected_drugs[0], result.detected_drugs[1]])
        } else if (result.detected_drugs.length === 1) {
          setSelectedDrugs([result.detected_drugs[0]])
        }
      } else {
        toast.error('No drugs detected. Try a clearer image.')
      }
    } catch (error) {
      console.error('OCR error:', error)
      toast.error('Error processing image. Please try again.')
    } finally {
      setIsProcessing(false)
    }
  }

  const toggleDrugSelection = (drug) => {
    setSelectedDrugs((prev) => {
      if (prev.includes(drug)) {
        return prev.filter((d) => d !== drug)
      }
      if (prev.length < 2) {
        return [...prev, drug]
      }
      // Replace second drug if already 2 selected
      return [prev[0], drug]
    })
  }

  const checkSelectedDrugs = async () => {
    if (selectedDrugs.length !== 2) {
      toast.error('Please select exactly 2 drugs to check')
      return
    }

    setIsLoading(true)
    setResults(null)
    setAlternatives(null)

    try {
      const interactionResult = await checkInteraction(selectedDrugs[0], selectedDrugs[1])
      setResults(interactionResult)

      if (interactionResult.has_interaction && interactionResult.interaction?.severity !== 'minor') {
        try {
          const alternativesResult = await getAlternatives(selectedDrugs[0], selectedDrugs[1])
          setAlternatives(alternativesResult)
        } catch (altError) {
          console.error('Could not fetch alternatives:', altError)
        }
      }

      if (!interactionResult.has_interaction) {
        toast.success('No known interaction found!')
      } else {
        const severity = interactionResult.interaction?.severity
        if (severity === 'contraindicated') {
          toast.error('CONTRAINDICATED - Do not use together!')
        } else if (severity === 'major') {
          toast.error('Major interaction detected!')
        } else {
          toast('Interaction detected', { icon: '⚠️' })
        }
      }
    } catch (error) {
      console.error('Error:', error)
      toast.error('Error checking interaction')
    } finally {
      setIsLoading(false)
    }
  }

  const reset = () => {
    setCapturedImage(null)
    setDetectedDrugs([])
    setSelectedDrugs([])
    setShowCamera(false)
  }

  return (
    <div className="glass rounded-3xl p-8 max-w-2xl mx-auto">
      <AnimatePresence mode="wait">
        {!capturedImage ? (
          <motion.div
            key="input"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="space-y-6"
          >
            {!showCamera ? (
              <>
                {/* Camera and upload buttons */}
                <div className="grid grid-cols-2 gap-4">
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => setShowCamera(true)}
                    className="flex flex-col items-center justify-center gap-3 p-8 bg-slate-800/50 hover:bg-slate-700/50 border border-slate-700/50 hover:border-medical-500/30 rounded-2xl transition-all group"
                  >
                    <div className="w-16 h-16 rounded-2xl bg-medical-500/10 flex items-center justify-center text-medical-400 group-hover:scale-110 transition-transform">
                      <Camera className="w-8 h-8" />
                    </div>
                    <div>
                      <p className="text-white font-medium">Use Camera</p>
                      <p className="text-slate-500 text-sm">Take a photo of medication</p>
                    </div>
                  </motion.button>

                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => fileInputRef.current?.click()}
                    className="flex flex-col items-center justify-center gap-3 p-8 bg-slate-800/50 hover:bg-slate-700/50 border border-slate-700/50 hover:border-medical-500/30 rounded-2xl transition-all group"
                  >
                    <div className="w-16 h-16 rounded-2xl bg-medical-500/10 flex items-center justify-center text-medical-400 group-hover:scale-110 transition-transform">
                      <Upload className="w-8 h-8" />
                    </div>
                    <div>
                      <p className="text-white font-medium">Upload Image</p>
                      <p className="text-slate-500 text-sm">Select from gallery</p>
                    </div>
                  </motion.button>
                </div>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileUpload}
                  className="hidden"
                />

                {/* Tips */}
                <div className="p-4 bg-medical-500/5 border border-medical-500/10 rounded-xl">
                  <p className="text-medical-400 text-sm font-medium mb-2">📷 Tips for best results:</p>
                  <ul className="text-slate-400 text-sm space-y-1">
                    <li>• Ensure good lighting on the medication label</li>
                    <li>• Keep the text in focus and straight</li>
                    <li>• Include the drug name clearly in the frame</li>
                  </ul>
                </div>
              </>
            ) : (
              /* Camera view */
              <div className="space-y-4">
                <div className="relative rounded-2xl overflow-hidden bg-slate-900">
                  <Webcam
                    ref={webcamRef}
                    audio={false}
                    screenshotFormat="image/jpeg"
                    videoConstraints={videoConstraints}
                    className="w-full rounded-2xl"
                  />
                  <div className="absolute inset-0 border-4 border-medical-500/30 rounded-2xl pointer-events-none" />
                  <div className="absolute top-4 left-4 px-3 py-1 bg-slate-900/80 rounded-full">
                    <p className="text-white text-sm">Position medication label in frame</p>
                  </div>
                </div>

                <div className="flex gap-3">
                  <button
                    onClick={() => setShowCamera(false)}
                    className="flex-1 py-3 bg-slate-800 hover:bg-slate-700 text-white rounded-xl transition-colors flex items-center justify-center gap-2"
                  >
                    <X className="w-5 h-5" />
                    Cancel
                  </button>
                  <button
                    onClick={capture}
                    className="flex-1 py-3 bg-medical-500 hover:bg-medical-400 text-white rounded-xl transition-colors flex items-center justify-center gap-2"
                  >
                    <Camera className="w-5 h-5" />
                    Capture
                  </button>
                </div>
              </div>
            )}
          </motion.div>
        ) : (
          <motion.div
            key="results"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="space-y-6"
          >
            {/* Captured image preview */}
            <div className="relative rounded-2xl overflow-hidden">
              <img
                src={capturedImage}
                alt="Captured medication"
                className="w-full rounded-2xl"
              />
              <button
                onClick={reset}
                className="absolute top-3 right-3 p-2 bg-slate-900/80 hover:bg-slate-800 rounded-full transition-colors"
              >
                <X className="w-5 h-5 text-white" />
              </button>
            </div>

            {/* Processing indicator */}
            {isProcessing && (
              <div className="flex items-center justify-center gap-3 py-4">
                <Loader2 className="w-6 h-6 text-medical-400 animate-spin" />
                <p className="text-medical-400">Analyzing image...</p>
              </div>
            )}

            {/* Detected drugs */}
            {!isProcessing && detectedDrugs.length > 0 && (
              <div className="space-y-4">
                <p className="text-slate-400 text-sm">
                  Select 2 drugs to check for interactions:
                </p>
                <div className="flex flex-wrap gap-2">
                  {detectedDrugs.map((drug, index) => (
                    <button
                      key={index}
                      onClick={() => toggleDrugSelection(drug)}
                      className={`px-4 py-2 rounded-xl border transition-all flex items-center gap-2 ${
                        selectedDrugs.includes(drug)
                          ? 'bg-medical-500/20 border-medical-500/50 text-medical-400'
                          : 'bg-slate-800/50 border-slate-700/50 text-slate-300 hover:border-medical-500/30'
                      }`}
                    >
                      {selectedDrugs.includes(drug) && (
                        <Check className="w-4 h-4" />
                      )}
                      {drug}
                    </button>
                  ))}
                </div>

                <div className="flex gap-3">
                  <button
                    onClick={reset}
                    className="flex-1 py-3 bg-slate-800 hover:bg-slate-700 text-white rounded-xl transition-colors flex items-center justify-center gap-2"
                  >
                    <RefreshCw className="w-5 h-5" />
                    Try Again
                  </button>
                  <button
                    onClick={checkSelectedDrugs}
                    disabled={selectedDrugs.length !== 2}
                    className={`flex-1 py-3 rounded-xl transition-colors flex items-center justify-center gap-2 ${
                      selectedDrugs.length === 2
                        ? 'bg-medical-500 hover:bg-medical-400 text-white'
                        : 'bg-slate-700 text-slate-500 cursor-not-allowed'
                    }`}
                  >
                    Check Interaction
                  </button>
                </div>
              </div>
            )}

            {/* No drugs found */}
            {!isProcessing && detectedDrugs.length === 0 && (
              <div className="text-center py-4">
                <p className="text-slate-400 mb-4">
                  No drugs detected in the image. Please try again with a clearer photo.
                </p>
                <button
                  onClick={reset}
                  className="px-6 py-3 bg-medical-500 hover:bg-medical-400 text-white rounded-xl transition-colors flex items-center justify-center gap-2 mx-auto"
                >
                  <RefreshCw className="w-5 h-5" />
                  Try Again
                </button>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default CameraCapture

