import { useState, useRef, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Webcam from 'react-webcam'
import { Camera, Upload, X, Loader2, RefreshCw, Check } from 'lucide-react'
import toast from 'react-hot-toast'
import { extractFromImage } from '../services/api'
import { isNative } from '../utils/platform'
import useDrugStore from '../stores/useDrugStore'

// Conditionally import Capacitor Camera for native platforms
let CapacitorCamera = null
let CameraResultType = null
let CameraSource = null

// Dynamic import for Capacitor Camera (only loaded on native)
if (typeof window !== 'undefined') {
  import('@capacitor/camera').then((module) => {
    CapacitorCamera = module.Camera
    CameraResultType = module.CameraResultType
    CameraSource = module.CameraSource
  }).catch(() => {
    // Capacitor Camera not available (web fallback)
  })
}

function CameraCapture() {
  const runInteractionCheck = useDrugStore((state) => state.checkInteraction)
  const [showCamera, setShowCamera] = useState(false)
  const [capturedImage, setCapturedImage] = useState(null)
  const [detectedDrugs, setDetectedDrugs] = useState([])
  const [selectedDrugs, setSelectedDrugs] = useState([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [useNativeCamera, setUseNativeCamera] = useState(false)
  const webcamRef = useRef(null)
  const fileInputRef = useRef(null)

  // Check platform on mount
  useEffect(() => {
    setUseNativeCamera(isNative())
  }, [])

  const videoConstraints = {
    width: 1280,
    height: 720,
    facingMode: 'environment', // Use back camera on mobile
  }

  // Native camera capture using Capacitor
  const captureNative = async () => {
    if (!CapacitorCamera) {
      toast.error('Camera not available')
      return
    }

    try {
      const image = await CapacitorCamera.getPhoto({
        quality: 90,
        allowEditing: false,
        resultType: CameraResultType.Base64,
        source: CameraSource.Camera,
        correctOrientation: true,
      })

      const base64Image = `data:image/${image.format};base64,${image.base64String}`
      setCapturedImage(base64Image)
      setShowCamera(false)
      processImage(base64Image)
    } catch (error) {
      if (error.message !== 'User cancelled photos app') {
        console.error('Camera error:', error)
        toast.error('Could not access camera')
      }
    }
  }

  // Native gallery picker using Capacitor
  const pickFromGalleryNative = async () => {
    if (!CapacitorCamera) {
      // Fallback to file input
      fileInputRef.current?.click()
      return
    }

    try {
      const image = await CapacitorCamera.getPhoto({
        quality: 90,
        allowEditing: false,
        resultType: CameraResultType.Base64,
        source: CameraSource.Photos,
      })

      const base64Image = `data:image/${image.format};base64,${image.base64String}`
      setCapturedImage(base64Image)
      processImage(base64Image)
    } catch (error) {
      if (error.message !== 'User cancelled photos app') {
        console.error('Gallery error:', error)
        toast.error('Could not access gallery')
      }
    }
  }

  // Web camera capture using Webcam
  const captureWeb = useCallback(() => {
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

  const handleCameraClick = () => {
    if (useNativeCamera) {
      captureNative()
    } else {
      setShowCamera(true)
    }
  }

  const handleUploadClick = () => {
    if (useNativeCamera) {
      pickFromGalleryNative()
    } else {
      fileInputRef.current?.click()
    }
  }

  // Handle keyboard navigation for upload button triggering file input
  const handleUploadKeyDown = (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      handleUploadClick()
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

  // Handle keyboard navigation for drug selection
  const handleDrugKeyDown = (event, drug) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      toggleDrugSelection(drug)
    }
  }

  const checkSelectedDrugs = async () => {
    if (selectedDrugs.length !== 2) {
      toast.error('Please select exactly 2 drugs to check')
      return
    }

    try {
      const interactionResult = await runInteractionCheck(selectedDrugs[0], selectedDrugs[1])

      if (!interactionResult.has_interaction) {
        toast.success('No known interaction found!')
      } else {
        const severity = interactionResult.interaction?.severity
        if (severity === 'contraindicated') {
          toast.error('CONTRAINDICATED - Do not use together!')
        } else if (severity === 'major') {
          toast.error('Major interaction detected!')
        } else {
          toast('Interaction detected')
        }
      }
    } catch (error) {
      console.error('Error:', error)
      toast.error('Error checking interaction')
    }
  }

  const reset = () => {
    setCapturedImage(null)
    setDetectedDrugs([])
    setSelectedDrugs([])
    setShowCamera(false)
  }

  return (
    <div className="glass rounded-3xl p-8 max-w-2xl mx-auto" role="region" aria-label="Camera drug scanner">
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
                <div className="grid grid-cols-2 gap-4" role="group" aria-label="Image capture options">
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={handleCameraClick}
                    className="flex flex-col items-center justify-center gap-3 p-8 bg-slate-800/50 hover:bg-slate-700/50 border border-slate-700/50 hover:border-medical-500/30 rounded-2xl transition-all group focus:outline-none focus:ring-2 focus:ring-medical-500/50"
                    aria-label="Open camera to take a photo of medication"
                  >
                    <div className="w-16 h-16 rounded-2xl bg-medical-500/10 flex items-center justify-center text-medical-400 group-hover:scale-110 transition-transform" aria-hidden="true">
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
                    onClick={handleUploadClick}
                    onKeyDown={handleUploadKeyDown}
                    className="flex flex-col items-center justify-center gap-3 p-8 bg-slate-800/50 hover:bg-slate-700/50 border border-slate-700/50 hover:border-medical-500/30 rounded-2xl transition-all group focus:outline-none focus:ring-2 focus:ring-medical-500/50"
                    aria-label="Upload an image of medication from gallery"
                  >
                    <div className="w-16 h-16 rounded-2xl bg-medical-500/10 flex items-center justify-center text-medical-400 group-hover:scale-110 transition-transform" aria-hidden="true">
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
                  aria-label="Upload medication image file"
                  tabIndex={-1}
                />

                {/* Tips */}
                <div className="p-4 bg-medical-500/5 border border-medical-500/10 rounded-xl" role="note" aria-label="Tips for best results">
                  <p className="text-medical-400 text-sm font-medium mb-2">Tips for best results:</p>
                  <ul className="text-slate-400 text-sm space-y-1" aria-label="Photo tips list">
                    <li>• Ensure good lighting on the medication label</li>
                    <li>• Keep the text in focus and straight</li>
                    <li>• Include the drug name clearly in the frame</li>
                  </ul>
                </div>
              </>
            ) : (
              /* Web Camera view - only shown when not using native */
              <div className="space-y-4" role="region" aria-label="Camera viewfinder">
                <div className="relative rounded-2xl overflow-hidden bg-slate-900">
                  <Webcam
                    ref={webcamRef}
                    audio={false}
                    screenshotFormat="image/jpeg"
                    videoConstraints={videoConstraints}
                    className="w-full rounded-2xl"
                    aria-label="Camera preview"
                  />
                  <div className="absolute inset-0 border-4 border-medical-500/30 rounded-2xl pointer-events-none" aria-hidden="true" />
                  <div className="absolute top-4 left-4 px-3 py-1 bg-slate-900/80 rounded-full" role="status">
                    <p className="text-white text-sm">Position medication label in frame</p>
                  </div>
                </div>

                <div className="flex gap-3" role="group" aria-label="Camera controls">
                  <button
                    onClick={() => setShowCamera(false)}
                    className="flex-1 py-3 bg-slate-800 hover:bg-slate-700 text-white rounded-xl transition-colors flex items-center justify-center gap-2 focus:outline-none focus:ring-2 focus:ring-slate-500"
                    aria-label="Cancel and close camera"
                  >
                    <X className="w-5 h-5" aria-hidden="true" />
                    Cancel
                  </button>
                  <button
                    onClick={captureWeb}
                    className="flex-1 py-3 bg-medical-500 hover:bg-medical-400 text-white rounded-xl transition-colors flex items-center justify-center gap-2 focus:outline-none focus:ring-2 focus:ring-medical-300"
                    aria-label="Capture photo"
                  >
                    <Camera className="w-5 h-5" aria-hidden="true" />
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
            role="region"
            aria-label="Captured image and detected drugs"
          >
            {/* Captured image preview */}
            <div className="relative rounded-2xl overflow-hidden">
              <img
                src={capturedImage}
                alt="Captured medication label showing detected drugs"
                className="w-full rounded-2xl"
              />
              <button
                onClick={reset}
                className="absolute top-3 right-3 p-2 bg-slate-900/80 hover:bg-slate-800 rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-white/50"
                aria-label="Remove captured image and start over"
              >
                <X className="w-5 h-5 text-white" aria-hidden="true" />
              </button>
            </div>

            {/* Processing indicator */}
            {isProcessing && (
              <div className="flex items-center justify-center gap-3 py-4" role="status" aria-live="polite">
                <Loader2 className="w-6 h-6 text-medical-400 animate-spin" aria-hidden="true" />
                <p className="text-medical-400">Analyzing image...</p>
              </div>
            )}

            {/* Detected drugs */}
            {!isProcessing && detectedDrugs.length > 0 && (
              <div className="space-y-4">
                <p className="text-slate-400 text-sm" id="drug-selection-label">
                  Select 2 drugs to check for interactions ({selectedDrugs.length}/2 selected):
                </p>
                <div 
                  className="flex flex-wrap gap-2" 
                  role="listbox" 
                  aria-labelledby="drug-selection-label"
                  aria-multiselectable="true"
                >
                  {detectedDrugs.map((drug, index) => {
                    const isSelected = selectedDrugs.includes(drug)
                    return (
                      <button
                        key={index}
                        onClick={() => toggleDrugSelection(drug)}
                        onKeyDown={(e) => handleDrugKeyDown(e, drug)}
                        role="option"
                        aria-selected={isSelected}
                        className={`px-4 py-2 rounded-xl border transition-all flex items-center gap-2 focus:outline-none focus:ring-2 focus:ring-medical-500/50 ${
                          isSelected
                            ? 'bg-medical-500/20 border-medical-500/50 text-medical-400'
                            : 'bg-slate-800/50 border-slate-700/50 text-slate-300 hover:border-medical-500/30'
                        }`}
                        aria-label={`${drug}${isSelected ? ', selected' : ', not selected'}`}
                      >
                        {isSelected && (
                          <Check className="w-4 h-4" aria-hidden="true" />
                        )}
                        {drug}
                      </button>
                    )
                  })}
                </div>

                <div className="flex gap-3" role="group" aria-label="Actions">
                  <button
                    onClick={reset}
                    className="flex-1 py-3 bg-slate-800 hover:bg-slate-700 text-white rounded-xl transition-colors flex items-center justify-center gap-2 focus:outline-none focus:ring-2 focus:ring-slate-500"
                    aria-label="Discard results and try again with a new image"
                  >
                    <RefreshCw className="w-5 h-5" aria-hidden="true" />
                    Try Again
                  </button>
                  <button
                    onClick={checkSelectedDrugs}
                    disabled={selectedDrugs.length !== 2}
                    className={`flex-1 py-3 rounded-xl transition-colors flex items-center justify-center gap-2 focus:outline-none focus:ring-2 ${
                      selectedDrugs.length === 2
                        ? 'bg-medical-500 hover:bg-medical-400 text-white focus:ring-medical-300'
                        : 'bg-slate-700 text-slate-500 cursor-not-allowed'
                    }`}
                    aria-label={
                      selectedDrugs.length === 2
                        ? `Check interaction between ${selectedDrugs[0]} and ${selectedDrugs[1]}`
                        : 'Select 2 drugs to enable interaction check'
                    }
                    aria-disabled={selectedDrugs.length !== 2}
                  >
                    Check Interaction
                  </button>
                </div>
              </div>
            )}

            {/* No drugs found */}
            {!isProcessing && detectedDrugs.length === 0 && (
              <div className="text-center py-4" role="alert">
                <p className="text-slate-400 mb-4">
                  No drugs detected in the image. Please try again with a clearer photo.
                </p>
                <button
                  onClick={reset}
                  className="px-6 py-3 bg-medical-500 hover:bg-medical-400 text-white rounded-xl transition-colors flex items-center justify-center gap-2 mx-auto focus:outline-none focus:ring-2 focus:ring-medical-300"
                  aria-label="Try again with a new image"
                >
                  <RefreshCw className="w-5 h-5" aria-hidden="true" />
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
