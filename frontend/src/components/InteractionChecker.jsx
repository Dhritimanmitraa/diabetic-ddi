import { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Search, X, ArrowRight, Loader2 } from 'lucide-react'
import toast from 'react-hot-toast'
import { searchDrugs, checkInteraction, getAlternatives, getMLPrediction } from '../services/api'
import debounce from '../utils/debounce'

function InteractionChecker({ setResults, setAlternatives, setIsLoading, setMlPrediction, setMlLoading }) {
  const [drug1, setDrug1] = useState('')
  const [drug2, setDrug2] = useState('')
  const [suggestions1, setSuggestions1] = useState([])
  const [suggestions2, setSuggestions2] = useState([])
  const [showSuggestions1, setShowSuggestions1] = useState(false)
  const [showSuggestions2, setShowSuggestions2] = useState(false)
  const [searching1, setSearching1] = useState(false)
  const [searching2, setSearching2] = useState(false)

  // Debounced search function
  const debouncedSearch1 = useCallback(
    debounce(async (query) => {
      if (query.length < 2) {
        setSuggestions1([])
        return
      }
      setSearching1(true)
      try {
        const results = await searchDrugs(query)
        setSuggestions1(results)
        setShowSuggestions1(true)
      } catch (error) {
        console.error('Search error:', error)
      } finally {
        setSearching1(false)
      }
    }, 300),
    []
  )

  const debouncedSearch2 = useCallback(
    debounce(async (query) => {
      if (query.length < 2) {
        setSuggestions2([])
        return
      }
      setSearching2(true)
      try {
        const results = await searchDrugs(query)
        setSuggestions2(results)
        setShowSuggestions2(true)
      } catch (error) {
        console.error('Search error:', error)
      } finally {
        setSearching2(false)
      }
    }, 300),
    []
  )

  useEffect(() => {
    debouncedSearch1(drug1)
  }, [drug1, debouncedSearch1])

  useEffect(() => {
    debouncedSearch2(drug2)
  }, [drug2, debouncedSearch2])

  const handleSelectDrug1 = (drug) => {
    setDrug1(drug.name)
    setShowSuggestions1(false)
  }

  const handleSelectDrug2 = (drug) => {
    setDrug2(drug.name)
    setShowSuggestions2(false)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!drug1.trim() || !drug2.trim()) {
      toast.error('Please enter both drug names')
      return
    }

    if (drug1.trim().toLowerCase() === drug2.trim().toLowerCase()) {
      toast.error('Please enter two different drugs')
      return
    }

    setIsLoading(true)
    setResults(null)
    setAlternatives(null)
    if (setMlPrediction) setMlPrediction(null)

    try {
      // Check interaction
      const interactionResult = await checkInteraction(drug1, drug2)
      setResults(interactionResult)

      // Fetch ML prediction in parallel (if available)
      if (setMlPrediction && setMlLoading) {
        setMlLoading(true)
        getMLPrediction(drug1, drug2)
          .then(mlResult => {
            if (!mlResult.error) {
              setMlPrediction(mlResult)
            }
          })
          .catch(err => {
            console.log('ML prediction not available:', err.message)
          })
          .finally(() => {
            setMlLoading(false)
          })
      }

      // If there's an interaction, fetch alternatives
      if (interactionResult.has_interaction && interactionResult.interaction?.severity !== 'minor') {
        try {
          const alternativesResult = await getAlternatives(drug1, drug2)
          setAlternatives(alternativesResult)
        } catch (altError) {
          console.error('Could not fetch alternatives:', altError)
        }
      }

      // Show appropriate toast
      if (!interactionResult.has_interaction) {
        toast.success('No known interaction found!')
      } else if (interactionResult.interaction?.severity === 'minor') {
        toast('Minor interaction detected')
      } else if (interactionResult.interaction?.severity === 'moderate') {
        toast('Moderate interaction detected')
      } else if (interactionResult.interaction?.severity === 'major') {
        toast.error('Major interaction detected!')
      } else if (interactionResult.interaction?.severity === 'contraindicated') {
        toast.error('CONTRAINDICATED - Do not use together!')
      }
    } catch (error) {
      console.error('Error checking interaction:', error)
      toast.error('Error checking interaction. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  const clearDrug1 = () => {
    setDrug1('')
    setSuggestions1([])
    setShowSuggestions1(false)
  }

  const clearDrug2 = () => {
    setDrug2('')
    setSuggestions2([])
    setShowSuggestions2(false)
  }

  return (
    <div className="glass rounded-3xl p-8 max-w-2xl mx-auto">
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Drug 1 Input */}
        <div className="relative">
          <label className="block text-sm font-medium text-slate-400 mb-2">
            First Drug
          </label>
          <div className="relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
            <input
              type="text"
              value={drug1}
              onChange={(e) => setDrug1(e.target.value)}
              onFocus={() => drug1.length >= 2 && setShowSuggestions1(true)}
              onBlur={() => setTimeout(() => setShowSuggestions1(false), 200)}
              placeholder="Enter first drug name (e.g., Aspirin)"
              className="w-full pl-12 pr-12 py-4 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white placeholder-slate-500 focus:border-medical-500/50 transition-colors"
            />
            {drug1 && (
              <button
                type="button"
                onClick={clearDrug1}
                className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-500 hover:text-white transition-colors"
              >
                {searching1 ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <X className="w-5 h-5" />
                )}
              </button>
            )}
          </div>
          
          {/* Suggestions dropdown */}
          <AnimatePresence>
            {showSuggestions1 && suggestions1.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="absolute z-50 w-full mt-2 bg-slate-800 border border-slate-700 rounded-xl overflow-hidden shadow-2xl max-h-80 overflow-y-auto"
              >
                {suggestions1.map((drug, index) => (
                  <button
                    key={drug.id || index}
                    type="button"
                    onClick={() => handleSelectDrug1(drug)}
                    className="w-full px-4 py-3 text-left hover:bg-medical-500/10 transition-colors border-b border-slate-700/50 last:border-b-0"
                  >
                    <p className="text-white font-medium">{drug.name}</p>
                    {drug.generic_name && (
                      <p className="text-slate-400 text-sm">{drug.generic_name}</p>
                    )}
                  </button>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Arrow between inputs */}
        <div className="flex justify-center">
          <div className="w-10 h-10 rounded-full bg-slate-800/50 flex items-center justify-center">
            <ArrowRight className="w-5 h-5 text-medical-400 rotate-90" />
          </div>
        </div>

        {/* Drug 2 Input */}
        <div className="relative">
          <label className="block text-sm font-medium text-slate-400 mb-2">
            Second Drug
          </label>
          <div className="relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
            <input
              type="text"
              value={drug2}
              onChange={(e) => setDrug2(e.target.value)}
              onFocus={() => drug2.length >= 2 && setShowSuggestions2(true)}
              onBlur={() => setTimeout(() => setShowSuggestions2(false), 200)}
              placeholder="Enter second drug name (e.g., Warfarin)"
              className="w-full pl-12 pr-12 py-4 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white placeholder-slate-500 focus:border-medical-500/50 transition-colors"
            />
            {drug2 && (
              <button
                type="button"
                onClick={clearDrug2}
                className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-500 hover:text-white transition-colors"
              >
                {searching2 ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <X className="w-5 h-5" />
                )}
              </button>
            )}
          </div>
          
          {/* Suggestions dropdown */}
          <AnimatePresence>
            {showSuggestions2 && suggestions2.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="absolute z-50 w-full mt-2 bg-slate-800 border border-slate-700 rounded-xl overflow-hidden shadow-2xl max-h-80 overflow-y-auto"
              >
                {suggestions2.map((drug, index) => (
                  <button
                    key={drug.id || index}
                    type="button"
                    onClick={() => handleSelectDrug2(drug)}
                    className="w-full px-4 py-3 text-left hover:bg-medical-500/10 transition-colors border-b border-slate-700/50 last:border-b-0"
                  >
                    <p className="text-white font-medium">{drug.name}</p>
                    {drug.generic_name && (
                      <p className="text-slate-400 text-sm">{drug.generic_name}</p>
                    )}
                  </button>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Submit button */}
        <motion.button
          type="submit"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          className="w-full py-4 bg-gradient-to-r from-medical-500 to-medical-600 hover:from-medical-400 hover:to-medical-500 text-white font-semibold rounded-xl shadow-lg shadow-medical-500/25 transition-all btn-hover"
        >
          Check Interaction
        </motion.button>
      </form>

      {/* Quick examples */}
      <div className="mt-6 pt-6 border-t border-slate-700/50">
        <p className="text-sm text-slate-500 mb-3">Quick examples:</p>
        <div className="flex flex-wrap gap-2">
          {[
            { drug1: 'Aspirin', drug2: 'Warfarin' },
            { drug1: 'Simvastatin', drug2: 'Clarithromycin' },
            { drug1: 'Metformin', drug2: 'Lisinopril' },
          ].map((example, index) => (
            <button
              key={index}
              onClick={() => {
                setDrug1(example.drug1)
                setDrug2(example.drug2)
              }}
              className="px-3 py-1.5 text-xs bg-slate-800/50 hover:bg-slate-700/50 text-slate-400 hover:text-white rounded-full transition-colors"
            >
              {example.drug1} + {example.drug2}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

export default InteractionChecker

