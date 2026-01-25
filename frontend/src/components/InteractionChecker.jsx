import { useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Search, X, ArrowRight, Loader2 } from 'lucide-react'
import toast from 'react-hot-toast'
import { searchDrugs, checkInteraction, getAlternatives, getMLPrediction } from '../services/api'
import { useDebouncedSearch } from '../hooks'

/**
 * DrugSearchInput - Reusable drug search input with autocomplete
 */
function DrugSearchInput({
  label,
  placeholder,
  searchState,
  onSelect,
  inputId,
}) {
  const {
    query,
    setQuery,
    results,
    isLoading,
    showResults,
    setShowResults,
    clear
  } = searchState

  return (
    <div className="relative">
      <label
        htmlFor={inputId}
        className="block text-sm font-medium text-slate-400 mb-2"
      >
        {label}
      </label>
      <div className="relative">
        <Search
          className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500"
          aria-hidden="true"
        />
        <input
          id={inputId}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => query.length >= 2 && setShowResults(true)}
          onBlur={() => setTimeout(() => setShowResults(false), 200)}
          placeholder={placeholder}
          className="w-full pl-12 pr-12 py-4 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white placeholder-slate-500 focus:border-medical-500/50 focus:outline-none focus:ring-2 focus:ring-medical-500/20 transition-colors"
          autoComplete="off"
          aria-autocomplete="list"
          aria-controls={`${inputId}-suggestions`}
          aria-expanded={showResults && results.length > 0}
        />
        {query && (
          <button
            type="button"
            onClick={clear}
            className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-500 hover:text-white transition-colors"
            aria-label={`Clear ${label.toLowerCase()}`}
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" aria-hidden="true" />
            ) : (
              <X className="w-5 h-5" aria-hidden="true" />
            )}
          </button>
        )}
      </div>

      {/* Suggestions dropdown */}
      <AnimatePresence>
        {showResults && results.length > 0 && (
          <motion.ul
            id={`${inputId}-suggestions`}
            role="listbox"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="absolute z-50 w-full mt-2 bg-slate-800 border border-slate-700 rounded-xl overflow-hidden shadow-2xl max-h-80 overflow-y-auto"
          >
            {results.map((drug, index) => (
              <li key={drug.id || index} role="option">
                <button
                  type="button"
                  onClick={() => onSelect(drug)}
                  className="w-full px-4 py-3 text-left hover:bg-medical-500/10 transition-colors border-b border-slate-700/50 last:border-b-0 focus:bg-medical-500/10 focus:outline-none"
                >
                  <p className="text-white font-medium">{drug.name}</p>
                  {drug.generic_name && (
                    <p className="text-slate-400 text-sm">{drug.generic_name}</p>
                  )}
                </button>
              </li>
            ))}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  )
}

/**
 * InteractionChecker - Main component for checking drug interactions
 */
function InteractionChecker({ setResults, setAlternatives, setIsLoading, setMlPrediction, setMlLoading }) {
  // Use the custom hook for both drug searches
  const drug1Search = useDebouncedSearch(searchDrugs, { delay: 300, minLength: 2 })
  const drug2Search = useDebouncedSearch(searchDrugs, { delay: 300, minLength: 2 })

  // Handle drug selection
  const handleSelectDrug1 = useCallback((drug) => {
    drug1Search.selectItem(drug)
  }, [drug1Search])

  const handleSelectDrug2 = useCallback((drug) => {
    drug2Search.selectItem(drug)
  }, [drug2Search])

  // Swap drugs
  const handleSwap = useCallback(() => {
    const temp = drug1Search.query
    drug1Search.setQuery(drug2Search.query)
    drug2Search.setQuery(temp)
  }, [drug1Search, drug2Search])

  // Keyboard shortcut: Ctrl+Enter to check interaction
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (drug1Search.query.trim() && drug2Search.query.trim()) {
          document.getElementById('check-interaction-btn')?.click()
        }
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [drug1Search.query, drug2Search.query])

  const handleSubmit = async (e) => {
    e.preventDefault()

    const drug1 = drug1Search.query.trim()
    const drug2 = drug2Search.query.trim()

    if (!drug1 || !drug2) {
      toast.error('Please enter both drug names')
      return
    }

    if (drug1.toLowerCase() === drug2.toLowerCase()) {
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

      // Show appropriate toast based on severity
      showInteractionToast(interactionResult)
    } catch (error) {
      console.error('Error checking interaction:', error)
      toast.error('Error checking interaction. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  // Quick example buttons
  const quickExamples = [
    { drug1: 'Aspirin', drug2: 'Warfarin', severity: 'major', color: 'text-orange-400 bg-orange-500/10 border-orange-500/30' },
    { drug1: 'Simvastatin', drug2: 'Clarithromycin', severity: 'major', color: 'text-orange-400 bg-orange-500/10 border-orange-500/30' },
    { drug1: 'Metformin', drug2: 'Lisinopril', severity: 'safe', color: 'text-medical-400 bg-medical-500/10 border-medical-500/30' },
  ]

  const handleExampleClick = useCallback((example) => {
    drug1Search.setQuery(example.drug1)
    drug2Search.setQuery(example.drug2)
  }, [drug1Search, drug2Search])

  return (
    <div className="glass rounded-3xl p-8 max-w-2xl mx-auto">
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Drug 1 Input */}
        <DrugSearchInput
          label="First Drug"
          placeholder="Enter first drug name (e.g., Aspirin)"
          searchState={drug1Search}
          onSelect={handleSelectDrug1}
          inputId="drug1-input"
        />

        {/* Swap button between inputs */}
        <div className="flex justify-center">
          <motion.button
            type="button"
            onClick={handleSwap}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            className="w-12 h-12 rounded-full bg-slate-800/50 hover:bg-medical-500/20 border border-slate-700/50 hover:border-medical-500/50 flex items-center justify-center transition-colors swap-rotate group"
            aria-label="Swap drug names"
          >
            <ArrowRight
              className="w-5 h-5 text-slate-400 group-hover:text-medical-400 rotate-90 transition-colors"
              aria-hidden="true"
            />
          </motion.button>
        </div>

        {/* Drug 2 Input */}
        <DrugSearchInput
          label="Second Drug"
          placeholder="Enter second drug name (e.g., Warfarin)"
          searchState={drug2Search}
          onSelect={handleSelectDrug2}
          inputId="drug2-input"
        />

        {/* Submit button */}
        <motion.button
          id="check-interaction-btn"
          type="submit"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          className="w-full py-4 bg-gradient-to-r from-medical-500 to-medical-600 hover:from-medical-400 hover:to-medical-500 text-white font-semibold rounded-xl shadow-lg shadow-medical-500/25 transition-all btn-hover relative group"
        >
          <span>Check Interaction</span>
          <span className="absolute right-4 top-1/2 -translate-y-1/2 text-xs text-white/50 hidden md:inline group-hover:text-white/70 transition-colors">
            Ctrl + Enter
          </span>
        </motion.button>
      </form>

      {/* Quick examples */}
      <div className="mt-6 pt-6 border-t border-slate-700/50">
        <p className="text-sm text-slate-500 mb-3">Quick examples:</p>
        <div className="flex flex-wrap gap-2" role="group" aria-label="Quick example drug pairs">
          {quickExamples.map((example, index) => (
            <motion.button
              key={index}
              onClick={() => handleExampleClick(example)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="px-3 py-2 text-xs bg-slate-800/50 hover:bg-slate-700/50 text-slate-300 rounded-xl transition-colors flex items-center gap-2 border border-slate-700/50 hover:border-slate-600/50"
              aria-label={`Try ${example.drug1} and ${example.drug2} - ${example.severity} interaction`}
            >
              <span>{example.drug1} + {example.drug2}</span>
              <span className={`px-1.5 py-0.5 rounded-full text-[10px] font-medium uppercase border ${example.color}`}>
                {example.severity}
              </span>
            </motion.button>
          ))}
        </div>
      </div>
    </div>
  )
}

/**
 * Show toast notification based on interaction severity
 */
function showInteractionToast(result) {
  if (!result.has_interaction) {
    toast.success('No known interaction found!')
    return
  }

  const severity = result.interaction?.severity
  switch (severity) {
    case 'minor':
      toast('Minor interaction detected')
      break
    case 'moderate':
      toast.error('Moderate interaction detected')
      break
    case 'major':
      toast.error('Major interaction detected!')
      break
    case 'contraindicated':
      toast.error('CONTRAINDICATED - Do not use together!')
      break
    default:
      toast('Interaction detected')
  }
}

export default InteractionChecker
