import { useEffect, useCallback, useState, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Search, X, ArrowDown, Loader2, AlertCircle } from 'lucide-react'
import toast from 'react-hot-toast'
import { searchDrugs } from '../services/api'
import { useDebouncedSearch } from '../hooks'
import useDrugStore from '../stores/useDrugStore'

/**
 * DrugSearchInput — Search with autocomplete, keyboard nav, and error display
 */
function DrugSearchInput({ label, placeholder, searchState, onSelect, inputId }) {
  const { query, setQuery, results, isLoading, showResults, setShowResults, error, clear } = searchState
  const [highlightIndex, setHighlightIndex] = useState(-1)
  const containerRef = useRef(null)
  const isOpen = showResults && results.length > 0

  useEffect(() => { setHighlightIndex(-1) }, [results])

  const handleKeyDown = (e) => {
    if (!isOpen) return
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault()
        setHighlightIndex(i => (i + 1) % results.length)
        break
      case 'ArrowUp':
        e.preventDefault()
        setHighlightIndex(i => (i <= 0 ? results.length - 1 : i - 1))
        break
      case 'Enter':
        if (highlightIndex >= 0 && highlightIndex < results.length) {
          e.preventDefault()
          onSelect(results[highlightIndex])
          setHighlightIndex(-1)
        }
        break
      case 'Escape':
        setShowResults(false)
        setHighlightIndex(-1)
        break
    }
  }

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setShowResults(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [setShowResults])

  const activeDescendant = highlightIndex >= 0 ? `${inputId}-option-${highlightIndex}` : undefined

  return (
    <div className="relative" ref={containerRef}>
      <label htmlFor={inputId} className="block text-xs font-semibold text-[var(--text-muted)] mb-2 uppercase tracking-wider">
        {label}
      </label>
      <div className="relative">
        <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--text-muted)]" aria-hidden="true" />
        <input
          id={inputId}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => query.length >= 2 && setShowResults(true)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className="w-full pl-11 pr-11 py-3.5 bg-[var(--bg-elevated)] border border-[var(--border)] rounded-xl text-[var(--text-primary)] placeholder-[var(--text-muted)] text-sm transition-all duration-200 focus:border-medical-500/40"
          autoComplete="off"
          aria-autocomplete="list"
          aria-controls={`${inputId}-suggestions`}
          aria-expanded={isOpen}
          aria-activedescendant={activeDescendant}
        />
        {query && (
          <button
            type="button"
            onClick={clear}
            className="absolute right-3.5 top-1/2 -translate-y-1/2 text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors p-0.5"
            aria-label={`Clear ${label.toLowerCase()}`}
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" />
            ) : (
              <X className="w-4 h-4" aria-hidden="true" />
            )}
          </button>
        )}
      </div>

      {error && (
        <div className="flex items-center gap-1.5 mt-1.5 text-red-400 text-xs" role="alert">
          <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
          <span>Search failed. Check your connection and try again.</span>
        </div>
      )}

      {/* Suggestions dropdown */}
      <AnimatePresence>
        {isOpen && (
          <motion.ul
            id={`${inputId}-suggestions`}
            role="listbox"
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.15 }}
            className="absolute z-50 w-full mt-1.5 bg-[var(--bg-elevated)] border border-[var(--border)] rounded-xl overflow-hidden shadow-xl max-h-64 overflow-y-auto"
          >
            {results.map((drug, index) => (
              <li
                key={drug.id || index}
                id={`${inputId}-option-${index}`}
                role="option"
                aria-selected={highlightIndex === index}
              >
                <button
                  type="button"
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => { onSelect(drug); setHighlightIndex(-1) }}
                  className={`w-full px-4 py-3 text-left transition-colors border-b border-[var(--border)] last:border-b-0 ${highlightIndex === index ? 'bg-medical-500/12' : 'hover:bg-medical-500/8'
                    }`}
                >
                  <p className="text-medical-300 font-semibold text-sm">{drug.name}</p>
                  {drug.generic_name && (
                    <p className="text-[var(--text-muted)] text-xs mt-0.5">{drug.generic_name}</p>
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
 * InteractionChecker — Main drug pair input form
 */
function InteractionChecker() {
  const runInteractionCheck = useDrugStore((state) => state.checkInteraction)
  const drug1Search = useDebouncedSearch(searchDrugs, { delay: 300, minLength: 2 })
  const drug2Search = useDebouncedSearch(searchDrugs, { delay: 300, minLength: 2 })

  const handleSelectDrug1 = useCallback((drug) => {
    drug1Search.selectItem(drug)
  }, [drug1Search.selectItem])

  const handleSelectDrug2 = useCallback((drug) => {
    drug2Search.selectItem(drug)
  }, [drug2Search.selectItem])

  const handleSwap = useCallback(() => {
    const temp1Query = drug1Search.query
    const temp2Query = drug2Search.query
    // Use selectItem to mark as selected, preventing re-search
    drug1Search.selectItem({ name: temp2Query })
    drug2Search.selectItem({ name: temp1Query })
  }, [drug1Search.query, drug2Search.query, drug1Search.selectItem, drug2Search.selectItem])

  // Ctrl+Enter shortcut
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

  // Ctrl+K or "/" shortcut to focus search
  useEffect(() => {
    const handleFocus = (e) => {
      const isTyping = ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault()
        document.getElementById('drug1-input')?.focus()
      } else if (e.key === '/' && !isTyping) {
        e.preventDefault()
        document.getElementById('drug1-input')?.focus()
      }
    }
    window.addEventListener('keydown', handleFocus)
    return () => window.removeEventListener('keydown', handleFocus)
  }, [])

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

    try {
      const interactionResult = await runInteractionCheck(drug1, drug2)
      showInteractionToast(interactionResult)
    } catch (error) {
      console.error('Error checking interaction:', error)
      toast.error('Error checking interaction. Please try again.')
    }
  }

  const quickExamples = [
    { drug1: 'Aspirin', drug2: 'Warfarin', severity: 'major' },
    { drug1: 'Simvastatin', drug2: 'Clarithromycin', severity: 'major' },
    { drug1: 'Metformin', drug2: 'Lisinopril', severity: 'safe' },
  ]

  const handleExampleClick = useCallback((example) => {
    drug1Search.setQuery(example.drug1)
    drug2Search.setQuery(example.drug2)
  }, [drug1Search.setQuery, drug2Search.setQuery])

  const severityColors = {
    major: 'text-orange-400 bg-orange-500/8 border-orange-500/20',
    safe: 'text-emerald-400 bg-emerald-500/8 border-emerald-500/20',
  }

  return (
    <div className="bg-[var(--bg-elevated)] rounded-2xl p-6 sm:p-8 max-w-2xl mx-auto border border-[var(--border)]">
      <form onSubmit={handleSubmit} className="space-y-5">
        <DrugSearchInput
          label="First Drug"
          placeholder="e.g., Aspirin"
          searchState={drug1Search}
          onSelect={handleSelectDrug1}
          inputId="drug1-input"
        />

        {/* Swap button */}
        <div className="flex justify-center">
          <button
            type="button"
            onClick={handleSwap}
            className="w-9 h-9 rounded-full bg-[var(--bg-primary)] border border-[var(--border)] hover:border-medical-500/30 flex items-center justify-center transition-all swap-rotate group"
            aria-label="Swap drug names"
          >
            <ArrowDown className="w-3.5 h-3.5 text-[var(--text-muted)] group-hover:text-medical-400 transition-colors" aria-hidden="true" />
          </button>
        </div>

        <DrugSearchInput
          label="Second Drug"
          placeholder="e.g., Warfarin"
          searchState={drug2Search}
          onSelect={handleSelectDrug2}
          inputId="drug2-input"
        />

        {/* Submit */}
        <button
          id="check-interaction-btn"
          type="submit"
          className="w-full py-3.5 bg-medical-600 hover:bg-medical-500 text-white font-semibold text-sm rounded-xl transition-colors relative group"
        >
          <span>Check Interaction</span>
          <span className="absolute right-4 top-1/2 -translate-y-1/2 text-[10px] text-white/40 hidden md:inline group-hover:text-white/60 transition-colors font-medium">
            Ctrl+Enter
          </span>
        </button>
      </form>

      {/* Quick examples */}
      <div className="mt-5 pt-5 border-t border-[var(--border)]">
        <p className="text-[10px] text-[var(--text-muted)] mb-2.5 uppercase tracking-wider font-medium">Examples</p>
        <div className="flex flex-wrap gap-2" role="group" aria-label="Example drug pairs">
          {quickExamples.map((example, index) => (
            <button
              key={index}
              onClick={() => handleExampleClick(example)}
              className="px-3 py-1.5 text-xs bg-[var(--bg-primary)] hover:bg-[var(--bg-primary)]/80 text-[var(--text-secondary)] rounded-lg transition-colors flex items-center gap-2 border border-[var(--border)] hover:border-[var(--border-hover)]"
              aria-label={`Try ${example.drug1} and ${example.drug2}`}
            >
              <span>{example.drug1} + {example.drug2}</span>
              <span className={`px-1.5 py-0.5 rounded text-[9px] font-semibold uppercase border ${severityColors[example.severity]}`}>
                {example.severity}
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

function showInteractionToast(result) {
  if (!result.has_interaction) {
    toast.success('No known interaction found!')
    return
  }
  const severity = result.interaction?.severity
  switch (severity) {
    case 'minor': toast('Minor interaction detected'); break
    case 'moderate': toast.error('Moderate interaction detected'); break
    case 'major': toast.error('Major interaction detected!'); break
    case 'contraindicated': toast.error('CONTRAINDICATED — Do not use together!'); break
    default: toast('Interaction detected')
  }
}

export default InteractionChecker
