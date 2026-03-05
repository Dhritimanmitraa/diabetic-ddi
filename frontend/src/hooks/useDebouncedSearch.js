import { useState, useEffect, useRef, useCallback } from 'react'

/**
 * Custom hook for debounced search with loading state
 * 
 * @param {Function} searchFn - Async function that performs the search
 * @param {Object} options - Configuration options
 * @param {number} options.delay - Debounce delay in milliseconds (default: 300)
 * @param {number} options.minLength - Minimum query length to trigger search (default: 2)
 * @returns {Object} Hook state and controls
 * 
 * @example
 * const { 
 *   query, setQuery, 
 *   results, 
 *   isLoading, 
 *   showResults, setShowResults,
 *   clear 
 * } = useDebouncedSearch(searchDrugs, { delay: 300, minLength: 2 })
 */
export function useDebouncedSearch(searchFn, options = {}) {
  const { delay = 300, minLength = 2 } = options
  
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [showResults, setShowResults] = useState(false)
  const [error, setError] = useState(null)
  
  const timeoutRef = useRef(null)
  const abortControllerRef = useRef(null)
  const searchFnRef = useRef(searchFn)
  searchFnRef.current = searchFn

  // Perform the search
  useEffect(() => {
    if (query.length < minLength) {
      setResults([])
      setError(null)
      return
    }

    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
    }

    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }

    timeoutRef.current = setTimeout(async () => {
      setIsLoading(true)
      setError(null)
      
      const controller = new AbortController()
      abortControllerRef.current = controller
      
      try {
        const searchResults = await searchFnRef.current(query, 10, { signal: controller.signal })
        if (!controller.signal.aborted) {
          setResults(searchResults)
          setShowResults(true)
        }
      } catch (err) {
        if (err.name !== 'AbortError') {
          console.error('Search error:', err)
          setError(err)
          setResults([])
        }
      } finally {
        if (!controller.signal.aborted) {
          setIsLoading(false)
        }
      }
    }, delay)

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
    }
  }, [query, delay, minLength])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
    }
  }, [])

  // Clear function to reset all state
  const clear = useCallback(() => {
    setQuery('')
    setResults([])
    setShowResults(false)
    setError(null)
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
    }
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
  }, [])

  // Select item and close dropdown
  const selectItem = useCallback((item) => {
    setQuery(item.name || item)
    setShowResults(false)
  }, [])

  return {
    query,
    setQuery,
    results,
    isLoading,
    showResults,
    setShowResults,
    error,
    clear,
    selectItem,
  }
}

export default useDebouncedSearch
