/**
 * Zustand store for drug interaction checking state.
 *
 * Centralises results, alternatives, ML predictions, loading flags,
 * and the active input tab that were previously prop-drilled via App.jsx.
 */
import { create } from 'zustand'
import {
  checkInteraction,
  getAlternatives,
  getMLPrediction,
  searchDrugs,
} from '../services/api'

const useDrugStore = create((set, get) => ({
  // ── State ──────────────────────────────────────────────────
  results: null,
  alternatives: null,
  mlPrediction: null,
  isLoading: false,
  mlLoading: false,
  activeTab: 'text', // 'text' | 'camera'

  // Drug search
  searchQuery: '',
  searchResults: [],
  searchLoading: false,

  // ── Actions ────────────────────────────────────────────────

  setActiveTab: (tab) => set({ activeTab: tab }),

  /** Clear all results (e.g. when user starts a new check). */
  clearResults: () =>
    set({ results: null, alternatives: null, mlPrediction: null }),

  /**
   * Full interaction check flow:
   *  1. Call rules + DB interaction endpoint
   *  2. In parallel, request ML prediction
   *  3. If interaction found, fetch alternatives
   */
  checkInteraction: async (drug1, drug2) => {
    set({ isLoading: true, results: null, alternatives: null, mlPrediction: null })

    try {
      // Step 1 — rules / DB check
      const result = await checkInteraction(drug1, drug2)
      set({ results: result })

      // Step 2 — ML prediction (fire-and-forget style)
      set({ mlLoading: true })
      getMLPrediction(drug1, drug2)
        .then((ml) => set({ mlPrediction: ml }))
        .catch(() => {})
        .finally(() => set({ mlLoading: false }))

      // Step 3 — alternatives if interaction detected
      if (result?.has_interaction) {
        getAlternatives(drug1, drug2)
          .then((alt) => set({ alternatives: alt }))
          .catch(() => {})
      }
    } catch (err) {
      console.error('Interaction check failed', err)
    } finally {
      set({ isLoading: false })
    }
  },

  /** Debounced drug name search. */
  searchDrugs: async (query) => {
    if (!query || query.length < 2) {
      set({ searchResults: [], searchQuery: query })
      return
    }
    set({ searchLoading: true, searchQuery: query })
    try {
      const drugs = await searchDrugs(query, 10)
      set({ searchResults: drugs })
    } catch {
      set({ searchResults: [] })
    } finally {
      set({ searchLoading: false })
    }
  },

  // ── Direct setters (for camera capture & other callers) ───
  setResults: (results) => set({ results }),
  setAlternatives: (alternatives) => set({ alternatives }),
  setMlPrediction: (mlPrediction) => set({ mlPrediction }),
  setIsLoading: (isLoading) => set({ isLoading }),
  setMlLoading: (mlLoading) => set({ mlLoading }),
}))

export default useDrugStore
