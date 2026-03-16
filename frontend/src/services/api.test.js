/**
 * Unit tests for the API service module.
 * 
 * Tests cover:
 * - API request functions
 * - Error handling
 * - Response parsing
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

// Mock the platform utility
vi.mock('../utils/platform', () => ({
  getApiBaseUrl: () => 'http://localhost:8001',
}))

// Import after mock
import {
  searchDrugs,
  getDrugById,
  getDrugByName,
  checkInteraction,
  getDrugInteractions,
  getAlternatives,
  extractFromImage,
  getStats,
  healthCheck,
  getSystemStatus,
  getMLPrediction,
  getMLModelInfo,
  uploadPrescription,
  setAdminApiKey,
} from './api'

describe('API Service', () => {
  // Save original fetch
  const originalFetch = global.fetch

  beforeEach(() => {
    // Reset fetch mock before each test
    global.fetch = vi.fn()
  })

  afterEach(() => {
    // Restore original fetch
    global.fetch = originalFetch
    localStorage.clear()
    vi.resetAllMocks()
  })

  describe('getSystemStatus', () => {
    it('should send admin API key for protected system status endpoint', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ status: 'healthy' }),
      })

      setAdminApiKey('secret-key')
      const result = await getSystemStatus()

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8001/admin/system-status',
        expect.objectContaining({
          headers: expect.objectContaining({
            'X-API-Key': 'secret-key',
          }),
        })
      )
      expect(result.status).toBe('healthy')
    })
  })

  describe('searchDrugs', () => {
    it('should search for drugs successfully', async () => {
      const mockDrugs = [
        { id: 1, name: 'Aspirin', generic_name: 'Acetylsalicylic Acid' },
        { id: 2, name: 'Aspirin Complex', generic_name: null },
      ]

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockDrugs),
      })

      const result = await searchDrugs('aspirin')

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8001/drugs/search?query=aspirin&limit=10',
        expect.objectContaining({
          headers: { 'Content-Type': 'application/json' },
        })
      )
      expect(result).toEqual(mockDrugs)
    })

    it('should encode query parameters', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve([]),
      })

      await searchDrugs('aspirin & warfarin', 5)

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('query=aspirin%20%26%20warfarin'),
        expect.any(Object)
      )
    })

    it('should respect limit parameter', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve([]),
      })

      await searchDrugs('test', 20)

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('limit=20'),
        expect.any(Object)
      )
    })

    it('should throw error on failed request', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.resolve({ detail: 'Server error' }),
      })

      await expect(searchDrugs('test')).rejects.toThrow('Server error')
    })
  })

  describe('getDrugById', () => {
    it('should get drug by ID successfully', async () => {
      const mockDrug = { id: 1, name: 'Aspirin' }

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockDrug),
      })

      const result = await getDrugById(1)

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8001/drugs/1',
        expect.any(Object)
      )
      expect(result).toEqual(mockDrug)
    })

    it('should throw error for non-existent drug', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: () => Promise.resolve({ detail: 'Drug not found' }),
      })

      await expect(getDrugById(99999)).rejects.toThrow('Drug not found')
    })
  })

  describe('getDrugByName', () => {
    it('should get drug by name successfully', async () => {
      const mockDrug = { id: 1, name: 'Aspirin' }

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockDrug),
      })

      const result = await getDrugByName('Aspirin')

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8001/drugs/name/Aspirin',
        expect.any(Object)
      )
      expect(result).toEqual(mockDrug)
    })
  })

  describe('checkInteraction', () => {
    it('should check interaction successfully', async () => {
      const mockResult = {
        has_interaction: true,
        is_safe: false,
        interaction: {
          severity: 'major',
          effect: 'Increased bleeding risk',
        },
      }

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResult),
      })

      const result = await checkInteraction('Aspirin', 'Warfarin')

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8001/interactions/check',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            drug1_name: 'Aspirin',
            drug2_name: 'Warfarin',
          }),
        })
      )
      expect(result).toEqual(mockResult)
    })

    it('should handle no interaction found', async () => {
      const mockResult = {
        has_interaction: false,
        is_safe: true,
        interaction: null,
      }

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResult),
      })

      const result = await checkInteraction('Metformin', 'Lisinopril')

      expect(result.has_interaction).toBe(false)
      expect(result.is_safe).toBe(true)
    })
  })

  describe('getDrugInteractions', () => {
    it('should get all interactions for a drug', async () => {
      const mockResult = {
        drug: 'Aspirin',
        total_interactions: 2,
        interactions: [
          { other_drug: 'Warfarin', severity: 'major' },
          { other_drug: 'Ibuprofen', severity: 'moderate' },
        ],
      }

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResult),
      })

      const result = await getDrugInteractions('Aspirin')

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8001/interactions/drug/Aspirin',
        expect.any(Object)
      )
      expect(result.total_interactions).toBe(2)
    })

    it('should filter by severity', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ interactions: [] }),
      })

      await getDrugInteractions('Aspirin', 'major')

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8001/interactions/drug/Aspirin?severity=major',
        expect.any(Object)
      )
    })
  })

  describe('getAlternatives', () => {
    it('should get alternatives successfully', async () => {
      const mockResult = {
        alternatives_for_drug1: [{ name: 'Ibuprofen' }],
        alternatives_for_drug2: [{ name: 'Heparin' }],
      }

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResult),
      })

      const result = await getAlternatives('Aspirin', 'Warfarin')

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8001/alternatives',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            drug1_name: 'Aspirin',
            drug2_name: 'Warfarin',
          }),
        })
      )
      expect(result).toEqual(mockResult)
    })
  })

  describe('extractFromImage', () => {
    it('should extract drugs from image', async () => {
      const mockResult = {
        detected_drugs: ['Aspirin', 'Metformin'],
        confidence: 0.95,
        extracted_text: 'Aspirin 100mg...',
      }

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResult),
      })

      const result = await extractFromImage('base64imagedata')

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8001/ocr/extract',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ image_base64: 'base64imagedata' }),
        })
      )
      expect(result.detected_drugs).toHaveLength(2)
    })
  })

  describe('getStats', () => {
    it('should get database statistics', async () => {
      const mockStats = {
        total_drugs: 5000,
        total_interactions: 100000,
        interactions_by_severity: {
          minor: 30000,
          moderate: 40000,
          major: 25000,
          contraindicated: 5000,
        },
      }

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockStats),
      })

      const result = await getStats()

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8001/stats',
        expect.any(Object)
      )
      expect(result.total_drugs).toBe(5000)
    })
  })

  describe('healthCheck', () => {
    it('should return healthy status', async () => {
      const mockHealth = {
        status: 'healthy',
        timestamp: '2025-01-01T00:00:00Z',
      }

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockHealth),
      })

      const result = await healthCheck()

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8001/health',
        expect.any(Object)
      )
      expect(result.status).toBe('healthy')
    })

    it('should handle server unavailable', async () => {
      global.fetch.mockRejectedValueOnce(new Error('Network error'))

      await expect(healthCheck()).rejects.toThrow('Network error')
    })
  })

  describe('getMLPrediction', () => {
    it('should get ML prediction successfully', async () => {
      const mockPrediction = {
        drug1: 'Aspirin',
        drug2: 'Warfarin',
        interaction_probability: 0.85,
        predicted_interaction: true,
        severity_prediction: 'major',
        confidence: 0.92,
        model_predictions: {
          random_forest: 0.87,
          xgboost: 0.83,
        },
      }

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockPrediction),
      })

      const result = await getMLPrediction('Aspirin', 'Warfarin')

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8001/ml/predict',
        expect.objectContaining({
          method: 'POST',
        })
      )
      expect(result.interaction_probability).toBe(0.85)
    })
  })

  describe('getMLModelInfo', () => {
    it('should get model information', async () => {
      const mockInfo = {
        status: 'loaded',
        models: ['random_forest', 'xgboost', 'lightgbm'],
        optimal_threshold: 0.5,
      }

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockInfo),
      })

      const result = await getMLModelInfo()

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8001/ml/model-info',
        expect.any(Object)
      )
      expect(result.status).toBe('loaded')
    })
  })

  describe('Error Handling', () => {
    it('should handle JSON parse error gracefully', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.reject(new Error('Invalid JSON')),
      })

      await expect(searchDrugs('test')).rejects.toThrow()
    })

    it('should include HTTP status in error message', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 503,
        json: () => Promise.resolve({}),
      })

      await expect(searchDrugs('test')).rejects.toThrow(/503/)
    })

    it('should handle network failures', async () => {
      global.fetch.mockRejectedValueOnce(new Error('Network failure'))

      await expect(checkInteraction('Drug1', 'Drug2')).rejects.toThrow('Network failure')
    })
  })
})

describe('API Service - File Upload', () => {
  const originalFetch = global.fetch

  beforeEach(() => {
    global.fetch = vi.fn()
  })

  afterEach(() => {
    global.fetch = originalFetch
    vi.resetAllMocks()
  })

  describe('uploadPrescription', () => {
    it('should upload prescription file', async () => {
      const mockResult = {
        prescription_id: 1,
        medicines: ['Aspirin', 'Metformin'],
        raw_text: 'Prescription text...',
      }

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResult),
      })

      const mockFile = new File(['test'], 'prescription.jpg', { type: 'image/jpeg' })
      const result = await uploadPrescription(mockFile)

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8001/prescription/upload',
        expect.objectContaining({
          method: 'POST',
        })
      )
      expect(result.prescription_id).toBe(1)
    })

    it('should handle upload failure', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({ detail: 'Invalid file format' }),
      })

      const mockFile = new File(['test'], 'test.txt', { type: 'text/plain' })

      await expect(uploadPrescription(mockFile)).rejects.toThrow('Invalid file format')
    })
  })
})
