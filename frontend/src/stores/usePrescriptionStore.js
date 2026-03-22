import { create } from 'zustand'
import {
  chatWithPrescription,
  checkPrescriptionInteractions,
  deletePrescription,
  ensureAuthenticatedSession,
  getPrescriptionChatHistory,
  getPrescriptionHistory,
  uploadPrescription,
} from '../services/api'

const usePrescriptionStore = create((set, get) => ({
  activeSection: 'upload',
  isUploading: false,
  prescription: null,
  history: [],
  chatMessages: [],
  isChatLoading: false,
  drugWarnings: null,
  isCheckingWarnings: false,

  setActiveSection: (activeSection) => set({ activeSection }),
  resetCurrentPrescription: () =>
    set({
      prescription: null,
      chatMessages: [],
      drugWarnings: null,
      activeSection: 'upload',
    }),

  loadHistory: async () => {
    await ensureAuthenticatedSession()
    const data = await getPrescriptionHistory(10, 0)
    set({ history: data.prescriptions || [] })
    return data
  },

  checkDrugWarnings: async (medicines) => {
    if (!medicines || medicines.length < 2) {
      set({ drugWarnings: null })
      return null
    }

    set({ isCheckingWarnings: true })
    try {
      const result = await checkPrescriptionInteractions(medicines.map((medicine) => medicine.name))
      set({ drugWarnings: result })
      return result
    } finally {
      set({ isCheckingWarnings: false })
    }
  },

  uploadFile: async (file) => {
    set({
      isUploading: true,
      prescription: null,
      chatMessages: [],
      drugWarnings: null,
    })

    try {
      const result = await uploadPrescription(file)
      const nextState = {
        prescription: result.status === 'completed' ? result : null,
        activeSection: result.status === 'completed' ? 'result' : 'upload',
      }
      set(nextState)

      await get().loadHistory()
      if (result.id) {
        try {
          const chatHistory = await getPrescriptionChatHistory(result.id)
          set({ chatMessages: chatHistory.messages || [] })
        } catch (error) {
          console.warn('Failed to load prescription chat history', error)
          set({ chatMessages: [] })
        }
      }
      if (result.medicines?.length >= 2) {
        await get().checkDrugWarnings(result.medicines)
      }
      return result
    } finally {
      set({ isUploading: false })
    }
  },

  sendChatMessage: async (message) => {
    const prescriptionId = get().prescription?.id
    if (!message?.trim() || !prescriptionId) {
      return null
    }

    const userMessage = message.trim()
    set((state) => ({
      chatMessages: [...state.chatMessages, { role: 'user', content: userMessage }],
      isChatLoading: true,
    }))

    try {
      const response = await chatWithPrescription(prescriptionId, userMessage)
      set((state) => ({
        chatMessages: [
          ...state.chatMessages,
          {
            role: 'assistant',
            content: response.assistant_message,
            model_used: response.model_used,
          },
        ],
      }))
      return response
    } catch (error) {
      set((state) => ({
        chatMessages: [
          ...state.chatMessages,
          {
            role: 'assistant',
            content: 'Sorry, I encountered an error. Please try again.',
            error: true,
          },
        ],
      }))
      throw error
    } finally {
      set({ isChatLoading: false })
    }
  },

  selectPrescriptionFromHistory: async (item) => {
    set({
      prescription: item,
      activeSection: 'result',
      chatMessages: [],
    })

    try {
      const chatHistory = await getPrescriptionChatHistory(item.id)
      set({ chatMessages: chatHistory.messages || [] })
    } catch (error) {
      console.warn('Failed to load chat history from selected prescription', error)
      set({ chatMessages: [] })
    }

    if (item.medicines?.length >= 2) {
      await get().checkDrugWarnings(item.medicines)
    } else {
      set({ drugWarnings: null })
    }
  },

  removePrescription: async (id) => {
    await deletePrescription(id)
    await get().loadHistory()
    if (get().prescription?.id === id) {
      set({
        prescription: null,
        chatMessages: [],
        drugWarnings: null,
        activeSection: 'upload',
      })
    }
  },
}))

export default usePrescriptionStore
