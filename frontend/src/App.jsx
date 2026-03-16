import { lazy, Suspense } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { motion, AnimatePresence } from 'framer-motion'

import { ThemeProvider } from './context/ThemeContext'
import useDrugStore from './stores/useDrugStore'
import Navbar from './components/Navbar'
import Hero from './components/Hero'
import InteractionChecker from './components/InteractionChecker'
import CameraCapture from './components/CameraCapture'
import ResultsDisplay from './components/ResultsDisplay'
import AlternativesDisplay from './components/AlternativesDisplay'
import MLPrediction from './components/MLPrediction'
import Footer from './components/Footer'
import FloatingElements from './components/FloatingElements'
import { ErrorBoundary, RouteErrorBoundary } from './components/ErrorBoundary'

const ModelDashboard = lazy(() => import('./components/ModelDashboard'))
const DiabetesManager = lazy(() => import('./components/DiabetesManager'))
const PrescriptionRAG = lazy(() => import('./components/PrescriptionRAG'))
const PatientPrescriptionScanner = lazy(() => import('./components/PatientPrescriptionScanner'))
const SystemStatus = lazy(() => import('./components/SystemStatus'))

function App() {
  const results = useDrugStore((s) => s.results)
  const alternatives = useDrugStore((s) => s.alternatives)
  const mlPrediction = useDrugStore((s) => s.mlPrediction)
  const isLoading = useDrugStore((s) => s.isLoading)
  const mlLoading = useDrugStore((s) => s.mlLoading)
  const activeTab = useDrugStore((s) => s.activeTab)
  const setActiveTab = useDrugStore((s) => s.setActiveTab)

  return (
    <ThemeProvider>
      <RouteErrorBoundary>
        <Router>
          <div className="min-h-screen relative overflow-hidden" style={{ backgroundColor: 'var(--bg-primary)' }}>
            {/* Ambient background */}
            <FloatingElements />

            {/* Toast notifications */}
            <Toaster
              position="top-right"
              toastOptions={{
                duration: 4000,
                style: {
                  background: 'var(--bg-elevated)',
                  color: 'var(--text-primary)',
                  border: '1px solid var(--border)',
                  borderRadius: '12px',
                  fontSize: '14px',
                  boxShadow: '0 8px 30px rgba(0,0,0,0.12)',
                },
                success: {
                  iconTheme: {
                    primary: '#14b8a6',
                    secondary: 'var(--bg-elevated)',
                  },
                },
                error: {
                  iconTheme: {
                    primary: '#ef4444',
                    secondary: 'var(--bg-elevated)',
                  },
                },
              }}
            />

            {/* Navigation */}
            <ErrorBoundary>
              <Navbar />
            </ErrorBoundary>

            {/* Main content */}
            <main className="relative z-10">
              <Suspense fallback={
                <div className="flex items-center justify-center min-h-[60vh]">
                  <div className="text-center">
                    <div className="spinner" />
                    <p className="text-[var(--text-muted)] text-sm mt-4">Loading...</p>
                  </div>
                </div>
              }>
                <Routes>
                  <Route
                    path="/"
                    element={
                      <ErrorBoundary>
                        <>
                          <Hero />

                          {/* Input method tabs */}
                          <section className="max-w-3xl mx-auto px-5 py-6">
                            <div
                              className="flex justify-center gap-1 mb-6 p-1 bg-[var(--bg-elevated)] rounded-lg w-fit mx-auto border border-[var(--border)]"
                              role="tablist"
                              aria-label="Input method"
                            >
                              <button
                                onClick={() => setActiveTab('text')}
                                role="tab"
                                aria-selected={activeTab === 'text'}
                                aria-controls="text-panel"
                                id="text-tab"
                                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-150 ${activeTab === 'text'
                                  ? 'bg-[var(--bg-primary)] text-[var(--text-primary)] shadow-sm border border-[var(--border)]'
                                  : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
                                  }`}
                              >
                                <span className="flex items-center gap-2">
                                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                                  </svg>
                                  Type Names
                                </span>
                              </button>
                              <button
                                onClick={() => setActiveTab('camera')}
                                role="tab"
                                aria-selected={activeTab === 'camera'}
                                aria-controls="camera-panel"
                                id="camera-tab"
                                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-150 ${activeTab === 'camera'
                                  ? 'bg-[var(--bg-primary)] text-[var(--text-primary)] shadow-sm border border-[var(--border)]'
                                  : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
                                  }`}
                              >
                                <span className="flex items-center gap-2">
                                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                                  </svg>
                                  Scan Label
                                </span>
                              </button>
                            </div>

                            {/* Input section */}
                            <AnimatePresence mode="wait">
                              {activeTab === 'text' ? (
                                <motion.div
                                  key="text"
                                  id="text-panel"
                                  role="tabpanel"
                                  aria-labelledby="text-tab"
                                  initial={{ opacity: 0, y: 12 }}
                                  animate={{ opacity: 1, y: 0 }}
                                  exit={{ opacity: 0, y: -12 }}
                                  transition={{ duration: 0.2 }}
                                >
                                  <ErrorBoundary>
                                    <InteractionChecker />
                                  </ErrorBoundary>
                                </motion.div>
                              ) : (
                                <motion.div
                                  key="camera"
                                  id="camera-panel"
                                  role="tabpanel"
                                  aria-labelledby="camera-tab"
                                  initial={{ opacity: 0, y: 12 }}
                                  animate={{ opacity: 1, y: 0 }}
                                  exit={{ opacity: 0, y: -12 }}
                                  transition={{ duration: 0.2 }}
                                >
                                  <ErrorBoundary>
                                    <CameraCapture />
                                  </ErrorBoundary>
                                </motion.div>
                              )}
                            </AnimatePresence>
                          </section>

                          {/* Loading overlay */}
                          <AnimatePresence>
                            {isLoading && (
                              <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                className="fixed inset-0 z-50 flex items-center justify-center"
                                style={{ backgroundColor: 'var(--loading-overlay-bg, rgba(10, 15, 28, 0.8))' }}
                                role="status"
                                aria-live="polite"
                                aria-label="Loading"
                              >
                                <div className="text-center">
                                  <div className="flex flex-col items-center gap-4">
                                    <div className="spinner" />
                                    <p className="text-[var(--text-secondary)] text-sm">Checking interactions…</p>
                                  </div>
                                </div>
                              </motion.div>
                            )}
                          </AnimatePresence>

                          {/* Results */}
                          <AnimatePresence>
                            {results && (
                              <motion.div
                                initial={{ opacity: 0, y: 24 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: 24 }}
                                transition={{ duration: 0.35 }}
                              >
                                <ErrorBoundary>
                                  <ResultsDisplay results={results} />
                                </ErrorBoundary>

                                <section className="max-w-3xl mx-auto px-5 py-4">
                                  <ErrorBoundary>
                                    <MLPrediction prediction={mlPrediction} isLoading={mlLoading} />
                                  </ErrorBoundary>
                                </section>
                              </motion.div>
                            )}
                          </AnimatePresence>

                          {/* Alternatives */}
                          <AnimatePresence>
                            {alternatives && results?.has_interaction && (
                              <motion.div
                                initial={{ opacity: 0, y: 24 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: 24 }}
                                transition={{ duration: 0.35, delay: 0.1 }}
                              >
                                <ErrorBoundary>
                                  <AlternativesDisplay alternatives={alternatives} />
                                </ErrorBoundary>
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </>
                      </ErrorBoundary>
                    }
                  />

                  <Route path="/ml-dashboard" element={<ErrorBoundary><ModelDashboard /></ErrorBoundary>} />
                  <Route path="/diabetes" element={<ErrorBoundary><DiabetesManager /></ErrorBoundary>} />
                  <Route path="/prescription" element={<ErrorBoundary><PrescriptionRAG /></ErrorBoundary>} />
                  <Route path="/patient-prescription" element={<ErrorBoundary><PatientPrescriptionScanner /></ErrorBoundary>} />
                  <Route path="/system-status" element={<ErrorBoundary><SystemStatus /></ErrorBoundary>} />
                  <Route path="*" element={
                    <div className="flex flex-col items-center justify-center min-h-[60vh] text-center px-5">
                      <h1 className="text-6xl font-bold text-medical-400 mb-4">404</h1>
                      <p className="text-[var(--text-secondary)] text-lg mb-6">Page not found</p>
                      <a href="/" className="px-5 py-2.5 bg-medical-500 hover:bg-medical-400 text-white rounded-xl text-sm font-medium transition-colors">
                        Back to Home
                      </a>
                    </div>
                  } />
                </Routes>
              </Suspense>
            </main>

            <Footer />
          </div>
        </Router>
      </RouteErrorBoundary>
    </ThemeProvider>
  )
}

export default App
