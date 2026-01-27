import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { motion, AnimatePresence } from 'framer-motion'

import { ThemeProvider } from './context/ThemeContext'
import Navbar from './components/Navbar'
import Hero from './components/Hero'
import InteractionChecker from './components/InteractionChecker'
import CameraCapture from './components/CameraCapture'
import ResultsDisplay from './components/ResultsDisplay'
import AlternativesDisplay from './components/AlternativesDisplay'
import MLPrediction from './components/MLPrediction'
import ModelDashboard from './components/ModelDashboard'
import DiabetesManager from './components/DiabetesManager'
import PrescriptionRAG from './components/PrescriptionRAG'
import PatientPrescriptionScanner from './components/PatientPrescriptionScanner'
import Footer from './components/Footer'
import FloatingElements from './components/FloatingElements'
import { ErrorBoundary, RouteErrorBoundary } from './components/ErrorBoundary'

function App() {
  const [results, setResults] = useState(null)
  const [alternatives, setAlternatives] = useState(null)
  const [mlPrediction, setMlPrediction] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [mlLoading, setMlLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('text') // 'text' or 'camera'

  return (
    <ThemeProvider>
      <RouteErrorBoundary>
        <Router>
          <div className="min-h-screen animated-gradient grid-bg relative overflow-hidden">
            {/* Floating background elements */}
            <FloatingElements />

            {/* Toast notifications */}
            <Toaster
              position="top-right"
              toastOptions={{
                duration: 4000,
                style: {
                  background: '#172033',
                  color: '#e2e8f0',
                  border: '1px solid rgba(20, 184, 154, 0.2)',
                },
                success: {
                  iconTheme: {
                    primary: '#14b89a',
                    secondary: '#0d1321',
                  },
                },
                error: {
                  iconTheme: {
                    primary: '#ef4444',
                    secondary: '#0d1321',
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
              <Routes>
                <Route
                  path="/"
                  element={
                    <ErrorBoundary>
                      <>
                        <Hero />

                        {/* Tab switcher */}
                        <section className="max-w-4xl mx-auto px-4 py-8">
                          <div
                            className="flex justify-center gap-4 mb-8"
                            role="tablist"
                            aria-label="Input method selection"
                          >
                            <button
                              onClick={() => setActiveTab('text')}
                              role="tab"
                              aria-selected={activeTab === 'text'}
                              aria-controls="text-panel"
                              id="text-tab"
                              className={`px-6 py-3 rounded-xl font-medium transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-medical-500/50 ${activeTab === 'text'
                                ? 'bg-medical-500 text-white shadow-lg shadow-medical-500/25'
                                : 'bg-slate-800/50 text-slate-400 hover:bg-slate-800 hover:text-white'
                                }`}
                            >
                              <span className="flex items-center gap-2">
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                                </svg>
                                Type Drug Names
                              </span>
                            </button>
                            <button
                              onClick={() => setActiveTab('camera')}
                              role="tab"
                              aria-selected={activeTab === 'camera'}
                              aria-controls="camera-panel"
                              id="camera-tab"
                              className={`px-6 py-3 rounded-xl font-medium transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-medical-500/50 ${activeTab === 'camera'
                                ? 'bg-medical-500 text-white shadow-lg shadow-medical-500/25'
                                : 'bg-slate-800/50 text-slate-400 hover:bg-slate-800 hover:text-white'
                                }`}
                            >
                              <span className="flex items-center gap-2">
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                                </svg>
                                Scan with Camera
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
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -20 }}
                                transition={{ duration: 0.3 }}
                              >
                                <ErrorBoundary>
                                  <InteractionChecker
                                    setResults={setResults}
                                    setAlternatives={setAlternatives}
                                    setIsLoading={setIsLoading}
                                    setMlPrediction={setMlPrediction}
                                    setMlLoading={setMlLoading}
                                  />
                                </ErrorBoundary>
                              </motion.div>
                            ) : (
                              <motion.div
                                key="camera"
                                id="camera-panel"
                                role="tabpanel"
                                aria-labelledby="camera-tab"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -20 }}
                                transition={{ duration: 0.3 }}
                              >
                                <ErrorBoundary>
                                  <CameraCapture
                                    setResults={setResults}
                                    setAlternatives={setAlternatives}
                                    setIsLoading={setIsLoading}
                                    setMlPrediction={setMlPrediction}
                                    setMlLoading={setMlLoading}
                                  />
                                </ErrorBoundary>
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </section>

                        {/* Loading state */}
                        <AnimatePresence>
                          {isLoading && (
                            <motion.div
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              exit={{ opacity: 0 }}
                              className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center"
                              role="status"
                              aria-live="polite"
                              aria-label="Loading"
                            >
                              <div className="text-center">
                                <div className="flex flex-col items-center gap-4">
                                  {/* Animated pills */}
                                  <div className="flex gap-3" aria-hidden="true">
                                    <motion.div
                                      animate={{ y: [0, -15, 0] }}
                                      transition={{ duration: 0.6, repeat: Infinity, delay: 0 }}
                                      className="pill-loader"
                                    />
                                    <motion.div
                                      animate={{ y: [0, -15, 0] }}
                                      transition={{ duration: 0.6, repeat: Infinity, delay: 0.2 }}
                                      className="pill-loader"
                                    />
                                    <motion.div
                                      animate={{ y: [0, -15, 0] }}
                                      transition={{ duration: 0.6, repeat: Infinity, delay: 0.4 }}
                                      className="pill-loader"
                                    />
                                  </div>
                                  <p className="text-medical-400 font-medium">Analyzing drug interactions...</p>
                                  <p className="text-slate-500 text-sm">Checking 42M+ interactions database</p>
                                </div>
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>

                        {/* Results section */}
                        <AnimatePresence>
                          {results && (
                            <motion.div
                              initial={{ opacity: 0, y: 40 }}
                              animate={{ opacity: 1, y: 0 }}
                              exit={{ opacity: 0, y: 40 }}
                              transition={{ duration: 0.5 }}
                            >
                              <ErrorBoundary>
                                <ResultsDisplay results={results} />
                              </ErrorBoundary>

                              {/* ML Prediction */}
                              <section className="max-w-4xl mx-auto px-4 py-4">
                                <ErrorBoundary>
                                  <MLPrediction prediction={mlPrediction} isLoading={mlLoading} />
                                </ErrorBoundary>
                              </section>
                            </motion.div>
                          )}
                        </AnimatePresence>

                        {/* Alternatives section */}
                        <AnimatePresence>
                          {alternatives && results?.has_interaction && (
                            <motion.div
                              initial={{ opacity: 0, y: 40 }}
                              animate={{ opacity: 1, y: 0 }}
                              exit={{ opacity: 0, y: 40 }}
                              transition={{ duration: 0.5, delay: 0.2 }}
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

                {/* ML Dashboard Route */}
                <Route
                  path="/ml-dashboard"
                  element={
                    <ErrorBoundary>
                      <ModelDashboard />
                    </ErrorBoundary>
                  }
                />

                {/* Diabetic Patient DDI Route */}
                <Route
                  path="/diabetes"
                  element={
                    <ErrorBoundary>
                      <DiabetesManager />
                    </ErrorBoundary>
                  }
                />

                {/* Prescription RAG Route */}
                <Route
                  path="/prescription"
                  element={
                    <ErrorBoundary>
                      <PrescriptionRAG />
                    </ErrorBoundary>
                  }
                />

                {/* Patient Prescription Scanner (Integrated) */}
                <Route
                  path="/patient-prescription"
                  element={
                    <ErrorBoundary>
                      <PatientPrescriptionScanner />
                    </ErrorBoundary>
                  }
                />
              </Routes>
            </main>

            {/* Footer */}
            <Footer />
          </div>
        </Router>
      </RouteErrorBoundary>
    </ThemeProvider >
  )
}

export default App
