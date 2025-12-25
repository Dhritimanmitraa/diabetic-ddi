import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { motion, AnimatePresence } from 'framer-motion'

import Navbar from './components/Navbar'
import Hero from './components/Hero'
import InteractionChecker from './components/InteractionChecker'
import CameraCapture from './components/CameraCapture'
import ResultsDisplay from './components/ResultsDisplay'
import AlternativesDisplay from './components/AlternativesDisplay'
import MLPrediction from './components/MLPrediction'
import ModelDashboard from './components/ModelDashboard'
import DiabetesManager from './components/DiabetesManager'
import Footer from './components/Footer'
import FloatingElements from './components/FloatingElements'

function App() {
  const [results, setResults] = useState(null)
  const [alternatives, setAlternatives] = useState(null)
  const [mlPrediction, setMlPrediction] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [mlLoading, setMlLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('text') // 'text' or 'camera'

  return (
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
        <Navbar />

        {/* Main content */}
        <main className="relative z-10">
          <Routes>
            <Route
              path="/"
              element={
                <>
                  <Hero />
                  
                  {/* Tab switcher */}
                  <section className="max-w-4xl mx-auto px-4 py-8">
                    <div className="flex justify-center gap-4 mb-8">
                      <button
                        onClick={() => setActiveTab('text')}
                        className={`px-6 py-3 rounded-xl font-medium transition-all duration-300 ${
                          activeTab === 'text'
                            ? 'bg-medical-500 text-white shadow-lg shadow-medical-500/25'
                            : 'bg-slate-800/50 text-slate-400 hover:bg-slate-800 hover:text-white'
                        }`}
                      >
                        <span className="flex items-center gap-2">
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                          </svg>
                          Type Drug Names
                        </span>
                      </button>
                      <button
                        onClick={() => setActiveTab('camera')}
                        className={`px-6 py-3 rounded-xl font-medium transition-all duration-300 ${
                          activeTab === 'camera'
                            ? 'bg-medical-500 text-white shadow-lg shadow-medical-500/25'
                            : 'bg-slate-800/50 text-slate-400 hover:bg-slate-800 hover:text-white'
                        }`}
                      >
                        <span className="flex items-center gap-2">
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -20 }}
                          transition={{ duration: 0.3 }}
                        >
                          <InteractionChecker
                            setResults={setResults}
                            setAlternatives={setAlternatives}
                            setIsLoading={setIsLoading}
                            setMlPrediction={setMlPrediction}
                            setMlLoading={setMlLoading}
                          />
                        </motion.div>
                      ) : (
                        <motion.div
                          key="camera"
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -20 }}
                          transition={{ duration: 0.3 }}
                        >
                          <CameraCapture
                            setResults={setResults}
                            setAlternatives={setAlternatives}
                            setIsLoading={setIsLoading}
                            setMlPrediction={setMlPrediction}
                            setMlLoading={setMlLoading}
                          />
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
                      >
                        <div className="text-center">
                          <div className="spinner mx-auto mb-4"></div>
                          <p className="text-medical-400 font-medium">Analyzing drug interactions...</p>
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
                        <ResultsDisplay results={results} />
                        
                        {/* ML Prediction */}
                        <section className="max-w-4xl mx-auto px-4 py-4">
                          <MLPrediction prediction={mlPrediction} isLoading={mlLoading} />
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
                        <AlternativesDisplay alternatives={alternatives} />
                      </motion.div>
                    )}
                  </AnimatePresence>
                </>
              }
            />
            
            {/* ML Dashboard Route */}
            <Route
              path="/ml-dashboard"
              element={<ModelDashboard />}
            />
            
            {/* Diabetic Patient DDI Route */}
            <Route
              path="/diabetes"
              element={<DiabetesManager />}
            />
          </Routes>
        </main>

        {/* Footer */}
        <Footer />
      </div>
    </Router>
  )
}

export default App

