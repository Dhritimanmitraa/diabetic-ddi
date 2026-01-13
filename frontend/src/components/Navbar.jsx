import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Link, useLocation } from 'react-router-dom'
import { Pill, Activity, Shield, Brain, Heart, Menu, X, FileText } from 'lucide-react'
import { healthCheck } from '../services/api'

function Navbar() {
  const [isHealthy, setIsHealthy] = useState(null)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const location = useLocation()

  // Check health on mount and periodically
  useEffect(() => {
    const checkHealth = async () => {
      try {
        await healthCheck()
        setIsHealthy(true)
      } catch {
        setIsHealthy(false)
      }
    }
    checkHealth()
    const interval = setInterval(checkHealth, 30000) // Check every 30s
    return () => clearInterval(interval)
  }, [])

  const isActive = (path) => location.pathname === path

  return (
    <motion.nav
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
      className="fixed top-0 left-0 right-0 z-50 glass"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <motion.a
            href="/"
            className="flex items-center gap-3 group"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <div className="relative">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-medical-400 to-medical-600 flex items-center justify-center shadow-lg shadow-medical-500/25 group-hover:shadow-medical-500/40 transition-shadow">
                <Pill className="w-5 h-5 text-white" />
              </div>
              <div className="absolute -bottom-1 -right-1 w-4 h-4 rounded-full bg-medical-400 flex items-center justify-center">
                <Shield className="w-2.5 h-2.5 text-white" />
              </div>
            </div>
            <div>
              <h1 className="font-display font-bold text-xl text-white">
                Drug<span className="gradient-text">Guard</span>
              </h1>
              <p className="text-[10px] text-slate-500 font-medium tracking-wide uppercase">
                Drug Interaction Checker
              </p>
            </div>
          </motion.a>

          {/* Desktop Navigation links */}
          <div className="hidden md:flex items-center gap-6">
            <NavLink href="/#how-it-works">How it Works</NavLink>
            <NavLink href="/#features">Features</NavLink>
            <Link
              to="/prescription"
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition-colors text-sm font-medium ${isActive('/prescription')
                  ? 'bg-cyan-500/20 border-cyan-500/40 text-cyan-300'
                  : 'bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 hover:bg-cyan-500/20'
                }`}
            >
              <FileText className="w-4 h-4" />
              Prescription
            </Link>
            <Link
              to="/diabetes"
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition-colors text-sm font-medium ${isActive('/diabetes')
                  ? 'bg-rose-500/20 border-rose-500/40 text-rose-300'
                  : 'bg-rose-500/10 border border-rose-500/20 text-rose-400 hover:bg-rose-500/20'
                }`}
            >
              <Heart className="w-4 h-4" />
              Diabetes DDI
            </Link>
            <Link
              to="/ml-dashboard"
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition-colors text-sm font-medium ${isActive('/ml-dashboard')
                  ? 'bg-purple-500/20 border-purple-500/40 text-purple-300'
                  : 'bg-purple-500/10 border border-purple-500/20 text-purple-400 hover:bg-purple-500/20'
                }`}
            >
              <Brain className="w-4 h-4" />
              ML Dashboard
            </Link>
          </div>

          {/* Status indicator & Mobile menu button */}
          <div className="flex items-center gap-3">
            {/* Health Status */}
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-800/50 border border-slate-700/50">
              <div className={`relative flex h-2 w-2`}>
                {isHealthy && (
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-medical-400 opacity-75"></span>
                )}
                <span className={`relative inline-flex rounded-full h-2 w-2 ${isHealthy === null ? 'bg-slate-500' : isHealthy ? 'bg-medical-500' : 'bg-red-500'
                  }`}></span>
              </div>
              <span className={`text-xs font-medium ${isHealthy === null ? 'text-slate-400' : isHealthy ? 'text-medical-400' : 'text-red-400'
                }`}>
                {isHealthy === null ? 'Checking...' : isHealthy ? 'Online' : 'Offline'}
              </span>
            </div>

            {/* Mobile menu button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-2 rounded-lg bg-slate-800/50 hover:bg-slate-700/50 text-slate-400 hover:text-white transition-colors"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Mobile menu */}
        <AnimatePresence>
          {mobileMenuOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="md:hidden border-t border-slate-700/50 py-4 space-y-2"
            >
              <a href="/#how-it-works" onClick={() => setMobileMenuOpen(false)} className="block px-4 py-2 text-slate-400 hover:text-white hover:bg-slate-800/50 rounded-lg transition-colors">
                How it Works
              </a>
              <a href="/#features" onClick={() => setMobileMenuOpen(false)} className="block px-4 py-2 text-slate-400 hover:text-white hover:bg-slate-800/50 rounded-lg transition-colors">
                Features
              </a>
              <Link to="/prescription" onClick={() => setMobileMenuOpen(false)} className="flex items-center gap-2 px-4 py-2 text-cyan-400 hover:bg-cyan-500/10 rounded-lg transition-colors">
                <FileText className="w-4 h-4" />
                Prescription
              </Link>
              <Link to="/diabetes" onClick={() => setMobileMenuOpen(false)} className="flex items-center gap-2 px-4 py-2 text-rose-400 hover:bg-rose-500/10 rounded-lg transition-colors">
                <Heart className="w-4 h-4" />
                Diabetes DDI
              </Link>
              <Link to="/ml-dashboard" onClick={() => setMobileMenuOpen(false)} className="flex items-center gap-2 px-4 py-2 text-purple-400 hover:bg-purple-500/10 rounded-lg transition-colors">
                <Brain className="w-4 h-4" />
                ML Dashboard
              </Link>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.nav>
  )
}

function NavLink({ href, children }) {
  return (
    <motion.a
      href={href}
      className="text-slate-400 hover:text-white text-sm font-medium transition-colors relative group"
      whileHover={{ y: -2 }}
    >
      {children}
      <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-medical-400 group-hover:w-full transition-all duration-300" />
    </motion.a>
  )
}

export default Navbar
