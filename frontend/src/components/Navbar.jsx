import { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Link, useLocation } from 'react-router-dom'
import { Pill, Shield, Brain, Heart, Menu, X, FileText } from 'lucide-react'
import { healthCheck } from '../services/api'
import ThemeToggle from './ThemeToggle'

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

  // Close mobile menu on route change
  useEffect(() => {
    setMobileMenuOpen(false)
  }, [location.pathname])

  // Close mobile menu on Escape key
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape' && mobileMenuOpen) {
        setMobileMenuOpen(false)
      }
    }
    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [mobileMenuOpen])

  const isActive = (path) => location.pathname === path

  const toggleMobileMenu = useCallback(() => {
    setMobileMenuOpen(prev => !prev)
  }, [])

  const getHealthStatus = () => {
    if (isHealthy === null) return { text: 'Checking...', color: 'text-slate-400', bgColor: 'bg-slate-500' }
    if (isHealthy) return { text: 'Online', color: 'text-medical-400', bgColor: 'bg-medical-500' }
    return { text: 'Offline', color: 'text-red-400', bgColor: 'bg-red-500' }
  }

  const healthStatus = getHealthStatus()

  return (
    <motion.nav
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
      className="fixed top-0 left-0 right-0 z-50 glass"
      role="navigation"
      aria-label="Main navigation"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <motion.a
            href="/"
            className="flex items-center gap-3 group focus:outline-none focus:ring-2 focus:ring-medical-500/50 rounded-lg"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            aria-label="DrugGuard - Go to home page"
          >
            <div className="relative">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-medical-400 to-medical-600 flex items-center justify-center shadow-lg shadow-medical-500/25 group-hover:shadow-medical-500/40 transition-shadow" aria-hidden="true">
                <Pill className="w-5 h-5 text-white" />
              </div>
              <div className="absolute -bottom-1 -right-1 w-4 h-4 rounded-full bg-medical-400 flex items-center justify-center" aria-hidden="true">
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
          <div className="hidden md:flex items-center gap-6" role="menubar">
            <NavLink href="/#how-it-works">How it Works</NavLink>
            <NavLink href="/#features">Features</NavLink>
            <NavMenuItem
              to="/prescription"
              isActive={isActive('/prescription')}
              icon={<FileText className="w-4 h-4" aria-hidden="true" />}
              activeColor="cyan"
              label="Prescription"
            />
            <NavMenuItem
              to="/diabetes"
              isActive={isActive('/diabetes')}
              icon={<Heart className="w-4 h-4" aria-hidden="true" />}
              activeColor="rose"
              label="Diabetes DDI"
            />
            <NavMenuItem
              to="/ml-dashboard"
              isActive={isActive('/ml-dashboard')}
              icon={<Brain className="w-4 h-4" aria-hidden="true" />}
              activeColor="purple"
              label="ML Dashboard"
            />
          </div>

          {/* Status indicator, Theme toggle & Mobile menu button */}
          <div className="flex items-center gap-3">
            {/* Theme Toggle */}
            <ThemeToggle />

            {/* Health Status */}
            <div
              className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-800/50 border border-slate-700/50"
              role="status"
              aria-live="polite"
              aria-label={`API status: ${healthStatus.text}`}
            >
              <div className="relative flex h-2 w-2">
                {isHealthy && (
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-medical-400 opacity-75" aria-hidden="true"></span>
                )}
                <span className={`relative inline-flex rounded-full h-2 w-2 ${healthStatus.bgColor}`} aria-hidden="true"></span>
              </div>
              <span className={`text-xs font-medium ${healthStatus.color}`}>
                {healthStatus.text}
              </span>
            </div>

            {/* Mobile menu button */}
            <button
              onClick={toggleMobileMenu}
              className="md:hidden p-2 rounded-lg bg-slate-800/50 hover:bg-slate-700/50 text-slate-400 hover:text-white transition-colors focus:outline-none focus:ring-2 focus:ring-medical-500/50"
              aria-expanded={mobileMenuOpen}
              aria-controls="mobile-menu"
              aria-label={mobileMenuOpen ? 'Close navigation menu' : 'Open navigation menu'}
            >
              {mobileMenuOpen ? (
                <X className="w-5 h-5" aria-hidden="true" />
              ) : (
                <Menu className="w-5 h-5" aria-hidden="true" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile menu */}
        <AnimatePresence>
          {mobileMenuOpen && (
            <motion.div
              id="mobile-menu"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="md:hidden border-t border-slate-700/50 py-4 space-y-2"
              role="menu"
              aria-label="Mobile navigation"
            >
              <MobileNavLink
                href="/#how-it-works"
                onClick={() => setMobileMenuOpen(false)}
              >
                How it Works
              </MobileNavLink>
              <MobileNavLink
                href="/#features"
                onClick={() => setMobileMenuOpen(false)}
              >
                Features
              </MobileNavLink>
              <MobileNavMenuItem
                to="/prescription"
                onClick={() => setMobileMenuOpen(false)}
                icon={<FileText className="w-4 h-4" aria-hidden="true" />}
                color="cyan"
              >
                Prescription
              </MobileNavMenuItem>
              <MobileNavMenuItem
                to="/diabetes"
                onClick={() => setMobileMenuOpen(false)}
                icon={<Heart className="w-4 h-4" aria-hidden="true" />}
                color="rose"
              >
                Diabetes DDI
              </MobileNavMenuItem>
              <MobileNavMenuItem
                to="/ml-dashboard"
                onClick={() => setMobileMenuOpen(false)}
                icon={<Brain className="w-4 h-4" aria-hidden="true" />}
                color="purple"
              >
                ML Dashboard
              </MobileNavMenuItem>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.nav>
  )
}

/**
 * Desktop navigation link component
 */
function NavLink({ href, children }) {
  return (
    <motion.a
      href={href}
      className="text-slate-400 hover:text-white text-sm font-medium transition-colors relative group focus:outline-none focus:text-white"
      whileHover={{ y: -2 }}
      role="menuitem"
    >
      {children}
      <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-medical-400 group-hover:w-full transition-all duration-300" aria-hidden="true" />
    </motion.a>
  )
}

/**
 * Desktop navigation menu item with icon and active state
 */
function NavMenuItem({ to, isActive, icon, activeColor, label }) {
  const activeClasses = {
    cyan: 'bg-cyan-500/20 border-cyan-500/40 text-cyan-300',
    rose: 'bg-rose-500/20 border-rose-500/40 text-rose-300',
    purple: 'bg-purple-500/20 border-purple-500/40 text-purple-300',
  }

  const inactiveClasses = {
    cyan: 'bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 hover:bg-cyan-500/20',
    rose: 'bg-rose-500/10 border border-rose-500/20 text-rose-400 hover:bg-rose-500/20',
    purple: 'bg-purple-500/10 border border-purple-500/20 text-purple-400 hover:bg-purple-500/20',
  }

  return (
    <Link
      to={to}
      className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition-colors text-sm font-medium focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 focus:ring-${activeColor}-500/50 ${isActive ? activeClasses[activeColor] : inactiveClasses[activeColor]
        }`}
      role="menuitem"
      aria-current={isActive ? 'page' : undefined}
    >
      {icon}
      {label}
    </Link>
  )
}

/**
 * Mobile navigation link component
 */
function MobileNavLink({ href, onClick, children }) {
  return (
    <a
      href={href}
      onClick={onClick}
      className="block px-4 py-2 text-slate-400 hover:text-white hover:bg-slate-800/50 rounded-lg transition-colors focus:outline-none focus:bg-slate-800/50 focus:text-white"
      role="menuitem"
    >
      {children}
    </a>
  )
}

/**
 * Mobile navigation menu item with icon
 */
function MobileNavMenuItem({ to, onClick, icon, color, children }) {
  const colorClasses = {
    cyan: 'text-cyan-400 hover:bg-cyan-500/10 focus:bg-cyan-500/10',
    rose: 'text-rose-400 hover:bg-rose-500/10 focus:bg-rose-500/10',
    purple: 'text-purple-400 hover:bg-purple-500/10 focus:bg-purple-500/10',
  }

  return (
    <Link
      to={to}
      onClick={onClick}
      className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors focus:outline-none ${colorClasses[color]}`}
      role="menuitem"
    >
      {icon}
      {children}
    </Link>
  )
}

export default Navbar
