import { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Link, useLocation } from 'react-router-dom'
import { Shield, Brain, Heart, Menu, X, FileText } from 'lucide-react'
import { healthCheck } from '../services/api'
import ThemeToggle from './ThemeToggle'

function Navbar() {
  const [isHealthy, setIsHealthy] = useState(null)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [scrolled, setScrolled] = useState(false)
  const location = useLocation()

  // Scroll detection for navbar backdrop
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  // Health check
  useEffect(() => {
    const check = async () => {
      try {
        await healthCheck()
        setIsHealthy(true)
      } catch {
        setIsHealthy(false)
      }
    }
    check()
    const interval = setInterval(check, 30000)
    return () => clearInterval(interval)
  }, [])

  // Close mobile menu on route change
  useEffect(() => {
    setMobileMenuOpen(false)
  }, [location.pathname])

  // Close on Escape
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape' && mobileMenuOpen) setMobileMenuOpen(false)
    }
    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [mobileMenuOpen])

  const isActive = (path) => location.pathname === path

  const toggleMobileMenu = useCallback(() => {
    setMobileMenuOpen(prev => !prev)
  }, [])

  const healthDot = isHealthy === null
    ? 'bg-slate-500'
    : isHealthy
      ? 'bg-emerald-500'
      : 'bg-red-500'

  const healthLabel = isHealthy === null
    ? 'Checking...'
    : isHealthy
      ? 'Online'
      : 'Offline'

  return (
    <motion.nav
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.4, ease: 'easeOut' }}
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${scrolled
          ? 'glass shadow-lg shadow-black/5'
          : 'bg-transparent'
        }`}
      role="navigation"
      aria-label="Main navigation"
    >
      <div className="max-w-6xl mx-auto px-5 sm:px-6">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link
            to="/"
            className="flex items-center gap-2.5 group focus:outline-none focus:ring-2 focus:ring-medical-500/50 rounded-lg"
            aria-label="DrugGuard — Go to home page"
          >
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-medical-500 to-medical-600 flex items-center justify-center shadow-md shadow-medical-500/15">
              <Shield className="w-4.5 h-4.5 text-white" />
            </div>
            <span className="font-display font-bold text-lg text-[var(--text-primary)] tracking-tight">
              Drug<span className="gradient-text">Guard</span>
            </span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-1" role="menubar">
            <NavLink href="/#how-it-works">How it Works</NavLink>
            <NavLink href="/#features">Features</NavLink>
            <NavItem
              to="/prescription"
              active={isActive('/prescription')}
              icon={<FileText className="w-3.5 h-3.5" />}
            >
              Prescription
            </NavItem>
            <NavItem
              to="/diabetes"
              active={isActive('/diabetes')}
              icon={<Heart className="w-3.5 h-3.5" />}
            >
              Diabetes DDI
            </NavItem>
            <NavItem
              to="/ml-dashboard"
              active={isActive('/ml-dashboard')}
              icon={<Brain className="w-3.5 h-3.5" />}
            >
              ML Dashboard
            </NavItem>
          </div>

          {/* Right side */}
          <div className="flex items-center gap-2.5">
            <ThemeToggle />

            {/* Health indicator */}
            <div
              className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-[var(--bg-elevated)]/60 border border-[var(--border)]"
              role="status"
              aria-live="polite"
              aria-label={`API status: ${healthLabel}`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${healthDot}`} />
              <span className="text-xs font-medium text-[var(--text-secondary)]">
                {healthLabel}
              </span>
            </div>

            {/* Mobile menu toggle */}
            <button
              onClick={toggleMobileMenu}
              className="md:hidden p-2 rounded-lg text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-elevated)] transition-colors"
              aria-expanded={mobileMenuOpen}
              aria-controls="mobile-menu"
              aria-label={mobileMenuOpen ? 'Close menu' : 'Open menu'}
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
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
              className="md:hidden border-t border-[var(--border)] py-3 space-y-1"
              role="menu"
            >
              <MobileLink href="/#how-it-works" onClick={() => setMobileMenuOpen(false)}>
                How it Works
              </MobileLink>
              <MobileLink href="/#features" onClick={() => setMobileMenuOpen(false)}>
                Features
              </MobileLink>
              <MobileNavItem to="/prescription" onClick={() => setMobileMenuOpen(false)} icon={<FileText className="w-4 h-4" />}>
                Prescription
              </MobileNavItem>
              <MobileNavItem to="/diabetes" onClick={() => setMobileMenuOpen(false)} icon={<Heart className="w-4 h-4" />}>
                Diabetes DDI
              </MobileNavItem>
              <MobileNavItem to="/ml-dashboard" onClick={() => setMobileMenuOpen(false)} icon={<Brain className="w-4 h-4" />}>
                ML Dashboard
              </MobileNavItem>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.nav>
  )
}

function NavLink({ href, children }) {
  return (
    <a
      href={href}
      className="px-3 py-2 text-sm font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors rounded-lg hover:bg-[var(--bg-elevated)]/50"
      role="menuitem"
    >
      {children}
    </a>
  )
}

function NavItem({ to, active, icon, children }) {
  return (
    <Link
      to={to}
      className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${active
          ? 'bg-medical-500/10 text-medical-400'
          : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-elevated)]/50'
        }`}
      role="menuitem"
      aria-current={active ? 'page' : undefined}
    >
      {icon}
      {children}
    </Link>
  )
}

function MobileLink({ href, onClick, children }) {
  return (
    <a
      href={href}
      onClick={onClick}
      className="block px-4 py-2.5 text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-elevated)] rounded-lg transition-colors"
      role="menuitem"
    >
      {children}
    </a>
  )
}

function MobileNavItem({ to, onClick, icon, children }) {
  return (
    <Link
      to={to}
      onClick={onClick}
      className="flex items-center gap-2.5 px-4 py-2.5 text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-elevated)] rounded-lg transition-colors"
      role="menuitem"
    >
      {icon}
      {children}
    </Link>
  )
}

export default Navbar
