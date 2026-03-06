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

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

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

  useEffect(() => {
    setMobileMenuOpen(false)
  }, [location.pathname])

  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape' && mobileMenuOpen) setMobileMenuOpen(false)
    }
    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [mobileMenuOpen])

  const isActive = (path) => location.pathname === path
  const toggleMobileMenu = useCallback(() => setMobileMenuOpen(prev => !prev), [])

  const healthColor = isHealthy === null ? 'bg-slate-500' : isHealthy ? 'bg-emerald-500' : 'bg-red-500'
  const healthLabel = isHealthy === null ? 'Checking' : isHealthy ? 'Online' : 'Offline'

  return (
    <motion.nav
      initial={{ y: -16, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.35, ease: 'easeOut' }}
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${scrolled ? 'bg-[var(--bg-primary)]/90 backdrop-blur-xl border-b border-[var(--border)]' : 'bg-transparent'}`}
      role="navigation"
      aria-label="Main navigation"
    >
      <div className="max-w-6xl mx-auto px-5 sm:px-6">
        <div className="flex items-center justify-between h-16">

          {/* Logo */}
          <Link
            to="/"
            className="flex items-center gap-2.5 focus:outline-none focus-visible:ring-2 focus-visible:ring-medical-500/50 rounded-lg"
            aria-label="DrugGuard — home"
          >
            <div className="w-8 h-8 rounded-lg bg-medical-500 flex items-center justify-center">
              <Shield className="w-4 h-4 text-white" />
            </div>
            <span className="font-display font-bold text-base text-[var(--text-primary)] tracking-tight">
              Drug<span className="text-medical-400">Guard</span>
            </span>
          </Link>

          {/* Desktop nav */}
          <div className="hidden md:flex items-center gap-0.5" role="menubar">
            <NavAnchor href="/#how-it-works">How it Works</NavAnchor>
            <NavAnchor href="/#features">Features</NavAnchor>
            <NavItem to="/prescription" active={isActive('/prescription')} icon={<FileText className="w-3.5 h-3.5" />}>Prescription</NavItem>
            <NavItem to="/diabetes" active={isActive('/diabetes')} icon={<Heart className="w-3.5 h-3.5" />}>Diabetes DDI</NavItem>
            <NavItem to="/ml-dashboard" active={isActive('/ml-dashboard')} icon={<Brain className="w-3.5 h-3.5" />}>ML Dashboard</NavItem>
          </div>

          {/* Right side */}
          <div className="flex items-center gap-2">
            <ThemeToggle />

            <div
              className="hidden sm:flex items-center gap-1.5 px-2.5 py-1.5 rounded-full border border-[var(--border)] text-xs font-medium text-[var(--text-muted)]"
              role="status"
              aria-live="polite"
              aria-label={`API: ${healthLabel}`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${healthColor}`} />
              {healthLabel}
            </div>

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
              transition={{ duration: 0.2 }}
              className="md:hidden border-t border-[var(--border)] py-3 space-y-0.5"
              role="menu"
            >
              <MobileAnchor href="/#how-it-works" onClick={() => setMobileMenuOpen(false)}>How it Works</MobileAnchor>
              <MobileAnchor href="/#features" onClick={() => setMobileMenuOpen(false)}>Features</MobileAnchor>
              <MobileNavItem to="/prescription" onClick={() => setMobileMenuOpen(false)} icon={<FileText className="w-4 h-4" />}>Prescription</MobileNavItem>
              <MobileNavItem to="/diabetes" onClick={() => setMobileMenuOpen(false)} icon={<Heart className="w-4 h-4" />}>Diabetes DDI</MobileNavItem>
              <MobileNavItem to="/ml-dashboard" onClick={() => setMobileMenuOpen(false)} icon={<Brain className="w-4 h-4" />}>ML Dashboard</MobileNavItem>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.nav>
  )
}

function NavAnchor({ href, children }) {
  return (
    <a
      href={href}
      className="px-3 py-2 text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors rounded-lg"
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
      className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm transition-colors ${active
        ? 'text-medical-400 font-medium'
        : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}`}
      role="menuitem"
      aria-current={active ? 'page' : undefined}
    >
      {icon}
      {children}
    </Link>
  )
}

function MobileAnchor({ href, onClick, children }) {
  return (
    <a
      href={href}
      onClick={onClick}
      className="block px-4 py-2.5 text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-elevated)] rounded-lg transition-colors"
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
      className="flex items-center gap-2.5 px-4 py-2.5 text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-elevated)] rounded-lg transition-colors"
      role="menuitem"
    >
      {icon}
      {children}
    </Link>
  )
}

export default Navbar
