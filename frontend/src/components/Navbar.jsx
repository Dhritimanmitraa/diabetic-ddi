import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { Pill, Activity, Shield, Brain, Heart } from 'lucide-react'

function Navbar() {
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

          {/* Navigation links */}
          <div className="hidden md:flex items-center gap-8">
            <NavLink href="#how-it-works">How it Works</NavLink>
            <NavLink href="#features">Features</NavLink>
            <Link
              to="/diabetes"
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-400 hover:bg-rose-500/20 transition-colors text-sm font-medium"
            >
              <Heart className="w-4 h-4" />
              Diabetes DDI
            </Link>
            <Link
              to="/ml-dashboard"
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-purple-500/10 border border-purple-500/20 text-purple-400 hover:bg-purple-500/20 transition-colors text-sm font-medium"
            >
              <Brain className="w-4 h-4" />
              ML Dashboard
            </Link>
          </div>

          {/* Status indicator */}
          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-medical-500/10 border border-medical-500/20">
              <Activity className="w-4 h-4 text-medical-400" />
              <span className="text-xs font-medium text-medical-400">System Online</span>
            </div>
          </div>
        </div>
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

