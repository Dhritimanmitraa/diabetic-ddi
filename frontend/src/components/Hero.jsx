import { motion } from 'framer-motion'
import { Shield, Zap, Search, AlertTriangle } from 'lucide-react'

function Hero() {
  return (
    <section className="pt-32 pb-16 px-4">
      <div className="max-w-6xl mx-auto text-center">
        {/* Badge */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-medical-500/10 border border-medical-500/20 mb-8"
        >
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-medical-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-medical-500"></span>
          </span>
          <span className="text-medical-400 text-sm font-medium">
            Powered by AI & 100,000+ Drug Interactions Database
          </span>
        </motion.div>

        {/* Main heading */}
        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="font-display font-bold text-5xl md:text-7xl text-white mb-6 leading-tight"
        >
          Check Drug Interactions
          <br />
          <span className="gradient-text">Stay Safe</span>
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="text-xl text-slate-400 max-w-2xl mx-auto mb-12 font-body"
        >
          Instantly verify if your medications are safe to use together. 
          Get AI-powered recommendations for safer alternatives when needed.
        </motion.p>

        {/* Feature cards */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-4xl mx-auto"
        >
          <FeatureCard
            icon={<Search className="w-6 h-6" />}
            title="Type or Scan"
            description="Enter drug names or use your camera to scan medication labels"
            delay={0.4}
          />
          <FeatureCard
            icon={<Zap className="w-6 h-6" />}
            title="Instant Results"
            description="Get real-time interaction analysis with severity levels"
            delay={0.5}
          />
          <FeatureCard
            icon={<Shield className="w-6 h-6" />}
            title="Safe Alternatives"
            description="Receive AI-recommended safer medication substitutes"
            delay={0.6}
          />
        </motion.div>

        {/* Warning notice */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.7 }}
          className="mt-12 inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-warning-500/10 border border-warning-500/20"
        >
          <AlertTriangle className="w-4 h-4 text-warning-400" />
          <span className="text-warning-400 text-sm">
            This tool is for informational purposes only. Always consult a healthcare professional.
          </span>
        </motion.div>
      </div>
    </section>
  )
}

function FeatureCard({ icon, title, description, delay }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      whileHover={{ y: -4, scale: 1.02 }}
      className="glass-light rounded-2xl p-6 text-left card-hover"
    >
      <div className="w-12 h-12 rounded-xl bg-medical-500/10 flex items-center justify-center text-medical-400 mb-4">
        {icon}
      </div>
      <h3 className="font-display font-semibold text-lg text-white mb-2">{title}</h3>
      <p className="text-slate-400 text-sm">{description}</p>
    </motion.div>
  )
}

export default Hero

