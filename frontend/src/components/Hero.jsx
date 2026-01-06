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

      {/* How it Works Section */}
      <div id="how-it-works" className="max-w-6xl mx-auto px-4 py-20">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <h2 className="font-display font-bold text-3xl md:text-4xl text-white mb-4">
            How <span className="gradient-text">DrugGuard</span> Works
          </h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            A simple 3-step process to check your medications for potential interactions.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <StepCard
            number="1"
            title="Enter Your Medications"
            description="Type the names of the drugs you're taking or scan their labels using your camera. Our system recognizes thousands of medication names."
            delay={0.1}
          />
          <StepCard
            number="2"
            title="AI Analyzes Interactions"
            description="Our hybrid system checks 42M+ interactions using clinical rules first, then ML models for personalized risk assessment."
            delay={0.2}
          />
          <StepCard
            number="3"
            title="Get Clear Results"
            description="Receive instant results with severity levels, explanations you can understand, and safe alternative suggestions when needed."
            delay={0.3}
          />
        </div>

        {/* Visual flow diagram */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="mt-12 glass-light rounded-2xl p-8 border border-slate-700/50"
        >
          <div className="flex flex-col md:flex-row items-center justify-center gap-4 text-center">
            <div className="flex flex-col items-center">
              <div className="w-16 h-16 rounded-full bg-medical-500/20 flex items-center justify-center text-medical-400 mb-2">
                <Search className="w-8 h-8" />
              </div>
              <span className="text-white font-medium">Input</span>
            </div>
            <div className="text-medical-400 text-2xl hidden md:block">→</div>
            <div className="text-medical-400 text-2xl md:hidden">↓</div>
            <div className="flex flex-col items-center">
              <div className="w-16 h-16 rounded-full bg-danger-500/20 flex items-center justify-center text-danger-400 mb-2">
                <Shield className="w-8 h-8" />
              </div>
              <span className="text-white font-medium">Rules Check</span>
            </div>
            <div className="text-medical-400 text-2xl hidden md:block">→</div>
            <div className="text-medical-400 text-2xl md:hidden">↓</div>
            <div className="flex flex-col items-center">
              <div className="w-16 h-16 rounded-full bg-warning-500/20 flex items-center justify-center text-warning-400 mb-2">
                <Zap className="w-8 h-8" />
              </div>
              <span className="text-white font-medium">ML Analysis</span>
            </div>
            <div className="text-medical-400 text-2xl hidden md:block">→</div>
            <div className="text-medical-400 text-2xl md:hidden">↓</div>
            <div className="flex flex-col items-center">
              <div className="w-16 h-16 rounded-full bg-success-500/20 flex items-center justify-center text-success-400 mb-2">
                <AlertTriangle className="w-8 h-8" />
              </div>
              <span className="text-white font-medium">Results</span>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Features Section */}
      <div id="features" className="max-w-6xl mx-auto px-4 py-20">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <h2 className="font-display font-bold text-3xl md:text-4xl text-white mb-4">
            Powerful Features for <span className="gradient-text">Medication Safety</span>
          </h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            DrugGuard combines AI, clinical rules, and real-time data to protect you from harmful drug interactions.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <FeatureBox
            icon={<Shield className="w-8 h-8" />}
            title="42M+ Drug Interactions"
            description="Powered by TWOSIDES, OFFSIDES, and DrugBank databases with over 42 million verified drug-drug interactions."
            delay={0.1}
          />
          <FeatureBox
            icon={<Zap className="w-8 h-8" />}
            title="ML-Powered Predictions"
            description="XGBoost and Random Forest models trained on real clinical data to predict interaction severity."
            delay={0.2}
          />
          <FeatureBox
            icon={<Search className="w-8 h-8" />}
            title="SHAP Explainability"
            description="Understand WHY a drug is risky with feature attribution that shows contributing factors."
            delay={0.3}
          />
          <FeatureBox
            icon={<AlertTriangle className="w-8 h-8" />}
            title="Diabetic Patient Safety"
            description="Specialized module for diabetic patients with eGFR monitoring and nephropathy considerations."
            delay={0.4}
          />
          <FeatureBox
            icon={<Shield className="w-8 h-8" />}
            title="Rules-First Architecture"
            description="Clinical rules ALWAYS take priority over ML predictions for contraindications and fatal combinations."
            delay={0.5}
          />
          <FeatureBox
            icon={<Zap className="w-8 h-8" />}
            title="Patient-Friendly Explanations"
            description="LLM-powered explanations translate complex medical findings into simple, understandable language."
            delay={0.6}
          />
        </div>
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

function FeatureBox({ icon, title, description, delay }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay }}
      whileHover={{ y: -4, borderColor: 'rgba(20, 184, 154, 0.5)' }}
      className="glass-light rounded-2xl p-6 border border-slate-700/50 hover:border-medical-500/50 transition-all duration-300"
    >
      <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-medical-500/20 to-medical-600/20 flex items-center justify-center text-medical-400 mb-4">
        {icon}
      </div>
      <h3 className="font-display font-semibold text-xl text-white mb-3">{title}</h3>
      <p className="text-slate-400">{description}</p>
    </motion.div>
  )
}

function StepCard({ number, title, description, delay }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay }}
      className="glass-light rounded-2xl p-6 border border-slate-700/50 text-center relative"
    >
      <div className="absolute -top-4 left-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-medical-500 text-white flex items-center justify-center font-bold text-lg shadow-lg shadow-medical-500/30">
        {number}
      </div>
      <div className="pt-4">
        <h3 className="font-display font-semibold text-xl text-white mb-3">{title}</h3>
        <p className="text-slate-400">{description}</p>
      </div>
    </motion.div>
  )
}

export default Hero

