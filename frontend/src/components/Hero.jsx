import { motion } from 'framer-motion'
import { Shield, Zap, Search, AlertTriangle, Database, Activity, Sparkles } from 'lucide-react'

function Hero() {
  return (
    <section className="pt-32 pb-10 px-5">
      <div className="max-w-4xl mx-auto">

        {/* Hero headline */}
        <div className="text-center mb-20">
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full border border-medical-500/20 bg-medical-500/6 mb-8"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-medical-400" />
            <span className="text-medical-400 text-xs font-medium tracking-wide">
              42M+ Drug Interactions Database
            </span>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, delay: 0.05 }}
            className="font-display font-bold text-4xl sm:text-5xl md:text-[3.5rem] text-[var(--text-primary)] mb-5 leading-[1.1] tracking-tight"
          >
            Verify Drug Interactions.
            <br />
            <span className="text-medical-400">Protect Patient Safety.</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, delay: 0.1 }}
            className="text-base sm:text-lg text-[var(--text-secondary)] max-w-lg mx-auto mb-10 leading-relaxed"
          >
            Check if your medications are safe together. AI-powered severity
            analysis and safer alternatives — instantly.
          </motion.p>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, delay: 0.15 }}
            className="flex flex-wrap justify-center gap-10"
          >
            <Stat icon={<Database className="w-4 h-4" />} value="42M+" label="Interactions" />
            <Stat icon={<Activity className="w-4 h-4" />} value="100K+" label="Drugs Indexed" />
            <Stat icon={<Shield className="w-4 h-4" />} value="99.9%" label="Uptime" />
          </motion.div>
        </div>

        {/* How it Works */}
        <div id="how-it-works" className="mb-24">
          <SectionHeader
            title="How DrugGuard Works"
            subtitle="Three steps to safer medication management."
          />
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <StepCard number="01" title="Enter Medications" description="Type drug names or scan labels. Thousands of medications recognized." icon={<Search className="w-4.5 h-4.5" />} delay={0.1} />
            <StepCard number="02" title="AI Analysis" description="Clinical rules check 42M+ interactions. ML models assess personalized risk." icon={<Sparkles className="w-4.5 h-4.5" />} delay={0.2} />
            <StepCard number="03" title="Clear Results" description="Severity levels, plain-language explanations, and safer alternatives." icon={<Shield className="w-4.5 h-4.5" />} delay={0.3} />
          </div>
        </div>

        {/* Features */}
        <div id="features" className="mb-16">
          <SectionHeader
            title="Built for Medication Safety"
            subtitle="Combining AI, clinical rules, and real-time data."
          />
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3.5">
            <FeatureCard icon={<Database className="w-4 h-4" />} title="42M+ Interactions" description="TWOSIDES, OFFSIDES, and DrugBank databases combined." delay={0.1} />
            <FeatureCard icon={<Zap className="w-4 h-4" />} title="ML Predictions" description="XGBoost and Random Forest trained on clinical data." delay={0.15} />
            <FeatureCard icon={<Search className="w-4 h-4" />} title="SHAP Explainability" description="Understand why a drug pair is flagged as risky." delay={0.2} />
            <FeatureCard icon={<AlertTriangle className="w-4 h-4" />} title="Diabetic Safety" description="eGFR monitoring and nephropathy considerations." delay={0.25} />
            <FeatureCard icon={<Shield className="w-4 h-4" />} title="Rules-First" description="Clinical contraindications always override ML." delay={0.3} />
            <FeatureCard icon={<Sparkles className="w-4 h-4" />} title="LLM Explanations" description="Complex findings translated to plain language." delay={0.35} />
          </div>
        </div>

        {/* Disclaimer */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg border border-[var(--border)] text-center"
        >
          <AlertTriangle className="w-3.5 h-3.5 text-[var(--text-muted)] flex-shrink-0" />
          <span className="text-[var(--text-muted)] text-xs">
            For informational purposes only. Always consult a healthcare professional.
          </span>
        </motion.div>
      </div>
    </section>
  )
}

function SectionHeader({ title, subtitle }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4 }}
      className="text-center mb-10"
    >
      <h2 className="font-display font-bold text-2xl sm:text-3xl text-[var(--text-primary)] mb-2.5 tracking-tight">
        {title}
      </h2>
      <p className="text-[var(--text-secondary)] text-sm max-w-md mx-auto">{subtitle}</p>
    </motion.div>
  )
}

function Stat({ icon, value, label }) {
  return (
    <div className="flex items-center gap-3">
      <div className="w-8 h-8 rounded-lg bg-[var(--bg-elevated)] border border-[var(--border)] flex items-center justify-center text-medical-400">
        {icon}
      </div>
      <div className="text-left">
        <div className="text-lg font-bold text-[var(--text-primary)] tracking-tight">{value}</div>
        <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">{label}</div>
      </div>
    </div>
  )
}

function StepCard({ number, title, description, icon, delay }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.35, delay }}
      className="p-5 rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)]/40 hover:border-[var(--border-hover)] transition-colors"
    >
      <div className="flex items-center gap-2.5 mb-3.5">
        <span className="text-xs font-bold text-medical-400 tabular-nums">{number}</span>
        <div className="w-7 h-7 rounded-lg bg-medical-500/8 flex items-center justify-center text-medical-400">
          {icon}
        </div>
      </div>
      <h3 className="font-display font-semibold text-sm text-[var(--text-primary)] mb-1.5">{title}</h3>
      <p className="text-xs text-[var(--text-secondary)] leading-relaxed">{description}</p>
    </motion.div>
  )
}

function FeatureCard({ icon, title, description, delay }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.3, delay }}
      className="p-4.5 rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)]/30 hover:border-[var(--border-hover)] transition-colors"
    >
      <div className="w-8 h-8 rounded-lg bg-medical-500/8 flex items-center justify-center text-medical-400 mb-3">
        {icon}
      </div>
      <h3 className="font-display font-semibold text-sm text-[var(--text-primary)] mb-1">{title}</h3>
      <p className="text-xs text-[var(--text-secondary)] leading-relaxed">{description}</p>
    </motion.div>
  )
}

export default Hero
