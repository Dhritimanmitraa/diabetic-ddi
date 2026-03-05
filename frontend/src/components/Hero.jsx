import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Shield, Zap, Search, AlertTriangle, Database, Activity, Sparkles, ArrowRight } from 'lucide-react'

function AnimatedCounter({ target, suffix = '', duration = 2000 }) {
  const [count, setCount] = useState(0)

  useEffect(() => {
    let start = 0
    const end = target
    const incrementTime = duration / end
    const timer = setInterval(() => {
      start += 1
      setCount(Math.min(start, end))
      if (start >= end) clearInterval(timer)
    }, incrementTime)
    return () => clearInterval(timer)
  }, [target, duration])

  return <span>{count.toLocaleString()}{suffix}</span>
}

function Hero() {
  return (
    <section className="pt-28 pb-8 px-5">
      <div className="max-w-5xl mx-auto">
        {/* Top section — centered intro */}
        <div className="text-center mb-20">
          {/* Status badge */}
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-medical-500/8 border border-medical-500/15 mb-7"
          >
            <span className="relative flex h-1.5 w-1.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-medical-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-medical-500"></span>
            </span>
            <span className="text-medical-400 text-xs font-medium tracking-wide">
              42M+ Drug Interactions Database
            </span>
          </motion.div>

          {/* Headline */}
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.05 }}
            className="font-display font-extrabold text-4xl sm:text-5xl md:text-6xl text-[var(--text-primary)] mb-5 leading-[1.1] tracking-tight"
          >
            Verify Drug Interactions.
            <br />
            <span className="gradient-text">Protect Patient Safety.</span>
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="text-base sm:text-lg text-[var(--text-secondary)] max-w-xl mx-auto mb-10 leading-relaxed"
          >
            Check if your medications are safe together. Get AI-powered severity
            analysis and safer alternatives — instantly.
          </motion.p>

          {/* Stats row */}
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.15 }}
            className="flex flex-wrap justify-center gap-8 sm:gap-12"
          >
            <Stat icon={<Database className="w-4 h-4" />} value={<AnimatedCounter target={42} suffix="M+" duration={1500} />} label="Interactions" />
            <Stat icon={<Activity className="w-4 h-4" />} value={<AnimatedCounter target={100} suffix="K+" duration={1500} />} label="Drugs Indexed" />
            <Stat icon={<Shield className="w-4 h-4" />} value="99.9%" label="Uptime" />
          </motion.div>
        </div>

        {/* How it Works */}
        <div id="how-it-works" className="mb-24">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="font-display font-bold text-2xl sm:text-3xl text-[var(--text-primary)] mb-3 tracking-tight">
              How <span className="gradient-text">DrugGuard</span> Works
            </h2>
            <p className="text-[var(--text-secondary)] text-sm max-w-lg mx-auto">
              Three steps to safer medication management.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            <StepCard
              number="01"
              title="Enter Medications"
              description="Type drug names or scan labels. Our system recognizes thousands of medications and their variants."
              icon={<Search className="w-5 h-5" />}
              delay={0.1}
            />
            <StepCard
              number="02"
              title="AI Analysis"
              description="A hybrid system: clinical rules check 42M+ interactions first, then ML models assess personalized risk."
              icon={<Sparkles className="w-5 h-5" />}
              delay={0.2}
            />
            <StepCard
              number="03"
              title="Clear Results"
              description="Get severity levels, plain-language explanations, and safe alternative suggestions when needed."
              icon={<Shield className="w-5 h-5" />}
              delay={0.3}
            />
          </div>
        </div>

        {/* Features */}
        <div id="features" className="mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="font-display font-bold text-2xl sm:text-3xl text-[var(--text-primary)] mb-3 tracking-tight">
              Built for <span className="gradient-text">Medication Safety</span>
            </h2>
            <p className="text-[var(--text-secondary)] text-sm max-w-lg mx-auto">
              Combining AI, clinical rules, and real-time data analysis.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <FeatureCard
              icon={<Database className="w-5 h-5" />}
              title="42M+ Interactions"
              description="TWOSIDES, OFFSIDES, and DrugBank databases combined."
              delay={0.1}
            />
            <FeatureCard
              icon={<Zap className="w-5 h-5" />}
              title="ML Predictions"
              description="XGBoost and Random Forest trained on clinical data."
              delay={0.15}
            />
            <FeatureCard
              icon={<Search className="w-5 h-5" />}
              title="SHAP Explainability"
              description="Understand why a drug pair is flagged as risky."
              delay={0.2}
            />
            <FeatureCard
              icon={<AlertTriangle className="w-5 h-5" />}
              title="Diabetic Safety"
              description="eGFR monitoring and nephropathy considerations."
              delay={0.25}
            />
            <FeatureCard
              icon={<Shield className="w-5 h-5" />}
              title="Rules-First"
              description="Clinical contraindications always override ML."
              delay={0.3}
            />
            <FeatureCard
              icon={<Sparkles className="w-5 h-5" />}
              title="LLM Explanations"
              description="Complex findings translated to plain language."
              delay={0.35}
            />
          </div>
        </div>

        {/* Disclaimer */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-warning-500/6 border border-warning-500/12 text-center"
        >
          <AlertTriangle className="w-3.5 h-3.5 text-warning-500 flex-shrink-0" />
          <span className="text-warning-600 dark:text-warning-400 text-xs">
            For informational purposes only. Always consult a healthcare professional.
          </span>
        </motion.div>
      </div>
    </section>
  )
}

/* -------------------------------------------------- */
/*  Sub-components                                     */
/* -------------------------------------------------- */

function Stat({ icon, value, label }) {
  return (
    <div className="flex items-center gap-3">
      <div className="w-9 h-9 rounded-lg bg-[var(--bg-elevated)] border border-[var(--border)] flex items-center justify-center text-medical-400">
        {icon}
      </div>
      <div className="text-left">
        <div className="text-xl font-bold text-[var(--text-primary)] tracking-tight">{value}</div>
        <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider font-medium">{label}</div>
      </div>
    </div>
  )
}

function StepCard({ number, title, description, icon, delay }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4, delay }}
      className="glass-subtle rounded-2xl p-6 group card-hover"
    >
      <div className="flex items-center gap-3 mb-4">
        <span className="text-xs font-bold text-medical-500 bg-medical-500/8 px-2.5 py-1 rounded-md tracking-wider">
          {number}
        </span>
        <div className="w-8 h-8 rounded-lg bg-medical-500/8 flex items-center justify-center text-medical-400">
          {icon}
        </div>
      </div>
      <h3 className="font-display font-semibold text-base text-[var(--text-primary)] mb-2">{title}</h3>
      <p className="text-sm text-[var(--text-secondary)] leading-relaxed">{description}</p>
    </motion.div>
  )
}

function FeatureCard({ icon, title, description, delay }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.35, delay }}
      className="glass-subtle rounded-xl p-5 group card-hover"
    >
      <div className="w-10 h-10 rounded-lg bg-medical-500/8 flex items-center justify-center text-medical-400 mb-3.5">
        {icon}
      </div>
      <h3 className="font-display font-semibold text-sm text-[var(--text-primary)] mb-1.5">{title}</h3>
      <p className="text-xs text-[var(--text-secondary)] leading-relaxed">{description}</p>
    </motion.div>
  )
}

export default Hero
