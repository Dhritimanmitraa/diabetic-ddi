import { AlertTriangle } from 'lucide-react'

function Hero() {
  return (
    <section className="pt-28 pb-6 px-5">
      <div className="max-w-2xl mx-auto text-center">
        <h1 className="font-display font-bold text-3xl sm:text-4xl text-[var(--text-primary)] mb-3 leading-tight tracking-tight">
          Drug Interaction Checker
        </h1>
        <p className="text-sm sm:text-base text-[var(--text-secondary)] max-w-md mx-auto mb-6 leading-relaxed">
          Check if your medications are safe to take together.
          Enter two drug names below to get started.
        </p>
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border border-[var(--border)] text-center">
          <AlertTriangle className="w-3.5 h-3.5 text-[var(--text-muted)] flex-shrink-0" />
          <span className="text-[var(--text-muted)] text-xs">
            For informational purposes only. Always consult a healthcare professional.
          </span>
        </div>
      </div>
    </section>
  )
}

export default Hero
