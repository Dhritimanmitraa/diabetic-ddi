import { memo } from 'react'
import { Shield, Github } from 'lucide-react'

const Footer = memo(function Footer() {
  return (
    <footer className="relative z-10 mt-16 border-t border-[var(--border)]">
      <div className="max-w-3xl mx-auto px-5 py-8">
        {/* Disclaimer */}
        <p className="text-[var(--text-muted)] text-xs leading-relaxed mb-6">
          <span className="font-semibold text-[var(--text-secondary)]">Medical Disclaimer — </span>
          This tool is for informational purposes only and does not replace professional medical advice.
          Always consult your physician or qualified health provider regarding medication interactions.
        </p>

        {/* Bottom bar */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-5 border-t border-[var(--border)]">
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded bg-medical-500 flex items-center justify-center">
              <Shield className="w-3 h-3 text-white" />
            </div>
            <span className="font-display font-semibold text-sm text-[var(--text-primary)]">
              Drug<span className="text-medical-400">Guard</span>
            </span>
          </div>
          <div className="flex items-center gap-4">
            <a
              href="https://github.com/Dhritimanmitraa/diabetic-ddi"
              target="_blank"
              rel="noopener noreferrer"
              className="text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
              aria-label="GitHub"
            >
              <Github className="w-4 h-4" />
            </a>
            <p className="text-[var(--text-muted)] text-xs">
              &copy; {new Date().getFullYear()} DrugGuard
            </p>
          </div>
        </div>
      </div>
    </footer>
  )
})

export default Footer
