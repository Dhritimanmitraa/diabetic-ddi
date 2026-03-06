import { memo } from 'react'
import { Shield, Github, Mail, ExternalLink } from 'lucide-react'

const Footer = memo(function Footer() {
  return (
    <footer className="relative z-10 mt-20 border-t border-[var(--border)]">
      <div className="max-w-6xl mx-auto px-5 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-10 mb-10">

          {/* Brand */}
          <div className="md:col-span-2">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-7 h-7 rounded-lg bg-medical-500 flex items-center justify-center">
                <Shield className="w-3.5 h-3.5 text-white" />
              </div>
              <span className="font-display font-bold text-base text-[var(--text-primary)] tracking-tight">
                Drug<span className="text-medical-400">Guard</span>
              </span>
            </div>
            <p className="text-[var(--text-secondary)] text-sm leading-relaxed mb-5 max-w-sm">
              AI-powered drug interaction checker. Verify medication safety before
              potential conflicts become problems.
            </p>
            <div className="flex items-center gap-2.5">
              <SocialLink href="https://github.com/Dhritimanmitraa/diabetic-ddi" icon={<Github className="w-4 h-4" />} label="GitHub" />
              <SocialLink href="mailto:contact@drugguard.ai" icon={<Mail className="w-4 h-4" />} label="Email" />
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="text-xs font-semibold text-[var(--text-muted)] mb-4 uppercase tracking-wider">Links</h4>
            <ul className="space-y-2.5">
              <FooterLink href="#how-it-works">How it Works</FooterLink>
              <FooterLink href="#features">Features</FooterLink>
              <FooterLink href="#about">About</FooterLink>
              <FooterLink href="#privacy">Privacy</FooterLink>
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h4 className="text-xs font-semibold text-[var(--text-muted)] mb-4 uppercase tracking-wider">Resources</h4>
            <ul className="space-y-2.5">
              <FooterLink href="https://www.drugbank.com" external>DrugBank</FooterLink>
              <FooterLink href="https://www.fda.gov" external>FDA Database</FooterLink>
              <FooterLink href="https://rxnav.nlm.nih.gov" external>RxNorm (NIH)</FooterLink>
              <FooterLink href="#api">API Docs</FooterLink>
            </ul>
          </div>
        </div>

        {/* Disclaimer */}
        <div className="p-4 bg-[var(--bg-elevated)]/40 rounded-xl border border-[var(--border)] mb-8">
          <p className="text-[var(--text-muted)] text-xs leading-relaxed">
            <span className="font-semibold text-[var(--text-secondary)]">Medical Disclaimer — </span>
            This tool is for informational purposes only and does not replace professional medical advice.
            Always consult your physician or qualified health provider regarding medication interactions.
          </p>
        </div>

        {/* Bottom bar */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-6 border-t border-[var(--border)]">
          <p className="text-[var(--text-muted)] text-xs">
            &copy; {new Date().getFullYear()} DrugGuard. All rights reserved.
          </p>
          <p className="text-[var(--text-muted)] text-xs">
            Built for patient safety.
          </p>
        </div>
      </div>
    </footer>
  )
})

function FooterLink({ href, children, external }) {
  return (
    <li>
      <a
        href={href}
        target={external ? '_blank' : undefined}
        rel={external ? 'noopener noreferrer' : undefined}
        className="text-[var(--text-muted)] hover:text-medical-400 transition-colors text-sm flex items-center gap-1.5 group"
      >
        {children}
        {external && (
          <ExternalLink className="w-3 h-3 opacity-0 group-hover:opacity-100 transition-opacity" />
        )}
      </a>
    </li>
  )
}

function SocialLink({ href, icon, label }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      aria-label={label}
      className="w-8 h-8 rounded-lg bg-[var(--bg-elevated)] hover:bg-medical-500/10 border border-[var(--border)] flex items-center justify-center text-[var(--text-muted)] hover:text-medical-400 transition-all"
    >
      {icon}
    </a>
  )
}

export default Footer
