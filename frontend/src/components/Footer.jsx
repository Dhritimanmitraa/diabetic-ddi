import { motion } from 'framer-motion'
import { Pill, Heart, Github, Mail, ExternalLink } from 'lucide-react'

function Footer() {
  return (
    <footer className="relative z-10 mt-20 border-t border-slate-800/50">
      <div className="max-w-7xl mx-auto px-4 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-12">
          {/* Brand */}
          <div className="md:col-span-2">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-medical-400 to-medical-600 flex items-center justify-center">
                <Pill className="w-5 h-5 text-white" />
              </div>
              <h3 className="font-display font-bold text-xl text-white">
                Drug<span className="gradient-text">Guard</span>
              </h3>
            </div>
            <p className="text-slate-400 mb-4 max-w-md">
              An AI-powered drug interaction checker designed to help you stay safe 
              by identifying potential medication conflicts before they become problems.
            </p>
            <div className="flex items-center gap-4">
              <SocialLink href="https://github.com" icon={<Github className="w-5 h-5" />} />
              <SocialLink href="mailto:contact@drugguard.ai" icon={<Mail className="w-5 h-5" />} />
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="font-display font-semibold text-white mb-4">Quick Links</h4>
            <ul className="space-y-2">
              <FooterLink href="#how-it-works">How it Works</FooterLink>
              <FooterLink href="#features">Features</FooterLink>
              <FooterLink href="#about">About Us</FooterLink>
              <FooterLink href="#privacy">Privacy Policy</FooterLink>
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h4 className="font-display font-semibold text-white mb-4">Resources</h4>
            <ul className="space-y-2">
              <FooterLink href="https://www.drugbank.com" external>DrugBank</FooterLink>
              <FooterLink href="https://www.fda.gov" external>FDA Drug Database</FooterLink>
              <FooterLink href="https://rxnav.nlm.nih.gov" external>RxNorm (NIH)</FooterLink>
              <FooterLink href="#api">API Documentation</FooterLink>
            </ul>
          </div>
        </div>

        {/* Disclaimer */}
        <div className="p-4 bg-slate-800/30 rounded-xl mb-8">
          <h5 className="text-white font-medium mb-2">Medical Disclaimer</h5>
          <p className="text-slate-400 text-sm">
            This tool is intended for informational purposes only and should not be used as a 
            substitute for professional medical advice, diagnosis, or treatment. Always seek 
            the advice of your physician or other qualified health provider with any questions 
            you may have regarding a medical condition or medication interactions. Never 
            disregard professional medical advice or delay in seeking it because of information 
            obtained from this application.
          </p>
        </div>

        {/* Bottom bar */}
        <div className="flex flex-col md:flex-row items-center justify-between pt-8 border-t border-slate-800/50">
          <p className="text-slate-500 text-sm mb-4 md:mb-0">
            © {new Date().getFullYear()} DrugGuard. All rights reserved.
          </p>
          <div className="flex items-center gap-1 text-slate-500 text-sm">
            Made with <Heart className="w-4 h-4 text-red-500 mx-1" /> for patient safety
          </div>
        </div>
      </div>
    </footer>
  )
}

function FooterLink({ href, children, external }) {
  return (
    <li>
      <a
        href={href}
        target={external ? '_blank' : undefined}
        rel={external ? 'noopener noreferrer' : undefined}
        className="text-slate-400 hover:text-medical-400 transition-colors flex items-center gap-1 group"
      >
        {children}
        {external && (
          <ExternalLink className="w-3 h-3 opacity-0 group-hover:opacity-100 transition-opacity" />
        )}
      </a>
    </li>
  )
}

function SocialLink({ href, icon }) {
  return (
    <motion.a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      whileHover={{ scale: 1.1, y: -2 }}
      whileTap={{ scale: 0.95 }}
      className="w-10 h-10 rounded-xl bg-slate-800 hover:bg-medical-500/20 flex items-center justify-center text-slate-400 hover:text-medical-400 transition-colors"
    >
      {icon}
    </motion.a>
  )
}

export default Footer

