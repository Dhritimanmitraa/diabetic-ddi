import { memo } from 'react'
import { motion } from 'framer-motion'

const FloatingElements = memo(function FloatingElements() {
  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none z-0" aria-hidden="true">
      {/* Subtle top-left ambient glow */}
      <motion.div
        animate={{ y: [0, -20, 0], x: [0, 10, 0] }}
        transition={{ duration: 24, repeat: Infinity, ease: 'easeInOut' }}
        className="absolute -top-40 -left-40 w-[500px] h-[500px] rounded-full blur-[140px]"
        style={{ background: 'radial-gradient(circle, hsla(162, 65%, 38%, 0.055) 0%, transparent 70%)' }}
      />
      {/* Subtle bottom-right ambient glow */}
      <motion.div
        animate={{ y: [0, 16, 0], x: [0, -14, 0] }}
        transition={{ duration: 30, repeat: Infinity, ease: 'easeInOut', delay: 6 }}
        className="absolute -bottom-56 -right-56 w-[420px] h-[420px] rounded-full blur-[140px]"
        style={{ background: 'radial-gradient(circle, hsla(210, 55%, 42%, 0.04) 0%, transparent 70%)' }}
      />
    </div>
  )
})

export default FloatingElements
