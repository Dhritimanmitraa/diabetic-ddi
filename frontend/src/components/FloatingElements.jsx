import { memo } from 'react'
import { motion } from 'framer-motion'

/**
 * FloatingElements — Subtle ambient background
 * 
 * Two soft gradient orbs that drift slowly, creating depth
 * without distracting from content. No particles, no stars.
 */
const FloatingElements = memo(function FloatingElements() {
  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none z-0" aria-hidden="true">
      {/* Primary ambient orb — top left */}
      <motion.div
        animate={{
          y: [0, -30, 0],
          x: [0, 15, 0],
          scale: [1, 1.08, 1],
        }}
        transition={{
          duration: 20,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
        className="absolute -top-32 -left-32 w-[600px] h-[600px] rounded-full blur-[120px]"
        style={{
          background: 'radial-gradient(circle, hsla(160, 70%, 40%, 0.08) 0%, transparent 70%)',
        }}
      />

      {/* Secondary ambient orb — bottom right */}
      <motion.div
        animate={{
          y: [0, 20, 0],
          x: [0, -20, 0],
          scale: [1, 1.1, 1],
        }}
        transition={{
          duration: 25,
          repeat: Infinity,
          ease: 'easeInOut',
          delay: 5,
        }}
        className="absolute -bottom-48 -right-48 w-[500px] h-[500px] rounded-full blur-[120px]"
        style={{
          background: 'radial-gradient(circle, hsla(200, 60%, 45%, 0.06) 0%, transparent 70%)',
        }}
      />
    </div>
  )
})

export default FloatingElements
