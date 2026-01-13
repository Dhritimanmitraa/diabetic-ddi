import { motion } from 'framer-motion'

function FloatingElements() {
  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
      {/* Animated gradient mesh background */}
      <div className="absolute inset-0 bg-gradient-mesh opacity-30" />

      {/* Large gradient orbs with enhanced animations */}
      <motion.div
        animate={{
          y: [0, -50, 0],
          x: [0, 30, 0],
          scale: [1, 1.2, 1],
          opacity: [0.1, 0.15, 0.1],
        }}
        transition={{
          duration: 10,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
        className="absolute top-1/4 -left-32 w-[500px] h-[500px] bg-gradient-to-br from-medical-500/20 to-teal-500/10 rounded-full blur-3xl"
      />
      <motion.div
        animate={{
          y: [0, 60, 0],
          x: [0, -40, 0],
          scale: [1, 1.3, 1],
          opacity: [0.1, 0.2, 0.1],
        }}
        transition={{
          duration: 12,
          repeat: Infinity,
          ease: 'easeInOut',
          delay: 2,
        }}
        className="absolute top-1/2 -right-32 w-[400px] h-[400px] bg-gradient-to-br from-purple-500/15 to-medical-600/10 rounded-full blur-3xl"
      />
      <motion.div
        animate={{
          y: [0, -30, 0],
          scale: [1, 1.2, 1],
        }}
        transition={{
          duration: 8,
          repeat: Infinity,
          ease: 'easeInOut',
          delay: 4,
        }}
        className="absolute bottom-1/4 left-1/3 w-[300px] h-[300px] bg-gradient-to-br from-cyan-400/10 to-medical-400/5 rounded-full blur-3xl"
      />

      {/* Floating pill particles */}
      {[...Array(8)].map((_, i) => (
        <motion.div
          key={i}
          animate={{
            y: [0, Math.random() * -60 - 20, 0],
            x: [0, Math.random() * 40 - 20, 0],
            rotate: [0, Math.random() * 30 - 15, 0],
            opacity: [0.1, 0.3, 0.1],
          }}
          transition={{
            duration: 5 + Math.random() * 5,
            repeat: Infinity,
            ease: 'easeInOut',
            delay: i * 0.5,
          }}
          className="absolute rounded-full"
          style={{
            top: `${10 + Math.random() * 80}%`,
            left: `${5 + Math.random() * 90}%`,
            width: `${8 + Math.random() * 8}px`,
            height: `${16 + Math.random() * 16}px`,
            background: `rgba(20, 184, 154, ${0.1 + Math.random() * 0.15})`,
          }}
        />
      ))}

      {/* Twinkling stars */}
      {[...Array(15)].map((_, i) => (
        <motion.div
          key={`star-${i}`}
          animate={{
            opacity: [0, 0.8, 0],
            scale: [0.5, 1, 0.5],
          }}
          transition={{
            duration: 2 + Math.random() * 2,
            repeat: Infinity,
            ease: 'easeInOut',
            delay: Math.random() * 3,
          }}
          className="absolute rounded-full bg-white"
          style={{
            top: `${Math.random() * 100}%`,
            left: `${Math.random() * 100}%`,
            width: `${2 + Math.random() * 2}px`,
            height: `${2 + Math.random() * 2}px`,
          }}
        />
      ))}

      {/* Shooting star effect */}
      <motion.div
        animate={{
          x: ['-100%', '200%'],
          y: ['0%', '100%'],
          opacity: [0, 1, 0],
        }}
        transition={{
          duration: 3,
          repeat: Infinity,
          ease: 'easeIn',
          repeatDelay: 8,
        }}
        className="absolute top-20 left-0 w-32 h-0.5 bg-gradient-to-r from-transparent via-medical-400 to-transparent rounded-full"
      />

      {/* Grid lines */}
      <svg
        className="absolute inset-0 w-full h-full opacity-20"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          <pattern
            id="grid"
            width="60"
            height="60"
            patternUnits="userSpaceOnUse"
          >
            <path
              d="M 60 0 L 0 0 0 60"
              fill="none"
              stroke="rgba(20, 184, 154, 0.05)"
              strokeWidth="1"
            />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />
      </svg>

      {/* Radial gradient overlay */}
      <div className="absolute inset-0 bg-gradient-radial from-transparent via-transparent to-slate-950/50" />
    </div>
  )
}

export default FloatingElements
