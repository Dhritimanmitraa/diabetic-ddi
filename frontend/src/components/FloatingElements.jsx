import { memo } from 'react'

/**
 * FloatingElements — Single static ambient gradient at the top of the page.
 * No animation, no particles — just enough depth to avoid a flat canvas.
 */
const FloatingElements = memo(function FloatingElements() {
  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none z-0" aria-hidden="true">
      {/* Single subtle top-edge glow */}
      <div
        className="absolute -top-64 left-1/2 -translate-x-1/2 w-[900px] h-[500px] rounded-full"
        style={{
          background: 'radial-gradient(ellipse at center top, hsla(160, 60%, 38%, 0.07) 0%, transparent 65%)',
          filter: 'blur(60px)',
        }}
      />
    </div>
  )
})

export default FloatingElements
