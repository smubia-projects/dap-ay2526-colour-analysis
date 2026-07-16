import { useEffect, useRef } from 'react'

const LINKS = [
  { label: 'Join DAP',               href: 'https://t.me/join_smubia_dap', primary: true },
  { label: 'Explore other projects', href: 'https://www.smubia.com/Projects' },
  { label: 'View on GitHub',         href: 'https://github.com/smubia-projects/dap-ay2526-colour-analysis' },
]

const ACCENT = '#9c5a1d'

export default function CtaModal({ count, onDismiss }) {
  const cardRef = useRef(null)
  const closeRef = useRef(null)

  // Focus trap: hold focus inside the card while it's open, and restore it on close.
  useEffect(() => {
    const previous = document.activeElement
    closeRef.current?.focus()

    const onKeyDown = (e) => {
      if (e.key === 'Escape') { onDismiss(); return }
      if (e.key !== 'Tab') return

      const focusable = cardRef.current?.querySelectorAll('a[href], button')
      if (!focusable?.length) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]

      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault(); last.focus()
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault(); first.focus()
      }
    }

    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('keydown', onKeyDown)
      previous?.focus?.()
    }
  }, [onDismiss])

  return (
    <div
      onClick={onDismiss}
      style={{
        position: 'fixed', inset: 0, zIndex: 1000,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: '1.5rem',
        background: 'rgba(26, 22, 18, 0.4)',
        backdropFilter: 'blur(12px)',
        animation: 'ctaFade 0.5s ease both',
      }}
    >
      <style>{`
        @keyframes ctaFade { from { opacity: 0 } to { opacity: 1 } }
        @keyframes ctaRise {
          from { opacity: 0; transform: translateY(16px) scale(0.95) }
          to   { opacity: 1; transform: none }
        }
      `}</style>

      <div
        ref={cardRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="cta-headline"
        onClick={(e) => e.stopPropagation()}
        style={{
          width: '100%', maxWidth: 460,
          background: '#f7f4ef',
          borderRadius: 12,
          border: '1px solid #e4ddd5',
          overflow: 'hidden',
          boxShadow: '0 24px 64px rgba(26, 22, 18, 0.22)',
          animation: 'ctaRise 0.6s cubic-bezier(0.16, 1, 0.3, 1) both',
        }}
      >
        {/* Banner */}
        <div style={{
          position: 'relative',
          padding: '2.25rem 2rem 2rem',
          background: '#fdf3e7',
          borderBottom: '1px solid #e8c49a',
          overflow: 'hidden',
        }}>
          <div style={{
            position: 'absolute', top: 0, left: 0,
            width: 4, height: '100%', background: ACCENT,
          }} />
          {/* Decorative swatches — echo the season palette without competing with the copy */}
          <div aria-hidden="true" style={{
            position: 'absolute', top: -28, right: -28,
            width: 130, height: 130, borderRadius: '50%',
            background: 'rgba(156, 90, 29, 0.07)',
          }} />
          <div aria-hidden="true" style={{
            position: 'absolute', top: 34, right: 42,
            width: 60, height: 60, borderRadius: '50%',
            background: 'rgba(156, 90, 29, 0.05)',
          }} />

          <p style={{
            fontSize: '0.65rem', letterSpacing: '0.14em', textTransform: 'uppercase',
            color: ACCENT, marginBottom: 10,
          }}>
            Colour Analysis · DAP
          </p>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, marginBottom: 10 }}>
            <h2 id="cta-headline" style={{
              fontFamily: 'Cormorant Garamond', fontSize: '2.1rem',
              fontWeight: 400, lineHeight: 1.05, color: '#1a1612',
            }}>
              You've found your season
            </h2>
            <span style={{ fontSize: '1.4rem' }}>🎨</span>
          </div>
          <p style={{ fontSize: '0.84rem', lineHeight: 1.65, color: '#4a4440', maxWidth: 330 }}>
            You've analysed {count} photos through FaRL segmentation, K-Means colour
            extraction, and a fine-tuned CLIP classifier. Curious how it was built?
          </p>
        </div>

        {/* Actions */}
        <div style={{ padding: '1.5rem 2rem 1.75rem', display: 'flex', flexDirection: 'column', gap: 10 }}>
          {LINKS.map(({ label, href, primary }) => (
            <a
              key={label}
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              style={{
                position: 'relative',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                padding: '0.85rem 1rem',
                borderRadius: 8,
                textDecoration: 'none',
                fontSize: '0.78rem', letterSpacing: '0.1em', textTransform: 'uppercase',
                background: primary ? `linear-gradient(135deg, ${ACCENT}, #b86b38)` : '#faf8f5',
                color: primary ? '#f7f4ef' : '#4a4440',
                border: primary ? 'none' : '1px solid #e4ddd5',
                transition: 'opacity 0.2s',
              }}
            >
              {label}
              <span aria-hidden="true" style={{
                position: 'absolute', right: 18,
                opacity: 0.7, fontSize: '0.9rem',
              }}>→</span>
            </a>
          ))}

          <button
            ref={closeRef}
            onClick={onDismiss}
            style={{
              marginTop: 4, padding: '0.6rem',
              background: 'none', border: 'none', cursor: 'pointer',
              fontFamily: 'DM Sans', fontSize: '0.76rem', color: '#9a9189',
              letterSpacing: '0.04em',
            }}
          >
            Maybe later
          </button>
        </div>
      </div>
    </div>
  )
}
