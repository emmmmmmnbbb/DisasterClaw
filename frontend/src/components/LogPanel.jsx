import { useEffect, useRef } from 'react'

function formatTime(ts) {
  if (!ts) return '--'
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

export default function LogPanel({ logs }) {
  const recent = (logs || []).slice(-120)
  const scrollRef = useRef(null)
  const pinnedRef = useRef(true)

  const handleScroll = () => {
    const el = scrollRef.current
    if (!el) return
    pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48
  }

  useEffect(() => {
    const el = scrollRef.current
    if (el && pinnedRef.current) {
      el.scrollTop = el.scrollHeight
    }
  }, [recent.length])

  return (
    <section className="log-panel" style={{ padding: 18 }}>
      <div className="section-title">Mission Log</div>
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="panel-scroll"
        style={{ display: 'flex', flexDirection: 'column', gap: 10 }}
      >
        {recent.length === 0 ? (
          <div style={{ color: 'var(--ink-soft)' }}>No events yet.</div>
        ) : (
          recent.map((entry, index) => (
            <div
              key={`${entry.ts}-${index}`}
              style={{
                display: 'grid',
                gridTemplateColumns: '90px 72px minmax(0, 1fr)',
                gap: 10,
                alignItems: 'start',
                padding: '10px 12px',
                borderRadius: 14,
                background: 'rgba(255,255,255,0.58)',
                border: '1px solid rgba(171, 152, 117, 0.16)',
                fontSize: 13,
              }}
            >
              <span style={{ color: 'var(--ink-soft)', fontFamily: 'var(--font-mono)' }}>{formatTime(entry.ts)}</span>
              <span style={{ color: levelColor(entry.level), fontWeight: 700, textTransform: 'uppercase', fontSize: 12 }}>
                {entry.level}
              </span>
              <span>{entry.msg}</span>
            </div>
          ))
        )}
      </div>
    </section>
  )
}

function levelColor(level) {
  if (level === 'error') return 'var(--danger)'
  if (level === 'warn') return 'var(--warning)'
  if (level === 'success') return 'var(--accent)'
  return 'var(--ink-soft)'
}

