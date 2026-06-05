export default function Header({ connected, systemStatus, onModeSwitch }) {
  const mode = systemStatus.mode || 'manual'

  return (
    <header className="topbar">
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        <div>
          <div style={{ fontSize: 24, fontWeight: 800, letterSpacing: '-0.03em' }}>DisasterClaw</div>
          <div style={{ color: 'var(--ink-soft)', fontSize: 13 }}>
            Local tile mission console for a single autonomous UAV
          </div>
        </div>
        <div className="pill">
          <span className="dot" style={{ background: connected ? 'var(--accent)' : 'var(--danger)' }} />
          {connected ? 'Socket Connected' : 'Socket Offline'}
        </div>
        <div className="pill">
          <span className="dot" style={{ background: systemStatus.is_executing ? 'var(--warning)' : 'var(--accent)' }} />
          {systemStatus.is_executing ? 'Mission Running' : 'UAV Hovering'}
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <button
          className={`btn ${mode === 'manual' ? 'btn-accent' : ''}`}
          onClick={() => onModeSwitch('manual')}
        >
          Manual
        </button>
        <button
          className={`btn ${mode === 'ai' ? 'btn-warm' : ''}`}
          onClick={() => onModeSwitch('ai')}
        >
          AI Auto
        </button>
      </div>
    </header>
  )
}

