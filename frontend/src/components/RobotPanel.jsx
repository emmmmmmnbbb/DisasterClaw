function formatNumber(value, digits = 2) {
  return Number.isFinite(value) ? value.toFixed(digits) : '--'
}

function StatRow({ label, value, muted = false }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, fontSize: 14 }}>
      <span style={{ color: 'var(--ink-soft)' }}>{label}</span>
      <span style={{ fontWeight: 600, color: muted ? 'var(--ink-soft)' : 'var(--ink)' }}>{value}</span>
    </div>
  )
}

export default function RobotPanel({ worldState, systemStatus }) {
  const robot = worldState.robots?.[systemStatus.current_robot || 'UAV_1']
  const anchor = worldState.map?.anchor
  const reportCount = worldState.map?.reports?.length || 0

  return (
    <aside className="side-panel" style={{ padding: 18 }}>
      <div className="section-title">UAV State</div>

      {!robot ? (
        <div style={{ color: 'var(--ink-soft)' }}>Waiting for robot state...</div>
      ) : (
        <div className="panel-scroll" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div style={{ padding: 16, borderRadius: 18, background: 'var(--panel-strong)', border: '1px solid rgba(171, 152, 117, 0.28)' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
              <div>
                <div style={{ fontSize: 20, fontWeight: 800 }}>UAV_1</div>
                <div style={{ color: 'var(--ink-soft)', fontSize: 13 }}>Single vehicle / mock geodetic runtime</div>
              </div>
              <div className="pill" style={{ padding: '6px 10px' }}>
                <span className="dot" style={{ background: robot.in_air ? 'var(--accent)' : 'var(--danger)' }} />
                {robot.in_air ? 'Airborne' : 'Grounded'}
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <StatRow label="Status" value={robot.status || '--'} />
              <StatRow label="Task" value={robot.task_state || '--'} muted />
              <StatRow label="Battery" value={`${formatNumber(robot.battery, 0)}%`} />
              <StatRow label="Altitude" value={`${formatNumber(robot.position?.alt, 1)} m`} />
              <StatRow label="Heading" value={`${formatNumber(robot.heading_deg, 0)}°`} />
              <StatRow label="Speed" value={`${formatNumber(robot.speed_mps, 1)} m/s`} muted />
            </div>
          </div>

          <div style={{ padding: 16, borderRadius: 18, background: 'rgba(255,255,255,0.44)', border: '1px solid rgba(171, 152, 117, 0.22)' }}>
            <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10, color: 'var(--accent)' }}>Geodetic Position</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <StatRow label="Latitude" value={formatNumber(robot.position?.lat, 6)} />
              <StatRow label="Longitude" value={formatNumber(robot.position?.lon, 6)} />
              <StatRow label="North Offset" value={`${formatNumber(robot.position?.north_m, 1)} m`} muted />
              <StatRow label="East Offset" value={`${formatNumber(robot.position?.east_m, 1)} m`} muted />
            </div>
          </div>

          <div style={{ padding: 16, borderRadius: 18, background: 'rgba(255,255,255,0.44)', border: '1px solid rgba(171, 152, 117, 0.22)' }}>
            <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10, color: 'var(--accent)' }}>Map Summary</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <StatRow label="Anchor" value={anchor?.label || '--'} />
              <StatRow label="Targets" value={`${worldState.targets?.length || 0}`} muted />
              <StatRow label="Reports" value={`${reportCount}`} muted />
              <StatRow label="Tile Mode" value="Local SVG / Offline" muted />
            </div>
          </div>
        </div>
      )}
    </aside>
  )
}

