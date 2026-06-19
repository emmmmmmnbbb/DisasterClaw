import { useState } from 'react'

const RISK_META = {
  none: { label: '未发现受灾', color: 'var(--accent)', bg: 'rgba(45, 134, 82, 0.12)' },
  low: { label: '轻度受灾', color: '#b07a16', bg: 'rgba(176, 122, 22, 0.14)' },
  moderate: { label: '局部受灾', color: 'var(--warning)', bg: 'rgba(219, 139, 45, 0.16)' },
  high: { label: '灾情较重', color: 'var(--danger)', bg: 'rgba(181, 61, 61, 0.18)' },
}

function formatTime(ts) {
  if (!ts) return '--'
  return new Date(ts).toLocaleTimeString([], {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  })
}

function topEntries(obj, n = 4) {
  return Object.entries(obj || {})
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .slice(0, n)
}

export default function PerceptionPanel({ perception, semanticMap, vlnThought }) {
  const [imageKind, setImageKind] = useState('patch') // patch / detection / overlay

  // 以语义地图作为 VLN 强信号（默认开启）；避免普通 AI 任务的 ai_thought 误触发。
  const hasVln = !!semanticMap

  if (!perception) {
    return (
      <section className="perception-panel" style={panelStyle}>
        <div className="section-title">Perception</div>
        {hasVln && <VlnStatus semanticMap={semanticMap} thought={vlnThought} />}
        <div style={{ color: 'var(--ink-soft)', fontSize: 13 }}>
          {hasVln
            ? 'VLN 语言导航进行中，等待首帧感知结果（YOLO + SegFormer）……'
            : '尚未运行 detect_disaster。发一个带“检测 / 受灾 / 评估”关键词的 AI 任务，或一条 VLN 语言导航指令，UAV 到位后会在此显示结果。'}
        </div>
      </section>
    )
  }

  const risk = RISK_META[perception.risk_level] || RISK_META.none
  const det = perception.detection || {}
  const seg = perception.segmentation || {}
  const pos = perception.position || {}

  const imageSrc = (() => {
    if (imageKind === 'detection' && perception.detection_url) return perception.detection_url
    if (imageKind === 'overlay' && perception.overlay_url) return perception.overlay_url
    return perception.patch_url
  })()

  const tabs = [
    { id: 'patch', label: 'UAV 视场', available: true },
    { id: 'detection', label: 'YOLO 标注', available: !!perception.detection_url },
    { id: 'overlay', label: 'SegFormer', available: !!perception.overlay_url },
  ]

  return (
    <section className="perception-panel" style={panelStyle}>
      <div
        className="section-title"
        style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}
      >
        <span>Perception</span>
        <span
          style={{
            fontSize: 11,
            padding: '3px 10px',
            borderRadius: 12,
            color: risk.color,
            background: risk.bg,
            border: `1px solid ${risk.color}`,
            textTransform: 'none',
            letterSpacing: 'normal',
          }}
        >
          {risk.label}
        </span>
      </div>

      {hasVln && <VlnStatus semanticMap={semanticMap} thought={vlnThought} />}

      <div style={{ display: 'grid', gridTemplateColumns: '180px minmax(0, 1fr)', gap: 12, minHeight: 0, flex: 1 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, minWidth: 0 }}>
          <div style={{ display: 'flex', gap: 4 }}>
            {tabs.map((t) => (
              <button
                key={t.id}
                type="button"
                disabled={!t.available}
                onClick={() => setImageKind(t.id)}
                style={{
                  flex: 1,
                  fontSize: 10,
                  padding: '3px 4px',
                  borderRadius: 6,
                  border: '1px solid rgba(171,152,117,0.35)',
                  background: imageKind === t.id ? 'var(--accent)' : 'rgba(255,255,255,0.6)',
                  color: imageKind === t.id ? '#fff' : 'var(--ink)',
                  opacity: t.available ? 1 : 0.35,
                  cursor: t.available ? 'pointer' : 'not-allowed',
                }}
              >
                {t.label}
              </button>
            ))}
          </div>
          <div
            style={{
              border: '1px solid rgba(171,152,117,0.3)',
              borderRadius: 8,
              overflow: 'hidden',
              aspectRatio: '1 / 1',
              background: 'rgba(0,0,0,0.05)',
            }}
          >
            {imageSrc ? (
              <img
                src={imageSrc}
                alt={imageKind}
                style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
              />
            ) : (
              <div style={{ padding: 10, fontSize: 11, color: 'var(--ink-soft)' }}>no image</div>
            )}
          </div>
          <div style={{ fontSize: 10, color: 'var(--ink-soft)' }}>
            {perception.patch_width}×{perception.patch_height}px ·
            半径≈{Number(perception.radius_m || 0).toFixed(0)}m ·
            {formatTime(perception.timestamp)}
          </div>
        </div>

        <div className="panel-scroll" style={{ display: 'flex', flexDirection: 'column', gap: 8, fontSize: 12 }}>
          <div style={{ color: 'var(--ink)', fontWeight: 600, lineHeight: 1.4 }}>
            {perception.risk_summary}
          </div>
          {perception.vlm_summary ? (
            <div
              style={{
                padding: '8px 10px',
                background: 'rgba(35, 97, 163, 0.08)',
                borderLeft: '3px solid #2361a3',
                borderRadius: 6,
                lineHeight: 1.5,
                color: 'var(--ink)',
              }}
            >
              <div style={{ fontSize: 10, color: '#2361a3', marginBottom: 3, fontWeight: 700 }}>
                Qwen-VL 结论
              </div>
              {perception.vlm_summary}
            </div>
          ) : null}

          <MetricsRow
            damaged={perception.damaged_buildings || 0}
            intact={perception.intact_buildings || 0}
            vehicles={perception.vehicles || 0}
            totalDet={det.num_objects || 0}
            totalSeg={seg.num_labels || 0}
          />

          {Object.keys(det.class_counts || {}).length > 0 && (
            <div>
              <div style={labelStyle}>YOLO 目标</div>
              <div style={chipRowStyle}>
                {topEntries(det.class_counts, 6).map(([name, count]) => (
                  <span key={name} style={chipStyle}>
                    {name} <b style={{ marginLeft: 4 }}>{count}</b>
                  </span>
                ))}
              </div>
            </div>
          )}

          {Object.keys(seg.stats || {}).length > 0 && (
            <div>
              <div style={labelStyle}>SegFormer 区域 (top)</div>
              <div style={chipRowStyle}>
                {topEntries(seg.stats, 5).map(([name]) => (
                  <span key={name} style={{ ...chipStyle, background: 'rgba(45,134,82,0.08)' }}>
                    {name}
                  </span>
                ))}
              </div>
            </div>
          )}

          <div style={{ fontSize: 10, color: 'var(--ink-soft)' }}>
            位置: {pos.lat?.toFixed?.(5)}, {pos.lon?.toFixed?.(5)} @ {pos.alt?.toFixed?.(1)}m
            {perception.degraded ? '  ·  ⚠ 视场回退为整图' : ''}
          </div>
        </div>
      </div>
    </section>
  )
}

const SKILL_META = {
  fly_relative: { label: '飞行', color: '#2361a3' },
  hover: { label: '悬停', color: 'var(--ink-soft)' },
  stop: { label: '到达', color: 'var(--accent)' },
  recheck: { label: '复核', color: 'var(--warning)' },
}

function VlnStatus({ semanticMap, thought }) {
  const stats = semanticMap?.stats || {}
  const instruction = semanticMap?.instruction || ''
  const objects = semanticMap?.objects || []
  const candidates = objects.filter((o) => o.layer === 'candidate_goals')

  const skill = thought?.skill
  const skillMeta = SKILL_META[skill] || { label: skill || '—', color: 'var(--ink-soft)' }
  const matched = thought?.matched
  const dist = thought?.target_dist_m
  const uncertainty = thought?.uncertainty

  const statChips = [
    { label: '步', value: stats.step_count },
    { label: '已探索格', value: stats.explored_cells },
    { label: '地标', value: stats.landmarks },
    { label: '物体', value: stats.surrounding_objects },
    { label: '候选', value: stats.candidate_goals },
  ].filter((c) => c.value != null)

  return (
    <div style={vlnWrapStyle}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
        <span style={vlnBadgeStyle}>VLN 语言导航</span>
        {instruction ? (
          <span style={{ fontSize: 12, color: 'var(--ink)', fontWeight: 600 }}>「{instruction}」</span>
        ) : (
          <span style={{ fontSize: 11, color: 'var(--ink-soft)' }}>等待指令 / 语义地图…</span>
        )}
      </div>

      {thought && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', fontSize: 11 }}>
          {thought.progress && <span style={vlnPillStyle}>{thought.progress}</span>}
          <span style={{ ...vlnPillStyle, color: skillMeta.color, borderColor: skillMeta.color }}>
            {skillMeta.label}
          </span>
          {matched != null && (
            <span
              style={{
                ...vlnPillStyle,
                color: matched ? 'var(--accent)' : 'var(--ink-soft)',
                borderColor: matched ? 'var(--accent)' : 'rgba(171,152,117,0.4)',
              }}
            >
              {matched ? `命中目标${dist != null ? ` ~${Number(dist).toFixed(0)}m` : ''}` : '搜索中'}
            </span>
          )}
        </div>
      )}

      {uncertainty != null && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: 10, color: 'var(--warning)', whiteSpace: 'nowrap' }}>
            复核不确定性
          </span>
          <div style={{ flex: 1, height: 6, borderRadius: 3, background: 'rgba(171,152,117,0.2)', overflow: 'hidden' }}>
            <div
              style={{
                width: `${Math.round(Math.min(Math.max(uncertainty, 0), 1) * 100)}%`,
                height: '100%',
                background: 'var(--warning)',
              }}
            />
          </div>
          <span style={{ fontSize: 10, color: 'var(--ink-soft)' }}>{Number(uncertainty).toFixed(2)}</span>
        </div>
      )}

      {thought?.thinking && (
        <div style={{ fontSize: 11, color: 'var(--ink-soft)', lineHeight: 1.45 }}>{thought.thinking}</div>
      )}

      {statChips.length > 0 && (
        <div style={chipRowStyle}>
          {statChips.map((c) => (
            <span key={c.label} style={{ ...chipStyle, background: 'rgba(35,97,163,0.08)' }}>
              {c.label} <b style={{ marginLeft: 3 }}>{c.value}</b>
            </span>
          ))}
        </div>
      )}

      {candidates.length > 0 && (
        <div>
          <div style={labelStyle}>候选 / 复核目标</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 3, maxHeight: 84, overflow: 'auto' }}>
            {candidates.slice(0, 6).map((o, i) => (
              <div key={`${o.gi},${o.gj},${i}`} style={candidateRowStyle}>
                <span style={{ color: 'var(--ink)', fontWeight: 600 }}>{o.class_name}</span>
                <span style={{ color: 'var(--ink-soft)' }}>
                  {o.risk && o.risk !== 'none' ? `risk ${o.risk} · ` : ''}
                  conf {Number(o.conf || 0).toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function MetricsRow({ damaged, intact, vehicles, totalDet, totalSeg }) {
  const cells = [
    { label: '受损建筑', value: damaged, tint: damaged > 0 ? 'var(--danger)' : undefined },
    { label: '完好建筑', value: intact },
    { label: '车辆', value: vehicles },
    { label: 'YOLO 总数', value: totalDet },
    { label: 'Seg 区域', value: totalSeg },
  ]
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 6 }}>
      {cells.map((c) => (
        <div
          key={c.label}
          style={{
            textAlign: 'center',
            padding: '6px 4px',
            background: 'rgba(255,255,255,0.55)',
            border: '1px solid rgba(171,152,117,0.2)',
            borderRadius: 6,
          }}
        >
          <div style={{ fontSize: 10, color: 'var(--ink-soft)' }}>{c.label}</div>
          <div style={{ fontSize: 17, fontWeight: 700, color: c.tint || 'var(--ink)' }}>
            {c.value ?? 0}
          </div>
        </div>
      ))}
    </div>
  )
}

const panelStyle = {
  padding: 16,
  display: 'flex',
  flexDirection: 'column',
  gap: 10,
  minHeight: 0,
  overflow: 'hidden',
}

const labelStyle = {
  fontSize: 10,
  color: 'var(--ink-soft)',
  textTransform: 'uppercase',
  letterSpacing: '0.08em',
  marginBottom: 4,
}

const chipRowStyle = { display: 'flex', flexWrap: 'wrap', gap: 4 }

const vlnWrapStyle = {
  display: 'flex',
  flexDirection: 'column',
  gap: 6,
  padding: '8px 10px',
  borderRadius: 8,
  background: 'rgba(35, 97, 163, 0.06)',
  border: '1px solid rgba(35, 97, 163, 0.2)',
  flex: 'none',
}

const vlnBadgeStyle = {
  fontSize: 10,
  fontWeight: 700,
  letterSpacing: '0.04em',
  padding: '2px 8px',
  borderRadius: 10,
  color: '#fff',
  background: '#2361a3',
}

const vlnPillStyle = {
  fontSize: 11,
  padding: '2px 8px',
  borderRadius: 10,
  background: 'rgba(255,255,255,0.7)',
  border: '1px solid rgba(171,152,117,0.4)',
  color: 'var(--ink)',
  whiteSpace: 'nowrap',
}

const candidateRowStyle = {
  display: 'flex',
  justifyContent: 'space-between',
  gap: 8,
  fontSize: 11,
  padding: '3px 8px',
  borderRadius: 6,
  background: 'rgba(255,255,255,0.55)',
  border: '1px solid rgba(171,152,117,0.25)',
}

const chipStyle = {
  fontSize: 11,
  padding: '3px 8px',
  borderRadius: 10,
  background: 'rgba(219, 139, 45, 0.14)',
  border: '1px solid rgba(171,152,117,0.3)',
  color: 'var(--ink)',
}
