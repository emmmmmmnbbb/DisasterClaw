import { useEffect, useState } from 'react'
import VisionPanel from './VisionPanel'

function StepList({ plan }) {
  const steps = plan?.steps || []
  if (!steps.length) {
    return <div style={{ color: 'var(--ink-soft)', fontSize: 13 }}>No AI plan yet.</div>
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {steps.map((step, index) => (
        <div key={`${step.action}-${index}`} style={{ padding: 12, borderRadius: 14, background: 'rgba(255,255,255,0.56)', border: '1px solid rgba(171,152,117,0.16)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
            <strong>{index + 1}. {step.action}</strong>
            <span style={{ color: 'var(--ink-soft)', fontSize: 12 }}>AI step</span>
          </div>
          <div style={{ marginTop: 6, color: 'var(--ink-soft)', fontSize: 13 }}>{step.reason}</div>
        </div>
      ))}
    </div>
  )
}

export default function TaskPanel({
  systemStatus,
  selectedPoint,
  lastActionResult,
  lastAiPlan,
  lastAiReport,
  onSubmitAiTask,
  onSubmitVlnTask,
  onStop,
  onHover,
  onFlyToPoint,
  onMarkPoint,
  onInspectPoint,
}) {
  const [draft, setDraft] = useState('飞到当前地图中心附近进行低空观察，并给出简短态势汇报。')
  const [vlnDraft, setVlnDraft] = useState('飞到北侧寻找完全损毁的建筑。')

  useEffect(() => {
    if (!selectedPoint) return
    const lat = Number(selectedPoint.lat)
    const lon = Number(selectedPoint.lon)
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return
    setDraft(
      `飞到纬度 ${lat.toFixed(6)}、经度 ${lon.toFixed(6)} 上空 30m 悬停观察，标记该位置并汇报灾害态势。`,
    )
  }, [selectedPoint])

  return (
    <aside className="task-panel" style={{ padding: 18 }}>
      <div className="section-title">Task Console</div>
      <div className="panel-scroll" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        <div style={{ padding: 16, borderRadius: 18, background: 'var(--panel-strong)', border: '1px solid rgba(171,152,117,0.24)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--accent)' }}>AI Task</div>
            <button
              className="btn btn-danger"
              onClick={onStop}
              disabled={!systemStatus.is_executing}
              style={{ padding: '4px 10px', fontSize: 12 }}
            >
              Stop
            </button>
          </div>
          <textarea
            rows={6}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            placeholder="Describe the task you want the UAV to perform."
          />
          <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
            <button className="btn btn-warm" onClick={() => onSubmitAiTask(draft)} disabled={!draft.trim()}>
              Run AI Mission
            </button>
            <button className="btn btn-soft" onClick={onHover}>
              Hover
            </button>
          </div>
        </div>

        <div style={{ padding: 16, borderRadius: 18, background: 'var(--panel-strong)', border: '1px solid rgba(171,152,117,0.24)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--accent)' }}>VLN 语言目标导航</div>
            <span style={{ color: 'var(--ink-soft)', fontSize: 11 }}>闭环 · BEV</span>
          </div>
          <div style={{ color: 'var(--ink-soft)', fontSize: 12, marginBottom: 8, lineHeight: 1.5 }}>
            用一句话描述要找的目标（如"完全损毁建筑/车辆/积水"）和方向，无人机会逐步感知并自主飞向目标。
          </div>
          <textarea
            rows={3}
            value={vlnDraft}
            onChange={(event) => setVlnDraft(event.target.value)}
            placeholder='例如："飞到北侧寻找完全损毁的建筑。"'
          />
          <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
            <button className="btn btn-accent" onClick={() => onSubmitVlnTask(vlnDraft)} disabled={!vlnDraft.trim() || !onSubmitVlnTask}>
              Run VLN
            </button>
            <button
              className="btn btn-danger"
              onClick={onStop}
              disabled={!systemStatus.is_executing}
              style={{ padding: '4px 10px', fontSize: 12 }}
            >
              Stop
            </button>
          </div>
        </div>

        <div style={{ padding: 16, borderRadius: 18, background: 'rgba(255,255,255,0.54)', border: '1px solid rgba(171,152,117,0.18)' }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10, color: 'var(--accent)' }}>Selected Point</div>
          {selectedPoint ? (
            <>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8, fontSize: 13 }}>
                <div><strong>Lat</strong> {Number(selectedPoint.lat || 0).toFixed(6)}</div>
                <div><strong>Lon</strong> {Number(selectedPoint.lon || 0).toFixed(6)}</div>
                <div style={{ color: 'var(--ink-soft)' }}>
                  local {Number(selectedPoint.north_m || 0).toFixed(1)}m N /{' '}
                  {Number(selectedPoint.east_m || 0).toFixed(1)}m E
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 12 }}>
                <button className="btn btn-accent" onClick={() => onFlyToPoint(selectedPoint)}>
                  Fly Here
                </button>
                <button className="btn btn-soft" onClick={() => onMarkPoint(selectedPoint)}>
                  Mark
                </button>
                <button className="btn btn-warm" style={{ gridColumn: '1 / -1' }} onClick={() => onInspectPoint(selectedPoint)}>
                  Ask AI To Inspect Here
                </button>
              </div>
            </>
          ) : (
            <div style={{ color: 'var(--ink-soft)', fontSize: 13 }}>
              在地图上点击目标点，即可在此处 Fly / Mark / Ask AI Inspect。
            </div>
          )}
        </div>

        <VisionPanel />

        <div style={{ padding: 16, borderRadius: 18, background: 'rgba(255,255,255,0.54)', border: '1px solid rgba(171,152,117,0.18)' }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10, color: 'var(--accent)' }}>Last Action</div>
          {lastActionResult ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, fontSize: 13 }}>
              <div><strong>{lastActionResult.action}</strong></div>
              <div style={{ color: 'var(--ink-soft)' }}>{lastActionResult.result?.message || 'No message'}</div>
            </div>
          ) : (
            <div style={{ color: 'var(--ink-soft)', fontSize: 13 }}>No manual or AI action has completed yet.</div>
          )}
        </div>

        <div style={{ padding: 16, borderRadius: 18, background: 'rgba(255,255,255,0.54)', border: '1px solid rgba(171,152,117,0.18)' }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10, color: 'var(--accent)' }}>AI Plan</div>
          {lastAiPlan?.summary && (
            <div style={{ marginBottom: 10, fontSize: 13, color: 'var(--ink-soft)' }}>{lastAiPlan.summary}</div>
          )}
          <StepList plan={lastAiPlan} />
        </div>

        <div style={{ padding: 16, borderRadius: 18, background: 'rgba(255,255,255,0.54)', border: '1px solid rgba(171,152,117,0.18)' }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10, color: 'var(--accent)' }}>Mission Report</div>
          <div style={{ color: lastAiReport ? 'var(--ink)' : 'var(--ink-soft)', fontSize: 13, lineHeight: 1.6 }}>
            {lastAiReport?.summary || 'No AI report yet.'}
          </div>
        </div>
      </div>
    </aside>
  )
}
