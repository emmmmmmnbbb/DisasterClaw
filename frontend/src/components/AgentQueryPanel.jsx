// Agent-VQA 问答结果展示 (D4, 计划 8.3)。
// 显示当前问题、答案与置信度、证据来源、当前动作、剩余预算、重观测前后对比、
// 结构化动作理由、最终状态与失败原因。不显示模型私有思维过程。

const DECISION_LABEL = {
  answer: '回答',
  continue_search: '继续搜索',
  reobserve: '重观测',
  abstain: '弃答',
}

const REASON_LABEL = {
  sufficient_evidence: '证据充分',
  target_missing: '目标缺失',
  low_confidence: '置信不足',
  budget_exhausted: '预算耗尽',
  invalid_question: '问题无效',
  planner_unavailable: '规划器不可用',
  out_of_coverage: '超出覆盖区',
  execution_error: '执行错误',
  invalid_output: '模型输出非法',
  vlm_unavailable: '视觉问答模型不可用',
  cancelled: '用户取消',
}

const ACTION_LABEL = {
  fly_relative: '机动飞行',
  report_observation: '上报观测',
  stop: '停止',
}

const TYPE_LABEL = {
  presence: '是否存在',
  damage: '损伤等级',
  count: '数量',
  spatial: '空间方位',
}

function pct(v) {
  const n = Number(v)
  if (!Number.isFinite(n)) return '--'
  return `${Math.round(n * 100)}%`
}

function StepRow({ step, index }) {
  const decision = DECISION_LABEL[step.decision] || step.decision
  const reason = REASON_LABEL[step.reason_code] || step.reason_code
  const action = ACTION_LABEL[step.action] || step.action
  const pos = step.position || {}
  return (
    <div style={{
      padding: '10px 12px', borderRadius: 12,
      background: 'rgba(255,255,255,0.5)',
      border: '1px solid rgba(171,152,117,0.16)',
      fontSize: 12, lineHeight: 1.5,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
        <strong>步 {index + 1}</strong>
        <span style={{ color: 'var(--ink-soft)' }}>
          预算 {step.budget_after ?? '--'}/{step.budget_before ?? '--'}
        </span>
      </div>
      <div style={{ marginTop: 4 }}>
        <span style={{ color: 'var(--ink-soft)' }}>候选：</span>
        <strong>{step.candidate_answer || '—'}</strong>
        <span style={{ marginLeft: 8, color: 'var(--ink-soft)' }}>置信 {pct(step.confidence)}</span>
      </div>
      <div style={{ marginTop: 4, color: 'var(--ink-soft)' }}>
        决策 <strong style={{ color: 'var(--accent)' }}>{decision}</strong>
        · 理由 {reason} · 动作 {action}
      </div>
      {pos.lat != null && (
        <div style={{ marginTop: 2, color: 'var(--ink-soft)', fontSize: 11 }}>
          位姿 {Number(pos.lat).toFixed(5)}, {Number(pos.lon).toFixed(5)} @ {Number(pos.alt || 0).toFixed(0)}m
        </div>
      )}
      {step.fallback_used && (
        <div style={{ marginTop: 4, color: 'var(--warning)', fontSize: 11 }}>
          规则回退{step.degraded_reason ? `（${step.degraded_reason}）` : ''}
        </div>
      )}
    </div>
  )
}

function ReobserveCompare({ steps }) {
  // decision=reobserve 的当前步是“观测前”；下一步才是机动后的新观测。
  const pairs = []
  for (let i = 0; i < steps.length - 1; i += 1) {
    if (steps[i].decision === 'reobserve') {
      pairs.push({ before: steps[i], after: steps[i + 1] })
    }
  }
  if (!pairs.length) return null
  return (
    <div style={{ marginTop: 10 }}>
      <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 6, color: 'var(--accent)' }}>
        重观测前后对比
      </div>
      {pairs.map((p, idx) => {
        const changed = p.before.candidate_answer !== p.after.candidate_answer
        return (
          <div key={idx} style={{
            display: 'flex', gap: 8, alignItems: 'center',
            padding: '8px 10px', borderRadius: 10,
            background: changed ? 'rgba(219,139,45,0.12)' : 'rgba(255,255,255,0.5)',
            border: '1px solid rgba(171,152,117,0.16)', fontSize: 12, marginBottom: 6,
          }}>
            <span>{p.before.candidate_answer || '—'} <span style={{ color: 'var(--ink-soft)' }}>{pct(p.before.confidence)}</span></span>
            <span style={{ color: 'var(--ink-soft)' }}>→</span>
            <span><strong>{p.after.candidate_answer || '—'}</strong> <span style={{ color: 'var(--ink-soft)' }}>{pct(p.after.confidence)}</span></span>
            <span style={{ marginLeft: 'auto', color: changed ? 'var(--warning)' : 'var(--ink-soft)', fontSize: 11 }}>
              {changed ? '答案翻转' : '答案稳定'}
            </span>
          </div>
        )
      })}
    </div>
  )
}

function FinalCard({ result }) {
  if (!result) return null
  const ans = result.answer || {}
  const abstain = ans.abstain
  const ok = result.ok !== false
  const cancelled = result.cancelled || ans.reason_code === 'cancelled'
  const tone = cancelled ? 'var(--warning)' : !ok ? 'var(--danger)' : abstain ? 'var(--warning)' : 'var(--accent)'
  const bg = cancelled ? 'rgba(219,139,45,0.1)' : !ok ? 'rgba(181,61,61,0.1)' : abstain ? 'rgba(219,139,45,0.1)' : 'rgba(45,134,82,0.1)'
  const ev = ans.evidence || {}
  return (
    <div style={{
      padding: 14, borderRadius: 14, background: bg,
      border: `1px solid ${tone}`, marginTop: 12,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ fontWeight: 700, color: tone, fontSize: 13 }}>
          {cancelled ? '已取消' : ok ? (abstain ? '弃答' : '最终回答') : '执行失败'}
        </div>
        <span style={{ color: 'var(--ink-soft)', fontSize: 11 }}>
          {result.n_steps != null ? `${result.n_steps} 步` : ''}
        </span>
      </div>
      {!ok && !cancelled ? (
        <div style={{ marginTop: 6, fontSize: 13, color: 'var(--danger)' }}>
          {result.error || '未知错误'}
        </div>
      ) : (
        <>
          <div style={{ marginTop: 8, fontSize: 15 }}>
            <strong>{ans.answer || '—'}</strong>
            <span style={{ marginLeft: 10, color: 'var(--ink-soft)', fontSize: 12 }}>
              置信 {pct(ans.confidence)}
            </span>
          </div>
          <div style={{ marginTop: 6, color: 'var(--ink-soft)', fontSize: 12 }}>
            决策 {DECISION_LABEL[ans.decision] || ans.decision || '—'}
            · 理由 {REASON_LABEL[ans.reason_code] || ans.reason_code || '—'}
          </div>
          {ev.target_subtype && (
            <div style={{ marginTop: 6, fontSize: 12 }}>
              <span style={{ color: 'var(--ink-soft)' }}>证据目标：</span>
              <strong>{ev.target_subtype}</strong>
              {ev.norm_xy && (
                <span style={{ marginLeft: 8, color: 'var(--ink-soft)' }}>
                  位置 ({Number(ev.norm_xy[0]).toFixed(2)}, {Number(ev.norm_xy[1]).toFixed(2)})
                </span>
              )}
            </div>
          )}
          {ev.source && (
            <div style={{ marginTop: 2, fontSize: 11, color: 'var(--ink-soft)' }}>
              来源 {ev.source}
            </div>
          )}
          {result.fallback_used && (
            <div style={{ marginTop: 6, color: 'var(--warning)', fontSize: 11 }}>
              规则回退{result.degraded_reason ? `（${result.degraded_reason}）` : ''}
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default function AgentQueryPanel({ agentQueryState }) {
  if (!agentQueryState) {
    return (
      <section className="agent-query-panel" style={{ padding: 16 }}>
        <div className="section-title">Agent-VQA</div>
        <div style={{ color: 'var(--ink-soft)', fontSize: 13, marginTop: 8 }}>
          在 Task Console 提交一个灾情问题（如"北侧是否存在完全损毁建筑？"），Agent 会搜索证据并在此展示回答、证据与动作轨迹。
        </div>
      </section>
    )
  }

  const { phase, question, questionType, steps = [], result } = agentQueryState
  const running = phase === 'running'

  return (
    <section className="agent-query-panel" style={{ padding: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div className="section-title" style={{ margin: 0 }}>Agent-VQA</div>
        <span style={{
          fontSize: 11, padding: '2px 8px', borderRadius: 8,
          color: running ? 'var(--accent)' : 'var(--ink-soft)',
          background: running ? 'rgba(45,134,82,0.1)' : 'rgba(255,255,255,0.5)',
        }}>
          {running ? '运行中' : '已完成'}
        </span>
      </div>

      <div style={{ marginTop: 10, padding: 12, borderRadius: 12, background: 'var(--panel-strong)', border: '1px solid rgba(171,152,117,0.24)' }}>
        <div style={{ fontSize: 11, color: 'var(--ink-soft)' }}>
          {TYPE_LABEL[questionType] || questionType || '问题'}
        </div>
        <div style={{ marginTop: 4, fontSize: 14, lineHeight: 1.5, wordBreak: 'break-word' }}>
          {question}
        </div>
      </div>

      {steps.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 6, color: 'var(--accent)' }}>
            动作轨迹（{steps.length}）
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {steps.map((s, i) => <StepRow key={i} step={s} index={i} />)}
          </div>
        </div>
      )}

      <ReobserveCompare steps={steps} />

      <FinalCard result={result} />
    </section>
  )
}
