import { useCallback, useEffect, useRef, useState } from 'react'
import { io } from 'socket.io-client'

const API_BASE = ''

export function useSocket() {
  const socketRef = useRef(null)
  const [connected, setConnected] = useState(false)
  const [systemStatus, setSystemStatus] = useState({
    initialized: false,
    mode: 'manual',
    current_robot: 'UAV_1',
    is_executing: false,
  })
  const [worldState, setWorldState] = useState({ robots: {}, targets: [], map: { reports: [] } })
  const [logs, setLogs] = useState([])
  const [lastActionResult, setLastActionResult] = useState(null)
  const [lastAiPlan, setLastAiPlan] = useState(null)
  const [lastAiReport, setLastAiReport] = useState(null)
  const [lastPerception, setLastPerception] = useState(null)
  const [semanticMap, setSemanticMap] = useState(null)
  const [vlnThought, setVlnThought] = useState(null)
  // Agent-VQA 问答状态 (D4, 计划 8.2)。与 VLN 状态独立，新任务开始时清理。
  const [agentQueryState, setAgentQueryState] = useState(null)

  useEffect(() => {
    const loadSnapshot = async () => {
      try {
        const [statusRes, worldRes, logsRes] = await Promise.all([
          fetch(`${API_BASE}/api/status`),
          fetch(`${API_BASE}/api/world`),
          fetch(`${API_BASE}/api/logs`),
        ])
        if (statusRes.ok) setSystemStatus(await statusRes.json())
        if (worldRes.ok) setWorldState(await worldRes.json())
        if (logsRes.ok) setLogs(await logsRes.json())
      } catch (error) {
        console.warn('[snapshot]', error)
      }
    }

    loadSnapshot()

    const socket = io({
      transports: ['polling'],
      upgrade: false,
      reconnectionAttempts: 10,
      reconnectionDelay: 800,
    })
    socketRef.current = socket

    socket.on('connect', () => setConnected(true))
    socket.on('disconnect', () => setConnected(false))
    socket.on('system_status', setSystemStatus)
    socket.on('world_state', setWorldState)
    socket.on('action_result', setLastActionResult)
    socket.on('ai_plan_result', setLastAiPlan)
    socket.on('ai_execution_report', setLastAiReport)
    socket.on('perception_result', setLastPerception)
    // VLN（语言导航）相关：语义地图快照 + 每步思考/grounding/复核状态
    socket.on('semantic_map', setSemanticMap)
    socket.on('ai_thought', setVlnThought)
    // Agent-VQA：开始 / 逐步更新 / 最终结果 (计划 8.2)
    socket.on('agent_query_started', (payload) => {
      setAgentQueryState({
        phase: 'running',
        question: payload?.question || '',
        questionType: payload?.question_type || '',
        source: payload?.source || '',
        tsMs: payload?.ts_ms || 0,
        steps: [],
        result: null,
      })
    })
    socket.on('agent_query_update', (step) => {
      setAgentQueryState((prev) => {
        if (!prev) return prev
        return { ...prev, steps: [...(prev.steps || []), step] }
      })
    })
    socket.on('agent_query_result', (result) => {
      setAgentQueryState((prev) => prev ? { ...prev, phase: 'done', result } : { phase: 'done', result })
    })
    socket.on('task_started', () => {
      // 新任务开始：清空上一次 VLN 状态与 Agent-VQA 状态，避免残留误导
      setVlnThought(null)
      setSemanticMap(null)
      setAgentQueryState(null)
    })
    socket.on('log', (entry) => {
      setLogs((prev) => {
        const next = [...prev, entry]
        return next.length > 300 ? next.slice(-300) : next
      })
    })

    return () => socket.disconnect()
  }, [])

  const setMode = useCallback((mode) => {
    socketRef.current?.emit('set_mode', { mode })
  }, [])

  const executeAction = useCallback((action, params = {}) => {
    socketRef.current?.emit('execute_action', { action, params })
  }, [])

  const submitAiTask = useCallback((task) => {
    socketRef.current?.emit('ai_task', { task })
  }, [])

  const submitVlnTask = useCallback((instruction) => {
    socketRef.current?.emit('vln_task', { instruction })
  }, [])

  const submitAgentQuery = useCallback((question) => {
    socketRef.current?.emit('agent_query', { question })
  }, [])

  const stopExecution = useCallback(() => {
    socketRef.current?.emit('stop_execution')
  }, [])

  return {
    connected,
    systemStatus,
    worldState,
    logs,
    lastActionResult,
    lastAiPlan,
    lastAiReport,
    lastPerception,
    semanticMap,
    vlnThought,
    agentQueryState,
    setMode,
    executeAction,
    submitAiTask,
    submitVlnTask,
    submitAgentQuery,
    stopExecution,
  }
}

