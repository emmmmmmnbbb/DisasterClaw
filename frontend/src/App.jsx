import { useCallback, useState } from 'react'
import { useSocket } from './hooks/useSocket'
import Header from './components/Header'
import RobotPanel from './components/RobotPanel'
import SituationMap from './components/SituationMap'
import TaskPanel from './components/TaskPanel'
import LogPanel from './components/LogPanel'
import PerceptionPanel from './components/PerceptionPanel'
import AgentQueryPanel from './components/AgentQueryPanel'

export default function App() {
  const {
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
  } = useSocket()

  const [selectedPoint, setSelectedPoint] = useState(null)

  const hoverAltitude = systemStatus.hover_altitude_m || 30

  const handleSubmitAiTask = useCallback((task) => {
    setMode('ai')
    submitAiTask(task)
  }, [setMode, submitAiTask])

  const handleSubmitVlnTask = useCallback((instruction) => {
    setMode('ai')
    submitVlnTask(instruction)
  }, [setMode, submitVlnTask])

  const handleSubmitAgentQuery = useCallback((question) => {
    setMode('ai')
    submitAgentQuery(question)
  }, [setMode, submitAgentQuery])

  const handleFlyToPoint = useCallback((point) => {
    setMode('manual')
    executeAction('fly_to_geo', {
      lat: point.lat,
      lon: point.lon,
      alt: hoverAltitude,
      speed: 14.0,
    })
  }, [executeAction, hoverAltitude, setMode])

  const handleMarkPoint = useCallback((point) => {
    setMode('manual')
    executeAction('mark_target', {
      lat: point.lat,
      lon: point.lon,
      label: 'Manual Marker',
      kind: 'manual-mark',
    })
  }, [executeAction, setMode])

  const handleInspectPoint = useCallback(async (point) => {
    const inLat = Number(point?.lat)
    const inLon = Number(point?.lon)
    if (!Number.isFinite(inLat) || !Number.isFinite(inLon)) {
      window.alert('坐标无效，请重新在地图上选点。')
      return
    }

    let targetLat = inLat
    let targetLon = inLon
    try {
      const response = await fetch(
        `/api/xbd/find-tile?lat=${inLat.toFixed(6)}&lon=${inLon.toFixed(6)}&stage=post`,
      )
      const data = await response.json().catch(() => null)
      if (!response.ok || !data?.ok) {
        window.alert('无法校验 POST 瓦片覆盖，请稍后重试。')
        return
      }
      if (!data.covered) {
        const nearest = data.nearest
        const nLat = Number(nearest?.center?.lat)
        const nLon = Number(nearest?.center?.lon)
        if (!nearest || !Number.isFinite(nLat) || !Number.isFinite(nLon)) {
          window.alert(
            `此位置 (${inLat.toFixed(5)}, ${inLon.toFixed(5)}) 不在任何 POST 灾后瓦片覆盖范围内，` +
            '且 manifest 里也没有找到最近的 POST 瓦片。'
          )
          return
        }
        const go = window.confirm(
          `此位置 (${inLat.toFixed(5)}, ${inLon.toFixed(5)}) 不在 POST 覆盖内。\n\n` +
          `最近的 POST 瓦片：${nearest.tile_id}\n` +
          `灾情：${nearest.disaster || '?'} (${nearest.disaster_type || '?'})\n` +
          `距离：约 ${nearest.distance_km} km\n\n` +
          '点"确定"自动跳转到该瓦片中心并进行 AI 受灾检测。'
        )
        if (!go) return
        targetLat = nLat
        targetLon = nLon
      }
    } catch (_error) {
      window.alert('预检查 POST 瓦片失败，请稍后重试。')
      return
    }

    // 仅在实际跳转（离开原点）时才更新 selectedPoint，避免无意中把
    // 原 selectedPoint 的 north_m / east_m / elevation 等字段冲掉。
    if (targetLat !== inLat || targetLon !== inLon) {
      setSelectedPoint((prev) => ({
        ...(prev || {}),
        lat: targetLat,
        lon: targetLon,
        north_m: undefined,
        east_m: undefined,
        elevation: undefined,
      }))
    }
    const task = (
      `飞到纬度 ${targetLat.toFixed(6)}、经度 ${targetLon.toFixed(6)} 上空 ${hoverAltitude}m 悬停，` +
      `对该位置进行受灾检测与灾情评估（YOLO + SegFormer + VLM），标记目标并汇报结果。`
    )
    handleSubmitAiTask(task)
  }, [handleSubmitAiTask, hoverAltitude, setSelectedPoint])

  const handleHover = useCallback(() => {
    executeAction('hover', { duration: 3.0 })
  }, [executeAction])

  return (
    <div className="app-shell">
      <Header connected={connected} systemStatus={systemStatus} onModeSwitch={setMode} />

      <div className="main-grid">
        <RobotPanel worldState={worldState} systemStatus={systemStatus} />

        <SituationMap
          worldState={worldState}
          selectedPoint={selectedPoint}
          onSelectPoint={setSelectedPoint}
          onFlyToPoint={handleFlyToPoint}
          onInspectPoint={handleInspectPoint}
        />

        <TaskPanel
          systemStatus={systemStatus}
          selectedPoint={selectedPoint}
          lastActionResult={lastActionResult}
          lastAiPlan={lastAiPlan}
          lastAiReport={lastAiReport}
          onSubmitAiTask={handleSubmitAiTask}
          onSubmitVlnTask={handleSubmitVlnTask}
          onSubmitAgentQuery={handleSubmitAgentQuery}
          onStop={stopExecution}
          onHover={handleHover}
          onFlyToPoint={handleFlyToPoint}
          onMarkPoint={handleMarkPoint}
          onInspectPoint={handleInspectPoint}
        />
      </div>

      <div className="bottom-row">
        <PerceptionPanel
          perception={lastPerception}
          semanticMap={semanticMap}
          vlnThought={vlnThought}
        />
        <AgentQueryPanel agentQueryState={agentQueryState} />
        <LogPanel logs={logs} />
      </div>
    </div>
  )
}

