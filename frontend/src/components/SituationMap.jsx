import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import L from 'leaflet'
import {
  CircleMarker,
  GeoJSON,
  ImageOverlay,
  LayersControl,
  MapContainer,
  Marker,
  Pane,
  Polygon,
  Polyline,
  Popup,
  TileLayer,
  Tooltip,
  useMap,
  useMapEvents,
} from 'react-leaflet'

const { BaseLayer, Overlay } = LayersControl

const MAX_TILE_OPTIONS = 400
const TRAIL_MAX_POINTS = 600

const DAMAGE_COLOR = {
  'no-damage': '#4ade80',
  minor: '#fde047',
  'minor-damage': '#fde047',
  major: '#fb923c',
  'major-damage': '#fb923c',
  destroyed: '#ef4444',
  'un-classified': '#94a3b8',
  unclassified: '#94a3b8',
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max)
}

function damageColor(level) {
  if (!level) return '#60a5fa'
  return DAMAGE_COLOR[String(level).toLowerCase()] || '#60a5fa'
}

function latLonToMeters(anchor, lat, lon) {
  if (!anchor) return { north_m: 0, east_m: 0 }
  return {
    north_m: (lat - anchor.lat) * 110540.0,
    east_m: (lon - anchor.lon) * 111320.0 * Math.cos((anchor.lat * Math.PI) / 180),
  }
}

function formatCoord(value, digits = 6) {
  return Number.isFinite(value) ? value.toFixed(digits) : '--'
}

function pickInitialDisaster(byDisaster) {
  const keys = Object.keys(byDisaster || {})
  if (!keys.length) return ''
  const preferred = ['hurricane-florence', 'socal-fire', 'hurricane-michael', 'midwest-flooding']
  const hit = preferred.find((name) => keys.includes(name))
  return hit || keys.sort((a, b) => a.localeCompare(b))[0]
}

function tileBoundsToLatLng(bounds) {
  if (!bounds) return null
  const south = Number(bounds.south)
  const north = Number(bounds.north)
  const west = Number(bounds.west)
  const east = Number(bounds.east)
  if (![south, north, west, east].every(Number.isFinite)) return null
  return [
    [south, west],
    [north, east],
  ]
}

function formatTileLabel(item) {
  if (!item) return 'No tile'
  const stage = (item.stage || 'tile').toUpperCase()
  return `${item.disaster || 'unknown'} · ${stage} · ${item.split || 'split'}`
}

function buildUavIcon() {
  return L.divIcon({
    className: 'dc-uav-icon',
    iconSize: [28, 28],
    iconAnchor: [14, 14],
    html: `
      <div style="
        width:28px;height:28px;border-radius:50%;
        background:rgba(15,118,110,0.22);
        display:flex;align-items:center;justify-content:center;
        box-shadow:0 0 0 2px rgba(15,118,110,0.55);
      ">
        <div style="
          width:0;height:0;
          border-left:8px solid transparent;
          border-right:8px solid transparent;
          border-bottom:14px solid #0f766e;
          transform:rotate(0deg);
        "></div>
      </div>`,
  })
}

function MapClickHandler({ onMapClick }) {
  useMapEvents({
    click(event) {
      onMapClick(event.latlng)
    },
  })
  return null
}

function MouseTracker({ onMove, onLeave }) {
  useMapEvents({
    mousemove(event) {
      onMove(event.latlng)
    },
    mouseout() {
      if (onLeave) onLeave()
    },
  })
  return null
}

function DoubleClickHandler({ onDoubleClick }) {
  useMapEvents({
    dblclick() {
      if (onDoubleClick) onDoubleClick()
    },
  })
  return null
}

function AutoFitToTile({ tileBounds, signalKey }) {
  const map = useMap()
  const lastKeyRef = useRef(null)
  useEffect(() => {
    if (!tileBounds || lastKeyRef.current === signalKey) return
    const latLngBounds = L.latLngBounds(tileBounds[0], tileBounds[1])
    if (!latLngBounds.isValid()) return
    map.fitBounds(latLngBounds, { padding: [40, 40], maxZoom: 19 })
    lastKeyRef.current = signalKey
  }, [map, tileBounds, signalKey])
  return null
}

function InvalidateOnResize({ boxRef }) {
  const map = useMap()
  useEffect(() => {
    const node = boxRef.current
    if (!node) return undefined
    const observer = new ResizeObserver(() => {
      window.requestAnimationFrame(() => map.invalidateSize())
    })
    observer.observe(node)
    return () => observer.disconnect()
  }, [map, boxRef])
  return null
}

export default function SituationMap({
  worldState,
  selectedPoint,
  onSelectPoint,
  onFlyToPoint,
  onInspectPoint,
}) {
  const boxRef = useRef(null)

  const anchor = worldState?.map?.anchor
  const activeTile = worldState?.map?.active_tile || null
  const activeTileId = worldState?.map?.active_tile_id || null
  const basemap = worldState?.map?.basemap || null
  const robot = worldState?.robots?.UAV_1
  const targets = Array.isArray(worldState?.targets) ? worldState.targets : []
  const reports = Array.isArray(worldState?.map?.reports) ? worldState.map.reports : []

  const [catalogSummary, setCatalogSummary] = useState(null)
  // POST-only 模式下 stage 永远是 post，保留字段以兼容 catalog 请求参数
  const [filters, setFilters] = useState({ split: '', disaster: '', stage: 'post' })
  const [tileList, setTileList] = useState([])
  const [annotations, setAnnotations] = useState({ features: [] })
  const [footprints, setFootprints] = useState(null)
  const [loadingCatalog, setLoadingCatalog] = useState(false)
  const [loadingTile, setLoadingTile] = useState(false)
  const [loadingAnnotations, setLoadingAnnotations] = useState(false)
  const [tileError, setTileError] = useState('')
  const [imageOpacity, setImageOpacity] = useState(0.85)
  const [showAnnotations, setShowAnnotations] = useState(true)
  const [showFootprints, setShowFootprints] = useState(true)
  const [trail, setTrail] = useState([])
  const [hoverLatLng, setHoverLatLng] = useState(null)
  const [hoverElevation, setHoverElevation] = useState(null)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [damageRanking, setDamageRanking] = useState([])
  const [damageRankingError, setDamageRankingError] = useState('')
  const [selectedRankingTile, setSelectedRankingTile] = useState('')
  const filterSeeded = useRef(false)
  const elevationTimerRef = useRef(null)

  const toggleFullscreen = useCallback(() => {
    setIsFullscreen((prev) => !prev)
  }, [])

  useEffect(() => {
    if (!isFullscreen) return undefined
    const onKey = (event) => {
      if (event.key === 'Escape') setIsFullscreen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [isFullscreen])

  const uavIcon = useMemo(buildUavIcon, [])

  const tileLatLngBounds = useMemo(
    () => tileBoundsToLatLng(activeTile?.bounds),
    [activeTile?.bounds],
  )

  // Catalog summary (for disaster filter options)
  useEffect(() => {
    let cancelled = false
    const loadSummary = async () => {
      try {
        const response = await fetch('/api/xbd/catalog?georef=true&limit=1')
        if (!response.ok) return
        const data = await response.json()
        if (cancelled || !data.ok) return
        setCatalogSummary(data.summary || null)
        if (!filterSeeded.current) {
          const initialDisaster =
            activeTile?.disaster ||
            pickInitialDisaster(data.summary?.by_disaster || {})
          if (initialDisaster) {
            filterSeeded.current = true
            setFilters((prev) => ({
              ...prev,
              disaster: initialDisaster,
              stage: activeTile?.stage || prev.stage,
            }))
          }
        }
      } catch (_error) {
        if (!cancelled) setTileError('读取 xBD 摘要失败')
      }
    }
    loadSummary()
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Tile list whenever filter changes
  useEffect(() => {
    let cancelled = false
    const params = new URLSearchParams({
      georef: 'true',
      limit: String(MAX_TILE_OPTIONS),
    })
    if (filters.split) params.set('split', filters.split)
    if (filters.disaster) params.set('disaster', filters.disaster)
    if (filters.stage) params.set('stage', filters.stage)

    const run = async () => {
      setLoadingCatalog(true)
      try {
        const response = await fetch(`/api/xbd/catalog?${params.toString()}`)
        const data = await response.json()
        if (cancelled) return
        if (!response.ok || !data.ok) {
          throw new Error(data.error || '读取 xBD 瓦片列表失败')
        }
        setCatalogSummary(data.summary || null)
        setTileList(Array.isArray(data.items) ? data.items : [])
        setTileError('')
      } catch (error) {
        if (!cancelled) {
          setTileList([])
          setTileError(error.message || '读取 xBD 瓦片列表失败')
        }
      } finally {
        if (!cancelled) setLoadingCatalog(false)
      }
    }
    run()
    return () => {
      cancelled = true
    }
  }, [filters.disaster, filters.split, filters.stage])

  // Footprints
  useEffect(() => {
    let cancelled = false
    // 带时间戳打破浏览器缓存，避免吃到旧版（含 PRE features）的 footprints
    fetch(`/api/xbd/footprints.geojson?_=${Date.now()}`, {
      cache: 'no-store',
    })
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => {
        if (cancelled) return
        // 防御性：即便缓存/旧后端返回了 PRE，前端也把它过滤掉
        if (data && Array.isArray(data.features)) {
          const kept = data.features.filter((feat) => {
            const stage = String(feat?.properties?.stage || '').toLowerCase()
            return stage === 'post_disaster' || stage === 'post'
          })
          data = { ...data, features: kept }
        }
        setFootprints(data)
      })
      .catch(() => {
        if (!cancelled) setFootprints(null)
      })
    return () => {
      cancelled = true
    }
  }, [])

  // Damage ranking (top destroyed tiles) — effect only; handler defined below activateTile
  useEffect(() => {
    let cancelled = false
    fetch(`/api/xbd/damage-ranking?limit=30&_=${Date.now()}`, { cache: 'no-store' })
      .then(async (response) => {
        const data = await response.json().catch(() => null)
        if (cancelled) return
        if (!response.ok || !data?.ok) {
          setDamageRankingError(data?.error || '未生成 damage_ranking.json')
          setDamageRanking([])
          return
        }
        setDamageRankingError('')
        setDamageRanking(Array.isArray(data.items) ? data.items : [])
      })
      .catch((err) => {
        if (cancelled) return
        setDamageRankingError(String(err?.message || err))
        setDamageRanking([])
      })
    return () => {
      cancelled = true
    }
  }, [])

  // Annotations for active tile
  useEffect(() => {
    if (!activeTileId) {
      setAnnotations({ features: [] })
      return undefined
    }
    let cancelled = false
    setLoadingAnnotations(true)
    fetch(`/api/xbd/annotations/${activeTileId}`)
      .then(async (response) => {
        const data = await response.json().catch(() => null)
        if (cancelled) return
        if (!response.ok || !data?.ok) {
          setAnnotations({ features: [] })
          return
        }
        setAnnotations(data.geojson || { features: [] })
      })
      .catch(() => {
        if (!cancelled) setAnnotations({ features: [] })
      })
      .finally(() => {
        if (!cancelled) setLoadingAnnotations(false)
      })
    return () => {
      cancelled = true
    }
  }, [activeTileId])

  // UAV trail
  useEffect(() => {
    const lat = robot?.position?.lat
    const lon = robot?.position?.lon
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return
    setTrail((prev) => {
      const last = prev[prev.length - 1]
      if (last && Math.abs(last[0] - lat) < 1e-6 && Math.abs(last[1] - lon) < 1e-6) {
        return prev
      }
      const next = [...prev, [lat, lon]]
      return next.length > TRAIL_MAX_POINTS ? next.slice(-TRAIL_MAX_POINTS) : next
    })
  }, [robot?.position?.lat, robot?.position?.lon])

  // Reset trail when active tile changes (local NED frame shift)
  useEffect(() => {
    setTrail([])
  }, [activeTileId])

  const activateTile = useCallback(async (tileId) => {
    if (!tileId || loadingTile) return
    setLoadingTile(true)
    setTileError('')
    try {
      const response = await fetch(`/api/xbd/activate/${tileId}`, { method: 'POST' })
      const data = await response.json()
      if (!response.ok || !data.ok) {
        throw new Error(data.error || '激活 xBD 瓦片失败')
      }
    } catch (error) {
      setTileError(error.message || '激活 xBD 瓦片失败')
    } finally {
      setLoadingTile(false)
    }
  }, [loadingTile])

  const handlePickRankingTile = useCallback(async (tileId) => {
    if (!tileId) return
    const row = damageRanking.find((item) => item.tile_id === tileId)
    if (!row) return
    const lat = Number(row.center?.lat)
    const lon = Number(row.center?.lon)
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return
    setSelectedRankingTile(tileId)
    await activateTile(tileId)
    const offsets = latLonToMeters(anchor, lat, lon)
    if (onSelectPoint) {
      onSelectPoint({
        lat,
        lon,
        north_m: offsets.north_m,
        east_m: offsets.east_m,
      })
    }
    if (onFlyToPoint) {
      onFlyToPoint({ lat, lon })
    }
  }, [damageRanking, anchor, onSelectPoint, onFlyToPoint, activateTile])

  const handleMapClick = useCallback(
    (latlng) => {
      if (!latlng) return
      const offsets = latLonToMeters(anchor, latlng.lat, latlng.lng)
      onSelectPoint({
        lat: latlng.lat,
        lon: latlng.lng,
        north_m: offsets.north_m,
        east_m: offsets.east_m,
      })
    },
    [anchor, onSelectPoint],
  )

  const scheduleElevation = useCallback((latlng) => {
    if (!latlng) return
    if (elevationTimerRef.current) {
      clearTimeout(elevationTimerRef.current)
    }
    elevationTimerRef.current = setTimeout(async () => {
      try {
        const response = await fetch(
          `/api/elevation?lat=${latlng.lat.toFixed(6)}&lon=${latlng.lng.toFixed(6)}`,
        )
        const data = await response.json()
        if (data?.ok && Number.isFinite(data.elevation)) {
          setHoverElevation(data.elevation)
        } else {
          setHoverElevation(null)
        }
      } catch (_error) {
        setHoverElevation(null)
      }
    }, 220)
  }, [])

  const handleHoverMove = useCallback(
    (latlng) => {
      setHoverLatLng(latlng)
      scheduleElevation(latlng)
    },
    [scheduleElevation],
  )

  const handleHoverLeave = useCallback(() => {
    if (elevationTimerRef.current) {
      clearTimeout(elevationTimerRef.current)
      elevationTimerRef.current = null
    }
    setHoverLatLng(null)
    setHoverElevation(null)
  }, [])

  useEffect(() => () => {
    if (elevationTimerRef.current) clearTimeout(elevationTimerRef.current)
  }, [])

  const disasterOptions = useMemo(
    () => Object.entries(catalogSummary?.by_disaster || {}).sort((a, b) => a[0].localeCompare(b[0])),
    [catalogSummary],
  )

  const annotationPolygons = useMemo(() => {
    if (!showAnnotations || !annotations?.features?.length) return []
    const out = []
    for (const feature of annotations.features) {
      const geometry = feature?.geometry
      const properties = feature?.properties || {}
      if (!geometry?.coordinates) continue
      const pushRing = (ring) => {
        const latlngs = (ring || [])
          .map((coord) => {
            if (!Array.isArray(coord) || coord.length < 2) return null
            return [coord[1], coord[0]]
          })
          .filter(Boolean)
        if (latlngs.length >= 3) {
          out.push({ latlngs, properties })
        }
      }
      if (geometry.type === 'Polygon') {
        pushRing(geometry.coordinates[0])
      } else if (geometry.type === 'MultiPolygon') {
        for (const polygon of geometry.coordinates) pushRing(polygon[0])
      }
    }
    return out
  }, [annotations, showAnnotations])

  // 后端已启用 POST_ONLY_MODE，footprints 只包含 POST 灾后瓦片
  const footprintStyle = useCallback(
    (feature) => {
      const id = feature?.properties?.tile_id
      const isActive = id && id === activeTileId
      if (isActive) {
        return {
          color: '#dc2626',
          weight: 3,
          opacity: 0.98,
          fillColor: '#ef4444',
          fillOpacity: 0.22,
        }
      }
      return {
        color: '#dc2626',
        weight: 1.4,
        opacity: 0.85,
        fillColor: '#ef4444',
        fillOpacity: 0.14,
      }
    },
    [activeTileId],
  )

  const footprintKey = useMemo(
    () => `footprints-${activeTileId || 'none'}`,
    [activeTileId],
  )

  const onEachFootprint = useCallback(
    (feature, layer) => {
      const props = feature?.properties || {}
      const tileId = props.tile_id
      const stage = String(props.stage || '').toLowerCase()
      const isPost = stage === 'post_disaster' || stage === 'post'
      const label = `${props.disaster || '?'} · POST · ${props.split || '?'}`
      layer.bindTooltip(`🟥 可检测<br/>${label}<br/>${tileId || ''}`, { sticky: true })
      layer.on('click', () => {
        if (!tileId || !isPost) return
        if (tileId !== activeTileId) activateTile(tileId)
      })
    },
    [activateTile, activeTileId],
  )

  // 预构建 POST footprints 的 bbox 列表（只算一次），供 selectedPoint 快速判断
  const postBBoxes = useMemo(() => {
    const out = []
    const feats = footprints?.features
    if (!Array.isArray(feats)) return out
    for (const feature of feats) {
      const props = feature?.properties || {}
      const stage = String(props.stage || '').toLowerCase()
      if (stage !== 'post_disaster' && stage !== 'post') continue
      const geom = feature.geometry
      if (!geom || !Array.isArray(geom.coordinates)) continue
      const rings = geom.type === 'Polygon'
        ? geom.coordinates
        : geom.type === 'MultiPolygon'
          ? geom.coordinates.flat()
          : null
      if (!rings || !rings.length) continue
      let west = Infinity
      let south = Infinity
      let east = -Infinity
      let north = -Infinity
      for (const ring of rings) {
        if (!Array.isArray(ring)) continue
        for (const coord of ring) {
          if (!Array.isArray(coord) || coord.length < 2) continue
          const [x, y] = coord
          if (x < west) west = x
          if (y < south) south = y
          if (x > east) east = x
          if (y > north) north = y
        }
      }
      if (!Number.isFinite(west)) continue
      out.push({ tile_id: props.tile_id, west, south, east, north })
    }
    return out
  }, [footprints])

  const selectedPostCoverage = useMemo(() => {
    if (!selectedPoint || !postBBoxes.length) return null
    const { lat, lon } = selectedPoint
    for (const bb of postBBoxes) {
      if (lon >= bb.west && lon <= bb.east && lat >= bb.south && lat <= bb.north) {
        return { covered: true, tile_id: bb.tile_id }
      }
    }
    return { covered: false }
  }, [selectedPoint, postBBoxes])

  const tileCenter = useMemo(() => {
    if (!tileLatLngBounds) return null
    const [south, west] = tileLatLngBounds[0]
    const [north, east] = tileLatLngBounds[1]
    return [(south + north) / 2, (west + east) / 2]
  }, [tileLatLngBounds])

  const initialCenter = useMemo(() => {
    if (tileCenter) return tileCenter
    if (anchor?.lat && anchor?.lon) return [anchor.lat, anchor.lon]
    return [31.2304, 121.4737]
  }, [tileCenter, anchor?.lat, anchor?.lon])

  const sectionStyle = isFullscreen
    ? {
        position: 'fixed',
        inset: 0,
        zIndex: 9500,
        background: '#0b1120',
        borderRadius: 0,
      }
    : { position: 'relative' }

  return (
    <section className="map-stage" style={sectionStyle}>
      <div ref={boxRef} style={{ position: 'absolute', inset: 0 }}>
        <MapContainer
          center={initialCenter}
          zoom={17}
          minZoom={3}
          maxZoom={19}
          zoomControl={false}
          doubleClickZoom={false}
          style={{ width: '100%', height: '100%', background: '#0b1120' }}
          worldCopyJump={false}
        >
          <InvalidateOnResize boxRef={boxRef} />

          <LayersControl position="topright">
            <BaseLayer checked name="Esri World Imagery">
              <TileLayer
                url={
                  basemap?.url ||
                  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
                }
                attribution={basemap?.attribution || 'Tiles © Esri'}
                maxZoom={basemap?.max_zoom || 19}
              />
            </BaseLayer>
            <BaseLayer name="OpenStreetMap">
              <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution="© OpenStreetMap contributors"
                maxZoom={19}
              />
            </BaseLayer>
            <BaseLayer name="Carto Dark">
              <TileLayer
                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"
                attribution="© OpenStreetMap, © CARTO"
                maxZoom={19}
              />
            </BaseLayer>
          </LayersControl>

          <Pane name="xbd-image" style={{ zIndex: 410 }} />
          <Pane name="xbd-annotations" style={{ zIndex: 430 }} />
          <Pane name="xbd-footprints" style={{ zIndex: 420 }} />
          <Pane name="operator" style={{ zIndex: 460 }} />

          {tileLatLngBounds && (
            <ImageOverlay
              key={`img-${activeTileId}`}
              url={`/api/xbd/images/${activeTileId}`}
              bounds={tileLatLngBounds}
              opacity={imageOpacity}
              pane="xbd-image"
            />
          )}

          {showFootprints && footprints && (
            <GeoJSON
              key={footprintKey}
              data={footprints}
              style={footprintStyle}
              onEachFeature={onEachFootprint}
              pane="xbd-footprints"
            />
          )}

          {annotationPolygons.map((polygon, idx) => {
            const level =
              polygon.properties?.damage ||
              polygon.properties?.subtype ||
              polygon.properties?.damage_level
            const color = damageColor(level)
            return (
              <Polygon
                key={`ann-${activeTileId}-${idx}`}
                positions={polygon.latlngs}
                pathOptions={{
                  color,
                  weight: 1.2,
                  fillColor: color,
                  fillOpacity: 0.18,
                }}
                pane="xbd-annotations"
              >
                <Tooltip sticky>
                  <div style={{ fontSize: 11, lineHeight: 1.5 }}>
                    <div><strong>{polygon.properties?.feature_type || 'building'}</strong></div>
                    <div>damage: {level || 'n/a'}</div>
                    {polygon.properties?.uid && <div>uid: {polygon.properties.uid}</div>}
                  </div>
                </Tooltip>
              </Polygon>
            )
          })}

          {trail.length > 1 && (
            <Polyline
              positions={trail}
              pathOptions={{ color: '#0f766e', weight: 4, opacity: 0.75 }}
              pane="operator"
            />
          )}

          {targets.map((target) => (
            <CircleMarker
              key={target.target_id}
              center={[target.lat, target.lon]}
              radius={8}
              pathOptions={{
                color: '#b45309',
                weight: 2,
                fillColor: '#f59e0b',
                fillOpacity: 0.85,
              }}
              pane="operator"
            >
              <Tooltip direction="top" offset={[0, -6]}>
                <div style={{ fontSize: 11 }}>
                  <strong>{target.label}</strong>
                  <div>{target.kind}</div>
                  <div>
                    {formatCoord(target.lat, 5)}, {formatCoord(target.lon, 5)}
                  </div>
                </div>
              </Tooltip>
            </CircleMarker>
          ))}

          {reports.map((report) => (
            <CircleMarker
              key={report.id}
              center={[report.lat, report.lon]}
              radius={6}
              pathOptions={{
                color: '#0f766e',
                weight: 2,
                fillColor: '#14b8a6',
                fillOpacity: 0.75,
              }}
              pane="operator"
            >
              <Popup>
                <div style={{ fontSize: 12, maxWidth: 260 }}>
                  <div style={{ fontWeight: 700, marginBottom: 4 }}>
                    {report.level?.toUpperCase() || 'INFO'}
                  </div>
                  <div>{report.content}</div>
                </div>
              </Popup>
            </CircleMarker>
          ))}

          {anchor && (
            <CircleMarker
              center={[anchor.lat, anchor.lon]}
              radius={5}
              pathOptions={{
                color: '#9a3412',
                weight: 2,
                fillColor: '#fdba74',
                fillOpacity: 0.9,
              }}
              pane="operator"
            >
              <Tooltip direction="top" offset={[0, -6]}>
                <div style={{ fontSize: 11 }}>
                  <strong>Anchor</strong>
                  <div>{anchor.label}</div>
                </div>
              </Tooltip>
            </CircleMarker>
          )}

          {robot?.position?.lat && robot?.position?.lon && (
            <Marker
              position={[robot.position.lat, robot.position.lon]}
              icon={uavIcon}
              pane="operator"
            >
              <Tooltip direction="top" offset={[0, -16]}>
                <div style={{ fontSize: 11 }}>
                  <strong>UAV_1</strong>
                  <div>alt {Number(robot.position.alt || 0).toFixed(1)}m</div>
                  <div>
                    {formatCoord(robot.position.lat, 5)}, {formatCoord(robot.position.lon, 5)}
                  </div>
                </div>
              </Tooltip>
            </Marker>
          )}

          {selectedPoint && Number.isFinite(Number(selectedPoint.lat)) && Number.isFinite(Number(selectedPoint.lon)) && (
            <CircleMarker
              center={[Number(selectedPoint.lat), Number(selectedPoint.lon)]}
              radius={10}
              pathOptions={{
                color: '#b91c1c',
                weight: 2,
                dashArray: '6 4',
                fillColor: '#fecaca',
                fillOpacity: 0.4,
              }}
              pane="operator"
            />
          )}

          <AutoFitToTile tileBounds={tileLatLngBounds} signalKey={activeTileId} />
          <MapClickHandler onMapClick={handleMapClick} />
          <MouseTracker onMove={handleHoverMove} onLeave={handleHoverLeave} />
          <DoubleClickHandler onDoubleClick={toggleFullscreen} />
        </MapContainer>
      </div>

      {/* ───────── Top HUD: filters + tile select + overlays ───────── */}
      <div
        style={{
          position: 'absolute',
          top: 12,
          left: 12,
          right: 12,
          display: 'grid',
          gridTemplateColumns: 'minmax(0, 1.1fr) minmax(220px, 1fr)',
          gap: 12,
          pointerEvents: 'none',
          zIndex: 500,
        }}
      >
        <div style={hudCardStyle}>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 8 }}>
            <select
              value={filters.disaster}
              onChange={(event) => setFilters((prev) => ({ ...prev, disaster: event.target.value }))}
              style={{ flex: '1 1 160px', minWidth: 140, ...selectStyle }}
            >
              <option value="">全部灾区</option>
              {disasterOptions.map(([name, stats]) => (
                <option key={name} value={name}>
                  {`${name} (${stats.has_georef || 0}/${stats.tiles || 0})`}
                </option>
              ))}
            </select>
            <span
              style={{
                padding: '6px 10px',
                borderRadius: 8,
                border: '1px solid #dc2626',
                background: 'rgba(239,68,68,0.10)',
                color: '#b91c1c',
                fontSize: 11,
                fontWeight: 700,
                letterSpacing: '0.04em',
                display: 'inline-flex',
                alignItems: 'center',
              }}
              title="POST_ONLY_MODE：系统已过滤所有灾前瓦片"
            >
              POST-only
            </span>
            <select
              value={filters.split}
              onChange={(event) => setFilters((prev) => ({ ...prev, split: event.target.value }))}
              style={{ width: 100, ...selectStyle }}
            >
              <option value="">All Split</option>
              <option value="train">Train</option>
              <option value="test">Test</option>
              <option value="tier3">Tier3</option>
            </select>
          </div>

          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <select
              value={activeTileId || ''}
              onChange={(event) => activateTile(event.target.value)}
              style={{ flex: '1 1 240px', minWidth: 200, ...selectStyle }}
            >
              <option value="" disabled>
                {loadingCatalog ? '载入瓦片列表中...' : '选择瓦片'}
              </option>
              {tileList.map((item) => (
                <option key={item.tile_id} value={item.tile_id}>
                  {`${formatTileLabel(item)} · ${item.tile_id}`}
                </option>
              ))}
            </select>
            <button
              className="btn btn-soft"
              onClick={() => setTrail([])}
              style={hudButtonStyle}
            >
              清轨迹
            </button>
            <button
              className="btn btn-soft"
              onClick={toggleFullscreen}
              style={{
                ...hudButtonStyle,
                background: isFullscreen ? '#fde68a' : '#fffdf7',
              }}
              title="双击地图也可切换"
            >
              {isFullscreen ? '退出全屏 (Esc)' : '地图全屏'}
            </button>
          </div>

          <div
            style={{
              marginTop: 10,
              paddingTop: 8,
              borderTop: '1px dashed rgba(220,38,38,0.35)',
            }}
          >
            <div
              style={{
                fontSize: 11,
                fontWeight: 700,
                color: '#b91c1c',
                letterSpacing: '0.04em',
                marginBottom: 6,
              }}
            >
              ⚠ 最严重受灾瓦片 · Top {damageRanking.length || '—'}
            </div>
            {damageRankingError ? (
              <div style={{ fontSize: 11, color: '#b91c1c' }}>
                {damageRankingError}
                <br />
                运行 <code>python scripts/rank_damage_tiles.py</code> 生成排名。
              </div>
            ) : (
              <select
                value={selectedRankingTile}
                onChange={(event) => handlePickRankingTile(event.target.value)}
                disabled={loadingTile || damageRanking.length === 0}
                style={{ width: '100%', ...selectStyle }}
              >
                <option value="" disabled>
                  {damageRanking.length === 0
                    ? '载入排名中…'
                    : '选择瓦片 → 自动激活 + 飞到中心'}
                </option>
                {damageRanking.map((row) => {
                  const c = row.counts || {}
                  const pct = Math.round((row.destroyed_ratio || 0) * 100)
                  return (
                    <option key={row.tile_id} value={row.tile_id}>
                      {`#${row.rank} · dest=${c.destroyed || 0}/${c.total_buildings || 0} (${pct}%) · ${row.disaster || '?'} · ${row.tile_id}`}
                    </option>
                  )
                })}
              </select>
            )}
            <div style={{ fontSize: 10, color: '#6b7280', marginTop: 4 }}>
              得分 = destroyed×5 + major×3 + minor×1；选中后会自动激活瓦片并让 UAV 飞到中心。
            </div>
          </div>
        </div>

        <div style={{ ...hudCardStyle, borderColor: 'rgba(234,88,12,0.35)' }}>
          <div style={{ fontSize: 12, color: '#9a3412', fontWeight: 700, marginBottom: 6 }}>
            {activeTile ? formatTileLabel(activeTile) : '未激活瓦片'}
          </div>
          <div style={metaRow}>
            Tile: <span>{activeTileId || 'none'}</span>
          </div>
          <div style={metaRow}>
            Type: <span>{activeTile?.disaster_type || '—'}</span> | Sensor:{' '}
            <span>{activeTile?.sensor || '—'}</span>
          </div>
          <div style={metaRow}>
            GSD:{' '}
            <span>
              {activeTile?.gsd ? `${Number(activeTile.gsd).toFixed(3)}m` : '—'}
            </span>{' '}
            | Fit:{' '}
            <span>
              {activeTile?.fit?.rms_error_m
                ? `${Number(activeTile.fit.rms_error_m).toFixed(2)}m`
                : '—'}
            </span>
          </div>
          <div style={metaRow}>
            Anchor:{' '}
            <span>
              {anchor
                ? `${formatCoord(anchor.lat, 5)}, ${formatCoord(anchor.lon, 5)}`
                : '—'}
            </span>
          </div>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 8 }}>
            <label style={toggleStyle}>
              <input
                type="checkbox"
                checked={showAnnotations}
                onChange={(event) => setShowAnnotations(event.target.checked)}
              />
              标注
              {loadingAnnotations && (
                <span style={{ color: '#b45309' }}> · 载入中</span>
              )}
            </label>
            <label style={toggleStyle}>
              <input
                type="checkbox"
                checked={showFootprints}
                onChange={(event) => setShowFootprints(event.target.checked)}
              />
              瓦片足迹
            </label>
            <span style={{ ...toggleStyle, gap: 4, cursor: 'default' }}>
              <span style={{
                display: 'inline-block',
                width: 10,
                height: 10,
                background: '#ef4444',
                border: '1px solid #dc2626',
                borderRadius: 2,
              }}
              />
              POST 灾后（可检测）
            </span>
            <label style={{ ...toggleStyle, gap: 6 }}>
              影像透明度
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={imageOpacity}
                onChange={(event) =>
                  setImageOpacity(clamp(Number(event.target.value), 0, 1))
                }
                style={{ width: 84 }}
              />
              <span style={{ fontFamily: 'monospace', width: 28 }}>
                {imageOpacity.toFixed(2)}
              </span>
            </label>
          </div>
        </div>
      </div>

      {/* ───────── Bottom-left: cursor target + elevation ───────── */}
      <div
        style={{
          position: 'absolute',
          left: 16,
          bottom: 16,
          padding: 14,
          borderRadius: 16,
          background: 'rgba(255,255,255,0.9)',
          border: '1px solid rgba(171,152,117,0.28)',
          minWidth: 300,
          zIndex: 500,
          boxShadow: '0 10px 28px rgba(15,23,42,0.16)',
        }}
      >
        <div
          style={{
            fontSize: 12,
            fontWeight: 700,
            color: 'var(--accent)',
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
            marginBottom: 8,
          }}
        >
          Cursor Target
        </div>

        {selectedPoint ? (
          <>
            <div style={{ fontSize: 13, color: 'var(--ink-soft)', lineHeight: 1.6 }}>
              {formatCoord(selectedPoint.lat)}, {formatCoord(selectedPoint.lon)}
            </div>
            <div style={{ fontSize: 13, color: 'var(--ink-soft)' }}>
              {Number(selectedPoint.north_m || 0).toFixed(1)}m N /{' '}
              {Number(selectedPoint.east_m || 0).toFixed(1)}m E
            </div>
            {selectedPostCoverage && (
              <div
                style={{
                  marginTop: 6,
                  display: 'inline-block',
                  padding: '2px 8px',
                  borderRadius: 10,
                  fontSize: 11,
                  fontWeight: 700,
                  color: selectedPostCoverage.covered ? '#b91c1c' : '#6b7280',
                  background: selectedPostCoverage.covered
                    ? 'rgba(239,68,68,0.12)'
                    : 'rgba(107,114,128,0.12)',
                  border: `1px solid ${selectedPostCoverage.covered ? '#dc2626' : '#9ca3af'}`,
                }}
                title={
                  selectedPostCoverage.covered
                    ? `位于 POST 瓦片 ${selectedPostCoverage.tile_id}，可 detect_disaster`
                    : 'detect_disaster 需要 POST 灾后瓦片覆盖，此点不满足'
                }
              >
                {selectedPostCoverage.covered ? '✓ 在 POST 覆盖内' : '✗ 无 POST 覆盖'}
              </div>
            )}
            <div
              style={{
                marginTop: 8,
                fontSize: 12,
                color: 'var(--ink-soft)',
                fontStyle: 'italic',
              }}
            >
              在右侧 Task Console · Selected Point 下达 Fly / Mark / Ask AI Inspect。
            </div>
          </>
        ) : (
          <div style={{ color: 'var(--ink-soft)', fontSize: 13 }}>
            在地图上点击目标点（或点击瓦片足迹切换灾区）。
          </div>
        )}

        <div
          style={{
            marginTop: 10,
            paddingTop: 10,
            borderTop: '1px dashed rgba(171,152,117,0.35)',
            fontSize: 12,
            color: 'var(--ink-soft)',
            lineHeight: 1.55,
          }}
        >
          <div style={{ fontWeight: 600, color: 'var(--accent)', marginBottom: 2 }}>
            Hover
          </div>
          <div>
            {hoverLatLng
              ? `${formatCoord(hoverLatLng.lat, 5)}, ${formatCoord(hoverLatLng.lng, 5)}`
              : '— 移动光标读取坐标 / 高程'}
          </div>
          <div>
            elevation:{' '}
            <span style={{ fontFamily: 'monospace' }}>
              {Number.isFinite(hoverElevation) ? `${hoverElevation.toFixed(1)} m` : '—'}
            </span>
          </div>
        </div>
      </div>

      {tileError && (
        <div
          style={{
            position: 'absolute',
            right: 16,
            bottom: 16,
            maxWidth: 360,
            padding: '10px 12px',
            borderRadius: 10,
            background: 'rgba(127,29,29,0.9)',
            color: '#fee2e2',
            fontSize: 12,
            zIndex: 500,
          }}
        >
          {tileError}
        </div>
      )}

      {(loadingTile || loadingCatalog) && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            pointerEvents: 'none',
            zIndex: 499,
          }}
        >
          <div
            style={{
              padding: '8px 14px',
              borderRadius: 999,
              background: 'rgba(15,23,42,0.7)',
              color: '#e2e8f0',
              fontSize: 12,
            }}
          >
            {loadingTile ? '激活灾害瓦片中...' : '载入瓦片列表中...'}
          </div>
        </div>
      )}
    </section>
  )
}

const hudCardStyle = {
  pointerEvents: 'auto',
  padding: 12,
  borderRadius: 12,
  background: 'rgba(255,255,255,0.92)',
  border: '1px solid rgba(171,152,117,0.28)',
  boxShadow: '0 6px 20px rgba(15,23,42,0.12)',
  backdropFilter: 'blur(6px)',
}

const selectStyle = {
  padding: '6px 8px',
  borderRadius: 8,
  border: '1px solid rgba(171,152,117,0.4)',
  background: '#fffdf7',
  color: '#1f2937',
  fontSize: 12,
}

const hudButtonStyle = {
  padding: '6px 12px',
  borderRadius: 8,
  border: '1px solid rgba(171,152,117,0.4)',
  background: '#fffdf7',
  color: '#1f2937',
  fontSize: 12,
  cursor: 'pointer',
}

const metaRow = {
  fontSize: 11,
  color: '#57534e',
  fontFamily: 'monospace',
  lineHeight: 1.6,
}

const toggleStyle = {
  display: 'flex',
  alignItems: 'center',
  gap: 4,
  fontSize: 11,
  color: '#57534e',
  cursor: 'pointer',
}
