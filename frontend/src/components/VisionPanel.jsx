import { useEffect, useState } from 'react'

export default function VisionPanel() {
  const [runtimeConfig, setRuntimeConfig] = useState(null)
  const [file, setFile] = useState(null)
  const [prompt, setPrompt] = useState('请分析这张图片中的灾害迹象、关键风险、可通行区域和值得标记的目标。')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    let ignore = false

    const loadConfig = async () => {
      try {
        const response = await fetch('/api/llm/config')
        if (!response.ok) return
        const payload = await response.json()
        if (!ignore) {
          setRuntimeConfig(payload.modules?.vlm || null)
        }
      } catch {
        // Keep the panel usable even if config fetch fails.
      }
    }

    loadConfig()
    return () => {
      ignore = true
    }
  }, [])

  const handleAnalyze = async () => {
    if (!file || busy) return

    setBusy(true)
    setError('')

    try {
      const formData = new FormData()
      formData.append('image', file)
      formData.append('prompt', prompt)

      const response = await fetch('/api/vlm/analyze', {
        method: 'POST',
        body: formData,
      })

      const payload = await response.json()
      if (!response.ok || !payload.ok) {
        throw new Error(payload.error || 'VLM analysis failed')
      }

      setResult(payload)
    } catch (err) {
      setResult(null)
      setError(err instanceof Error ? err.message : 'VLM analysis failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={{ padding: 16, borderRadius: 18, background: 'rgba(255,255,255,0.54)', border: '1px solid rgba(171,152,117,0.18)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center', marginBottom: 10 }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--accent)' }}>Vision Upload</div>
        <div style={{ color: 'var(--ink-soft)', fontSize: 12 }}>
          {runtimeConfig ? `${runtimeConfig.provider} / ${runtimeConfig.model}` : 'VLM config loading'}
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <input
          type="file"
          accept="image/*"
          onChange={(event) => setFile(event.target.files?.[0] || null)}
        />

        <div style={{ color: 'var(--ink-soft)', fontSize: 12 }}>
          {file ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB` : 'Upload a local image and send it to Qwen2-VL.'}
        </div>

        <textarea
          rows={5}
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          placeholder="Describe what you want the vision model to focus on."
        />

        <button className="btn btn-warm" onClick={handleAnalyze} disabled={!file || busy}>
          {busy ? 'Analyzing...' : 'Analyze Image'}
        </button>
      </div>

      {error && (
        <div style={{ marginTop: 12, color: 'var(--danger)', fontSize: 13, lineHeight: 1.6 }}>
          {error}
        </div>
      )}

      {result && (
        <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ color: 'var(--ink-soft)', fontSize: 12 }}>
            {result.model} via {result.provider_url} · {result.image_input_mode}
          </div>
          <pre style={{ margin: 0, whiteSpace: 'pre-wrap', fontFamily: 'var(--font-mono)', fontSize: 12, lineHeight: 1.7, color: 'var(--ink)' }}>
            {result.analysis}
          </pre>
        </div>
      )}
    </div>
  )
}
