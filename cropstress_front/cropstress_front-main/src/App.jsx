import React, { useState, useEffect, useContext, createContext, useCallback, useRef } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, AreaChart, Area, Legend
} from 'recharts'

// ── API base ──────────────────────────────────────────────
const API_BASE = 'https://crop-stress-detection.onrender.com'

// ── Context ───────────────────────────────────────────────
const AppCtx = createContext({})

// ── Translations ───────────────────────────────────────────
const T = {
  en: {
    login: 'Login', register: 'Register', logout: 'Logout',
    name: 'Full Name', email: 'Email', password: 'Password',
    welcome: 'Welcome to Crop Monitor', tagline: 'AI-powered field monitoring system',
    dashboard: 'Dashboard', upload: 'Upload Image', history: 'History',
    alerts: 'Alerts', analysis: 'AI Analysis', fields: 'Fields',
    cropHealth: 'Crop Health', stressScore: 'Stress Score',
    soilCondition: 'Soil Condition', confidence: 'Confidence',
    temperature: 'Temperature', humidity: 'Humidity',
    weather: 'Live Weather', stressType: 'Stress Type',
    disease: 'Disease', severity: 'Severity',
    explanation: 'Explanation', cause: 'Root Cause',
    remedy: 'First-Aid Remedies', prevention: 'Prevention',
    fertilizer: 'Fertilizer Tip', irrigation: 'Irrigation Advice',
    urgency: 'Urgency Level', uploadImage: 'Upload Field Image',
    chooseFile: 'Choose Image File', analyze: 'Analyze Crop',
    captureESP32: 'Capture from ESP32', orUpload: 'or upload from computer',
    latestReading: 'Latest Reading', trend: 'Stress Trend',
    noData: 'No data yet. Upload an image to begin.',
    loading: 'Loading...', analyzing: 'Analyzing...', error: 'Error',
    healthy: 'Healthy', stressed: 'Stressed', unknown: 'Unknown',
    dry: 'Dry', good_moisture: 'Good Moisture', wet_waterlogged: 'Waterlogged',
    not_detected: 'Not Detected',
    darkMode: 'Dark Mode', lightMode: 'Light Mode',
    language: 'Language', english: 'English', tamil: 'தமிழ்',
    field: 'Field', selectField: 'Select Field', addField: 'Add Field',
    fieldName: 'Field Name', latitude: 'Latitude', longitude: 'Longitude',
    save: 'Save', cancel: 'Cancel',
    alertsPanel: 'Active Alerts', noAlerts: 'No active alerts',
    trendIncreasing: 'Stress is increasing', trendDecreasing: 'Stress is decreasing',
    trendStable: 'Stress is stable', readings: 'readings',
    getAnalysis: 'Get AI Analysis', analysisNote: 'AI will analyze latest crop data with weather',
    noLatest: 'No recent data available for analysis',
    windspeed: 'Wind Speed', weatherCode: 'Condition',
    registerNow: 'Create Account', alreadyHave: 'Already have an account?',
    noAccount: "Don't have an account?", loginNow: 'Login',
    emailForAlerts: 'Email (for alerts)',
    plantType: 'Plant Identified', waterAlert: 'Irrigation Alert',
    drainageAlert: 'Drainage Alert', plantStressAlert: 'Plant Stress Alert',
    combinedAlert: 'Combined Alert',
  },
  ta: {
    login: 'உள்நுழை', register: 'பதிவு செய்', logout: 'வெளியேறு',
    name: 'முழு பெயர்', email: 'மின்னஞ்சல்', password: 'கடவுச்சொல்',
    welcome: 'பயிர் கண்காணிப்பிற்கு வரவேற்கிறோம்', tagline: 'AI-சக்தி வாய்ந்த வயல் கண்காணிப்பு',
    dashboard: 'டாஷ்போர்டு', upload: 'படம் பதிவேற்று', history: 'வரலாறு',
    alerts: 'எச்சரிக்கைகள்', analysis: 'AI பகுப்பாய்வு', fields: 'வயல்கள்',
    cropHealth: 'பயிர் ஆரோக்கியம்', stressScore: 'அழுத்த மதிப்பு',
    soilCondition: 'மண் நிலை', confidence: 'நம்பகத்தன்மை',
    temperature: 'வெப்பநிலை', humidity: 'ஈரப்பதம்',
    weather: 'நேரடி வானிலை', stressType: 'அழுத்த வகை',
    disease: 'நோய்', severity: 'தீவிரம்',
    explanation: 'விளக்கம்', cause: 'மூல காரணம்',
    remedy: 'முதலுதவி நடவடிக்கைகள்', prevention: 'தடுப்பு நடவடிக்கைகள்',
    fertilizer: 'உர ஆலோசனை', irrigation: 'பாசன ஆலோசனை',
    urgency: 'அவசர நிலை', uploadImage: 'வயல் படம் பதிவேற்று',
    chooseFile: 'படக்கோப்பு தேர்ந்தெடு', analyze: 'பயிர் பகுப்பாய்வு',
    captureESP32: 'ESP32 மூலம் படம் எடு', orUpload: 'அல்லது கணினியிலிருந்து பதிவேற்று',
    latestReading: 'கடைசி அளவீடு', trend: 'அழுத்த போக்கு',
    noData: 'தரவு இல்லை. படம் பதிவேற்றி தொடங்குங்கள்.',
    loading: 'ஏற்றுகிறது...', analyzing: 'பகுப்பாய்கிறது...', error: 'பிழை',
    healthy: 'ஆரோக்கியமான', stressed: 'அழுத்தமான', unknown: 'தெரியவில்லை',
    dry: 'உலர்', good_moisture: 'சரியான ஈரப்பதம்', wet_waterlogged: 'அதிக நீர்',
    not_detected: 'கண்டறியவில்லை',
    darkMode: 'இருண்ட பயன்முறை', lightMode: 'ஒளி பயன்முறை',
    language: 'மொழி', english: 'English', tamil: 'தமிழ்',
    field: 'வயல்', selectField: 'வயல் தேர்ந்தெடு', addField: 'வயல் சேர்',
    fieldName: 'வயல் பெயர்', latitude: 'அட்சரேகை', longitude: 'தீர்க்கரேகை',
    save: 'சேமி', cancel: 'ரத்து',
    alertsPanel: 'செயலில் உள்ள எச்சரிக்கைகள்', noAlerts: 'எச்சரிக்கைகள் இல்லை',
    trendIncreasing: 'அழுத்தம் அதிகரிக்கிறது', trendDecreasing: 'அழுத்தம் குறைகிறது',
    trendStable: 'அழுத்தம் நிலையானது', readings: 'அளவீடுகள்',
    getAnalysis: 'AI பகுப்பாய்வு பெறு', analysisNote: 'வானிலையுடன் AI பயிர் தரவை பகுப்பாய்யும்',
    noLatest: 'பகுப்பாய்விற்கு சமீபத்திய தரவு இல்லை',
    windspeed: 'காற்று வேகம்', weatherCode: 'நிலை',
    registerNow: 'கணக்கு உருவாக்கு', alreadyHave: 'கணக்கு உள்ளதா?',
    noAccount: 'கணக்கு இல்லையா?', loginNow: 'உள்நுழை',
    emailForAlerts: 'மின்னஞ்சல் (எச்சரிக்கைகளுக்கு)',
    plantType: 'கண்டறிந்த தாவரம்', waterAlert: 'நீர்ப்பாசன எச்சரிக்கை',
    drainageAlert: 'நீர்வடிகால் எச்சரிக்கை', plantStressAlert: 'தாவர அழுத்த எச்சரிக்கை',
    combinedAlert: 'இணைந்த எச்சரிக்கை',
  }
}

// ── API helper ────────────────────────────────────────────
async function apiFetch(path, opts = {}) {
  const token = localStorage.getItem('cm_token')
  const headers = { 'Content-Type': 'application/json', ...(opts.headers || {}) }
  if (token) headers['Authorization'] = `Bearer ${token}`
  if (opts.body instanceof FormData) delete headers['Content-Type']
  const r = await fetch(API_BASE + path, { ...opts, headers })
  if (r.status === 401) { localStorage.removeItem('cm_token'); window.location.reload(); return null }
  if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || 'Request failed') }
  return r.json()
}

// ── Severity colors ────────────────────────────────────────
const SEV_COLOR = { OK: '#22c55e', WARN: '#f59e0b', ALERT: '#ef4444', CRITICAL: '#7c3aed', INFO: '#6b7280' }
const SEV_BG    = { OK: '#dcfce7', WARN: '#fef9c3', ALERT: '#fee2e2', CRITICAL: '#ede9fe', INFO: '#f1f5f9' }

// ── Helper components ──────────────────────────────────────
function Badge({ label, color = '#22c55e', bg = '#dcfce7' }) {
  return <span style={{ background: bg, color, padding: '2px 10px', borderRadius: 12, fontSize: 12, fontWeight: 600 }}>{label}</span>
}

function Card({ children, style = {} }) {
  return <div style={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 14, padding: '18px 20px', ...style }}>{children}</div>
}

function MetricCard({ label, value, unit = '', color = 'var(--text)' }) {
  return (
    <Card style={{ textAlign: 'center', minWidth: 120 }}>
      <div style={{ fontSize: 13, color: 'var(--muted)', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 26, fontWeight: 700, color }}>{value}<span style={{ fontSize: 14, marginLeft: 3 }}>{unit}</span></div>
    </Card>
  )
}

function Spinner() {
  return <div style={{ display: 'flex', justifyContent: 'center', padding: 40 }}>
    <div className="spinner" />
  </div>
}

// ── Login / Register page ─────────────────────────────────
function AuthPage({ onAuth }) {
  const [mode, setMode] = useState('login')
  const [form, setForm] = useState({ name: '', email: '', password: '' })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const { lang } = useContext(AppCtx)
  const t = T[lang]

  async function submit(e) {
    e.preventDefault()
    setLoading(true); setError('')
    try {
      const path = mode === 'login' ? '/api/auth/login' : '/api/auth/register'
      const data = await apiFetch(path, { method: 'POST', body: JSON.stringify(form) })
      if (data) {
        localStorage.setItem('cm_token', data.token)
        localStorage.setItem('cm_user', JSON.stringify(data.user))
        onAuth(data.user)
      }
    } catch (err) { setError(err.message) }
    setLoading(false)
  }

  return (
    <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--bg)' }}>
      <div style={{ width: '100%', maxWidth: 420, padding: '0 20px' }}>
        <div style={{ textAlign: 'center', marginBottom: 32 }}>
          <div style={{ fontSize: 40 }}>🌱</div>
          <h1 style={{ fontSize: 24, fontWeight: 700, margin: '8px 0 4px' }}>{t.welcome}</h1>
          <p style={{ color: 'var(--muted)', fontSize: 14 }}>{t.tagline}</p>
        </div>
        <Card>
          <h2 style={{ fontSize: 18, marginBottom: 20, textAlign: 'center' }}>
            {mode === 'login' ? t.login : t.register}
          </h2>
          {error && <div style={{ color: '#ef4444', fontSize: 13, marginBottom: 12, padding: '8px 12px', background: '#fee2e2', borderRadius: 8 }}>{error}</div>}
          <form onSubmit={submit} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {mode === 'register' && (
              <input className="inp" placeholder={t.name} value={form.name}
                onChange={e => setForm({ ...form, name: e.target.value })} required />
            )}
            <input className="inp" type="email" placeholder={mode === 'register' ? t.emailForAlerts : t.email}
              value={form.email} onChange={e => setForm({ ...form, email: e.target.value })} required />
            <input className="inp" type="password" placeholder={t.password}
              value={form.password} onChange={e => setForm({ ...form, password: e.target.value })} required />
            <button className="btn-primary" disabled={loading} style={{ marginTop: 4 }}>
              {loading ? t.loading : (mode === 'login' ? t.login : t.registerNow)}
            </button>
          </form>
          <p style={{ textAlign: 'center', marginTop: 16, fontSize: 13, color: 'var(--muted)' }}>
            {mode === 'login' ? t.noAccount : t.alreadyHave}{' '}
            <button onClick={() => { setMode(mode === 'login' ? 'register' : 'login'); setError('') }}
              style={{ color: 'var(--accent)', background: 'none', border: 'none', cursor: 'pointer', fontWeight: 600 }}>
              {mode === 'login' ? t.registerNow : t.loginNow}
            </button>
          </p>
        </Card>
      </div>
    </div>
  )
}

// ── Sidebar ────────────────────────────────────────────────
function Sidebar({ page, setPage, user, onLogout }) {
  const { lang } = useContext(AppCtx)
  const t = T[lang]
  const nav = [
    { id: 'dashboard', icon: '🏠', label: t.dashboard },
    { id: 'upload',    icon: '📷', label: t.upload },
    { id: 'analysis',  icon: '🤖', label: t.analysis },
    { id: 'history',   icon: '📊', label: t.history },
    { id: 'alerts',    icon: '🔔', label: t.alerts },
  ]
  return (
    <aside style={{ width: 220, minHeight: '100vh', background: 'var(--sidebar)', borderRight: '1px solid var(--border)', display: 'flex', flexDirection: 'column', padding: '20px 0' }}>
      <div style={{ padding: '0 20px 20px', borderBottom: '1px solid var(--border)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 28 }}>🌱</span>
          <div>
            <div style={{ fontWeight: 700, fontSize: 15, color: 'var(--accent)' }}>CropMonitor</div>
            <div style={{ fontSize: 11, color: 'var(--muted)' }}>AI Field System</div>
          </div>
        </div>
      </div>
      <nav style={{ flex: 1, padding: '12px 10px' }}>
        {nav.map(n => (
          <button key={n.id} onClick={() => setPage(n.id)}
            style={{ display: 'flex', alignItems: 'center', gap: 10, width: '100%', padding: '10px 12px', borderRadius: 10, border: 'none', cursor: 'pointer', marginBottom: 2, fontSize: 14, fontWeight: page === n.id ? 600 : 400, background: page === n.id ? 'var(--accent-bg)' : 'transparent', color: page === n.id ? 'var(--accent)' : 'var(--text)', transition: 'all 0.15s' }}>
            <span style={{ fontSize: 17 }}>{n.icon}</span>{n.label}
          </button>
        ))}
      </nav>
      <div style={{ padding: '16px 20px', borderTop: '1px solid var(--border)' }}>
        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 2 }}>{user?.name}</div>
        <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 10, wordBreak: 'break-all' }}>{user?.email}</div>
        <button onClick={onLogout} style={{ fontSize: 13, color: '#ef4444', background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}>← {t.logout}</button>
      </div>
    </aside>
  )
}

// ── Topbar ─────────────────────────────────────────────────
function Topbar({ page }) {
  const { lang, setLang, dark, setDark } = useContext(AppCtx)
  const t = T[lang]
  return (
    <header style={{ height: 60, background: 'var(--card)', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', padding: '0 24px', gap: 16, position: 'sticky', top: 0, zIndex: 10 }}>
      <div style={{ flex: 1, fontWeight: 600, fontSize: 16 }}>
        {T[lang][page] || page}
      </div>
      <button onClick={() => setDark(!dark)} title={dark ? t.lightMode : t.darkMode}
        style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 8, padding: '5px 12px', cursor: 'pointer', fontSize: 14 }}>
        {dark ? '☀️' : '🌙'}
      </button>
      <select className="inp" style={{ width: 110, height: 34 }} value={lang} onChange={e => setLang(e.target.value)}>
        <option value="en">English</option>
        <option value="ta">தமிழ்</option>
      </select>
    </header>
  )
}

// ── Weather Widget ─────────────────────────────────────────
function WeatherWidget() {
  const [data, setData] = useState(null)
  const { lang } = useContext(AppCtx)
  const t = T[lang]

  useEffect(() => {
    apiFetch('/api/weather').then(setData).catch(() => {})
  }, [])

  const codeLabel = code => {
    if (code === 0) return lang === 'ta' ? 'தெளிவான வானம்' : 'Clear sky'
    if (code <= 3) return lang === 'ta' ? 'மேக மூட்டம்' : 'Partly cloudy'
    if (code <= 49) return lang === 'ta' ? 'மூடுபனி' : 'Fog'
    if (code <= 69) return lang === 'ta' ? 'மழை' : 'Rain'
    if (code <= 79) return lang === 'ta' ? 'பனி' : 'Snow'
    if (code <= 99) return lang === 'ta' ? 'இடி மழை' : 'Thunderstorm'
    return '—'
  }

  if (!data || data.error) return (
    <Card><div style={{ color: 'var(--muted)', fontSize: 13 }}>{t.weather}: unavailable</div></Card>
  )
  return (
    <Card>
      <div style={{ fontWeight: 600, marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
        🌤 {t.weather}
        <span style={{ fontSize: 11, color: 'var(--muted)', fontWeight: 400 }}>{data.location}</span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(100px,1fr))', gap: 12 }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#f97316' }}>{data.temperature}°C</div>
          <div style={{ fontSize: 12, color: 'var(--muted)' }}>{t.temperature}</div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#3b82f6' }}>{data.humidity ?? '—'}%</div>
          <div style={{ fontSize: 12, color: 'var(--muted)' }}>{t.humidity}</div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 28, fontWeight: 700, color: 'var(--text)' }}>{data.windspeed}</div>
          <div style={{ fontSize: 12, color: 'var(--muted)' }}>{t.windspeed} km/h</div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 18, fontWeight: 600, color: 'var(--text)', paddingTop: 4 }}>{codeLabel(data.weathercode)}</div>
          <div style={{ fontSize: 12, color: 'var(--muted)' }}>{t.weatherCode}</div>
        </div>
      </div>
    </Card>
  )
}

// ── Dashboard Page ────────────────────────────────────────
function DashboardPage({ selectedField }) {
  const [latest, setLatest] = useState(null)
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const { lang } = useContext(AppCtx)
  const t = T[lang]

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const [lat, hist] = await Promise.all([
        apiFetch('/api/latest'),
        apiFetch(`/api/history?limit=20${selectedField ? '&field_id=' + selectedField : ''}`)
      ])
      setLatest(lat)
      setHistory((hist || []).reverse().map(r => ({ ...r, ts: r.timestamp?.slice(0, 16).replace('T', ' ') })))
    } catch {}
    setLoading(false)
  }, [selectedField])

  useEffect(() => { load() }, [load])

  if (loading) return <Spinner />

  const noData = !latest || !latest.crop_health

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Status cards */}
      {noData ? (
        <Card><div style={{ textAlign: 'center', color: 'var(--muted)', padding: '30px 0' }}>{t.noData}</div></Card>
      ) : (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px,1fr))', gap: 14 }}>
            <MetricCard label={t.cropHealth}
              value={latest.crop_health === 'healthy' ? t.healthy : latest.crop_health === 'stressed' ? t.stressed : t.unknown}
              color={latest.crop_health === 'healthy' ? '#22c55e' : '#ef4444'} />
            <MetricCard label={t.stressScore} value={latest.stress_score} unit="/100"
              color={latest.stress_score > 60 ? '#ef4444' : latest.stress_score > 35 ? '#f59e0b' : '#22c55e'} />
            <MetricCard label={t.soilCondition} value={t[latest.soil_condition] || latest.soil_condition} />
            <MetricCard label={t.confidence} value={latest.confidence} unit="%" />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px,1fr))', gap: 14 }}>
            <Card>
              <div style={{ fontSize: 13, color: 'var(--muted)', marginBottom: 8 }}>{t.stressType}</div>
              <div style={{ fontWeight: 700, fontSize: 16 }}>{latest.stress_type?.replace(/_/g, ' ')}</div>
              <div style={{ fontSize: 12, color: 'var(--muted)', marginTop: 4 }}>{latest.disease_name}</div>
            </Card>
            <Card>
              <div style={{ fontSize: 13, color: 'var(--muted)', marginBottom: 8 }}>{t.severity}</div>
              <Badge label={latest.severity} color={SEV_COLOR[latest.severity]} bg={SEV_BG[latest.severity]} />
            </Card>
            {latest.thumbnail && (
              <Card style={{ padding: 10 }}>
                <img src={`data:image/jpeg;base64,${latest.thumbnail}`} alt="latest" style={{ width: '100%', borderRadius: 8, maxHeight: 160, objectFit: 'cover' }} />
              </Card>
            )}
          </div>
        </>
      )}

      {/* Weather */}
      <WeatherWidget />

      {/* Trend chart */}
      {history.length > 1 && (
        <Card>
          <div style={{ fontWeight: 600, marginBottom: 16 }}>📈 {t.trend}</div>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={history}>
              <defs>
                <linearGradient id="sg" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="ts" tick={{ fontSize: 10, fill: 'var(--muted)' }} interval="preserveStartEnd" />
              <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: 'var(--muted)' }} />
              <Tooltip contentStyle={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }} />
              <Area type="monotone" dataKey="stress_score" stroke="#3b82f6" fill="url(#sg)" name={t.stressScore} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </Card>
      )}
    </div>
  )
}

// ── Upload Page ───────────────────────────────────────────
function UploadPage({ fields, selectedField, onUploaded }) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [progressLabel, setProgressLabel] = useState('')
  const [result, setResult] = useState(null)
  const [esp32Loading, setEsp32Loading] = useState(false)
  const [esp32Progress, setEsp32Progress] = useState(0)
  const [error, setError] = useState('')
  const [esp32Ip, setEsp32Ip] = useState('')
  const [esp32Reachable, setEsp32Reachable] = useState(null)
  const [ipInput, setIpInput] = useState('')
  const [ipSaving, setIpSaving] = useState(false)
  const { lang } = useContext(AppCtx)
  const t = T[lang]
  const fileRef = useRef()
  const progressTimerRef = useRef(null)

  useEffect(() => {
    apiFetch('/api/esp32/status')
      .then(d => { setEsp32Ip(d.ip || ''); setEsp32Reachable(d.reachable); setIpInput(d.ip || '') })
      .catch(() => {})
  }, [])

  async function saveIp() {
    if (!ipInput.trim()) return
    setIpSaving(true)
    try {
      await apiFetch('/api/esp32/set-ip', { method: 'POST', body: JSON.stringify({ ip: ipInput.trim() }) })
      setEsp32Ip(ipInput.trim())
      const d = await apiFetch('/api/esp32/status')
      setEsp32Reachable(d.reachable)
    } catch (e) { setError(e.message) }
    setIpSaving(false)
  }

  // Staged progress simulation — realistic steps that match what the backend does:
  // upload → HSV analysis → segmentation model → classifier → save → done
  const STAGES = lang === 'ta' ? [
    { pct: 8,  label: '📤 படம் பதிவேற்றுகிறது...' },
    { pct: 22, label: '🎨 வண்ண பகுப்பாய்வு (HSV)...' },
    { pct: 40, label: '🔍 தாவர பகுதிகளை கண்டறிகிறது...' },
    { pct: 60, label: '🤖 AI மாதிரி இயக்குகிறது...' },
    { pct: 75, label: '🧪 தொற்று வகை சரிபார்க்கிறது...' },
    { pct: 88, label: '💾 முடிவுகளை சேமிக்கிறது...' },
    { pct: 95, label: '✅ முடிக்கிறது...' },
  ] : [
    { pct: 8,  label: '📤 Uploading image...' },
    { pct: 22, label: '🎨 Running HSV colour analysis...' },
    { pct: 40, label: '🔍 Detecting plant regions...' },
    { pct: 60, label: '🤖 Running AI model inference...' },
    { pct: 75, label: '🧪 Checking stress & lesion type...' },
    { pct: 88, label: '💾 Saving results to database...' },
    { pct: 95, label: '✅ Finalising...' },
  ]

  function startProgress(setP, setLabel) {
    let stageIdx = 0
    // Advance through stages at irregular intervals (mimics real backend timing)
    const delays = [300, 600, 900, 1200, 800, 600]
    function advance() {
      if (stageIdx >= STAGES.length) return
      const stage = STAGES[stageIdx]
      setP(stage.pct)
      setLabel(stage.label)
      stageIdx++
      if (stageIdx < STAGES.length) {
        progressTimerRef.current = setTimeout(advance, delays[stageIdx - 1] || 700)
      }
    }
    advance()
  }

  function finishProgress(setP, setLabel, successLabel) {
    clearTimeout(progressTimerRef.current)
    setP(100)
    setLabel(successLabel)
  }

  function resetProgress(setP, setLabel) {
    clearTimeout(progressTimerRef.current)
    setP(0)
    setLabel('')
  }

  function onFileChange(e) {
    const f = e.target.files[0]
    if (!f) return
    setResult(null)
    setError('')
    setFile(f)
    if (preview && preview.startsWith('blob:')) URL.revokeObjectURL(preview)
    setPreview(URL.createObjectURL(f))
    e.target.value = ''
  }

  function resetAll() {
    if (preview && preview.startsWith('blob:')) URL.revokeObjectURL(preview)
    setFile(null); setPreview(null); setResult(null); setError('')
    resetProgress(setProgress, setProgressLabel)
    if (fileRef.current) fileRef.current.value = ''
  }

  async function analyze() {
    if (!file) return
    setLoading(true); setError('')
    resetProgress(setProgress, setProgressLabel)
    startProgress(setProgress, setProgressLabel)
    try {
      const fd = new FormData()
      fd.append('file', file)
      const url = '/api/upload?source=manual' + (selectedField ? '&field_id=' + selectedField : '')
      const data = await apiFetch(url, { method: 'POST', body: fd })
      finishProgress(setProgress, setProgressLabel, lang === 'ta' ? '✅ பகுப்பாய்வு முடிந்தது!' : '✅ Analysis complete!')
      if (data.thumbnail) setPreview('data:image/jpeg;base64,' + data.thumbnail)
      // Small pause so user sees 100% before result renders
      await new Promise(r => setTimeout(r, 400))
      setResult(data)
      if (onUploaded) onUploaded()
    } catch (err) {
      resetProgress(setProgress, setProgressLabel)
      setError(err.message)
    }
    setLoading(false)
  }

  async function captureESP32() {
    resetAll()
    setEsp32Loading(true)
    setEsp32Progress(0)
    startProgress(setEsp32Progress, setProgressLabel)
    try {
      const data = await apiFetch('/api/esp32/capture')
      finishProgress(setEsp32Progress, setProgressLabel, lang === 'ta' ? '✅ பிடிப்பு முடிந்தது!' : '✅ Capture complete!')
      if (data.thumbnail) setPreview('data:image/jpeg;base64,' + data.thumbnail)
      await new Promise(r => setTimeout(r, 400))
      // Ensure all inference fields are present (same as upload result)
      setResult({
        ...data,
        colour_analysis: data.colour_analysis || {},
        alert_type:      data.alert_type      || 'NONE',
        alert_message:   data.alert_message   || '',
        alert_severity:  data.alert_severity  || 'OK',
        plant_type:      data.plant_type      || 'unknown',
        zones:           data.zones           || [],
      })
      if (onUploaded) onUploaded()
    } catch (err) {
      resetProgress(setEsp32Progress, setProgressLabel)
      setError(err.message || 'ESP32 not reachable. Check IP and connection.')
    }
    setEsp32Loading(false)
  }

  const isAnalyzing = loading || esp32Loading
  const activeProgress = loading ? progress : esp32Progress

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 700 }}>
      <Card>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <div style={{ fontWeight: 600 }}>📷 {t.uploadImage}</div>
          {(result || preview) && !isAnalyzing && (
            <button onClick={resetAll} style={{ fontSize: 12, color: 'var(--muted)', background: 'none', border: 'none', cursor: 'pointer' }}>
              ✕ Clear
            </button>
          )}
        </div>

        {/* ESP32 IP Status + Capture */}
        <div style={{ border: '1px solid var(--border)', borderRadius: 10, padding: 14, marginBottom: 16, background: 'var(--input-bg)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
            <span style={{ fontSize: 13, fontWeight: 600 }}>📡 ESP32-CAM</span>
            {esp32Reachable === true  && <span style={{ fontSize: 11, color: '#16a34a', background: '#dcfce7', padding: '2px 8px', borderRadius: 20 }}>● {lang === 'ta' ? 'இணைக்கப்பட்டது' : 'Connected'} — {esp32Ip}</span>}
            {esp32Reachable === false && <span style={{ fontSize: 11, color: '#dc2626', background: '#fee2e2', padding: '2px 8px', borderRadius: 20 }}>● {lang === 'ta' ? 'கிடைக்கவில்லை' : 'Unreachable'} — {esp32Ip || (lang === 'ta' ? 'IP அமைக்கப்படவில்லை' : 'No IP set')}</span>}
            {esp32Reachable === null  && <span style={{ fontSize: 11, color: 'var(--muted)' }}>{lang === 'ta' ? 'சரிபார்க்கிறது...' : 'Checking...'}</span>}
          </div>
          <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
            <input className="inp" style={{ flex: 1, height: 34, fontSize: 13 }}
              placeholder="e.g. 10.11.131.99"
              value={ipInput} onChange={e => setIpInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && saveIp()} />
            <button className="btn-secondary" onClick={saveIp} disabled={ipSaving}
              style={{ height: 34, fontSize: 13, whiteSpace: 'nowrap' }}>
              {ipSaving ? '...' : (lang === 'ta' ? 'IP அமை' : 'Set IP')}
            </button>
          </div>
          <button className='btn-secondary' onClick={captureESP32} disabled={isAnalyzing || !esp32Ip}
            style={{ width: '100%', fontSize: 14 }}>
            {esp32Loading ? '📡 ' + t.analyzing : '📡 ' + t.captureESP32}
          </button>
        </div>

        <div style={{ textAlign: 'center', color: 'var(--muted)', fontSize: 13, marginBottom: 16 }}>— {t.orUpload} —</div>

        <div onClick={() => !isAnalyzing && fileRef.current?.click()}
          style={{ border: '2px dashed var(--border)', borderRadius: 12, padding: '28px 20px', textAlign: 'center', cursor: isAnalyzing ? 'default' : 'pointer', marginBottom: 14, background: preview ? 'transparent' : 'var(--bg)', transition: 'all 0.15s' }}>
          {preview
            ? <img src={preview} alt='preview' style={{ maxHeight: 220, borderRadius: 8, maxWidth: '100%' }} />
            : <div><div style={{ fontSize: 36, marginBottom: 8 }}>🖼</div><div style={{ fontSize: 14, color: 'var(--muted)' }}>{t.chooseFile}</div></div>
          }
          <input ref={fileRef} type='file' accept='image/*' style={{ display: 'none' }} onChange={onFileChange} />
        </div>

        {/* ── Progress bar — shown during analysis ── */}
        {isAnalyzing && (
          <div style={{ marginBottom: 14 }}>
            {/* Stage label */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
              <span style={{ fontSize: 13, color: 'var(--text)', fontWeight: 500 }}>{progressLabel}</span>
              <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--accent)' }}>{activeProgress}%</span>
            </div>
            {/* Track */}
            <div style={{ width: '100%', height: 10, background: 'var(--border)', borderRadius: 99, overflow: 'hidden' }}>
              <div style={{
                height: '100%',
                width: `${activeProgress}%`,
                borderRadius: 99,
                background: 'linear-gradient(90deg, var(--accent), #86efac)',
                transition: 'width 0.5s cubic-bezier(0.4,0,0.2,1)',
                boxShadow: activeProgress > 0 && activeProgress < 100 ? '0 0 8px rgba(22,163,74,0.4)' : 'none',
              }} />
            </div>
            {/* Step dots */}
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6 }}>
              {[8,22,40,60,75,88,100].map((step, i) => (
                <div key={i} style={{
                  width: 8, height: 8, borderRadius: '50%',
                  background: activeProgress >= step ? 'var(--accent)' : 'var(--border)',
                  transition: 'background 0.3s',
                  flexShrink: 0,
                }} />
              ))}
            </div>
          </div>
        )}

        {file && !isAnalyzing && (
          <button className='btn-primary' onClick={analyze} style={{ width: '100%' }}>
            🔬 {t.analyze}
          </button>
        )}
        {error && <div style={{ color: '#ef4444', fontSize: 13, marginTop: 10, padding: '8px 12px', background: '#fee2e2', borderRadius: 8 }}>{error}</div>}
      </Card>

      {result && (
        <Card>
          {/* ── Smart Alert Banner ─────────────────────────────── */}
          {result.alert_type && result.alert_type !== 'NONE' && (
            <div style={{
              marginBottom: 16, padding: '12px 16px', borderRadius: 10,
              background: result.alert_type === 'WATER_ALERT' ? '#eff6ff'
                        : result.alert_type === 'DRAINAGE_ALERT' ? '#fff7ed'
                        : result.alert_type === 'COMBINED_ALERT' ? '#fef2f2'
                        : '#fefce8',
              border: `1.5px solid ${
                result.alert_type === 'WATER_ALERT' ? '#3b82f6'
                : result.alert_type === 'DRAINAGE_ALERT' ? '#f97316'
                : result.alert_type === 'COMBINED_ALERT' ? '#ef4444'
                : '#eab308'}`,
              display: 'flex', alignItems: 'flex-start', gap: 10,
            }}>
              <span style={{ fontSize: 22, flexShrink: 0 }}>
                {result.alert_type === 'WATER_ALERT' ? '🌊'
                : result.alert_type === 'DRAINAGE_ALERT' ? '🚰'
                : result.alert_type === 'COMBINED_ALERT' ? '🚨'
                : '🌿'}
              </span>
              <div>
                <div style={{ fontWeight: 700, fontSize: 14,
                  color: result.alert_type === 'WATER_ALERT' ? '#1d4ed8'
                       : result.alert_type === 'DRAINAGE_ALERT' ? '#c2410c'
                       : result.alert_type === 'COMBINED_ALERT' ? '#dc2626'
                       : '#854d0e'
                }}>
                  {result.alert_type === 'WATER_ALERT' ? (lang === 'ta' ? 'நீர்ப்பாசன எச்சரிக்கை' : 'Irrigation Alert')
                  : result.alert_type === 'DRAINAGE_ALERT' ? (lang === 'ta' ? 'நீர்வடிகால் எச்சரிக்கை' : 'Drainage Alert')
                  : result.alert_type === 'COMBINED_ALERT' ? (lang === 'ta' ? 'இணைந்த எச்சரிக்கை' : 'Combined Alert')
                  : (lang === 'ta' ? 'தாவர அழுத்த எச்சரிக்கை' : 'Plant Stress Alert')}
                </div>
                <div style={{ fontSize: 13, marginTop: 2, color: 'var(--text)', lineHeight: 1.5 }}>
                  {result.alert_message}
                </div>
              </div>
            </div>
          )}

          {/* Header with health status */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16, padding: '12px 16px', borderRadius: 10, background: result.crop_health === 'healthy' ? '#dcfce7' : '#fee2e2' }}>
            <div style={{ fontSize: 32 }}>{result.crop_health === 'healthy' ? '🌿' : '⚠️'}</div>
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 700, fontSize: 18, color: result.crop_health === 'healthy' ? '#15803d' : '#dc2626' }}>
                {result.crop_health === 'healthy' ? t.healthy : t.stressed}
              </div>
              <div style={{ fontSize: 12, color: result.crop_health === 'healthy' ? '#16a34a' : '#ef4444' }}>
                {result.disease_name}
              </div>
            </div>
            <Badge label={result.severity} color={SEV_COLOR[result.severity]} bg={SEV_BG[result.severity]} style={{ marginLeft: 'auto' }} />
          </div>

          {/* Plant type identified */}


          {/* Metric cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(130px,1fr))', gap: 12, marginBottom: 16 }}>
            <MetricCard label={t.stressScore} value={result.stress_score} unit='/100'
              color={result.stress_score > 60 ? '#dc2626' : result.stress_score > 35 ? '#d97706' : '#16a34a'} />
            <MetricCard label={t.confidence} value={result.confidence} unit='%'
              color={result.confidence > 70 ? '#16a34a' : '#d97706'} />
          </div>

          {/* Stress type */}
          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 6 }}>{t.stressType}</div>
            <Badge label={result.stress_type?.replace(/_/g, ' ')} color='#3b82f6' bg='#dbeafe' />
          </div>

          {/* Soil section — context-aware messaging */}
          <div style={{ padding: '10px 14px', background: 'var(--bg)', borderRadius: 8, marginBottom: 14 }}>
            <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 6 }}>{t.soilCondition}</div>
            {result.soil_condition === 'not_detected'
              ? <div style={{ fontSize: 14, color: 'var(--muted)', fontStyle: 'italic' }}>🪨 {t.not_detected}</div>
              : <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontSize: 20 }}>
                    {result.soil_condition === 'dry' ? '🏜️' : result.soil_condition === 'wet_waterlogged' ? '💧' : '🌱'}
                  </span>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: 14 }}>{t[result.soil_condition] || result.soil_condition}</div>
                    <div style={{ fontSize: 12, color: 'var(--muted)' }}>
                      {result.soil_condition === 'dry' && result.alert_type === 'WATER_ALERT'
                        ? (lang === 'ta' ? 'நீர் பாய்ச்சவும் — தாவரம் ஆரோக்கியமாக உள்ளது' : 'Pour water now — plant is still healthy')
                        : result.soil_condition === 'dry'
                        ? (lang === 'ta' ? 'நீர்ப்பாசனம் அவசியம்' : 'Consider irrigation')
                        : result.soil_condition === 'wet_waterlogged'
                        ? (lang === 'ta' ? 'வேர் அழுகல் ஆபத்து' : 'Risk of root rot — check drainage')
                        : (lang === 'ta' ? 'ஈரப்பதம் சரியாக உள்ளது' : 'Moisture level optimal')}
                    </div>
                  </div>
                </div>
            }
          </div>

          {/* Colour analysis breakdown */}
          {result.colour_analysis && (
            <div style={{ marginBottom: 14 }}>
              <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 8 }}>Colour analysis</div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {Object.entries(result.colour_analysis).map(([k, v]) => {
                  const colors = { green_pct: '#22c55e', yellow_pct: '#eab308', brown_pct: '#92400e', rust_pct: '#dc2626', necro_pct: '#4b5563', soil_pct: '#a16207' }
                  const col = colors[k] || 'var(--muted)'
                  return (
                    <div key={k} style={{ fontSize: 12, padding: '4px 10px', borderRadius: 20, border: '1px solid var(--border)', background: 'var(--bg)' }}>
                      <span style={{ color: col, fontWeight: 700 }}>{v}%</span>
                      <span style={{ color: 'var(--muted)', marginLeft: 4 }}>{k.replace(/_pct/g,'').replace(/_/g,' ')}</span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          <button onClick={resetAll} className='btn-secondary' style={{ width: '100%', fontSize: 13 }}>
            📷 Analyze another image
          </button>
        </Card>
      )}
    </div>
  )
}

// ── Analysis (LLM) Page ────────────────────────────────────
function AnalysisPage() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [latest, setLatest] = useState(null)
  const [error, setError] = useState('')
  const { lang } = useContext(AppCtx)
  const t = T[lang]
  const prevLangRef = useRef(lang)

  useEffect(() => {
    apiFetch('/api/latest').then(setLatest).catch(() => {})
  }, [])

  // When language changes AND a result is already showing, auto-retranslate it
  useEffect(() => {
    if (prevLangRef.current !== lang) {
      prevLangRef.current = lang
      if (result && latest?.crop_health) {
        // Clear old result immediately so user sees the language switched
        setResult(null)
        retranslate()
      }
    }
  }, [lang])

  async function retranslate() {
    if (!latest?.crop_health) return
    setLoading(true); setError('')
    try {
      const body = {
        crop_health:    latest.crop_health,
        stress_score:   latest.stress_score,
        stress_type:    latest.stress_type,
        soil_condition: latest.soil_condition,
        language:       lang,
        plant_type:     latest.plant_type  || 'unknown',
        alert_type:     latest.alert_type  || 'NONE',
      }
      const data = await apiFetch('/api/llm', { method: 'POST', body: JSON.stringify(body) })
      setResult(data)
    } catch (err) { setError(err.message) }
    setLoading(false)
  }

  async function getAnalysis() {
    if (!latest?.crop_health) return
    setLoading(true); setError('')
    try {
      const body = {
        crop_health:    latest.crop_health,
        stress_score:   latest.stress_score,
        stress_type:    latest.stress_type,
        soil_condition: latest.soil_condition,
        language:       lang,
        plant_type:     latest.plant_type  || 'unknown',
        alert_type:     latest.alert_type  || 'NONE',
      }
      const data = await apiFetch('/api/llm', { method: 'POST', body: JSON.stringify(body) })
      setResult(data)
    } catch (err) { setError(err.message) }
    setLoading(false)
  }

  const URGENCY_COLOR = { LOW: '#22c55e', MEDIUM: '#f59e0b', HIGH: '#ef4444', CRITICAL: '#7c3aed' }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 750 }}>
      <Card>
        <div style={{ fontWeight: 600, marginBottom: 8 }}>🤖 {t.analysis}</div>
        <p style={{ fontSize: 13, color: 'var(--muted)', marginBottom: 14 }}>{t.analysisNote}</p>
        {!latest?.crop_health && <div style={{ color: 'var(--muted)', fontSize: 13 }}>{t.noLatest}</div>}
        {latest?.crop_health && (
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 14 }}>
            <Badge label={`${t.stressScore}: ${latest.stress_score}/100`} color="#3b82f6" bg="#dbeafe" />
            <Badge label={latest.stress_type?.replace(/_/g, ' ')} color="#6b7280" bg="#f1f5f9" />
            <Badge label={latest.soil_condition} color="#22c55e" bg="#dcfce7" />

            {latest.alert_type && latest.alert_type !== 'NONE' && (
              <Badge
                label={latest.alert_type === 'WATER_ALERT' ? '🌊 Water Alert'
                     : latest.alert_type === 'DRAINAGE_ALERT' ? '🚰 Drainage Alert'
                     : latest.alert_type === 'PLANT_STRESS_ALERT' ? '🌿 Plant Stress'
                     : '🚨 Combined Alert'}
                color={latest.alert_type === 'WATER_ALERT' ? '#1d4ed8'
                     : latest.alert_type === 'DRAINAGE_ALERT' ? '#c2410c'
                     : latest.alert_type === 'COMBINED_ALERT' ? '#dc2626' : '#854d0e'}
                bg={latest.alert_type === 'WATER_ALERT' ? '#dbeafe'
                  : latest.alert_type === 'DRAINAGE_ALERT' ? '#ffedd5'
                  : latest.alert_type === 'COMBINED_ALERT' ? '#fee2e2' : '#fefce8'}
              />
            )}
          </div>
        )}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          <button className="btn-primary" onClick={getAnalysis} disabled={loading || !latest?.crop_health}>
            {loading ? t.loading : `🔮 ${t.getAnalysis}`}
          </button>
          {/* Language badge — shows which language the result will be in */}
          <span style={{ fontSize: 12, padding: '4px 10px', borderRadius: 20, background: 'var(--accent-bg)', color: 'var(--accent)', fontWeight: 600 }}>
            {lang === 'ta' ? '🇮🇳 தமிழ்' : '🇬🇧 English'}
          </span>
          {/* Retranslating indicator */}
          {loading && result === null && (
            <span style={{ fontSize: 12, color: 'var(--muted)', fontStyle: 'italic' }}>
              {lang === 'ta' ? 'தமிழில் மொழிபெயர்க்கிறது...' : 'Translating...'}
            </span>
          )}
        </div>
        {error && <div style={{ color: '#ef4444', fontSize: 13, marginTop: 10, padding: '8px 12px', background: '#fee2e2', borderRadius: 8 }}>{error}</div>}
      </Card>

      {result && (
        <>
          {/* Plant identified by LLM */}


          <Card style={{ borderLeft: `4px solid ${URGENCY_COLOR[result.urgency] || '#3b82f6'}` }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <div style={{ fontWeight: 600 }}>{t.explanation}</div>
              <Badge label={`${t.urgency}: ${result.urgency}`} color={URGENCY_COLOR[result.urgency] || '#6b7280'} bg={SEV_BG[result.urgency === 'CRITICAL' ? 'CRITICAL' : result.urgency === 'HIGH' ? 'ALERT' : 'WARN'] || '#f1f5f9'} />
            </div>
            <p style={{ fontSize: 14, lineHeight: 1.7, marginBottom: 12 }}>{result.explanation}</p>
            <div style={{ fontSize: 13, color: 'var(--muted)', fontWeight: 600, marginBottom: 4 }}>{t.cause}</div>
            <p style={{ fontSize: 13, color: 'var(--muted)' }}>{result.cause}</p>
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(300px,1fr))', gap: 16 }}>
            <Card>
              <div style={{ fontWeight: 600, marginBottom: 10 }}>💊 {t.remedy}</div>
              <pre style={{ fontSize: 13, lineHeight: 1.8, whiteSpace: 'pre-wrap', fontFamily: 'inherit', color: 'var(--text)', margin: 0 }}>{result.remedy}</pre>
            </Card>
            <Card>
              <div style={{ fontWeight: 600, marginBottom: 10 }}>🛡 {t.prevention}</div>
              <p style={{ fontSize: 13, lineHeight: 1.7 }}>{result.prevention}</p>
            </Card>
            <Card>
              <div style={{ fontWeight: 600, marginBottom: 10 }}>🌿 {t.fertilizer}</div>
              <p style={{ fontSize: 13, lineHeight: 1.7 }}>{result.fertilizer_tip}</p>
            </Card>
            <Card>
              <div style={{ fontWeight: 600, marginBottom: 10 }}>💧 {t.irrigation}</div>
              <p style={{ fontSize: 13, lineHeight: 1.7 }}>{result.irrigation_advice}</p>
            </Card>
          </div>
        </>
      )}
    </div>
  )
}

// ── History Page ───────────────────────────────────────────
function HistoryPage({ selectedField }) {
  const [records, setRecords] = useState([])
  const [loading, setLoading] = useState(true)
  const { lang } = useContext(AppCtx)
  const t = T[lang]

  useEffect(() => {
    setLoading(true)
    apiFetch(`/api/history?limit=50${selectedField ? '&field_id=' + selectedField : ''}`)
      .then(d => { setRecords(d || []); setLoading(false) })
      .catch(() => setLoading(false))
  }, [selectedField])

  const chartData = [...records].reverse().map(r => ({ ts: r.timestamp?.slice(5, 16).replace('T', ' '), stress_score: r.stress_score, confidence: r.confidence }))

  if (loading) return <Spinner />

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {chartData.length > 1 && (
        <Card>
          <div style={{ fontWeight: 600, marginBottom: 14 }}>📈 {t.trend}</div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="ts" tick={{ fontSize: 10, fill: 'var(--muted)' }} interval="preserveStartEnd" />
              <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: 'var(--muted)' }} />
              <Tooltip contentStyle={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line type="monotone" dataKey="stress_score" stroke="#ef4444" dot={false} name={t.stressScore} />
              <Line type="monotone" dataKey="confidence" stroke="#3b82f6" dot={false} name={t.confidence} />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      )}

      {records.length === 0
        ? <Card><div style={{ textAlign: 'center', color: 'var(--muted)', padding: '30px 0' }}>{t.noData}</div></Card>
        : records.map(r => (
          <Card key={r._id} style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
            {r.thumbnail ? null : <div style={{ width: 48, height: 48, borderRadius: 8, background: 'var(--bg)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 22, flexShrink: 0 }}>🌱</div>}
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6, flexWrap: 'wrap', gap: 6 }}>
                <span style={{ fontSize: 13, color: 'var(--muted)' }}>{r.timestamp?.slice(0, 16).replace('T', ' ')}</span>
                <div style={{ display: 'flex', gap: 6 }}>
                  <Badge label={r.severity || 'INFO'} color={SEV_COLOR[r.severity]} bg={SEV_BG[r.severity]} />
                  <Badge label={r.source || 'manual'} color="#6b7280" bg="#f1f5f9" />
                </div>
              </div>
              <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 13 }}>
                <span><strong>{t.stressScore}:</strong> {r.stress_score}/100</span>
                <span><strong>{t.cropHealth}:</strong> {t[r.crop_health] || r.crop_health}</span>
                <span><strong>{t.stressType}:</strong> {r.stress_type?.replace(/_/g, ' ')}</span>
                <span><strong>{t.soilCondition}:</strong> {t[r.soil_condition] || r.soil_condition}</span>
              </div>
              {r.disease_name && <div style={{ fontSize: 12, color: 'var(--muted)', marginTop: 4 }}>{r.disease_name}</div>}
            </div>
          </Card>
        ))
      }
    </div>
  )
}

// ── Alerts Page ────────────────────────────────────────────
function AlertsPage() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const { lang } = useContext(AppCtx)
  const t = T[lang]

  useEffect(() => {
    apiFetch('/api/alerts').then(setData).catch(() => {}).finally(() => setLoading(false))
  }, [])

  if (loading) return <Spinner />

  const trend = data?.trend
  const alerts = data?.alerts || []
  const trendLabel = trend?.direction === 'increasing' ? t.trendIncreasing : trend?.direction === 'decreasing' ? t.trendDecreasing : t.trendStable

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {trend && trend.scores?.length > 0 && (
        <Card style={{ borderLeft: `4px solid ${trend.direction === 'increasing' ? '#ef4444' : trend.direction === 'decreasing' ? '#22c55e' : '#3b82f6'}` }}>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>📊 {trendLabel}</div>
          <div style={{ fontSize: 13, color: 'var(--muted)' }}>
            {trend.scores.length} {t.readings} · avg change: {trend.avg_change > 0 ? '+' : ''}{trend.avg_change} pts
          </div>
          {trend.scores.length > 1 && (
            <ResponsiveContainer width="100%" height={80} style={{ marginTop: 12 }}>
              <AreaChart data={trend.scores.map((s, i) => ({ i, s }))}>
                <Area type="monotone" dataKey="s" stroke={trend.direction === 'increasing' ? '#ef4444' : '#22c55e'} fill={trend.direction === 'increasing' ? '#fee2e2' : '#dcfce7'} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </Card>
      )}

      <Card>
        <div style={{ fontWeight: 600, marginBottom: 14 }}>🔔 {t.alertsPanel}</div>
        {alerts.length === 0
          ? <div style={{ textAlign: 'center', color: 'var(--muted)', padding: '20px 0', fontSize: 14 }}>✅ {t.noAlerts}</div>
          : alerts.map(a => (
            <div key={a.id} style={{ display: 'flex', gap: 12, padding: '12px 0', borderBottom: '1px solid var(--border)', alignItems: 'flex-start' }}>
              <div style={{ width: 8, height: 8, borderRadius: '50%', background: SEV_COLOR[a.severity], marginTop: 5, flexShrink: 0 }} />
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 14, marginBottom: 3 }}>{a.message}</div>
                <div style={{ fontSize: 11, color: 'var(--muted)' }}>{a.timestamp?.slice(0, 16).replace('T', ' ')}</div>
              </div>
              <Badge label={a.severity} color={SEV_COLOR[a.severity]} bg={SEV_BG[a.severity]} />
            </div>
          ))
        }
      </Card>
    </div>
  )
}

// ── Fields Page ────────────────────────────────────────────
function FieldsPage({ onFieldsChange }) {
  const [fields, setFields] = useState([])
  const [showAdd, setShowAdd] = useState(false)
  const [form, setForm] = useState({ field_name: '', latitude: '', longitude: '' })
  const [loading, setLoading] = useState(false)
  const { lang } = useContext(AppCtx)
  const t = T[lang]

  useEffect(() => {
    apiFetch('/api/fields').then(d => { setFields(d || []) }).catch(() => {})
  }, [])

  async function addField() {
    setLoading(true)
    try {
      const data = await apiFetch('/api/fields', {
        method: 'POST',
        body: JSON.stringify({ field_name: form.field_name, latitude: parseFloat(form.latitude), longitude: parseFloat(form.longitude) })
      })
      setFields(prev => [...prev, data])
      setShowAdd(false)
      setForm({ field_name: '', latitude: '', longitude: '' })
      if (onFieldsChange) onFieldsChange()
    } catch (err) { alert(err.message) }
    setLoading(false)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16, maxWidth: 600 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ fontWeight: 600, fontSize: 16 }}>🌾 {t.fields}</div>
        <button className="btn-primary" onClick={() => setShowAdd(!showAdd)}>+ {t.addField}</button>
      </div>

      {showAdd && (
        <Card>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <input className="inp" placeholder={t.fieldName} value={form.field_name} onChange={e => setForm({ ...form, field_name: e.target.value })} />
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              <input className="inp" placeholder={t.latitude + ' (e.g. 13.08)'} value={form.latitude} onChange={e => setForm({ ...form, latitude: e.target.value })} />
              <input className="inp" placeholder={t.longitude + ' (e.g. 80.27)'} value={form.longitude} onChange={e => setForm({ ...form, longitude: e.target.value })} />
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <button className="btn-primary" onClick={addField} disabled={loading}>{t.save}</button>
              <button className="btn-secondary" onClick={() => setShowAdd(false)}>{t.cancel}</button>
            </div>
          </div>
        </Card>
      )}

      {fields.length === 0
        ? <Card><div style={{ textAlign: 'center', color: 'var(--muted)', padding: '24px 0' }}>No fields added yet</div></Card>
        : fields.map(f => (
          <Card key={f._id}>
            <div style={{ fontWeight: 600, marginBottom: 4 }}>{f.field_name}</div>
            <div style={{ fontSize: 13, color: 'var(--muted)' }}>
              {t.latitude}: {f.latitude} · {t.longitude}: {f.longitude}
            </div>
            <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 4 }}>
              Created: {f.created?.slice(0, 10)}
            </div>
          </Card>
        ))
      }
    </div>
  )
}

// ── Main App ───────────────────────────────────────────────
export default function App() {
  const [dark, setDark] = useState(() => localStorage.getItem('cm_dark') === 'true')
  const [lang, setLang] = useState(() => localStorage.getItem('cm_lang') || 'en')
  const [user, setUser] = useState(() => {
    try { return JSON.parse(localStorage.getItem('cm_user') || 'null') } catch { return null }
  })
  const [page, setPage] = useState('dashboard')
  const [fields, setFields] = useState([])
  const [selectedField, setSelectedField] = useState('')
  const [refreshKey, setRefreshKey] = useState(0)

  useEffect(() => {
    localStorage.setItem('cm_dark', dark)
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light')
  }, [dark])

  useEffect(() => { localStorage.setItem('cm_lang', lang) }, [lang])

  useEffect(() => {
    if (user) apiFetch('/api/fields').then(d => setFields(d || [])).catch(() => {})
  }, [user])

  function handleAuth(u) { setUser(u) }
  function handleLogout() {
    localStorage.removeItem('cm_token')
    localStorage.removeItem('cm_user')
    setUser(null)
  }

  const ctx = { dark, setDark, lang, setLang }

  if (!user) return (
    <AppCtx.Provider value={ctx}>
      <AuthPage onAuth={handleAuth} />
    </AppCtx.Provider>
  )

  return (
    <AppCtx.Provider value={ctx}>
      <div style={{ display: 'flex', minHeight: '100vh', background: 'var(--bg)' }}>
        <Sidebar page={page} setPage={setPage} user={user} onLogout={handleLogout} />
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
          <Topbar page={page} />
          <main style={{ flex: 1, padding: '24px', overflowY: 'auto' }}>
            {page === 'dashboard' && <DashboardPage key={refreshKey} selectedField={selectedField} />}
            {page === 'upload'    && <UploadPage fields={fields} selectedField={selectedField} onUploaded={() => setRefreshKey(k => k + 1)} />}
            {page === 'analysis'  && <AnalysisPage />}
            {page === 'history'   && <HistoryPage selectedField={selectedField} />}
            {page === 'alerts'    && <AlertsPage />}
          </main>
        </div>
      </div>
    </AppCtx.Provider>
  )
}