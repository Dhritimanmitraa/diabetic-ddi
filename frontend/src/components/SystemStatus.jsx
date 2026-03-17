import { useCallback, useEffect, useState } from 'react'
import { Activity, KeyRound, RefreshCw, ShieldAlert, Server } from 'lucide-react'

import { getAdminApiKey, getSystemStatus, setAdminApiKey } from '../services/api'

function StatusPill({ ok, label }) {
  return (
    <span className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border ${ok ? 'border-emerald-500/30 text-emerald-400 bg-emerald-500/10' : 'border-amber-500/30 text-amber-300 bg-amber-500/10'}`}>
      <span className={`w-2 h-2 rounded-full ${ok ? 'bg-emerald-400' : 'bg-amber-400'}`} />
      {label}
    </span>
  )
}

function MetricCard({ title, value, icon }) {
  return (
    <div className="rounded-2xl border border-[var(--border)] bg-[var(--bg-elevated)] p-5">
      <div className="flex items-center justify-between mb-3">
        <p className="text-sm text-[var(--text-muted)]">{title}</p>
        {icon}
      </div>
      <p className="text-2xl font-semibold text-[var(--text-primary)]">{value}</p>
    </div>
  )
}

export default function SystemStatus() {
  const [apiKey, setApiKey] = useState(() => getAdminApiKey())
  const [draftKey, setDraftKey] = useState(() => getAdminApiKey())
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const load = useCallback(async (overrideKey = apiKey) => {
    setLoading(true)
    setError('')
    try {
      const result = await getSystemStatus(overrideKey)
      setData(result)
    } catch (err) {
      setData(null)
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [apiKey])

  useEffect(() => {
    if (apiKey) {
      void load(apiKey)
    }
  }, [apiKey, load])

  const handleSave = async (e) => {
    e.preventDefault()
    setAdminApiKey(draftKey)
    setApiKey(draftKey)
    await load(draftKey)
  }

  return (
    <section className="max-w-6xl mx-auto px-5 sm:px-6 pt-28 pb-16">
      <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between mb-8">
        <div>
          <p className="text-sm uppercase tracking-[0.18em] text-[var(--text-muted)] mb-3">Operator Console</p>
          <h1 className="text-4xl font-display text-[var(--text-primary)]">System Status</h1>
          <p className="text-[var(--text-secondary)] mt-3 max-w-2xl">
            Protected diagnostics for deploy health, Gemini availability, Redis status, and background ML jobs.
          </p>
        </div>

        <button
          type="button"
          onClick={() => load()}
          className="inline-flex items-center gap-2 rounded-xl px-4 py-2.5 border border-[var(--border)] bg-[var(--bg-elevated)] text-[var(--text-primary)] hover:opacity-90 transition"
          disabled={loading}
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      <form onSubmit={handleSave} className="rounded-2xl border border-[var(--border)] bg-[var(--bg-elevated)] p-5 mb-8">
        <label className="block text-sm text-[var(--text-secondary)] mb-2">Admin API key</label>
        <div className="flex flex-col md:flex-row gap-3">
          <div className="relative flex-1">
            <KeyRound className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-muted)]" />
            <input
              value={draftKey}
              onChange={(e) => setDraftKey(e.target.value)}
              placeholder="Enter API key for protected admin endpoints"
              className="w-full rounded-xl border border-[var(--border)] bg-[var(--bg-primary)] pl-10 pr-4 py-3 text-[var(--text-primary)] outline-none focus:ring-2 focus:ring-medical-500/40"
            />
          </div>
          <button type="submit" className="rounded-xl px-4 py-3 bg-medical-500 text-white font-medium hover:bg-medical-400 transition">
            Save & Load
          </button>
        </div>
        {error && <p className="text-sm text-rose-400 mt-3">{error}</p>}
      </form>

      {!data && !loading ? (
        <div className="rounded-2xl border border-dashed border-[var(--border)] p-10 text-center text-[var(--text-secondary)]">
          Enter an admin API key to load protected system diagnostics.
        </div>
      ) : null}

      {data ? (
        <>
          <div className="grid gap-4 md:grid-cols-4 mb-8">
            <MetricCard title="Total Drugs" value={data.data.total_drugs} icon={<Activity className="w-4 h-4 text-medical-400" />} />
            <MetricCard title="Interactions" value={data.data.total_interactions} icon={<ShieldAlert className="w-4 h-4 text-medical-400" />} />
            <MetricCard title="Comparisons" value={data.data.total_comparisons} icon={<Server className="w-4 h-4 text-medical-400" />} />
            <MetricCard title="ML Predictions" value={data.data.total_ml_predictions} icon={<Activity className="w-4 h-4 text-medical-400" />} />
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <div className="rounded-2xl border border-[var(--border)] bg-[var(--bg-elevated)] p-5">
              <h2 className="text-lg font-semibold text-[var(--text-primary)] mb-4">Service Health</h2>
              <div className="flex flex-wrap gap-3 mb-4">
                <StatusPill ok={data.services.redis} label={data.services.redis ? 'Redis connected' : 'Redis unavailable'} />
                <StatusPill ok={data.services.gemini.available} label={data.services.gemini.available ? `Gemini via ${data.services.gemini.sdk}` : 'Gemini unavailable'} />
                <StatusPill ok={data.security.api_key_configured} label={data.security.api_key_configured ? 'Admin auth configured' : 'Admin auth not configured'} />
              </div>
              <pre className="text-xs whitespace-pre-wrap break-words text-[var(--text-secondary)] bg-[var(--bg-primary)] rounded-xl p-4 overflow-auto">
                {JSON.stringify(data.services.external_apis, null, 2)}
              </pre>
            </div>

            <div className="rounded-2xl border border-[var(--border)] bg-[var(--bg-elevated)] p-5">
              <h2 className="text-lg font-semibold text-[var(--text-primary)] mb-4">Background Jobs</h2>
              {data.jobs.length ? (
                <div className="space-y-3">
                  {data.jobs.map((job) => (
                    <div key={job.id} className="rounded-xl border border-[var(--border)] bg-[var(--bg-primary)] p-4">
                      <div className="flex items-center justify-between gap-3 mb-2">
                        <p className="font-medium text-[var(--text-primary)]">{job.name}</p>
                        <StatusPill ok={job.status === 'completed'} label={job.status} />
                      </div>
                      <p className="text-xs text-[var(--text-muted)] mb-2">{job.id}</p>
                      {job.error ? <p className="text-sm text-rose-400">{job.error}</p> : null}
                      {job.metadata ? <pre className="text-xs text-[var(--text-secondary)] whitespace-pre-wrap break-words">{JSON.stringify(job.metadata, null, 2)}</pre> : null}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-[var(--text-secondary)]">No tracked jobs yet.</p>
              )}
            </div>
          </div>
        </>
      ) : null}
    </section>
  )
}
