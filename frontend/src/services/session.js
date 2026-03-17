const AUTH_SESSION_STORAGE_KEY = 'drugguard_auth_session'
const ANON_CREDENTIALS_STORAGE_KEY = 'drugguard_anonymous_credentials'

function readJson(key) {
  try {
    const value = localStorage.getItem(key)
    return value ? JSON.parse(value) : null
  } catch {
    return null
  }
}

function writeJson(key, value) {
  try {
    if (value === null || value === undefined) {
      localStorage.removeItem(key)
      return
    }
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
    // Ignore storage failures in non-browser contexts.
  }
}

function randomId() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID().replace(/-/g, '')
  }
  return `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 12)}`
}

function normalizeTokenPair(payload) {
  return {
    accessToken: payload.access_token,
    refreshToken: payload.refresh_token,
    user: payload.user || null,
  }
}

async function authJsonRequest(baseUrl, endpoint, body) {
  const response = await fetch(`${baseUrl}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Authentication request failed' }))
    throw new Error(error.detail || `HTTP error! status: ${response.status}`)
  }

  return response.json()
}

function getAnonymousCredentials() {
  return readJson(ANON_CREDENTIALS_STORAGE_KEY)
}

function setAnonymousCredentials(credentials) {
  writeJson(ANON_CREDENTIALS_STORAGE_KEY, credentials)
}

function clearAnonymousCredentials() {
  writeJson(ANON_CREDENTIALS_STORAGE_KEY, null)
}

export function getAuthSession() {
  return readJson(AUTH_SESSION_STORAGE_KEY)
}

export function setAuthSession(session) {
  writeJson(AUTH_SESSION_STORAGE_KEY, session)
}

export function clearAuthSession() {
  writeJson(AUTH_SESSION_STORAGE_KEY, null)
}

export function getAccessToken() {
  return getAuthSession()?.accessToken || ''
}

export async function registerUser(baseUrl, payload) {
  const tokenPair = await authJsonRequest(baseUrl, '/auth/register', payload)
  const session = normalizeTokenPair(tokenPair)
  setAuthSession(session)
  return session
}

export async function loginUser(baseUrl, payload) {
  const tokenPair = await authJsonRequest(baseUrl, '/auth/login', payload)
  const session = normalizeTokenPair(tokenPair)
  setAuthSession(session)
  return session
}

export async function refreshAuthSession(baseUrl) {
  const session = getAuthSession()
  if (!session?.refreshToken) {
    throw new Error('No refresh token available')
  }

  const tokenPair = await authJsonRequest(baseUrl, '/auth/refresh', {
    refresh_token: session.refreshToken,
  })
  const nextSession = normalizeTokenPair(tokenPair)
  setAuthSession(nextSession)
  return nextSession
}

export async function ensureAnonymousSession(baseUrl) {
  const existingSession = getAuthSession()
  if (existingSession?.accessToken) {
    return existingSession
  }

  const savedCredentials = getAnonymousCredentials()
  if (savedCredentials?.username && savedCredentials?.password) {
    try {
      return await loginUser(baseUrl, {
        username: savedCredentials.username,
        password: savedCredentials.password,
      })
    } catch {
      clearAuthSession()
      clearAnonymousCredentials()
    }
  }

  const suffix = randomId().slice(0, 12)
  const credentials = {
    username: `guest_${suffix}`,
    email: `guest_${suffix}@local.drugguard.app`,
    password: `Device-${randomId().slice(0, 16)}!`,
  }
  setAnonymousCredentials(credentials)

  try {
    return await registerUser(baseUrl, credentials)
  } catch (error) {
    try {
      return await loginUser(baseUrl, {
        username: credentials.username,
        password: credentials.password,
      })
    } catch {
      clearAnonymousCredentials()
      throw error
    }
  }
}
