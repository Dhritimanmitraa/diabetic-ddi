/**
 * DrugGuard Service Worker
 * 
 * Provides offline capability by caching static assets and API responses.
 * Uses a cache-first strategy for static assets and network-first for API calls.
 */

const CACHE_NAME = 'drugguard-v1'
const STATIC_CACHE_NAME = 'drugguard-static-v1'
const API_CACHE_NAME = 'drugguard-api-v1'

// Static assets to cache on install
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/offline.html',
]

// API routes that can be cached
const CACHEABLE_API_ROUTES = [
  '/drugs/search',
  '/stats',
  '/ml/model-info',
  '/health',
]

// Cache duration for API responses (in seconds)
const API_CACHE_TTL = 3600 // 1 hour

/**
 * Install event - cache static assets
 */
self.addEventListener('install', (event) => {
  console.log('[ServiceWorker] Install')
  
  event.waitUntil(
    caches.open(STATIC_CACHE_NAME)
      .then((cache) => {
        console.log('[ServiceWorker] Caching static assets')
        return cache.addAll(STATIC_ASSETS)
      })
      .then(() => self.skipWaiting())
      .catch((error) => {
        console.error('[ServiceWorker] Install failed:', error)
      })
  )
})

/**
 * Activate event - clean up old caches
 */
self.addEventListener('activate', (event) => {
  console.log('[ServiceWorker] Activate')
  
  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames
            .filter((name) => {
              // Delete old versions of our caches
              return name.startsWith('drugguard-') && 
                     name !== CACHE_NAME && 
                     name !== STATIC_CACHE_NAME &&
                     name !== API_CACHE_NAME
            })
            .map((name) => {
              console.log('[ServiceWorker] Deleting old cache:', name)
              return caches.delete(name)
            })
        )
      })
      .then(() => self.clients.claim())
  )
})

/**
 * Fetch event - serve from cache or network
 */
self.addEventListener('fetch', (event) => {
  const { request } = event
  const url = new URL(request.url)
  
  // Skip non-GET requests
  if (request.method !== 'GET') {
    return
  }
  
  // Skip chrome-extension and other protocols
  if (!url.protocol.startsWith('http')) {
    return
  }
  
  // Handle API requests
  if (isApiRequest(url)) {
    event.respondWith(handleApiRequest(request, url))
    return
  }
  
  // Handle static assets with cache-first strategy
  event.respondWith(handleStaticRequest(request))
})

/**
 * Check if request is an API request
 */
function isApiRequest(url) {
  // Check if URL contains /api/ or is to the API server
  return url.pathname.includes('/api/') || 
         CACHEABLE_API_ROUTES.some(route => url.pathname.includes(route))
}

/**
 * Handle API requests with network-first, cache-fallback strategy
 */
async function handleApiRequest(request, url) {
  // Check if this route is cacheable
  const isCacheable = CACHEABLE_API_ROUTES.some(route => url.pathname.includes(route))
  
  try {
    // Try network first
    const networkResponse = await fetch(request)
    
    // Cache successful GET responses for cacheable routes
    if (networkResponse.ok && isCacheable) {
      const cache = await caches.open(API_CACHE_NAME)
      
      // Add timestamp to response for TTL checking
      const responseToCache = networkResponse.clone()
      const headers = new Headers(responseToCache.headers)
      headers.set('sw-cache-time', Date.now().toString())
      
      const modifiedResponse = new Response(responseToCache.body, {
        status: responseToCache.status,
        statusText: responseToCache.statusText,
        headers: headers
      })
      
      cache.put(request, modifiedResponse)
    }
    
    return networkResponse
  } catch (error) {
    console.log('[ServiceWorker] Network failed, trying cache:', url.pathname)
    
    // Network failed, try cache
    const cachedResponse = await caches.match(request)
    
    if (cachedResponse) {
      // Check if cache is still valid
      const cacheTime = cachedResponse.headers.get('sw-cache-time')
      if (cacheTime) {
        const age = (Date.now() - parseInt(cacheTime)) / 1000
        if (age < API_CACHE_TTL) {
          console.log('[ServiceWorker] Serving from cache:', url.pathname)
          return cachedResponse
        }
      }
      
      // Return stale cache if nothing else available
      console.log('[ServiceWorker] Serving stale cache:', url.pathname)
      return cachedResponse
    }
    
    // No cache available, return error response
    return new Response(
      JSON.stringify({ 
        error: 'Offline', 
        message: 'You are offline and this data is not cached.' 
      }),
      { 
        status: 503, 
        statusText: 'Service Unavailable',
        headers: { 'Content-Type': 'application/json' }
      }
    )
  }
}

/**
 * Handle static requests with cache-first strategy
 */
async function handleStaticRequest(request) {
  // Check cache first
  const cachedResponse = await caches.match(request)
  if (cachedResponse) {
    return cachedResponse
  }
  
  try {
    // Not in cache, fetch from network
    const networkResponse = await fetch(request)
    
    // Cache successful responses
    if (networkResponse.ok) {
      const cache = await caches.open(STATIC_CACHE_NAME)
      cache.put(request, networkResponse.clone())
    }
    
    return networkResponse
  } catch (error) {
    console.log('[ServiceWorker] Static fetch failed:', request.url)
    
    // Return offline page for navigation requests
    if (request.mode === 'navigate') {
      const offlinePage = await caches.match('/offline.html')
      if (offlinePage) {
        return offlinePage
      }
    }
    
    // Return a simple error response
    return new Response('Offline', { status: 503, statusText: 'Service Unavailable' })
  }
}

/**
 * Handle messages from the main thread
 */
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting()
  }
  
  if (event.data && event.data.type === 'CLEAR_CACHE') {
    event.waitUntil(
      caches.keys().then((cacheNames) => {
        return Promise.all(
          cacheNames.map((name) => caches.delete(name))
        )
      })
    )
  }
})

/**
 * Background sync for queued requests (if supported)
 */
self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-pending-requests') {
    event.waitUntil(syncPendingRequests())
  }
})

async function syncPendingRequests() {
  // This would sync any queued requests when back online
  // Implementation depends on your queuing strategy
  console.log('[ServiceWorker] Syncing pending requests')
}
