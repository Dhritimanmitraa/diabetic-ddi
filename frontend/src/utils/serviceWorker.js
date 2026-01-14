/**
 * Service Worker Registration Utilities
 * 
 * Handles registration, updates, and communication with the service worker.
 */

/**
 * Check if service workers are supported
 */
export function isServiceWorkerSupported() {
  return 'serviceWorker' in navigator
}

/**
 * Register the service worker
 * 
 * @param {Object} options - Registration options
 * @param {Function} options.onSuccess - Called when SW is registered
 * @param {Function} options.onUpdate - Called when a new SW is available
 * @param {Function} options.onOffline - Called when app goes offline
 * @param {Function} options.onOnline - Called when app comes back online
 */
export async function registerServiceWorker(options = {}) {
  const { onSuccess, onUpdate, onOffline, onOnline } = options

  if (!isServiceWorkerSupported()) {
    console.log('[SW] Service workers not supported')
    return null
  }

  // Only register in production or if explicitly enabled
  if (import.meta.env.DEV && !import.meta.env.VITE_ENABLE_SW) {
    console.log('[SW] Skipping registration in development')
    return null
  }

  try {
    const registration = await navigator.serviceWorker.register('/sw.js', {
      scope: '/',
    })

    console.log('[SW] Registered successfully:', registration.scope)

    // Handle updates
    registration.addEventListener('updatefound', () => {
      const newWorker = registration.installing
      
      if (newWorker) {
        newWorker.addEventListener('statechange', () => {
          if (newWorker.state === 'installed') {
            if (navigator.serviceWorker.controller) {
              // New content is available
              console.log('[SW] New content available, refresh to update')
              if (onUpdate) {
                onUpdate(registration)
              }
            } else {
              // Content is cached for offline use
              console.log('[SW] Content cached for offline use')
              if (onSuccess) {
                onSuccess(registration)
              }
            }
          }
        })
      }
    })

    // Setup online/offline listeners
    if (onOffline) {
      window.addEventListener('offline', onOffline)
    }
    
    if (onOnline) {
      window.addEventListener('online', onOnline)
    }

    return registration
  } catch (error) {
    console.error('[SW] Registration failed:', error)
    return null
  }
}

/**
 * Unregister all service workers
 */
export async function unregisterServiceWorker() {
  if (!isServiceWorkerSupported()) {
    return false
  }

  try {
    const registration = await navigator.serviceWorker.ready
    const success = await registration.unregister()
    
    if (success) {
      console.log('[SW] Unregistered successfully')
    }
    
    return success
  } catch (error) {
    console.error('[SW] Unregistration failed:', error)
    return false
  }
}

/**
 * Skip waiting and activate the new service worker
 */
export async function skipWaiting() {
  if (!isServiceWorkerSupported()) {
    return
  }

  const registration = await navigator.serviceWorker.ready
  
  if (registration.waiting) {
    registration.waiting.postMessage({ type: 'SKIP_WAITING' })
  }
}

/**
 * Clear all caches
 */
export async function clearCaches() {
  if (!isServiceWorkerSupported()) {
    return
  }

  const registration = await navigator.serviceWorker.ready
  
  if (registration.active) {
    registration.active.postMessage({ type: 'CLEAR_CACHE' })
  }
}

/**
 * Check if a new service worker update is available
 */
export async function checkForUpdates() {
  if (!isServiceWorkerSupported()) {
    return false
  }

  try {
    const registration = await navigator.serviceWorker.ready
    await registration.update()
    return true
  } catch (error) {
    console.error('[SW] Update check failed:', error)
    return false
  }
}

/**
 * Get the current service worker state
 */
export async function getServiceWorkerState() {
  if (!isServiceWorkerSupported()) {
    return 'unsupported'
  }

  try {
    const registration = await navigator.serviceWorker.getRegistration()
    
    if (!registration) {
      return 'unregistered'
    }
    
    if (registration.installing) {
      return 'installing'
    }
    
    if (registration.waiting) {
      return 'waiting'
    }
    
    if (registration.active) {
      return 'active'
    }
    
    return 'unknown'
  } catch (error) {
    console.error('[SW] State check failed:', error)
    return 'error'
  }
}

/**
 * Hook to use service worker in React components
 * 
 * Usage:
 *   import { useServiceWorker } from './utils/serviceWorker'
 *   
 *   function App() {
 *     const { isOnline, needsRefresh, updateServiceWorker } = useServiceWorker()
 *     
 *     if (needsRefresh) {
 *       return <button onClick={updateServiceWorker}>Update Available</button>
 *     }
 *   }
 */
export function createServiceWorkerHook() {
  let isOnline = navigator.onLine
  let needsRefresh = false
  let registration = null
  const listeners = new Set()

  const notify = () => {
    listeners.forEach(listener => listener({ isOnline, needsRefresh }))
  }

  // Register service worker
  registerServiceWorker({
    onSuccess: (reg) => {
      registration = reg
      notify()
    },
    onUpdate: (reg) => {
      registration = reg
      needsRefresh = true
      notify()
    },
    onOffline: () => {
      isOnline = false
      notify()
    },
    onOnline: () => {
      isOnline = true
      notify()
    },
  })

  return {
    subscribe: (listener) => {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
    getState: () => ({ isOnline, needsRefresh }),
    updateServiceWorker: async () => {
      if (registration?.waiting) {
        await skipWaiting()
        window.location.reload()
      }
    },
  }
}
