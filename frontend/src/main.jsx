import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'
import { registerServiceWorker } from './utils/serviceWorker'

// Register service worker for offline support
registerServiceWorker({
  onSuccess: () => {
    console.log('DrugGuard is now available offline!')
  },
  onUpdate: () => {
    console.log('New version of DrugGuard available. Refresh to update.')
    // Could show a toast notification here
  },
  onOffline: () => {
    console.log('You are now offline. Some features may be limited.')
  },
  onOnline: () => {
    console.log('Back online!')
  },
})

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
