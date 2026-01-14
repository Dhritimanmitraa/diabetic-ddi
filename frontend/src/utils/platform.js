/**
 * Platform Detection Utilities for Capacitor Mobile App
 * Helps distinguish between web and native Android environments
 */

import { Capacitor } from '@capacitor/core'

/**
 * Check if running in native Capacitor environment
 * @returns {boolean} True if running in native app
 */
export function isNative() {
    return Capacitor.isNativePlatform()
}

/**
 * Check if running on Android
 * @returns {boolean} True if running on Android
 */
export function isAndroid() {
    return Capacitor.getPlatform() === 'android'
}

/**
 * Check if running on web
 * @returns {boolean} True if running in browser
 */
export function isWeb() {
    return Capacitor.getPlatform() === 'web'
}

/**
 * Get current platform name
 * @returns {string} Platform name ('android', 'ios', 'web')
 */
export function getPlatform() {
    return Capacitor.getPlatform()
}

/**
 * Get appropriate API base URL based on platform
 * For native apps, uses the configured mobile API URL
 * For web, uses the default localhost or environment URL
 * @returns {string} API base URL
 */
export function getApiBaseUrl() {
    const envUrl = import.meta.env.VITE_API_URL
    const mobileUrl = import.meta.env.VITE_API_URL_MOBILE

    // In native app, prefer mobile URL if set
    if (isNative() && mobileUrl) {
        return mobileUrl
    }

    // Fall back to environment URL or default
    return envUrl || 'http://localhost:8000'
}

/**
 * Check if camera is available on this platform
 * @returns {boolean} True if camera can be used
 */
export function hasCameraSupport() {
    // Native always has camera through Capacitor
    if (isNative()) return true

    // Web needs navigator.mediaDevices
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)
}

export default {
    isNative,
    isAndroid,
    isWeb,
    getPlatform,
    getApiBaseUrl,
    hasCameraSupport,
}
