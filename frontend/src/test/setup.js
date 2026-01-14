/**
 * Vitest Test Setup
 * 
 * This file is run before each test file.
 * It sets up global test utilities and mocks.
 */
import '@testing-library/jest-dom'

// Mock window.matchMedia for components that use media queries
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: (query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => {},
  }),
})

// Mock IntersectionObserver
class MockIntersectionObserver {
  constructor() {
    this.observe = () => null
    this.unobserve = () => null
    this.disconnect = () => null
  }
}

Object.defineProperty(window, 'IntersectionObserver', {
  writable: true,
  configurable: true,
  value: MockIntersectionObserver,
})

// Mock ResizeObserver
class MockResizeObserver {
  constructor() {
    this.observe = () => null
    this.unobserve = () => null
    this.disconnect = () => null
  }
}

Object.defineProperty(window, 'ResizeObserver', {
  writable: true,
  configurable: true,
  value: MockResizeObserver,
})

// Suppress console errors during tests (optional)
// const originalError = console.error
// beforeAll(() => {
//   console.error = (...args) => {
//     if (args[0]?.includes('Warning:')) return
//     originalError.call(console, ...args)
//   }
// })
// afterAll(() => {
//   console.error = originalError
// })
