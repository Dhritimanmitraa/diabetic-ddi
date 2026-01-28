import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import Navbar from '../Navbar'
import * as api from '../../services/api'

// Mock the API module
vi.mock('../../services/api', () => ({
  healthCheck: vi.fn(),
}))

// Mock ThemeToggle component to avoid its internal logic/context requirements
vi.mock('../ThemeToggle', () => ({
    default: () => <button>ThemeToggle</button>
}))

describe('Navbar Component', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the logo correctly', () => {
    render(
      <MemoryRouter>
        <Navbar />
      </MemoryRouter>
    )

    // There might be multiple "Drug" texts (Logo + Description)
    const drugElements = screen.getAllByText(/Drug/i)
    expect(drugElements.length).toBeGreaterThan(0)

    expect(screen.getByText(/Guard/i)).toBeInTheDocument()
  })

  it('displays "Online" status when health check passes', async () => {
    api.healthCheck.mockResolvedValue({ status: 'healthy' })

    render(
      <MemoryRouter>
        <Navbar />
      </MemoryRouter>
    )

    // Wait for the health check to complete
    await waitFor(() => {
        const statusElement = screen.getByText('Online')
        expect(statusElement).toBeInTheDocument()
    })
  })

  it('displays "Offline" status when health check fails', async () => {
    api.healthCheck.mockRejectedValue(new Error('Network Error'))

    render(
      <MemoryRouter>
        <Navbar />
      </MemoryRouter>
    )

    await waitFor(() => {
        const statusElement = screen.getByText('Offline')
        expect(statusElement).toBeInTheDocument()
    })
  })
})
