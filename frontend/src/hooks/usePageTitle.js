import { useEffect } from 'react'

/**
 * Sets the document title when a component mounts.
 * Restores the base title on unmount.
 */
export default function usePageTitle(title) {
    useEffect(() => {
        const base = 'DrugGuard'
        document.title = title ? `${title} | ${base}` : base
        return () => { document.title = base }
    }, [title])
}
