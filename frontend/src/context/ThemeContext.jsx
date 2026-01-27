import { createContext, useContext, useEffect, useState } from 'react';

/**
 * Theme Context for Dark/Light mode support
 * 
 * Provides theme state management with localStorage persistence
 * and system preference detection.
 */

const ThemeContext = createContext(undefined);

export function ThemeProvider({ children }) {
    const [theme, setTheme] = useState(() => {
        // Check localStorage first
        if (typeof window !== 'undefined') {
            const saved = localStorage.getItem('drugguard-theme');
            if (saved) return saved;

            // Check system preference
            if (window.matchMedia('(prefers-color-scheme: light)').matches) {
                return 'light';
            }
        }
        return 'dark'; // Default to dark
    });

    useEffect(() => {
        const root = window.document.documentElement;

        // Remove both classes first
        root.classList.remove('light', 'dark');

        // Add current theme class
        root.classList.add(theme);

        // Save to localStorage
        localStorage.setItem('drugguard-theme', theme);

        // Update meta theme-color for mobile browsers
        const metaTheme = document.querySelector('meta[name="theme-color"]');
        if (metaTheme) {
            metaTheme.setAttribute('content', theme === 'dark' ? '#0d1321' : '#f0fdf9');
        }
    }, [theme]);

    // Listen for system preference changes
    useEffect(() => {
        const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

        const handleChange = (e) => {
            const savedTheme = localStorage.getItem('drugguard-theme');
            // Only auto-switch if user hasn't set a preference
            if (!savedTheme) {
                setTheme(e.matches ? 'dark' : 'light');
            }
        };

        mediaQuery.addEventListener('change', handleChange);
        return () => mediaQuery.removeEventListener('change', handleChange);
    }, []);

    const toggleTheme = () => {
        setTheme(prev => prev === 'dark' ? 'light' : 'dark');
    };

    const setDarkMode = () => setTheme('dark');
    const setLightMode = () => setTheme('light');

    return (
        <ThemeContext.Provider value={{
            theme,
            toggleTheme,
            setDarkMode,
            setLightMode,
            isDark: theme === 'dark',
            isLight: theme === 'light'
        }}>
            {children}
        </ThemeContext.Provider>
    );
}

export function useTheme() {
    const context = useContext(ThemeContext);
    if (context === undefined) {
        throw new Error('useTheme must be used within a ThemeProvider');
    }
    return context;
}

export default ThemeContext;
