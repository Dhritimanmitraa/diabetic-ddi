import { Moon, Sun } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

/**
 * ThemeToggle — Clean, minimal dark/light mode switch
 */
export default function ThemeToggle({ className = '' }) {
    const { isDark, toggleTheme } = useTheme();

    return (
        <button
            onClick={toggleTheme}
            className={`p-2 rounded-lg transition-colors duration-200 ${isDark
                    ? 'text-slate-400 hover:text-amber-300 hover:bg-[var(--bg-elevated)]'
                    : 'text-slate-500 hover:text-amber-500 hover:bg-slate-100'
                } ${className}`}
            aria-label={`Switch to ${isDark ? 'light' : 'dark'} mode`}
            title={`Switch to ${isDark ? 'light' : 'dark'} mode`}
        >
            {isDark ? <Moon className="w-4.5 h-4.5" /> : <Sun className="w-4.5 h-4.5" />}
        </button>
    );
}

/**
 * ThemeToggleExpanded - Segmented control variant
 */
export function ThemeToggleExpanded({ className = '' }) {
    const { setDarkMode, setLightMode, isDark, isLight } = useTheme();

    return (
        <div className={`flex items-center gap-0.5 p-1 rounded-lg bg-[var(--bg-elevated)] border border-[var(--border)] ${className}`}>
            <button
                onClick={setLightMode}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-all text-sm font-medium ${isLight
                        ? 'bg-white text-amber-600 shadow-sm'
                        : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
                    }`}
            >
                <Sun className="w-3.5 h-3.5" />
                Light
            </button>
            <button
                onClick={setDarkMode}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-all text-sm font-medium ${isDark
                        ? 'bg-[var(--bg-secondary)] text-amber-300 shadow-sm'
                        : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
                    }`}
            >
                <Moon className="w-3.5 h-3.5" />
                Dark
            </button>
        </div>
    );
}
