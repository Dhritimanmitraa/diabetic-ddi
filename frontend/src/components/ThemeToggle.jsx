import { Moon, Sun, Monitor } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';
import { motion } from 'framer-motion';

/**
 * ThemeToggle Component
 * 
 * A beautiful animated toggle button for switching between dark and light modes.
 */
export default function ThemeToggle({ className = '' }) {
    const { theme, toggleTheme, isDark } = useTheme();

    return (
        <motion.button
            onClick={toggleTheme}
            className={`relative p-2 rounded-xl transition-all duration-300 ${isDark
                    ? 'bg-slate-800 hover:bg-slate-700 text-yellow-400'
                    : 'bg-amber-100 hover:bg-amber-200 text-amber-600'
                } ${className}`}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            aria-label={`Switch to ${isDark ? 'light' : 'dark'} mode`}
            title={`Switch to ${isDark ? 'light' : 'dark'} mode`}
        >
            <motion.div
                initial={false}
                animate={{ rotate: isDark ? 0 : 180 }}
                transition={{ duration: 0.3 }}
            >
                {isDark ? (
                    <Moon className="w-5 h-5" />
                ) : (
                    <Sun className="w-5 h-5" />
                )}
            </motion.div>

            {/* Glow effect */}
            <motion.div
                className={`absolute inset-0 rounded-xl ${isDark ? 'bg-yellow-400' : 'bg-amber-400'
                    }`}
                initial={{ opacity: 0 }}
                animate={{ opacity: 0 }}
                whileHover={{ opacity: 0.1 }}
            />
        </motion.button>
    );
}

/**
 * ThemeToggleExpanded - Larger toggle with labels
 */
export function ThemeToggleExpanded({ className = '' }) {
    const { theme, setDarkMode, setLightMode, isDark, isLight } = useTheme();

    return (
        <div className={`flex items-center gap-1 p-1 rounded-xl bg-slate-800/50 dark:bg-slate-800/50 light:bg-slate-200/50 ${className}`}>
            <button
                onClick={setLightMode}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all ${isLight
                        ? 'bg-white text-amber-600 shadow-md'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
            >
                <Sun className="w-4 h-4" />
                <span className="text-sm font-medium">Light</span>
            </button>

            <button
                onClick={setDarkMode}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all ${isDark
                        ? 'bg-slate-700 text-yellow-400 shadow-md'
                        : 'text-slate-400 hover:text-slate-600'
                    }`}
            >
                <Moon className="w-4 h-4" />
                <span className="text-sm font-medium">Dark</span>
            </button>
        </div>
    );
}
