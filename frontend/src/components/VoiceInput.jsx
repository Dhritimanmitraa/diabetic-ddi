import { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Mic, MicOff, Loader2, Volume2 } from 'lucide-react';

/**
 * VoiceInput Component
 * 
 * Provides voice-to-text input using the Web Speech API.
 * Supports continuous listening with visual feedback.
 */
export default function VoiceInput({
    onResult,
    onInterimResult,
    placeholder = "Speak now...",
    className = "",
    disabled = false
}) {
    const [isListening, setIsListening] = useState(false);
    const [isSupported, setIsSupported] = useState(true);
    const [transcript, setTranscript] = useState('');
    const [interimTranscript, setInterimTranscript] = useState('');
    const [error, setError] = useState(null);
    const recognitionRef = useRef(null);

    // Check for browser support
    useEffect(() => {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            setIsSupported(false);
            setError('Voice input is not supported in this browser');
            return;
        }

        // Initialize speech recognition
        const recognition = new SpeechRecognition();
        recognition.continuous = true;  // Keep listening for longer speech
        recognition.interimResults = true;
        recognition.lang = 'en-US';
        recognition.maxAlternatives = 3;  // More alternatives for better accuracy

        recognition.onstart = () => {
            setIsListening(true);
            setError(null);
            console.log('[VoiceInput] Started listening...');
        };

        recognition.onend = () => {
            console.log('[VoiceInput] Recognition ended');
            // Auto-restart if still supposed to be listening (handles browser auto-stop)
            if (recognitionRef.current && recognitionRef.current._shouldListen) {
                try {
                    recognition.start();
                    console.log('[VoiceInput] Auto-restarting...');
                } catch (e) {
                    // Failed to restart, stop listening
                    setIsListening(false);
                    recognitionRef.current._shouldListen = false;
                }
            } else {
                setIsListening(false);
            }
        };

        recognition.onerror = (event) => {
            console.log('[VoiceInput] Error:', event.error);
            switch (event.error) {
                case 'no-speech':
                    // Don't stop on no-speech, just show a hint
                    setError('No speech detected. Keep speaking...');
                    // Clear error after 2 seconds
                    setTimeout(() => setError(null), 2000);
                    return;  // Don't stop listening
                case 'audio-capture':
                    setError('No microphone found. Please connect a microphone.');
                    break;
                case 'not-allowed':
                    setError('Microphone access denied. Please allow microphone access.');
                    break;
                case 'aborted':
                    // User stopped, don't show error
                    break;
                default:
                    setError(`Error: ${event.error}`);
            }
            setIsListening(false);
            if (recognitionRef.current) {
                recognitionRef.current._shouldListen = false;
            }
        };

        recognition.onresult = (event) => {
            let finalTranscript = '';
            let interimText = '';

            for (let i = event.resultIndex; i < event.results.length; i++) {
                const result = event.results[i];
                if (result.isFinal) {
                    finalTranscript += result[0].transcript;
                } else {
                    interimText += result[0].transcript;
                }
            }

            if (interimText) {
                setInterimTranscript(interimText);
                onInterimResult?.(interimText);
            }

            if (finalTranscript) {
                setTranscript(finalTranscript);
                setInterimTranscript('');
                onResult?.(finalTranscript.trim());
            }
        };

        recognitionRef.current = recognition;

        return () => {
            if (recognitionRef.current) {
                recognitionRef.current.abort();
            }
        };
    }, [onResult, onInterimResult]);

    const toggleListening = useCallback(() => {
        if (!recognitionRef.current) return;

        if (isListening) {
            recognitionRef.current._shouldListen = false;
            recognitionRef.current.stop();
            setIsListening(false);
        } else {
            setTranscript('');
            setInterimTranscript('');
            setError(null);
            recognitionRef.current._shouldListen = true;
            try {
                recognitionRef.current.start();
            } catch (e) {
                console.log('[VoiceInput] Start error:', e);
                // Already started, try to restart
                recognitionRef.current.stop();
                setTimeout(() => {
                    try {
                        recognitionRef.current?.start();
                    } catch (e2) {
                        console.log('[VoiceInput] Restart failed:', e2);
                    }
                }, 100);
            }
        }
    }, [isListening]);

    if (!isSupported) {
        return (
            <div className={`flex items-center gap-2 text-slate-500 text-sm ${className}`}>
                <MicOff className="w-4 h-4" />
                <span>Voice input not supported</span>
            </div>
        );
    }

    return (
        <div className={`relative ${className}`}>
            {/* Voice Input Button */}
            <motion.button
                type="button"
                onClick={toggleListening}
                disabled={disabled}
                className={`relative flex items-center justify-center p-3 rounded-xl transition-all duration-300 ${isListening
                    ? 'bg-red-500 text-white shadow-lg shadow-red-500/30'
                    : 'bg-slate-200 dark:bg-slate-800 text-slate-400 hover:text-white hover:bg-slate-300 dark:hover:bg-slate-700'
                    } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
                whileHover={{ scale: disabled ? 1 : 1.05 }}
                whileTap={{ scale: disabled ? 1 : 0.95 }}
                aria-label={isListening ? 'Stop listening' : 'Start voice input'}
                title={isListening ? 'Click to stop' : 'Click to speak'}
            >
                {/* Pulsing animation when listening */}
                <AnimatePresence>
                    {isListening && (
                        <>
                            <motion.div
                                className="absolute inset-0 rounded-xl bg-red-500"
                                initial={{ scale: 1, opacity: 0.5 }}
                                animate={{ scale: 1.5, opacity: 0 }}
                                exit={{ opacity: 0 }}
                                transition={{ duration: 1, repeat: Infinity }}
                            />
                            <motion.div
                                className="absolute inset-0 rounded-xl bg-red-500"
                                initial={{ scale: 1, opacity: 0.5 }}
                                animate={{ scale: 1.8, opacity: 0 }}
                                exit={{ opacity: 0 }}
                                transition={{ duration: 1, repeat: Infinity, delay: 0.3 }}
                            />
                        </>
                    )}
                </AnimatePresence>

                {isListening ? (
                    <motion.div
                        animate={{ scale: [1, 1.2, 1] }}
                        transition={{ duration: 0.5, repeat: Infinity }}
                    >
                        <Volume2 className="w-5 h-5 relative z-10" />
                    </motion.div>
                ) : (
                    <Mic className="w-5 h-5" />
                )}
            </motion.button>

            {/* Transcript display */}
            <AnimatePresence>
                {(isListening || interimTranscript || transcript) && (
                    <motion.div
                        initial={{ opacity: 0, y: 10, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: -10, scale: 0.95 }}
                        className="absolute top-full mt-2 left-0 right-0 min-w-[200px] p-3 rounded-lg bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 shadow-xl z-50"
                    >
                        {isListening && !interimTranscript && !transcript ? (
                            <div className="flex items-center gap-2 text-slate-400">
                                <Loader2 className="w-4 h-4 animate-spin" />
                                <span className="text-sm">{placeholder}</span>
                            </div>
                        ) : (
                            <div className="text-sm">
                                {transcript && (
                                    <p className="text-slate-900 dark:text-white font-medium">
                                        {transcript}
                                    </p>
                                )}
                                {interimTranscript && (
                                    <p className="text-slate-400 italic">
                                        {interimTranscript}
                                    </p>
                                )}
                            </div>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Error display */}
            <AnimatePresence>
                {error && (
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                        className="absolute top-full mt-2 left-0 right-0 p-2 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-xs"
                    >
                        {error}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

/**
 * VoiceInputInline - Inline variant for search fields
 */
export function VoiceInputInline({ onResult, className = "" }) {
    const [isListening, setIsListening] = useState(false);
    const [isSupported, setIsSupported] = useState(true);
    const recognitionRef = useRef(null);

    useEffect(() => {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            setIsSupported(false);
            return;
        }

        const recognition = new SpeechRecognition();
        recognition.continuous = true;  // Keep listening for longer speech
        recognition.interimResults = true;  // Get results as user speaks
        recognition.lang = 'en-US';
        recognition.maxAlternatives = 3;

        recognition.onstart = () => setIsListening(true);

        recognition.onend = () => {
            // Auto-restart if still supposed to be listening
            if (recognitionRef.current && recognitionRef.current._shouldListen) {
                try {
                    recognition.start();
                } catch (e) {
                    setIsListening(false);
                    recognitionRef.current._shouldListen = false;
                }
            } else {
                setIsListening(false);
            }
        };

        recognition.onerror = (event) => {
            if (event.error === 'no-speech') {
                // Don't stop on no-speech, keep listening
                return;
            }
            setIsListening(false);
            if (recognitionRef.current) {
                recognitionRef.current._shouldListen = false;
            }
        };

        recognition.onresult = (event) => {
            // Get the most recent final result
            for (let i = event.resultIndex; i < event.results.length; i++) {
                if (event.results[i].isFinal) {
                    const result = event.results[i][0].transcript;
                    onResult?.(result.trim());
                }
            }
        };

        recognitionRef.current = recognition;

        return () => recognitionRef.current?.abort();
    }, [onResult]);

    const toggleListening = useCallback(() => {
        if (!recognitionRef.current) return;

        if (isListening) {
            recognitionRef.current._shouldListen = false;
            recognitionRef.current.stop();
            setIsListening(false);
        } else {
            recognitionRef.current._shouldListen = true;
            try {
                recognitionRef.current.start();
            } catch (e) {
                // Already started, restart
                recognitionRef.current.stop();
                setTimeout(() => {
                    try {
                        recognitionRef.current?.start();
                    } catch (e2) {
                        // Ignore
                    }
                }, 100);
            }
        }
    }, [isListening]);

    if (!isSupported) return null;

    return (
        <button
            type="button"
            onClick={toggleListening}
            className={`p-2 rounded-lg transition-all ${isListening
                ? 'text-red-400 bg-red-500/10 animate-pulse'
                : 'text-slate-400 hover:text-medical-400 hover:bg-slate-800/50'
                } ${className}`}
            aria-label={isListening ? 'Stop listening' : 'Voice input'}
            title="Voice input"
        >
            {isListening ? (
                <Volume2 className="w-5 h-5" />
            ) : (
                <Mic className="w-5 h-5" />
            )}
        </button>
    );
}
