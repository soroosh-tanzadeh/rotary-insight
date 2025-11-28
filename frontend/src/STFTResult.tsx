import React from 'react';

interface STFTResultProps {
    stftData: { file_name: string; file_type?: string; image_base64: string } | null;
    windowSize: number | '';
    hopLength: number | '';
    onHopLengthChange: (value: number | '') => void;
    onRecalculate: () => void;
    isLoading: boolean;
    darkMode: boolean;
    translations: any;
    language: 'fa' | 'en';
}

export const STFTResult: React.FC<STFTResultProps> = ({
    stftData,
    windowSize,
    hopLength,
    onHopLengthChange,
    onRecalculate,
    isLoading,
    darkMode,
    translations,
    language,
}) => {
    if (!stftData || !stftData.image_base64) return null;

    return (
        <div className={`p-4 rounded-2xl ${darkMode ? 'bg-gray-900/30' : 'bg-gray-50'}`}>
            {/* Hop Length Input and Recalculate Button */}
            <div className="mb-6 flex flex-wrap justify-center items-center gap-4">
                <div className={`flex items-center gap-3 px-4 py-2 rounded-xl ${darkMode ? 'bg-gray-800' : 'bg-white'} shadow-sm`}>
                    <label className={`text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                        {translations[language].hopLength}
                    </label>
                    <input
                        type="number"
                        value={hopLength === '' ? '' : hopLength}
                        onChange={(e) => {
                            const value = e.target.value === '' ? '' : parseInt(e.target.value);
                            if (value === '' || (!isNaN(value as number) && value > 0)) {
                                onHopLengthChange(value);
                            }
                        }}
                        min="1"
                        className={`w-24 px-3 py-1.5 rounded-lg border text-center ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-gray-50 border-gray-200'} focus:outline-none focus:ring-2 focus:ring-primary/50`}
                        placeholder={windowSize && typeof windowSize === 'number' ? Math.floor(windowSize / 4).toString() : ''}
                    />
                </div>
                <button
                    onClick={onRecalculate}
                    disabled={isLoading}
                    className={`group px-4 py-2 rounded-xl font-medium transition-all duration-200 flex items-center gap-2 border-2 ${darkMode
                        ? 'border-purple-500/50 bg-purple-500/10 text-purple-400 hover:bg-purple-500/20 hover:border-purple-400 disabled:border-gray-700 disabled:bg-gray-800 disabled:text-gray-500'
                        : 'border-purple-300 bg-purple-50 text-purple-600 hover:bg-purple-100 hover:border-purple-400 disabled:border-gray-200 disabled:bg-gray-100 disabled:text-gray-400'
                        } ${!isLoading && 'hover:scale-105'}`}
                >
                    {isLoading ? (
                        <>
                            <div className="animate-spin rounded-full h-4 w-4 border-2 border-purple-400/30 border-t-purple-400"></div>
                            <span className="text-sm">{language === 'fa' ? 'در حال محاسبه...' : 'Calculating...'}</span>
                        </>
                    ) : (
                        <>
                            <svg className="w-4 h-4 transition-transform group-hover:rotate-180 duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                            </svg>
                            <span className="text-sm">{language === 'fa' ? 'محاسبه مجدد' : 'Recalculate'}</span>
                        </>
                    )}
                </button>
            </div>

            {/* STFT Image Container */}
            <div className={`relative rounded-2xl overflow-hidden ${darkMode ? 'bg-gray-800' : 'bg-white'} shadow-lg`}>
                {isLoading && (
                    <div className={`absolute inset-0 z-10 flex items-center justify-center backdrop-blur-sm ${darkMode ? 'bg-gray-900/60' : 'bg-white/60'}`}>
                        <div className="flex flex-col items-center gap-3">
                            <div className="animate-spin rounded-full h-10 w-10 border-3 border-primary/30 border-t-primary"></div>
                            <span className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                                {language === 'fa' ? 'در حال پردازش...' : 'Processing...'}
                            </span>
                        </div>
                    </div>
                )}

                <div className="p-4">
                    <img
                        src={`data:${stftData.file_type || 'image/png'};base64,${stftData.image_base64}`}
                        alt="STFT Spectrogram"
                        className={`w-full h-full object-contain transition-opacity duration-300 rounded-lg ${isLoading ? 'opacity-40' : 'opacity-100'}`}
                    />
                </div>

                {/* Parameters Badge */}
                <div className={`absolute top-3 right-3 flex gap-2`}>
                    {windowSize && typeof windowSize === 'number' && (
                        <span className={`px-2.5 py-1 rounded-lg text-xs font-medium ${darkMode ? 'bg-blue-500/20 text-blue-300' : 'bg-blue-100 text-blue-700'}`}>
                            {language === 'fa' ? 'پنجره' : 'Window'}: {windowSize}
                        </span>
                    )}
                    {hopLength && typeof hopLength === 'number' && (
                        <span className={`px-2.5 py-1 rounded-lg text-xs font-medium ${darkMode ? 'bg-green-500/20 text-green-300' : 'bg-green-100 text-green-700'}`}>
                            Hop: {hopLength}
                        </span>
                    )}
                </div>
            </div>

            {/* Caption */}
            <p className={`mt-4 text-center text-sm ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                {translations[language].stftDescription || 'Time-Frequency representation showing signal energy distribution'}
            </p>
        </div>
    );
};
