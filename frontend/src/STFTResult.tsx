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
        <div className="mb-16">
            <h3 className={`text-xl font-semibold mb-2 text-center ${darkMode ? 'text-gray-200' : 'text-gray-700'} font-persian`}>
                {translations[language].stft}
            </h3>

            {/* Hop Length Input and Recalculate Button */}
            <div className="mb-4 flex justify-center items-center gap-4">
                <label className={`font-semibold ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
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
                    className={`w-32 px-4 py-2 rounded-lg border ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300'} focus:outline-none focus:ring-2 focus:ring-primary`}
                    placeholder={windowSize && typeof windowSize === 'number' ? Math.floor(windowSize / 4).toString() : ''}
                />
                <button
                    onClick={onRecalculate}
                    disabled={isLoading}
                    className={`px-4 py-2 rounded-lg font-semibold transition-colors ${darkMode
                        ? 'bg-primary text-white hover:bg-blue-600 disabled:bg-gray-700 disabled:text-gray-500'
                        : 'bg-primary text-white hover:bg-blue-600 disabled:bg-gray-300 disabled:text-gray-500'
                        }`}
                >
                    {isLoading ? (translations[language].calculating || 'Calculating...') : (translations[language].calculate || 'Calculate')}
                </button>
            </div>

            <div className="h-64 relative">
                {isLoading && (
                    <div className={`absolute inset-0 z-10 flex items-center justify-center backdrop-blur-sm ${darkMode ? 'bg-gray-900/50' : 'bg-white/50'} rounded-lg`}>
                        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
                    </div>
                )}
                <img
                    src={`data:${stftData.file_type || 'image/png'};base64,${stftData.image_base64}`}
                    alt="STFT Spectrogram"
                    className={`w-full h-full object-contain transition-opacity duration-300 ${isLoading ? 'opacity-50' : 'opacity-100'}`}
                />
                {/* Legend/Info Overlay */}
                <div className={`absolute top-2 right-2 ${darkMode ? 'bg-gray-800/90' : 'bg-white/90'} rounded-lg p-3 text-xs shadow-lg border ${darkMode ? 'border-gray-600' : 'border-gray-300'}`}>
                    <div className="space-y-1">
                        <div className={`font-semibold mb-2 ${darkMode ? 'text-white' : 'text-gray-800'}`}>
                            {translations[language].parameters || 'Parameters'}
                        </div>
                        {windowSize && typeof windowSize === 'number' && (
                            <div className={`${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                                <span className="font-medium">{translations[language].windowSize}:</span> {windowSize}
                            </div>
                        )}
                        {hopLength && typeof hopLength === 'number' && (
                            <div className={`${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                                <span className="font-medium">{translations[language].hopLength}:</span> {hopLength}
                            </div>
                        )}
                        {windowSize && typeof windowSize === 'number' && (
                            <div className={`${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                                <span className="font-medium">N_FFT:</span> {windowSize}
                            </div>
                        )}
                    </div>
                </div>
                {/* Axis Labels */}
                <div className={`absolute bottom-2 left-1/2 transform -translate-x-1/2 ${darkMode ? 'text-gray-300' : 'text-gray-700'} text-xs font-medium`}>
                    Time →
                </div>
                <div className={`absolute left-2 top-1/2 transform -translate-y-1/2 -rotate-90 ${darkMode ? 'text-gray-300' : 'text-gray-700'} text-xs font-medium`}>
                    Frequency →
                </div>
            </div>
            {/* Caption */}
            <div className={`mt-2 text-center text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                <p className="italic">
                    {translations[language].stftDescription || 'Time-Frequency representation showing signal energy distribution'}
                </p>
            </div>
        </div>
    );
};
