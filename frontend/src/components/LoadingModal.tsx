import React from 'react';
import { Language } from '../types';
import { translations } from '../constants/translations';

interface LoadingModalProps {
  darkMode: boolean;
  language: Language;
  isVisible: boolean;
}

export const LoadingModal: React.FC<LoadingModalProps> = ({
  darkMode,
  language,
  isVisible,
}) => {
  const t = translations[language];

  if (!isVisible) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 backdrop-blur-sm transition-opacity duration-300">
      <div className={`relative p-8 rounded-xl ${darkMode ? 'bg-gray-800' : 'bg-white'} shadow-2xl transform transition-all duration-300 scale-100`}>
        <div className="flex flex-col items-center justify-center space-y-4">
          {/* Spinner Animation */}
          <div className="relative w-16 h-16">
            <div className={`absolute inset-0 border-4 ${darkMode ? 'border-gray-700' : 'border-gray-200'} rounded-full`}></div>
            <div className={`absolute inset-0 border-4 border-t-transparent ${darkMode ? 'border-primary' : 'border-primary'} rounded-full animate-spin`}></div>
          </div>
          <p className={`text-lg font-semibold ${darkMode ? 'text-gray-200' : 'text-gray-700'} font-persian`}>
            {t.loadingCharts}
          </p>
        </div>
      </div>
    </div>
  );
};

