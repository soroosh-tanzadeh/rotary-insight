import React from 'react';
import { Language } from '../types';

interface ErrorMessageProps {
  darkMode: boolean;
  error: string;
  language?: Language;
}

export const ErrorMessage: React.FC<ErrorMessageProps> = ({ darkMode, error, language = 'fa' }) => {
  if (!error) {
    return null;
  }

  const isRtl = language === 'fa';
  const lines = error.split('\n');
  const hasMultipleLines = lines.length > 1;

  return (
    <div 
      className={`mb-6 rounded-xl overflow-hidden shadow-lg ${darkMode ? 'bg-gray-800 border border-red-500/50' : 'bg-white border border-red-300'}`}
      dir={isRtl ? 'rtl' : 'ltr'}
    >
      {/* Header */}
      <div className={`px-4 py-3 flex items-center gap-3 ${darkMode ? 'bg-red-900/50' : 'bg-red-50'}`}>
        <div className={`p-2 rounded-full ${darkMode ? 'bg-red-500/20' : 'bg-red-100'}`}>
          <svg 
            className={`w-5 h-5 ${darkMode ? 'text-red-400' : 'text-red-600'}`} 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </div>
        <span className={`font-bold ${darkMode ? 'text-red-300' : 'text-red-700'}`}>
          {isRtl ? 'توجه' : 'Attention'}
        </span>
      </div>

      {/* Content */}
      <div className={`px-4 py-4 ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
        {hasMultipleLines ? (
          <div className="space-y-3">
            <p className={`font-semibold ${darkMode ? 'text-gray-100' : 'text-gray-800'}`}>
              {lines[0]}
            </p>
            <ul className={`space-y-2 ${isRtl ? 'pr-4' : 'pl-4'}`}>
              {lines.slice(1).map((line, index) => (
                <li 
                  key={index} 
                  className={`flex items-start gap-2 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}
                >
                  <span className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold flex-shrink-0 ${darkMode ? 'bg-primary/30 text-primary' : 'bg-primary/10 text-primary'}`}>
                    {index + 1}
                  </span>
                  <span className="pt-0.5">{line}</span>
                </li>
              ))}
            </ul>
          </div>
        ) : (
          <p className="font-medium">{error}</p>
        )}
      </div>
    </div>
  );
};

