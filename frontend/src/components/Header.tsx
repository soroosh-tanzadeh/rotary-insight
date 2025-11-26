import React from 'react';
import { Language } from '../types';
import { translations } from '../constants/translations';

interface HeaderProps {
  darkMode: boolean;
  language: Language;
  onDarkModeToggle: () => void;
  onLanguageChange: (lang: Language) => void;
  showBackButton?: boolean;
  onBackClick?: () => void;
  title?: string;
}

export const Header: React.FC<HeaderProps> = ({
  darkMode,
  language,
  onDarkModeToggle,
  onLanguageChange,
  showBackButton = false,
  onBackClick,
  title,
}) => {
  const t = translations[language];
  const displayTitle = title || t.title;

  return (
    <header className={`sticky top-0 z-50 w-full border-b ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} shadow-sm transition-colors duration-300`}>
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Left side */}
          <div className="flex items-center gap-4 flex-1">
            {showBackButton && (
              <button
                onClick={onBackClick}
                className={`p-3 rounded-lg ${darkMode ? 'bg-gray-700 text-white hover:bg-gray-600' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'} transition-colors`}
                title={t.back}
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M19 12H5M12 19l-7-7 7-7" />
                </svg>
              </button>
            )}
          </div>

          {/* Center - Logo and Title */}
          <div className="flex items-center justify-center gap-3 flex-1">
            <img
              src="/logo.png"
              alt="Rotary Insight Logo"
              className="h-16 w-16 object-contain"
            />
            <h1 className={`text-3xl font-bold ${darkMode ? 'text-white' : 'text-primary'} font-persian`}>
              {displayTitle}
            </h1>
          </div>

          {/* Right side - Controls */}
          <div className="flex items-center gap-4 flex-1 justify-end">
            {/* Language Toggle */}
            <div className="flex items-center border rounded-lg overflow-hidden">
              <button
                onClick={() => onLanguageChange('fa')}
                className={`px-3 py-1.5 text-sm ${language === 'fa' ? 'bg-primary text-white' : (darkMode ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' : 'bg-gray-200 text-gray-600 hover:bg-gray-300')} transition-colors`}
              >
                فا
              </button>
              <button
                onClick={() => onLanguageChange('en')}
                className={`px-3 py-1.5 text-sm border-r ${language === 'en' ? 'bg-primary text-white' : (darkMode ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' : 'bg-gray-200 text-gray-600 hover:bg-gray-300')} transition-colors ${darkMode ? 'border-gray-600' : 'border-gray-300'}`}
              >
                en
              </button>
            </div>

            {/* Dark Mode Toggle */}
            <button
              onClick={onDarkModeToggle}
              className={`p-2 rounded-lg ${darkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-200 hover:bg-gray-300'} transition-colors`}
              title={darkMode ? t.light : t.dark}
            >
              {darkMode ? (
                <svg className="w-6 h-6 text-yellow-400" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="5"></circle>
                  <line x1="12" y1="1" x2="12" y2="3"></line>
                  <line x1="12" y1="21" x2="12" y2="23"></line>
                  <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                  <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                  <line x1="1" y1="12" x2="3" y2="12"></line>
                  <line x1="21" y1="12" x2="23" y2="12"></line>
                  <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                  <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
                </svg>
              ) : (
                <svg className="w-6 h-6 text-gray-700" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

