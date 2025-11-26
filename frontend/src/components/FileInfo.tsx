import React from 'react';
import { Language } from '../types';
import { translations } from '../constants/translations';

interface FileInfoProps {
  darkMode: boolean;
  language: Language;
  fileName: string | undefined;
  selectedModel: string;
  windowSize: number | '';
}

export const FileInfo: React.FC<FileInfoProps> = ({
  darkMode,
  language,
  fileName,
  selectedModel,
  windowSize,
}) => {
  const t = translations[language];

  return (
    <div className={`mb-6 p-4 rounded-lg ${darkMode ? 'bg-gray-800' : 'bg-white'} shadow-lg transition-colors duration-300`}>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
        <div>
          <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'} font-persian`}>{t.file}</p>
          <p className={`text-lg font-semibold ${darkMode ? 'text-white' : 'text-gray-800'} font-persian`}>
            {fileName || '-'}
          </p>
        </div>
        <div>
          <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'} font-persian`}>{t.model}</p>
          <p className={`text-lg font-semibold ${darkMode ? 'text-white' : 'text-gray-800'} font-persian`}>
            {selectedModel || '-'}
          </p>
        </div>
        <div>
          <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'} font-persian`}>{t.windowSize}</p>
          <p className={`text-lg font-semibold ${darkMode ? 'text-white' : 'text-gray-800'} font-persian`}>
            {windowSize || '-'}
          </p>
        </div>
      </div>
    </div>
  );
};

