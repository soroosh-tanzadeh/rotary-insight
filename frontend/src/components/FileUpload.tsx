import React, { useRef, useState } from 'react';
import { Language } from '../types';
import { translations } from '../constants/translations';

interface FileUploadProps {
  darkMode: boolean;
  language: Language;
  csvFile: File | null;
  loading: boolean;
  onFileSelect: (file: File) => void;
  onCalculate: () => void;
}

export const FileUpload: React.FC<FileUploadProps> = ({
  darkMode,
  language,
  csvFile,
  loading,
  onFileSelect,
  onCalculate,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const t = translations[language];

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      onFileSelect(file);
    }
  };

  return (
    <section className={`mb-8 p-6 rounded-lg shadow-lg ${darkMode ? 'bg-gray-800' : 'bg-white'} transition-colors duration-300`}>
      <h2 className={`text-2xl font-bold mb-4 text-center ${darkMode ? 'text-white' : 'text-primary'} font-persian`}>
        {t.uploadCsv}
      </h2>
      
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-16 text-center cursor-pointer transition-all duration-300 ${isDragging
          ? darkMode
            ? 'border-primary bg-primary/20 scale-105 shadow-lg'
            : 'border-primary bg-primary/10 scale-105 shadow-lg'
          : darkMode
            ? 'border-gray-600 hover:border-primary hover:bg-gray-700/50'
            : 'border-gray-300 hover:border-primary hover:bg-primary/5'
          }`}
      >
        <div className="mb-6 flex justify-center">
          <svg
            className={`w-20 h-20 ${isDragging ? 'scale-110' : ''} transition-transform duration-300`}
            fill="none"
            stroke={darkMode ? '#E9E9E9' : '#385F8C'}
            strokeWidth="2"
            viewBox="0 0 24 24"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="17 8 12 3 7 8"></polyline>
            <line x1="12" y1="3" x2="12" y2="15"></line>
          </svg>
        </div>
        <p className={`text-xl mb-3 font-semibold ${darkMode ? 'text-gray-200' : 'text-gray-700'} font-persian`}>
          {t.dragDrop}
        </p>
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onFileSelect(file);
          }}
          className="hidden"
        />
      </div>

      {csvFile && (
        <div className={`mt-4 p-4 rounded-xl text-center border-2 ${darkMode ? 'bg-green-500/10 border-green-500/50' : 'bg-green-50 border-green-300'}`}>
          <p className={`${darkMode ? 'text-green-400' : 'text-green-700'} font-semibold`}>
            {language === 'fa' ? `فایل "${csvFile.name}" انتخاب شد` : `File "${csvFile.name}" selected`}
          </p>
        </div>
      )}

      {/* Calculate Button - Always visible */}
      <div className="mt-6 text-center">
        <button
          onClick={onCalculate}
          disabled={loading}
          className={`px-8 py-3 rounded-lg text-lg ${darkMode ? 'bg-primary hover:bg-primary/80' : 'bg-primary hover:bg-primary/90'} text-white font-semibold disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2 mx-auto`}
        >
          {loading ? (
            <>
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
              <span>{t.loading}</span>
            </>
          ) : (
            <>
              <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" strokeLinecap="round" strokeLinejoin="round">
                <path d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
              </svg>
              <span>{t.calculate}</span>
            </>
          )}
        </button>
      </div>
    </section>
  );
};

