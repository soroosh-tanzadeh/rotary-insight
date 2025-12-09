import React, { useState, useRef, useEffect } from 'react';
import { ExamplesResponse, Language } from '../types';
import { translations } from '../constants/translations';

interface SampleFilesProps {
  darkMode: boolean;
  language: Language;
  exampleFiles: ExamplesResponse;
  loadingExamples: boolean;
  selectedFileName: string | null;
  onLoadExample: (filename: string) => void;
}

export const SampleFiles: React.FC<SampleFilesProps> = ({
  darkMode,
  language,
  exampleFiles,
  loadingExamples,
  selectedFileName,
  onLoadExample,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [contentHeight, setContentHeight] = useState(0);
  const [activeTab, setActiveTab] = useState<string>('');
  const contentRef = useRef<HTMLDivElement>(null);
  const t = translations[language];

  // Get dataset names from the response
  const datasets = Object.keys(exampleFiles || {});

  // Set initial active tab when datasets are loaded
  useEffect(() => {
    if (datasets.length > 0 && !activeTab) {
      setActiveTab(datasets[0]);
    }
  }, [datasets.length, activeTab]);

  // Check if a sample is selected and find it
  let hasSelection = false;
  let selectedExample: any = null;
  let selectedDataset: string = '';

  datasets.forEach(datasetName => {
    const datasetFiles = exampleFiles[datasetName];
    if (Array.isArray(datasetFiles)) {
      const found = datasetFiles.find(e => e.filename === selectedFileName);
      if (found) {
        hasSelection = true;
        selectedExample = found;
        selectedDataset = datasetName;
      }
    }
  });

  // Update content height when expanded or content changes
  useEffect(() => {
    if (contentRef.current) {
      setContentHeight(contentRef.current.scrollHeight);
    }
  }, [exampleFiles, isExpanded, activeTab]);

  // Get current files for active tab
  const currentFiles = activeTab && exampleFiles[activeTab] && Array.isArray(exampleFiles[activeTab])
    ? exampleFiles[activeTab]
    : [];

  // Calculate total files count
  const totalFilesCount = datasets.reduce((count, dataset) => {
    const files = exampleFiles[dataset];
    return count + (Array.isArray(files) ? files.length : 0);
  }, 0);

  return (
    <section className={`mb-8 rounded-xl shadow-lg overflow-hidden ${darkMode ? 'bg-gray-800' : 'bg-white'} transition-colors duration-300`}>
      {/* Clickable Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={`w-full p-5 flex items-center justify-between transition-colors duration-200 ${darkMode
          ? 'hover:bg-gray-700/50'
          : 'hover:bg-gray-50'
          }`}
      >
        <div className="flex items-center gap-3">
          {/* Folder Icon with animation */}
          <div className={`p-2 rounded-lg transition-all duration-300 ${isExpanded
            ? (darkMode ? 'bg-primary/30 scale-110' : 'bg-primary/20 scale-110')
            : (darkMode ? 'bg-primary/20' : 'bg-primary/10')
            }`}>
            <svg
              className={`w-6 h-6 transition-transform duration-300 ${isExpanded ? 'scale-110' : ''} ${darkMode ? 'text-primary' : 'text-primary'}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              {isExpanded ? (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 19a2 2 0 01-2-2V7a2 2 0 012-2h4l2 2h4a2 2 0 012 2v1M5 19h14a2 2 0 002-2v-5a2 2 0 00-2-2H9a2 2 0 00-2 2v5a2 2 0 01-2 2z" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
              )}
            </svg>
          </div>

          <div className="text-start">
            <h2 className={`text-lg font-bold ${darkMode ? 'text-white' : 'text-gray-800'} font-persian`}>
              {t.orSelectSample}
            </h2>
            {hasSelection && selectedExample && (
              <p className={`text-sm ${darkMode ? 'text-green-400' : 'text-green-600'} flex items-center gap-1`}>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                </svg>
                {language === 'fa' ? 'انتخاب شده:' : 'Selected:'} {selectedExample.label} ({selectedDataset})
              </p>
            )}
            {!hasSelection && totalFilesCount > 0 && (
              <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                {totalFilesCount} {language === 'fa' ? 'فایل نمونه موجود' : 'sample files available'}
              </p>
            )}
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Badge showing count */}
          {totalFilesCount > 0 && (
            <span className={`px-3 py-1 rounded-full text-sm font-semibold transition-all duration-300 ${isExpanded
              ? (darkMode ? 'bg-primary text-white' : 'bg-primary text-white')
              : (darkMode ? 'bg-primary/20 text-primary' : 'bg-primary/10 text-primary')
              }`}>
              {totalFilesCount}
            </span>
          )}

          {/* Chevron Icon */}
          <div className={`p-1 rounded-full transition-all duration-300 ${isExpanded
            ? (darkMode ? 'bg-gray-700' : 'bg-gray-200')
            : ''
            }`}>
            <svg
              className={`w-5 h-5 transition-transform duration-300 ${isExpanded ? 'rotate-180' : ''} ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
            </svg>
          </div>
        </div>
      </button>

      {/* Expandable Content with smooth animation */}
      <div
        style={{
          maxHeight: isExpanded ? `${contentHeight}px` : '0px',
          transition: 'max-height 0.4s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.3s ease-in-out'
        }}
        className={`overflow-hidden ${isExpanded ? 'opacity-100' : 'opacity-0'}`}
      >
        <div
          ref={contentRef}
          className={`p-5 pt-0 border-t ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}
        >
          {loadingExamples ? (
            <div className="text-center py-8">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-primary mb-4"></div>
              <p className={`${darkMode ? 'text-gray-400' : 'text-gray-600'} font-persian`}>
                {t.loading}
              </p>
            </div>
          ) : datasets.length === 0 ? (
            <div className="text-center py-8">
              <p className={`${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                {t.noSamplesAvailable}
              </p>
            </div>
          ) : (
            <>
              {/* Tabs */}
              <div className="flex space-x-2 mb-4 border-b border-gray-200 dark:border-gray-700">
                {datasets.map((dataset) => {
                  const datasetFiles = exampleFiles[dataset];
                  const fileCount = Array.isArray(datasetFiles) ? datasetFiles.length : 0;

                  return (
                    <button
                      key={dataset}
                      onClick={() => setActiveTab(dataset)}
                      className={`px-4 py-2 text-sm font-medium transition-colors duration-200 border-b-2 ${activeTab === dataset
                        ? 'border-primary text-primary'
                        : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
                        }`}
                    >
                      {dataset} ({fileCount})
                    </button>
                  );
                })}
              </div>

              {/* Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 mt-4">
                {currentFiles.map((example, index) => {
                  const isSelected = selectedFileName === example.filename;
                  return (
                    <div
                      key={example.filename}
                      onClick={() => {
                        onLoadExample(example.filename);
                        setIsExpanded(false);
                      }}
                      style={{
                        animationDelay: isExpanded ? `${index * 50}ms` : '0ms'
                      }}
                      className={`
                        cursor-pointer rounded-xl border-2 p-4 transition-all duration-200
                        ${isExpanded ? 'animate-fadeInUp' : ''}
                        ${isSelected
                          ? (darkMode ? 'border-primary bg-primary/20' : 'border-primary bg-primary/10')
                          : (darkMode ? 'border-gray-700 bg-gray-800/50 hover:border-gray-600 hover:bg-gray-700/50 hover:scale-[1.02]' : 'border-gray-200 bg-gray-50 hover:border-gray-300 hover:bg-gray-100 hover:scale-[1.02]')
                        }
                      `}
                    >
                      <div className="flex justify-between items-start mb-2">
                        <h4 className={`font-bold text-sm ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                          {language === 'fa' ? `نمونه ${index + 1}` : `Sample ${index + 1}`}
                        </h4>
                        {isSelected && (
                          <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary text-white animate-bounce">
                            <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                            </svg>
                          </span>
                        )}
                      </div>

                      <div className="flex items-center gap-2 mb-2 flex-wrap">
                        <span className={`text-xs px-2 py-1 rounded-full ${darkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-200 text-gray-600'}`}>
                          {example.label}
                        </span>
                        <span className={`text-xs px-2 py-1 rounded-full ${darkMode ? 'bg-blue-900/50 text-blue-300' : 'bg-blue-100 text-blue-700'}`}>
                          {example.sampling_rate} Hz
                        </span>
                      </div>

                      <p className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-400'} truncate`}>
                        {example.filename}
                      </p>
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </div>
      </div>
    </section>
  );
};
