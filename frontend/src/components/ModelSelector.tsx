import React from 'react';
import { Model, Language } from '../types';
import { translations } from '../constants/translations';

interface ModelSelectorProps {
  darkMode: boolean;
  language: Language;
  models: { [key: string]: Model };
  selectedModel: string;
  windowSizeFilter: number | 'all';
  loading: boolean;
  onModelSelect: (modelName: string) => void;
  onWindowSizeFilterChange: (filter: number | 'all') => void;
  samplingRate: number;
  onSamplingRateChange: (rate: number) => void;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  darkMode,
  language,
  models,
  selectedModel,
  windowSizeFilter,
  loading,
  onModelSelect,
  onWindowSizeFilterChange,
  samplingRate,
  onSamplingRateChange,
}) => {
  const t = translations[language];

  // Filter models based on window size filter
  const filteredModelKeys = Object.keys(models).filter((key) => {
    if (windowSizeFilter === 'all') return true;
    return models[key].window_size === windowSizeFilter;
  });

  // Group filtered models by dataset
  const modelsByDataset = filteredModelKeys.reduce((acc, key) => {
    const dataset = models[key].dataset_name;
    if (!acc[dataset]) {
      acc[dataset] = [];
    }
    acc[dataset].push(key);
    return acc;
  }, {} as { [key: string]: string[] });

  // Get unique window sizes from all models
  const availableWindowSizes = Array.from(
    new Set(Object.keys(models).map(key => models[key].window_size))
  ).sort((a, b) => a - b);

  return (
    <section className={`mb-8 p-6 rounded-xl shadow-lg ${darkMode ? 'bg-gray-800' : 'bg-white'} transition-colors duration-300`}>
      <h2 className={`text-2xl font-bold mb-6 text-center ${darkMode ? 'text-white' : 'text-primary'} font-persian`}>
        {t.configuration}
      </h2>

      {/* Window Size Filter */}
      <div className="mb-6">
        <label className={`block mb-2 font-semibold text-center ${darkMode ? 'text-gray-200' : 'text-gray-700'} transition-colors duration-300`}>
          {t.filterByWindowSize}
        </label>
        <div className="flex justify-center">
          <select
            value={windowSizeFilter}
            onChange={(e) => onWindowSizeFilterChange(e.target.value === 'all' ? 'all' : parseInt(e.target.value))}
            className={`w-full md:w-1/3 px-4 py-2 rounded-lg border transition-colors duration-300 ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300 text-gray-900'} focus:outline-none focus:ring-2 focus:ring-primary`}
          >
            <option value="all">{t.allWindowSizes}</option>
            {availableWindowSizes.map(size => (
              <option key={size} value={size}>{size}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Sampling Rate Input */}
      <div className="mb-6">
        <label className={`block mb-2 font-semibold text-center ${darkMode ? 'text-gray-200' : 'text-gray-700'} transition-colors duration-300`}>
          {language === 'fa' ? 'نرخ نمونه‌برداری (Hz)' : 'Sampling Rate (Hz)'}
        </label>
        <div className="flex justify-center">
          <input
            type="number"
            value={samplingRate}
            onChange={(e) => onSamplingRateChange(parseInt(e.target.value) || 0)}
            className={`w-full md:w-1/3 px-4 py-2 rounded-lg border transition-colors duration-300 ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300 text-gray-900'} focus:outline-none focus:ring-2 focus:ring-primary text-center`}
          />
        </div>
      </div>

      {loading ? (
        <div className="text-center mt-4">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-primary mb-4"></div>
          <p className={`${darkMode ? 'text-gray-400' : 'text-gray-600'} font-persian`}>
            {t.loading}
          </p>
        </div>
      ) : (
        <div className="space-y-6">
          {Object.keys(modelsByDataset).length === 0 ? (
            <div className="text-center py-8">
              <p className={`${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                {t.loadModelsFirst}
              </p>
            </div>
          ) : (
            Object.keys(modelsByDataset).map((dataset) => (
              <div key={dataset} className="space-y-3">
                <h3 className={`text-lg font-semibold text-center ${darkMode ? 'text-gray-300' : 'text-gray-700'} border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'} pb-2`}>
                  {dataset === 'PU' ? t.puDataset : dataset === 'CWRU' ? t.cwruDataset : dataset}
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {modelsByDataset[dataset].map((modelKey) => {
                    const model = models[modelKey];
                    const isSelected = selectedModel === modelKey;
                    return (
                      <div
                        key={modelKey}
                        onClick={() => onModelSelect(modelKey)}
                        className={`
                          relative cursor-pointer rounded-xl border-2 p-4 transition-colors duration-200
                          ${isSelected
                            ? (darkMode ? 'border-primary bg-primary/20' : 'border-primary bg-primary/10')
                            : (darkMode
                              ? 'border-gray-600 bg-gray-800 hover:border-primary/50'
                              : 'border-gray-200 bg-white hover:border-primary/50')
                          }
                        `}
                      >
                        {/* Radio-style indicator */}
                        <div className="absolute top-3 left-3">
                          <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center transition-all duration-300 ${isSelected
                            ? 'border-primary bg-primary'
                            : (darkMode ? 'border-gray-500 hover:border-primary/50' : 'border-gray-300 hover:border-primary/50')
                            }`}>
                            {isSelected && (
                              <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M5 13l4 4L19 7" />
                              </svg>
                            )}
                          </div>
                        </div>

                        {/* Selected badge */}
                        {isSelected && (
                          <div className="absolute -top-2 -right-2">
                            <span className={`px-2 py-0.5 rounded-full text-xs font-bold bg-green-500 text-white shadow-lg`}>
                              {language === 'fa' ? 'انتخاب شده' : 'Selected'}
                            </span>
                          </div>
                        )}

                        <div className="flex justify-between items-start mb-2 pr-6">
                          <h4 className={`font-bold text-sm ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                            {model.name}
                          </h4>
                        </div>

                        <div className="flex items-center gap-2 mb-2">
                          <span className={`text-xs px-2 py-1 rounded-full ${isSelected
                            ? (darkMode ? 'bg-primary/30 text-primary' : 'bg-primary/20 text-primary')
                            : (darkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-600')
                            }`}>
                            {t.windowSize}: {model.window_size}
                          </span>
                        </div>
                        {/* Click hint for unselected cards */}
                        {!isSelected && (
                          <div className={`mt-3 pt-2 border-t ${darkMode ? 'border-gray-700' : 'border-gray-100'}`}>
                            <p className={`text-xs flex items-center gap-1 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" />
                              </svg>
                              {language === 'fa' ? 'انتخاب کنید' : 'Click to select'}
                            </p>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            ))
          )}
        </div>
      )}
    </section>
  );
};
