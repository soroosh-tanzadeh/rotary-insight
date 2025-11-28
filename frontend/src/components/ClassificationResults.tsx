import React from 'react';
import { Model, Language, Dataset } from '../types';
import { translations } from '../constants/translations';

interface ClassificationResultsProps {
  darkMode: boolean;
  language: Language;
  classificationResults: any;
  classificationLoading: boolean;
  error: string;
  models: { [key: string]: Model };
  selectedModel: string;
  selectedDataset: Dataset;
  windowSize: number | '';
}

// Helper function to format class name for display
const formatClassName = (className: string): string => {
  if (className === 'Normal') {
    return 'Normal';
  }

  const match = className.match(/(\d+\.\d+)-?(\w+)/);
  if (match) {
    const severity = match[1];
    const faultType = match[2];

    let displayType = '';
    if (faultType === 'OuterRace') {
      displayType = 'Outer Ring';
    } else if (faultType === 'InnerRace') {
      displayType = 'Inner Ring';
    } else if (faultType === 'Ball') {
      displayType = 'Ball Fault';
    } else {
      displayType = faultType;
    }

    return `${displayType} ${severity}`;
  }

  return className;
};

// Helper function to get color based on class type
const getBarColor = (className: string, isMax: boolean): string => {
  if (isMax) return 'bg-green-500';

  if (className.includes('OuterRace') || className.includes('Outer Ring')) {
    if (className.includes('0.007')) return 'bg-yellow-500';
    return 'bg-blue-500';
  }
  if (className.includes('InnerRace') || className.includes('Inner Ring')) {
    if (className.includes('0.007')) return 'bg-red-500';
    return 'bg-blue-500';
  }
  if (className.includes('Ball') || className.includes('Ball Fault')) {
    return 'bg-blue-500';
  }
  return 'bg-blue-500';
};

export const ClassificationResults: React.FC<ClassificationResultsProps> = ({
  darkMode,
  language,
  classificationResults,
  classificationLoading,
  error,
  models,
  selectedModel,
  selectedDataset,
  windowSize,
}) => {
  const t = translations[language];

  if (classificationLoading) {
    return (
      <section className={`mb-8 p-6 rounded-lg shadow-lg ${darkMode ? 'bg-gray-800' : 'bg-white'} transition-colors duration-300`}>
        <h2 className={`text-2xl font-bold mb-4 text-center ${darkMode ? 'text-white' : 'text-primary'} font-persian`}>
          {t.classificationResults}
        </h2>
        <div className="text-center py-8">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-primary mb-4"></div>
          <p className={`${darkMode ? 'text-gray-300' : 'text-gray-700'} font-persian`}>
            {t.loadingClassification}
          </p>
        </div>
      </section>
    );
  }

  if (error && error.includes('classification')) {
    return (
      <section className={`mb-8 p-6 rounded-lg shadow-lg ${darkMode ? 'bg-gray-800' : 'bg-white'} transition-colors duration-300`}>
        <h2 className={`text-2xl font-bold mb-4 text-center ${darkMode ? 'text-white' : 'text-primary'} font-persian`}>
          {t.classificationResults}
        </h2>
        <div className={`text-center py-8 p-4 rounded-lg ${darkMode ? 'bg-red-900 text-red-200' : 'bg-red-100 text-red-800'}`}>
          <p>{error}</p>
        </div>
      </section>
    );
  }

  if (!classificationResults) {
    return null;
  }

  // Check different possible response structures
  const predictions = classificationResults.predictions || classificationResults.prediction || classificationResults;
  const prediction = Array.isArray(predictions) ? predictions[0] : predictions;
  const probabilities = prediction?.probabilities || prediction?.probs || prediction?.prob || [];

  if (!prediction || probabilities.length === 0) {
    return (
      <section className={`mb-8 p-6 rounded-lg shadow-lg ${darkMode ? 'bg-gray-800' : 'bg-white'} transition-colors duration-300`}>
        <h2 className={`text-2xl font-bold mb-4 text-center ${darkMode ? 'text-white' : 'text-primary'} font-persian`}>
          {t.classificationResults}
        </h2>
        <div className="text-center py-8">
          <p className={`${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
            No classification data available.
          </p>
          <p className={`text-xs mt-2 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            Response structure: {JSON.stringify(classificationResults).substring(0, 300)}
          </p>
        </div>
      </section>
    );
  }

  // Get model info
  let modelInfo = models[selectedModel];
  if (!modelInfo && classificationResults.model_name) {
    modelInfo = models[classificationResults.model_name];
  }
  if (!modelInfo && windowSize && typeof windowSize === 'number') {
    const matchingModels = Object.keys(models).filter((key) => {
      const model = models[key];
      return model.window_size === windowSize &&
        ((selectedDataset === 'PU' && model.dataset_name === 'PU') ||
          (selectedDataset === 'CWRU' && model.dataset_name === 'CWRU'));
    });
    if (matchingModels.length > 0) {
      modelInfo = models[matchingModels[0]];
    }
  }
  const classNames = modelInfo?.class_names || [];

  if (probabilities.length === 0 || classNames.length === 0) {
    return (
      <section className={`mb-8 p-6 rounded-lg shadow-lg ${darkMode ? 'bg-gray-800' : 'bg-white'} transition-colors duration-300`}>
        <h2 className={`text-2xl font-bold mb-4 text-center ${darkMode ? 'text-white' : 'text-primary'} font-persian`}>
          {t.classificationResults}
        </h2>
        <div className="text-center py-8">
          <p className={`${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
            No classification data available.
          </p>
          <p className={`text-xs mt-2 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            Probabilities: {probabilities.length}, ClassNames: {classNames.length}
          </p>
          <p className={`text-xs mt-1 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            Model: {selectedModel || 'None'}, Dataset: {selectedDataset}
          </p>
        </div>
      </section>
    );
  }

  const maxProb = Math.max(...probabilities);
  const maxIndex = probabilities.indexOf(maxProb);

  // For CWRU Dataset: Combined single block (10 classes)
  if (selectedDataset === 'CWRU' && classNames.length === 10) {
    // Combine and order all indices
    const orderedIndices = [7, 8, 9, 4, 5, 6, 1, 2, 3, 0];

    // Get all predictions if available (for multi-window)
    const allPredictions = Array.isArray(predictions) ? predictions : [predictions];

    return (
      <section className={`mb-6 p-4 rounded-lg shadow-lg ${darkMode ? 'bg-gray-800' : 'bg-white'} transition-colors duration-300`}>
        <h2 className={`text-xl font-bold mb-3 text-center ${darkMode ? 'text-white' : 'text-primary'} font-persian`}>
          {t.classificationResults}
        </h2>
        <div>
          <h3 className={`text-lg font-semibold mb-3 text-center ${darkMode ? 'text-gray-300' : 'text-gray-700'} font-persian`}>
            {t.cwruDataset} - Fault Classification
          </h3>

          {/* Individual Window Predictions */}
          {allPredictions.length > 1 && (
            <div className={`mb-4 p-3 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
              <h4 className={`text-sm font-semibold mb-2 text-center ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                Predictions for Each Window
              </h4>
              <div className="grid grid-cols-3 md:grid-cols-4 gap-2">
                {allPredictions.map((pred: any, windowIdx: number) => {
                  const windowProbs = pred?.probabilities || pred?.probs || pred?.prob || [];
                  const windowMaxProb = Math.max(...windowProbs);
                  const windowMaxIndex = windowProbs.indexOf(windowMaxProb);
                  const windowClassName = formatClassName(classNames[windowMaxIndex] || `Class ${windowMaxIndex}`);

                  return (
                    <div
                      key={windowIdx}
                      className={`p-2 rounded text-center ${darkMode ? 'bg-gray-600' : 'bg-white'} border ${darkMode ? 'border-gray-500' : 'border-gray-200'}`}
                    >
                      <div className={`text-[10px] font-medium mb-0.5 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        Win {windowIdx + 1}
                      </div>
                      <div className={`text-xs font-semibold truncate ${darkMode ? 'text-white' : 'text-gray-800'}`} title={windowClassName}>
                        {windowClassName}
                      </div>
                      <div className={`text-[10px] ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                        {(windowMaxProb * 100).toFixed(0)}%
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Combined Classification Results */}
          <div className={`p-3 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
            <h4 className={`text-sm font-semibold mb-2 text-center ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
              {allPredictions.length > 1 ? 'Aggregated Results' : 'Classification Results'}
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-2">
              {orderedIndices.map((idx) => {
                const prob = probabilities[idx];
                const originalClassName = classNames[idx] || `Class ${idx}`;
                const className = formatClassName(originalClassName);
                const isMax = idx === maxIndex;

                return (
                  <div key={idx} className="flex items-center gap-2" dir="ltr">
                    <div className="flex-1">
                      <div className="flex justify-between items-center mb-0.5" dir="ltr">
                        <span className={`text-xs font-semibold truncate ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                          {className}
                        </span>
                        <span className={`text-[10px] ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                          {prob.toFixed(4)}
                        </span>
                      </div>
                      <div className={`h-3 rounded-full overflow-hidden ${darkMode ? 'bg-gray-600' : 'bg-gray-300'}`} dir="ltr">
                        <div
                          className={`h-full transition-all duration-500 ${getBarColor(originalClassName, isMax)}`}
                          style={{ width: `${Math.min(prob * 100, 100)}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Predicted Class Summary */}
          <div className={`mt-4 p-2 rounded-lg text-center ${darkMode ? 'bg-gray-700' : 'bg-secondary'}`}>
            <p className={`text-base font-semibold ${darkMode ? 'text-white' : 'text-gray-800'}`}>
              {t.predictedClass}: <span className="text-green-500">{formatClassName(classNames[maxIndex] || `Class ${maxIndex}`)}</span> ({t.confidence}: {(maxProb * 100).toFixed(2)}%)
            </p>
          </div>
        </div>
      </section>
    );
  }

  const allPredictions = Array.isArray(predictions) ? predictions : [predictions];

  return (
    <section className={`mb-6 p-4 rounded-lg shadow-lg ${darkMode ? 'bg-gray-800' : 'bg-white'} transition-colors duration-300`}>
      <h2 className={`text-xl font-bold mb-3 text-center ${darkMode ? 'text-white' : 'text-primary'} font-persian`}>
        {t.classificationResults}
      </h2>
      <div>
        <h3 className={`text-lg font-semibold mb-3 text-center ${darkMode ? 'text-gray-300' : 'text-gray-700'} font-persian`}>
          {selectedDataset === 'PU' ? t.puDataset : t.cwruDataset} - Fault Classification
        </h3>

        {/* Individual Window Predictions */}
        {allPredictions.length > 1 && (
          <div className={`mb-4 p-3 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
            <h4 className={`text-sm font-semibold mb-2 text-center ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
              Predictions for Each Window
            </h4>
            <div className="grid grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
              {allPredictions.map((pred: any, windowIdx: number) => {
                const windowProbs = pred?.probabilities || pred?.probs || pred?.prob || [];
                const windowMaxProb = Math.max(...windowProbs);
                const windowMaxIndex = windowProbs.indexOf(windowMaxProb);
                const windowClassName = formatClassName(classNames[windowMaxIndex] || `Class ${windowMaxIndex}`);

                return (
                  <div
                    key={windowIdx}
                    className={`p-2 rounded text-center ${darkMode ? 'bg-gray-600' : 'bg-white'} border ${darkMode ? 'border-gray-500' : 'border-gray-200'}`}
                  >
                    <div className={`text-[10px] font-medium mb-0.5 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                      Win {windowIdx + 1}
                    </div>
                    <div className={`text-xs font-semibold truncate ${darkMode ? 'text-white' : 'text-gray-800'}`} title={windowClassName}>
                      {windowClassName}
                    </div>
                    <div className={`text-[10px] ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                      {(windowMaxProb * 100).toFixed(0)}%
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        <div className="space-y-2">
          {probabilities.map((prob: number, idx: number) => {
            const isMax = idx === maxIndex;
            const originalClassName = classNames[idx] || `Class ${idx}`;
            const className = formatClassName(originalClassName);

            return (
              <div key={idx} className="flex items-center gap-2" dir="ltr">
                <div className="flex-1">
                  <div className="flex justify-between items-center mb-0.5" dir="ltr">
                    <span className={`text-xs font-semibold truncate ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                      {className}
                    </span>
                    <span className={`text-[10px] ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                      {prob.toFixed(4)}
                    </span>
                  </div>
                  <div className={`h-3 rounded-full overflow-hidden ${darkMode ? 'bg-gray-700' : 'bg-gray-200'}`} dir="ltr">
                    <div
                      className={`h-full transition-all duration-500 ${getBarColor(originalClassName, isMax)}`}
                      style={{ width: `${Math.min(prob * 100, 100)}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            );
          })}

          {/* Predicted Class Summary */}
          <div className={`mt-4 p-2 rounded-lg text-center ${darkMode ? 'bg-gray-700' : 'bg-secondary'}`}>
            <p className={`text-base font-semibold ${darkMode ? 'text-white' : 'text-gray-800'}`}>
              {t.predictedClass}: <span className="text-green-500">{formatClassName(classNames[maxIndex] || `Class ${maxIndex}`)}</span> ({t.confidence}: {(maxProb * 100).toFixed(2)}%)
            </p>
          </div>
        </div>
      </div>
    </section>
  );
};

