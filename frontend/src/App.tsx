import React, { useState, useEffect } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import Papa from 'papaparse';
import './App.css';

// Types
import { Model, ExampleFile, Language, Dataset, FFTData, STFTData } from './types';

// Constants
import { translations, MODEL_DESCRIPTIONS, MODEL_DESCRIPTIONS_EN } from './constants/translations';

// Components
import {
  Header,
  FileUpload,
  ModelSelector,
  SampleFiles,
  ClassificationResults,
  Charts,
  LoadingModal,
  FileInfo,
  ErrorMessage,
} from './components';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

function App() {
  // Theme & Language State
  const [darkMode, setDarkMode] = useState(false);
  const [language, setLanguage] = useState<Language>('fa');

  // API State
  const [apiUrl, setApiUrl] = useState('https://rotary-insight.ir');
  const [apiKey, setApiKey] = useState('PzJo3KDcHdpcgLQ88qH6AYPsnNYXE58M');
  const [isAuthenticated, setIsAuthenticated] = useState(true);

  // Models State
  const [models, setModels] = useState<{ [key: string]: Model }>({});
  const [selectedModel, setSelectedModel] = useState('');
  const [windowSize, setWindowSize] = useState<number | ''>('');
  const [windowSizeFilter, setWindowSizeFilter] = useState<number | 'all'>('all');
  const [hopLength, setHopLength] = useState<number | ''>('');

  // File State
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [exampleFiles, setExampleFiles] = useState<ExampleFile[]>([]);

  // Signal Data State
  const [signalData, setSignalData] = useState<number[]>([]);
  const [fftData, setFftData] = useState<FFTData | null>(null);
  const [stftData, setStftData] = useState<STFTData | null>(null);

  // Classification State
  const [selectedDataset] = useState<Dataset>('CWRU');
  const [classificationResults, setClassificationResults] = useState<any>(null);

  // Loading States
  const [loading, setLoading] = useState(false);
  const [loadingExamples, setLoadingExamples] = useState(false);
  const [chartsLoading, setChartsLoading] = useState(false);
  const [classificationLoading, setClassificationLoading] = useState(false);
  const [stftLoading, setStftLoading] = useState(false);

  // UI State
  const [error, setError] = useState('');
  const [isCalculated, setIsCalculated] = useState(false);
  const [showResults, setShowResults] = useState(false);

  // Initialize app
  useEffect(() => {
    const defaultApiUrl = 'https://rotary-insight.ir';
    const defaultApiKey = 'PzJo3KDcHdpcgLQ88qH6AYPsnNYXE58M';

    setApiUrl(defaultApiUrl);
    setApiKey(defaultApiKey);
    setIsAuthenticated(true);

    localStorage.setItem('apiUrl', defaultApiUrl);
    localStorage.setItem('apiKey', defaultApiKey);
    localStorage.setItem('isAuthenticated', 'true');

    const savedDarkMode = localStorage.getItem('darkMode') === 'true';
    const savedLanguage = localStorage.getItem('language') as Language | null;

    if (savedDarkMode) setDarkMode(savedDarkMode);
    if (savedLanguage) setLanguage(savedLanguage);
  }, []);

  // Dark mode effect
  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
    localStorage.setItem('darkMode', darkMode.toString());
  }, [darkMode]);

  // Save language preference
  useEffect(() => {
    localStorage.setItem('language', language);
  }, [language]);

  // Auto-load models when authenticated
  useEffect(() => {
    if (isAuthenticated && apiUrl && apiKey) {
      loadModels();
      loadExamples();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAuthenticated, apiUrl, apiKey]);

  // Turn off loading when FFT is ready
  useEffect(() => {
    if (fftData && chartsLoading) {
      const timer = setTimeout(() => {
        setChartsLoading(false);
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [fftData, chartsLoading]);

  // API Functions
  const loadModels = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch(`${apiUrl}/models/`, {
        headers: { 'X-API-Key': apiKey },
      });

      if (!response.ok) {
        throw new Error('Failed to load models. Check your API URL and API Key.');
      }

      const data = await response.json();
      const modelsWithDescriptions: { [key: string]: Model } = {};

      Object.keys(data.models).forEach((key) => {
        modelsWithDescriptions[key] = {
          ...data.models[key],
          description: language === 'fa'
            ? (MODEL_DESCRIPTIONS[key] || MODEL_DESCRIPTIONS_EN[key] || translations[language].select)
            : (MODEL_DESCRIPTIONS_EN[key] || MODEL_DESCRIPTIONS[key] || translations[language].select),
        };
      });

      setModels(modelsWithDescriptions);
    } catch (err: any) {
      setError(err.message || translations[language].failedLoadModels);
    } finally {
      setLoading(false);
    }
  };

  const loadExamples = async () => {
    setLoadingExamples(true);
    try {
      const response = await fetch(`${apiUrl}/examples/`, {
        headers: { 'X-API-Key': apiKey },
      });

      if (!response.ok) {
        console.error('Failed to load examples');
        setExampleFiles([]);
        return;
      }

      const data = await response.json();
      setExampleFiles(data.examples || []);
    } catch (err: any) {
      console.error('Error loading examples:', err);
      setExampleFiles([]);
    } finally {
      setLoadingExamples(false);
    }
  };

  const handleLoadExample = async (filename: string) => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch(`${apiUrl}/examples/${filename}`, {
        headers: { 'X-API-Key': apiKey },
      });

      if (!response.ok) {
        throw new Error('Failed to load example file');
      }

      const data = await response.json();
      const blob = new Blob([data.signal.map((v: number) => `ch1\n${v}`).join('\n')], { type: 'text/csv' });
      const file = new File([blob], filename, { type: 'text/csv' });

      setCsvFile(file);
      setIsCalculated(false);
      setShowResults(false);
    } catch (err: any) {
      setError(err.message || 'Failed to load example file');
    } finally {
      setLoading(false);
    }
  };

  // Model Selection
  const handleModelSelect = (modelName: string) => {
    setSelectedModel(modelName);
    setIsCalculated(false);
    setShowResults(false);

    if (models[modelName]) {
      const size = models[modelName].window_size;
      setWindowSize(size);
      setHopLength(Math.floor(size / 4));
    }
  };

  // File Handling
  const handleFileSelect = (file: File) => {
    if (!file.name.endsWith('.csv')) {
      setError(translations[language].invalidFileType);
      return;
    }
    setCsvFile(file);
    setIsCalculated(false);
    setShowResults(false);
    setError('');
  };

  const parseCSV = (file: File) => {
    setChartsLoading(true);
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        const data: number[] = [];
        results.data.forEach((row: any) => {
          const value = parseFloat(row.ch1 || row[Object.keys(row)[1]] || row[Object.keys(row)[0]]);
          if (!isNaN(value)) {
            data.push(value);
          }
        });

        if (data.length === 0) {
          setError(translations[language].noValidData);
          setChartsLoading(false);
          return;
        }

        setSignalData(data);
        computeFFT(data);
        computeSTFT(data);
        setIsCalculated(true);
        setShowResults(true);
        setError('');

        if (selectedModel) {
          loadClassificationResults(data, selectedModel);
        }
      },
      error: (parseError) => {
        setError(translations[language].failedParseCsv + parseError.message);
        setChartsLoading(false);
      },
    });
  };

  const handleCalculate = () => {
    const t = translations[language];
    const errors: string[] = [];

    // Check all validation conditions
    if (!selectedModel) {
      errors.push(t.step1SelectModel);
    }
    if (!csvFile) {
      errors.push(t.step2UploadFile);
    }

    // If there are multiple errors, show them as a list
    if (errors.length > 0) {
      if (errors.length === 1) {
        // Single error - show directly
        if (!selectedModel) {
          setError(t.selectModelError);
        } else if (!csvFile) {
          setError(t.uploadFileError);
        }
      } else {
        // Multiple errors - show as list
        setError(`${t.validationSteps}\n${errors.join('\n')}`);
      }
      return;
    }

    if (!windowSize || windowSize <= 0) {
      setError(t.windowSizeError);
      return;
    }

    // TypeScript guard - csvFile is guaranteed to be non-null here
    if (!csvFile) return;

    setError('');
    parseCSV(csvFile);
  };

  // Signal Processing
  const computeFFT = async (data: number[]) => {
    try {
      const fftLength = Math.min(windowSize || 512, data.length);
      const signal = data.slice(0, fftLength);

      const response = await fetch(`${apiUrl}/processing/fft`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-API-Key': apiKey },
        body: JSON.stringify({ signal, n: fftLength }),
      });

      if (!response.ok) {
        throw new Error(translations[language].failedFft.replace(': ', ''));
      }

      const result = await response.json();
      setFftData(result);
    } catch (err: any) {
      setError(translations[language].failedFft + (err.message || ''));
    } finally {
      setChartsLoading(false);
    }
  };

  const computeSTFT = async (data: number[]) => {
    try {
      if (windowSize === '' || !windowSize) return;

      setStftLoading(true);
      const currentWindowSize = typeof windowSize === 'number' ? windowSize : 512;
      const currentHopLength = (hopLength !== '' && typeof hopLength === 'number') ? hopLength : Math.floor(currentWindowSize / 4);

      let signal = [...data];
      if (signal.length < currentWindowSize) {
        signal = [...signal, ...Array(currentWindowSize - signal.length).fill(0)];
      }

      const response = await fetch(`${apiUrl}/processing/stft`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-API-Key': apiKey },
        body: JSON.stringify({
          signal,
          n_fft: currentWindowSize,
          hop_length: currentHopLength,
          win_length: currentWindowSize,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to compute STFT: ${errorText}`);
      }

      const contentType = response.headers.get('content-type') || '';

      if (contentType.includes('application/json')) {
        const result = await response.json();
        setStftData(result);
      } else {
        const blob = await response.blob();
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64String = reader.result as string;
          const base64 = base64String.includes(',') ? base64String.split(',')[1] : base64String;
          setStftData({
            file_name: 'stft_spectrogram.png',
            file_type: blob.type || 'image/png',
            image_base64: base64,
          });
        };
        reader.readAsDataURL(blob);
      }
    } catch (err: any) {
      console.error('STFT error:', err);
    } finally {
      setStftLoading(false);
    }
  };

  const handleHopLengthChange = (value: number | '') => {
    setHopLength(value);
  };

  const handleRecalculateSTFT = () => {
    if (signalData.length > 0) {
      computeSTFT(signalData);
    }
  };

  // Classification
  const loadClassificationResults = async (data: number[], modelName?: string) => {
    if (!windowSize || (typeof windowSize === 'string' && windowSize === '')) {
      setClassificationLoading(false);
      return;
    }

    const currentWindowSize = typeof windowSize === 'number' ? windowSize : 512;
    let modelToUse = modelName || selectedModel;

    if (!modelToUse || (selectedDataset === 'PU' && models[modelToUse]?.dataset_name !== 'PU') ||
      (selectedDataset === 'CWRU' && models[modelToUse]?.dataset_name !== 'CWRU')) {
      const matchingModels = Object.keys(models).filter((key) => {
        const model = models[key];
        return model.window_size === currentWindowSize &&
          ((selectedDataset === 'PU' && model.dataset_name === 'PU') ||
            (selectedDataset === 'CWRU' && model.dataset_name === 'CWRU'));
      });

      if (matchingModels.length > 0) {
        modelToUse = matchingModels[0];
      } else {
        const fallbackModels = Object.keys(models).filter((key) => models[key].window_size === currentWindowSize);
        if (fallbackModels.length > 0) {
          modelToUse = fallbackModels[0];
        } else {
          setClassificationLoading(false);
          return;
        }
      }
    }

    if (!modelToUse) {
      setClassificationLoading(false);
      return;
    }

    setClassificationLoading(true);
    try {
      let windowData: number[] = data.length >= currentWindowSize
        ? data.slice(0, currentWindowSize)
        : [...data, ...Array(currentWindowSize - data.length).fill(0)];

      if (windowData.length === 0) {
        setClassificationLoading(false);
        setError('No data available for classification');
        return;
      }

      const response = await fetch(`${apiUrl}/predict/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-API-Key': apiKey },
        body: JSON.stringify({
          data: [[windowData]],
          model_name: modelToUse,
          return_probabilities: true,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to get classification results: ${response.status} ${errorText}`);
      }

      const result = await response.json();

      if (!result || (!result.predictions && !result.prediction && !result.probabilities)) {
        setError('Invalid response from API');
        setClassificationResults(null);
      } else {
        setClassificationResults(result);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to load classification results');
      setClassificationResults(null);
    } finally {
      setClassificationLoading(false);
    }
  };

  // Results Page
  if (showResults && isCalculated) {
    return (
      <div className={`min-h-screen transition-colors duration-300 ${darkMode ? 'dark bg-gray-900' : 'bg-secondary'}`}>
        <Header
          darkMode={darkMode}
          language={language}
          onDarkModeToggle={() => setDarkMode(!darkMode)}
          onLanguageChange={setLanguage}
          showBackButton={true}
          onBackClick={() => setShowResults(false)}
          title={translations[language].resultsTitle}
        />

        <div className="container mx-auto px-4 py-8">
          <FileInfo
            darkMode={darkMode}
            language={language}
            fileName={csvFile?.name}
            selectedModel={selectedModel}
            windowSize={windowSize}
          />

          <LoadingModal
            darkMode={darkMode}
            language={language}
            isVisible={chartsLoading}
          />

          {signalData.length > 0 && (
            <>
              <ClassificationResults
                darkMode={darkMode}
                language={language}
                classificationResults={classificationResults}
                classificationLoading={classificationLoading}
                error={error}
                models={models}
                selectedModel={selectedModel}
                selectedDataset={selectedDataset}
                windowSize={windowSize}
              />

              <Charts
                darkMode={darkMode}
                language={language}
                signalData={signalData}
                fftData={fftData}
                stftData={stftData}
                stftLoading={stftLoading}
                windowSize={windowSize}
                hopLength={hopLength}
                onHopLengthChange={handleHopLengthChange}
                onRecalculateSTFT={handleRecalculateSTFT}
              />
            </>
          )}
        </div>
      </div>
    );
  }

  // Main Application Page
  return (
    <div className={`min-h-screen transition-colors duration-300 ${darkMode ? 'dark bg-gray-900' : 'bg-secondary'}`}>
      <Header
        darkMode={darkMode}
        language={language}
        onDarkModeToggle={() => setDarkMode(!darkMode)}
        onLanguageChange={setLanguage}
      />

      <div className="container mx-auto px-4 py-8">
        <ErrorMessage darkMode={darkMode} error={error} language={language} />

        <ModelSelector
          darkMode={darkMode}
          language={language}
          models={models}
          selectedModel={selectedModel}
          windowSizeFilter={windowSizeFilter}
          loading={loading}
          onModelSelect={handleModelSelect}
          onWindowSizeFilterChange={setWindowSizeFilter}
        />

        <FileUpload
          darkMode={darkMode}
          language={language}
          csvFile={csvFile}
          loading={loading || chartsLoading}
          onFileSelect={handleFileSelect}
          onCalculate={handleCalculate}
        />

        <SampleFiles
          darkMode={darkMode}
          language={language}
          exampleFiles={exampleFiles}
          loadingExamples={loadingExamples}
          selectedFileName={csvFile?.name || null}
          onLoadExample={handleLoadExample}
        />
      </div>
    </div>
  );
}

export default App;
