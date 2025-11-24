import React, { useState, useEffect, useRef } from 'react';
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
import { Line, Bar } from 'react-chartjs-2';
import Papa from 'papaparse';
import './App.css';
import { STFTResult } from './STFTResult';

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

interface Model {
  name: string;
  window_size: number;
  dataset_name: string;
  class_names: string[];
  description: string;
}

interface ExampleFile {
  filename: string;
  sample_index: number;
  fault_name: string;
}

interface SignalStats {
  min: number;
  max: number;
  mean: number;
  std: number;
  extrema: { min: { value: number; index: number }; max: { value: number; index: number } };
  range: number;
  peakToPeak: number;
  rms: number;
  variance: number;
  median: number;
  zeroCrossings: number;
  energy: number;
  crestFactor: number;
}

const MODEL_DESCRIPTIONS: { [key: string]: string } = {
  'transformer_encoder_cwru_512': 'مدل Transformer با اندازه پنجره 512 نمونه - مناسب برای سیگنال‌های کوتاه و پردازش سریع',
  'transformer_encoder_cwru_1024': 'مدل Transformer با اندازه پنجره 1024 نمونه - تعادل بین دقت و سرعت',
  'transformer_encoder_cwru_2048': 'مدل Transformer با اندازه پنجره 2048 نمونه - بالاترین دقت برای سیگنال‌های بلند',
};

const MODEL_DESCRIPTIONS_EN: { [key: string]: string } = {
  'transformer_encoder_cwru_512': 'Transformer model with 512 sample window - suitable for short signals and fast processing',
  'transformer_encoder_cwru_1024': 'Transformer model with 1024 sample window - balance between accuracy and speed',
  'transformer_encoder_cwru_2048': 'Transformer model with 2048 sample window - highest accuracy for long signals',
};

const translations = {
  fa: {
    title: 'Rotary Insight',
    resultsTitle: 'نتایج تحلیل',
    back: 'بازگشت',
    logout: 'خروج',
    light: 'روشن',
    dark: 'تاریک',
    configuration: 'تنظیمات',
    selectModel: 'انتخاب مدل',
    windowSize: 'اندازه پنجره',
    filterByWindowSize: 'فیلتر بر اساس اندازه پنجره',
    allWindowSizes: 'همه اندازه‌ها',
    sampleFiles: 'فایل‌های نمونه',
    loadSample: 'بارگذاری نمونه',
    orSelectSample: 'یا یک نمونه انتخاب کنید',
    noSamplesAvailable: 'فایل نمونه‌ای موجود نیست',
    hopLength: 'Hop Length',
    parameters: 'پارامترها',
    uploadCsv: 'آپلود فایل CSV',
    dragDrop: 'فایل CSV را اینجا بکشید یا کلیک کنید',
    calculate: 'محاسبه',
    signalInfo: 'اطلاعات سیگنال',
    charts: 'نمودارها',
    basicStats: 'آمار پایه',
    spreadStats: 'پراکندگی',
    advancedStats: 'آمار پیشرفته',
    timeDomain: 'سیگنال حوزه زمان',
    frequencyDomain: 'حوزه فرکانس (FFT)',
    stft: 'تبدیل فوریه کوتاه‌مدت (STFT)',
    stftDescription: 'نمایش زمان-فرکانس سیگنال',
    heatMap: 'نقشه حرارتی',
    login: 'ورود',
    validating: 'در حال بررسی...',
    file: 'فایل',
    model: 'مدل',
    loading: 'در حال بارگذاری مدل‌ها...',
    loadingCharts: 'در حال بارگیری اطلاعات',
    select: 'انتخاب کنید',
    loadModelsFirst: 'در حال بارگذاری مدل‌ها...',
    selectWindowSizeFirst: 'ابتدا اندازه پنجره را وارد کنید',
    min: 'حداقل',
    max: 'حداکثر',
    mean: 'میانگین',
    median: 'میانه',
    stdDev: 'انحراف معیار',
    variance: 'واریانس',
    range: 'بازه',
    peakToPeak: 'Peak-to-Peak',
    rms: 'RMS',
    energy: 'انرژی',
    crestFactor: 'Crest Factor',
    zeroCrossings: 'تعداد عبور از صفر',
    showing: 'نمایش',
    windows: 'پنجره',
    low: 'کم',
    high: 'زیاد',
    enterApiCredentials: 'لطفاً API URL و API Key را وارد کنید',
    invalidApiKey: 'API Key نامعتبر است. لطفاً API Key صحیح را وارد کنید.',
    apiConnectionError: 'خطا در اتصال به API. لطفاً URL و API Key را بررسی کنید.',
    serverConnectionError: 'خطا در اتصال به سرور. لطفاً URL را بررسی کنید.',
    selectModelError: 'لطفاً یک مدل انتخاب کنید',
    uploadFileError: 'لطفاً یک فایل CSV بارگذاری کنید',
    windowSizeError: 'اندازه پنجره باید بزرگتر از صفر باشد',
    invalidCsv: 'فایل CSV معتبر نیست',
    noValidData: 'داده معتبری در فایل CSV یافت نشد',
    failedParseCsv: 'خطا در خواندن فایل CSV: ',
    failedLoadModels: 'خطا در بارگذاری مدل‌ها. لطفاً API URL و API Key را بررسی کنید.',
    invalidFileType: 'لطفاً یک فایل CSV انتخاب کنید',
    failedFft: 'خطا در محاسبه FFT: ',
    classificationResults: 'نتایج طبقه‌بندی',
    puDataset: 'PU Dataset',
    cwruDataset: 'CWRU Dataset',
    predictedClass: 'کلاس پیش‌بینی شده',
    confidence: 'اطمینان',
    loadingClassification: 'در حال بارگیری نتایج طبقه‌بندی...',
  },
  en: {
    title: 'Rotary Insight',
    resultsTitle: 'Analysis Results',
    back: 'Back',
    logout: 'Logout',
    light: 'Light',
    dark: 'Dark',
    configuration: 'Configuration',
    selectModel: 'Select Model',
    windowSize: 'Window Size',
    filterByWindowSize: 'Filter by Window Size',
    allWindowSizes: 'All Window Sizes',
    sampleFiles: 'Sample Files',
    loadSample: 'Load Sample',
    orSelectSample: 'Or Select a Sample',
    noSamplesAvailable: 'No sample files available',
    hopLength: 'Hop Length',
    parameters: 'Parameters',
    uploadCsv: 'Upload CSV File',
    dragDrop: 'Drag & drop your CSV file here or click to browse',
    calculate: 'Calculate',
    signalInfo: 'Signal Information',
    charts: 'Charts',
    basicStats: 'Basic Statistics',
    spreadStats: 'Spread Statistics',
    advancedStats: 'Advanced Statistics',
    timeDomain: 'Time Domain Signal',
    frequencyDomain: 'Frequency Domain (FFT)',
    stft: 'Short-Time Fourier Transform (STFT)',
    stftDescription: 'Time-Frequency representation of the signal',
    heatMap: 'Heat Map',
    login: 'Login',
    validating: 'Validating...',
    file: 'File',
    model: 'Model',
    loading: 'Loading models...',
    loadingCharts: 'Loading data...',
    select: 'Select',
    loadModelsFirst: 'Load models first',
    selectWindowSizeFirst: 'Please select window size first',
    min: 'Min',
    max: 'Max',
    mean: 'Mean',
    median: 'Median',
    stdDev: 'Std Dev',
    variance: 'Variance',
    range: 'Range',
    peakToPeak: 'Peak-to-Peak',
    rms: 'RMS',
    energy: 'Energy',
    crestFactor: 'Crest Factor',
    zeroCrossings: 'Zero Crossings',
    showing: 'Showing',
    windows: 'windows',
    low: 'Low',
    high: 'High',
    enterApiCredentials: 'Please enter API URL and API Key',
    invalidApiKey: 'Invalid API Key. Please enter the correct API Key.',
    apiConnectionError: 'Error connecting to API. Please check your URL and API Key.',
    serverConnectionError: 'Error connecting to server. Please check your URL.',
    selectModelError: 'Please select a model',
    uploadFileError: 'Please upload a CSV file',
    windowSizeError: 'Window size must be greater than zero',
    invalidCsv: 'Invalid CSV file',
    noValidData: 'No valid data found in CSV file',
    failedParseCsv: 'Failed to parse CSV file: ',
    failedLoadModels: 'Failed to load models. Check your API URL and API Key.',
    invalidFileType: 'Please select a CSV file',
    failedFft: 'Failed to compute FFT: ',
    classificationResults: 'Classification Results',
    puDataset: 'PU Dataset',
    cwruDataset: 'CWRU Dataset',
    predictedClass: 'Predicted Class',
    confidence: 'Confidence',
    loadingClassification: 'Loading classification results...',
  },
};

function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [language, setLanguage] = useState<'fa' | 'en'>('fa');
  const [apiUrl, setApiUrl] = useState('https://rotary-insight.ir');
  const [apiKey, setApiKey] = useState('PzJo3KDcHdpcgLQ88qH6AYPsnNYXE58M');
  const [isAuthenticated, setIsAuthenticated] = useState(true);
  const [tempApiUrl, setTempApiUrl] = useState('https://rotary-insight.ir');
  const [tempApiKey, setTempApiKey] = useState('PzJo3KDcHdpcgLQ88qH6AYPsnNYXE58M');
  const [models, setModels] = useState<{ [key: string]: Model }>({});
  const [selectedModel, setSelectedModel] = useState('');
  const [windowSize, setWindowSize] = useState<number | ''>('');
  const [windowSizeFilter, setWindowSizeFilter] = useState<number | 'all'>('all');
  const [hopLength, setHopLength] = useState<number | ''>('');
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [signalData, setSignalData] = useState<number[]>([]);
  const [fftData, setFftData] = useState<{ frequencies: number[]; magnitudes: number[] } | null>(null);
  const [stftData, setStftData] = useState<{ file_name: string; file_type?: string; image_base64: string } | null>(null);
  const [signalStats, setSignalStats] = useState<SignalStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [validating, setValidating] = useState(false);
  const [isCalculated, setIsCalculated] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [chartsLoading, setChartsLoading] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState<'PU' | 'CWRU'>('CWRU');
  const [classificationResults, setClassificationResults] = useState<any>(null);
  const [classificationLoading, setClassificationLoading] = useState(false);
  const [stftLoading, setStftLoading] = useState(false);
  const [exampleFiles, setExampleFiles] = useState<ExampleFile[]>([]);
  const [loadingExamples, setLoadingExamples] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Set default API credentials
    const defaultApiUrl = 'https://rotary-insight.ir';
    const defaultApiKey = 'PzJo3KDcHdpcgLQ88qH6AYPsnNYXE58M';

    // Always use default credentials
    setApiUrl(defaultApiUrl);
    setApiKey(defaultApiKey);
    setTempApiUrl(defaultApiUrl);
    setTempApiKey(defaultApiKey);
    setIsAuthenticated(true);

    // Save to localStorage
    localStorage.setItem('apiUrl', defaultApiUrl);
    localStorage.setItem('apiKey', defaultApiKey);
    localStorage.setItem('isAuthenticated', 'true');

    // Load saved preferences
    const savedDarkMode = localStorage.getItem('darkMode') === 'true';
    const savedLanguage = localStorage.getItem('language') as 'fa' | 'en' | null;

    if (savedDarkMode) setDarkMode(savedDarkMode);
    if (savedLanguage) setLanguage(savedLanguage);
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
    localStorage.setItem('darkMode', darkMode.toString());
  }, [darkMode]);

  useEffect(() => {
    localStorage.setItem('apiUrl', apiUrl);
  }, [apiUrl]);

  useEffect(() => {
    localStorage.setItem('apiKey', apiKey);
  }, [apiKey]);

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

  const validateApiCredentials = async (url: string, key: string, lang: 'fa' | 'en' = language) => {
    try {
      // Try to fetch models endpoint which requires API Key
      const response = await fetch(`${url}/models/`, {
        method: 'GET',
        headers: {
          'X-API-Key': key,
        },
      });

      if (!response.ok) {
        if (response.status === 401 || response.status === 403) {
          throw new Error(translations[lang].invalidApiKey);
        }
        throw new Error(translations[lang].apiConnectionError);
      }

      return true;
    } catch (err: any) {
      if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
        throw new Error(translations[lang].serverConnectionError);
      }
      throw new Error(err.message || translations[lang].apiConnectionError);
    }
  };

  const handleLogin = async () => {
    if (!tempApiUrl.trim() || !tempApiKey.trim()) {
      setError(translations[language].enterApiCredentials);
      return;
    }

    setValidating(true);
    setError('');

    try {
      await validateApiCredentials(tempApiUrl.trim(), tempApiKey.trim(), language);

      // Save credentials
      setApiUrl(tempApiUrl.trim());
      setApiKey(tempApiKey.trim());
      setIsAuthenticated(true);
      localStorage.setItem('apiUrl', tempApiUrl.trim());
      localStorage.setItem('apiKey', tempApiKey.trim());
      localStorage.setItem('isAuthenticated', 'true');
    } catch (err: any) {
      setError(err.message);
    } finally {
      setValidating(false);
    }
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    setApiUrl('');
    setApiKey('');
    setTempApiUrl('http://localhost:8000');
    setTempApiKey('');
    localStorage.removeItem('apiUrl');
    localStorage.removeItem('apiKey');
    localStorage.removeItem('isAuthenticated');
    setModels({});
    setSelectedModel('');
    setCsvFile(null);
    setSignalData([]);
    setFftData(null);
    setStftData(null);
    setSignalStats(null);
  };

  const loadModels = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch(`${apiUrl}/models/`, {
        headers: {
          'X-API-Key': apiKey,
        },
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
        headers: {
          'X-API-Key': apiKey,
        },
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
        headers: {
          'X-API-Key': apiKey,
        },
      });

      if (!response.ok) {
        throw new Error('Failed to load example file');
      }

      const data = await response.json();

      // Create a virtual file object
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

  const handleModelSelect = (modelName: string) => {
    setSelectedModel(modelName);
    setIsCalculated(false);
    setShowResults(false);

    // Auto-set window size based on selected model
    if (models[modelName]) {
      const size = models[modelName].window_size;
      setWindowSize(size);
      setHopLength(Math.floor(size / 4));
    }
  };

  const handleWindowSizeChange = (size: string) => {
    const sizeValue = size === '' ? '' : parseInt(size);
    setWindowSize(sizeValue);
    setIsCalculated(false);
    setShowResults(false);
    // Set hop_length to 1/4 of window size by default
    if (sizeValue !== '') {
      setHopLength(Math.floor(sizeValue / 4));
    } else {
      setHopLength('');
    }
    // Auto-select first model with matching window size
    if (sizeValue !== '') {
      const matchingModels = Object.keys(models).filter((key) => {
        return models[key].window_size === sizeValue;
      });
      if (matchingModels.length > 0) {
        setSelectedModel(matchingModels[0]);
      } else {
        setSelectedModel('');
      }
    } else {
      setSelectedModel('');
    }
  };

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
  const availableWindowSizes = Array.from(new Set(Object.keys(models).map(key => models[key].window_size))).sort((a, b) => a - b);

  // Auto-select first model when models are loaded if none selected
  useEffect(() => {
    if (Object.keys(models).length > 0 && !selectedModel) {
      // Optional: Auto-select the first available model
      // const firstModel = Object.keys(models)[0];
      // handleModelSelect(firstModel);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [Object.keys(models).length]);

  const handleFileSelect = (file: File) => {
    if (!file.name.endsWith('.csv')) {
      setError(translations[language].invalidFileType);
      return;
    }
    setCsvFile(file);
    setIsCalculated(false);
    setShowResults(false);
    setError('');
    // Don't parse CSV automatically, wait for calculate button
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
        calculateStats(data);
        computeFFT(data);
        computeSTFT(data);
        setIsCalculated(true);
        setShowResults(true);
        setError('');
        // Load classification results with current selected model
        if (selectedModel) {
          loadClassificationResults(data, selectedModel);
        }
      },
      error: (error) => {
        setError(translations[language].failedParseCsv + error.message);
        setChartsLoading(false);
      },
    });
  };

  const handleCalculate = () => {
    if (!selectedModel) {
      setError(translations[language].selectModelError);
      return;
    }
    if (!csvFile) {
      setError(translations[language].uploadFileError);
      return;
    }
    if (!windowSize || windowSize <= 0) {
      setError(translations[language].windowSizeError);
      return;
    }

    setError('');
    parseCSV(csvFile);
  };

  const loadClassificationResults = async (data: number[], modelName?: string) => {
    if (!windowSize || (typeof windowSize === 'string' && windowSize === '')) {
      setClassificationLoading(false);
      return;
    }

    // Use provided model or find model based on dataset and window size
    const currentWindowSize = typeof windowSize === 'number' ? windowSize : 512;
    let modelToUse = modelName || selectedModel;

    // If no model provided/selected or model doesn't match dataset, find appropriate model
    if (!modelToUse || (selectedDataset === 'PU' && models[modelToUse]?.dataset_name !== 'PU') ||
      (selectedDataset === 'CWRU' && models[modelToUse]?.dataset_name !== 'CWRU')) {
      // Find model matching dataset and window size
      const matchingModels = Object.keys(models).filter((key) => {
        const model = models[key];
        return model.window_size === currentWindowSize &&
          ((selectedDataset === 'PU' && model.dataset_name === 'PU') ||
            (selectedDataset === 'CWRU' && model.dataset_name === 'CWRU'));
      });

      if (matchingModels.length > 0) {
        modelToUse = matchingModels[0];
      } else {
        // Fallback: use any model with matching window size
        const fallbackModels = Object.keys(models).filter((key) => {
          return models[key].window_size === currentWindowSize;
        });
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
      // Prepare window data - pad if necessary
      let windowData: number[] = [];

      if (data.length >= currentWindowSize) {
        // Use first window if data is long enough
        windowData = data.slice(0, currentWindowSize);
      } else {
        // Pad data with zeros if it's shorter than window size
        windowData = [...data];
        while (windowData.length < currentWindowSize) {
          windowData.push(0);
        }
      }

      if (windowData.length === 0) {
        setClassificationLoading(false);
        setError('No data available for classification');
        return;
      }

      // Use the window data for classification
      const response = await fetch(`${apiUrl}/predict/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': apiKey,
        },
        body: JSON.stringify({
          data: [[windowData]], // Shape: [batch_size, channels, signal_length]
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

  const handleDatasetChange = async (dataset: 'PU' | 'CWRU') => {
    setSelectedDataset(dataset);
    setClassificationResults(null);
    setError('');

    if (signalData.length > 0 && windowSize && typeof windowSize === 'number') {
      // Load classification results - it will automatically find the right model
      await loadClassificationResults(signalData);
    }
  };

  const calculateStats = (data: number[]) => {
    // Use reduce for large arrays to avoid "Maximum call stack size exceeded" error
    const min = data.reduce((a, b) => Math.min(a, b), data[0] || 0);
    const max = data.reduce((a, b) => Math.max(a, b), data[0] || 0);
    const mean = data.reduce((a, b) => a + b, 0) / data.length;
    const variance = data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / data.length;
    const std = Math.sqrt(variance);

    const minIndex = data.indexOf(min);
    const maxIndex = data.indexOf(max);

    // Calculate additional statistics
    const range = max - min;
    const peakToPeak = range;

    // RMS (Root Mean Square)
    const rms = Math.sqrt(data.reduce((sum, val) => sum + val * val, 0) / data.length);

    // Median
    const sortedData = [...data].sort((a, b) => a - b);
    const median = sortedData.length % 2 === 0
      ? (sortedData[sortedData.length / 2 - 1] + sortedData[sortedData.length / 2]) / 2
      : sortedData[Math.floor(sortedData.length / 2)];

    // Zero Crossings
    let zeroCrossings = 0;
    for (let i = 1; i < data.length; i++) {
      if ((data[i - 1] >= 0 && data[i] < 0) || (data[i - 1] < 0 && data[i] >= 0)) {
        zeroCrossings++;
      }
    }

    // Energy (sum of squares)
    const energy = data.reduce((sum, val) => sum + val * val, 0);

    // Crest Factor (peak / RMS)
    const peak = Math.max(Math.abs(max), Math.abs(min));
    const crestFactor = rms > 0 ? peak / rms : 0;

    setSignalStats({
      min,
      max,
      mean,
      std,
      extrema: {
        min: { value: min, index: minIndex },
        max: { value: max, index: maxIndex },
      },
      range,
      peakToPeak,
      rms,
      variance,
      median,
      zeroCrossings,
      energy,
      crestFactor,
    });
  };

  const computeFFT = async (data: number[]) => {
    try {
      const fftLength = Math.min(windowSize || 512, data.length);
      const signal = data.slice(0, fftLength);

      const response = await fetch(`${apiUrl}/processing/fft`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': apiKey,
        },
        body: JSON.stringify({
          signal: signal,
          n: fftLength,
        }),
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

  const handleHopLengthChange = (value: number | '') => {
    setHopLength(value);
    // Do NOT automatically re-compute STFT. User must click "Recalculate".
  };

  const handleRecalculateSTFT = () => {
    if (signalData.length > 0) {
      computeSTFT(signalData);
    }
  };

  const computeSTFT = async (data: number[], customHopLength?: number | '') => {
    try {
      if (windowSize === '' || !windowSize) {
        return;
      }

      setStftLoading(true);

      const currentWindowSize = typeof windowSize === 'number' ? windowSize : 512;
      // Use custom hop length if provided, otherwise use state
      const hopLenToUse = customHopLength !== undefined ? customHopLength : hopLength;
      const currentHopLength = (hopLenToUse !== '' && typeof hopLenToUse === 'number') ? hopLenToUse : Math.floor(currentWindowSize / 4);
      const currentNfft = currentWindowSize;

      // Pad signal if necessary - signal must be at least n_fft length
      let signal = [...data];
      if (signal.length < currentNfft) {
        signal = [...signal, ...Array(currentNfft - signal.length).fill(0)];
      }

      const response = await fetch(`${apiUrl}/processing/stft`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': apiKey,
        },
        body: JSON.stringify({
          signal: signal,
          n_fft: currentNfft,
          hop_length: currentHopLength,
          win_length: currentWindowSize,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to compute STFT: ${errorText}`);
      }

      // Check content type - API returns image/png directly
      const contentType = response.headers.get('content-type') || '';

      if (contentType.includes('application/json')) {
        // If JSON response (with base64 image)
        const result = await response.json();
        setStftData(result);
      } else {
        // If response is an image (PNG), convert to base64
        const blob = await response.blob();
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64String = reader.result as string;
          // Remove data URL prefix if present
          const base64 = base64String.includes(',') ? base64String.split(',')[1] : base64String;
          setStftData({
            file_name: 'stft_spectrogram.png',
            file_type: blob.type || 'image/png',
            image_base64: base64,
          });
        };
        reader.onerror = () => {
          console.error('Error reading STFT image blob');
        };
        reader.readAsDataURL(blob);
      }
    } catch (err: any) {
      console.error('STFT error:', err);
      // Don't set error state for STFT, just log it
    } finally {
      setStftLoading(false);
    }
  };

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
      handleFileSelect(file);
    }
  };


  const timeDomainChartData = {
    labels: signalData.map((_, i) => i),
    datasets: [
      {
        label: 'Signal Amplitude',
        data: signalData,
        borderColor: darkMode ? '#60A5FA' : '#385F8C',
        backgroundColor: darkMode ? 'rgba(96, 165, 250, 0.1)' : 'rgba(56, 95, 140, 0.1)',
        borderWidth: 1,
        pointRadius: 0,
        tension: 0.1,
      },
    ],
  };

  const fftChartData = fftData ? {
    labels: fftData.frequencies,
    datasets: [
      {
        label: 'Magnitude',
        data: fftData.magnitudes,
        borderColor: darkMode ? '#34D399' : '#10B981',
        backgroundColor: darkMode ? 'rgba(52, 211, 153, 0.1)' : 'rgba(16, 185, 129, 0.1)',
        borderWidth: 1,
        pointRadius: 0,
        fill: true,
      },
    ],
  } : null;


  // Turn off loading when FFT is ready
  useEffect(() => {
    if (fftData && chartsLoading) {
      // Small delay to ensure smooth transition
      const timer = setTimeout(() => {
        setChartsLoading(false);
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [fftData, chartsLoading]);

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        labels: {
          color: darkMode ? '#E9E9E9' : '#1F2937',
        },
      },
    },
    scales: {
      x: {
        ticks: {
          color: darkMode ? '#E9E9E9' : '#1F2937',
        },
        grid: {
          color: darkMode ? 'rgba(233, 233, 233, 0.1)' : 'rgba(31, 41, 55, 0.1)',
        },
      },
      y: {
        ticks: {
          color: darkMode ? '#E9E9E9' : '#1F2937',
        },
        grid: {
          color: darkMode ? 'rgba(233, 233, 233, 0.1)' : 'rgba(31, 41, 55, 0.1)',
        },
      },
    },
  };

  // Login page removed - auto-authenticated with default credentials

  // Results Page
  if (showResults && isCalculated) {
    return (
      <div className={`min-h-screen transition-colors duration-300 ${darkMode ? 'dark bg-gray-900' : 'bg-secondary'}`}>
        {/* Top Navigation Header */}
        <header className={`sticky top-0 z-50 w-full border-b ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} shadow-sm transition-colors duration-300`}>
          <div className="container mx-auto px-4">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center gap-4">
                <button
                  onClick={() => setShowResults(false)}
                  className={`p-3 rounded-lg ${darkMode ? 'bg-gray-700 text-white hover:bg-gray-600' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'} transition-colors`}
                  title={translations[language].back}
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M19 12H5M12 19l-7-7 7-7" />
                  </svg>
                </button>
              </div>
              <div className="flex items-center justify-center gap-3 flex-1">
                <img
                  src="/logo.png"
                  alt="Rotary Insight Logo"
                  className="h-16 w-16 object-contain"
                />
                <h1 className={`text-3xl font-bold ${darkMode ? 'text-white' : 'text-primary'} font-persian`}>
                  {translations[language].resultsTitle}
                </h1>
              </div>
              <div className="flex items-center gap-4">
                <div className="flex items-center border rounded-lg overflow-hidden">
                  <button
                    onClick={() => setLanguage('fa')}
                    className={`px-3 py-1.5 text-sm ${language === 'fa' ? (darkMode ? 'bg-primary text-white' : 'bg-primary text-white') : (darkMode ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' : 'bg-gray-200 text-gray-600 hover:bg-gray-300')} transition-colors`}
                  >
                    فا
                  </button>
                  <button
                    onClick={() => setLanguage('en')}
                    className={`px-3 py-1.5 text-sm border-r ${language === 'en' ? (darkMode ? 'bg-primary text-white' : 'bg-primary text-white') : (darkMode ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' : 'bg-gray-200 text-gray-600 hover:bg-gray-300')} transition-colors ${darkMode ? 'border-gray-600' : 'border-gray-300'}`}
                  >
                    en
                  </button>
                </div>
                <button
                  onClick={() => setDarkMode(!darkMode)}
                  className={`p-2 rounded-lg ${darkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-200 hover:bg-gray-300'} transition-colors`}
                  title={darkMode ? translations[language].light : translations[language].dark}
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

        <div className="container mx-auto px-4 py-8">
          {/* File Info */}
          <div className={`mb-6 p-4 rounded-lg ${darkMode ? 'bg-gray-800' : 'bg-white'} shadow-lg`}>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
              <div>
                <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'} font-persian`}>{translations[language].file}</p>
                <p className={`text-lg font-semibold ${darkMode ? 'text-white' : 'text-gray-800'} font-persian`}>
                  {csvFile?.name}
                </p>
              </div>
              <div>
                <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'} font-persian`}>{translations[language].model}</p>
                <p className={`text-lg font-semibold ${darkMode ? 'text-white' : 'text-gray-800'} font-persian`}>
                  {selectedModel}
                </p>
              </div>
              <div>
                <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'} font-persian`}>{translations[language].windowSize}</p>
                <p className={`text-lg font-semibold ${darkMode ? 'text-white' : 'text-gray-800'} font-persian`}>
                  {windowSize || '-'}
                </p>
              </div>
            </div>
          </div>

          {/* Signal Statistics Removed */}

          {/* Loading Modal */}
          {chartsLoading && (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 backdrop-blur-sm transition-opacity duration-300">
              <div className={`relative p-8 rounded-xl ${darkMode ? 'bg-gray-800' : 'bg-white'} shadow-2xl transform transition-all duration-300 scale-100`}>
                <div className="flex flex-col items-center justify-center space-y-4">
                  {/* Spinner Animation */}
                  <div className="relative w-16 h-16">
                    <div className={`absolute inset-0 border-4 ${darkMode ? 'border-gray-700' : 'border-gray-200'} rounded-full`}></div>
                    <div className={`absolute inset-0 border-4 border-t-transparent ${darkMode ? 'border-primary' : 'border-primary'} rounded-full animate-spin`}></div>
                  </div>
                  <p className={`text-lg font-semibold ${darkMode ? 'text-gray-200' : 'text-gray-700'} font-persian`}>
                    {translations[language].loadingCharts}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Classification Results Section */}
          {signalData.length > 0 && (
            <section className={`mb-8 p-6 rounded-lg shadow-lg ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
              <h2 className={`text-2xl font-bold mb-4 text-center ${darkMode ? 'text-white' : 'text-primary'} font-persian`}>
                {translations[language].classificationResults}
              </h2>

              {/* Dataset Selection Buttons Removed */}

              {/* Classification Results */}
              {classificationLoading ? (
                <div className="text-center py-8">
                  <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-primary mb-4"></div>
                  <p className={`${darkMode ? 'text-gray-300' : 'text-gray-700'} font-persian`}>
                    {translations[language].loadingClassification}
                  </p>
                </div>
              ) : error && error.includes('classification') ? (
                <div className={`text-center py-8 p-4 rounded-lg ${darkMode ? 'bg-red-900 text-red-200' : 'bg-red-100 text-red-800'}`}>
                  <p>{error}</p>
                </div>
              ) : classificationResults ? (
                <div>
                  {(() => {
                    // Check different possible response structures
                    const predictions = classificationResults.predictions || classificationResults.prediction || classificationResults;
                    const prediction = Array.isArray(predictions) ? predictions[0] : predictions;
                    const probabilities = prediction?.probabilities || prediction?.probs || prediction?.prob || [];

                    if (!prediction || probabilities.length === 0) {
                      return (
                        <div className="text-center py-8">
                          <p className={`${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                            No classification data available.
                          </p>
                          <p className={`text-xs mt-2 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                            Response structure: {JSON.stringify(classificationResults).substring(0, 300)}
                          </p>
                        </div>
                      );
                    }

                    // Get model info from classification result or selected model
                    let modelInfo = models[selectedModel];
                    if (!modelInfo && classificationResults.model_name) {
                      modelInfo = models[classificationResults.model_name];
                    }
                    // Also try to find model by dataset
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
                      );
                    }

                    const maxProb = Math.max(...probabilities);
                    const maxIndex = probabilities.indexOf(maxProb);

                    // Helper function to format class name for display
                    const formatClassName = (className: string) => {
                      // Convert API format to display format
                      // Examples: "0.007-OuterRace" -> "Outer Ring 0.007"
                      //           "0.014-Ball" -> "Ball Fault 0.014"
                      //           "Normal" -> "Normal"

                      if (className === 'Normal') {
                        return 'Normal';
                      }

                      // Match pattern: "0.XXX-Type" or "Type-0.XXX"
                      const match = className.match(/(\d+\.\d+)-?(\w+)/);
                      if (match) {
                        const severity = match[1]; // e.g., "0.007"
                        const faultType = match[2]; // e.g., "OuterRace", "Ball", "InnerRace"

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

                      // If no match, return original
                      return className;
                    };

                    // Helper function to get color based on class type and index
                    const getBarColor = (className: string, idx: number, isMax: boolean) => {
                      if (isMax) return 'bg-green-500';

                      // Color coding based on class type and severity
                      if (className.includes('OuterRace') || className.includes('Outer Ring')) {
                        // Outer Ring 0.007 (index 7) -> yellow, others -> blue
                        if (className.includes('0.007')) return 'bg-yellow-500';
                        return 'bg-blue-500';
                      }
                      if (className.includes('InnerRace') || className.includes('Inner Ring')) {
                        // Inner Ring 0.007 (index 4) -> red, others -> blue
                        if (className.includes('0.007')) return 'bg-red-500';
                        return 'bg-blue-500';
                      }
                      if (className.includes('Ball') || className.includes('Ball Fault')) {
                        return 'bg-blue-500';
                      }
                      // Normal
                      return 'bg-blue-500';
                    };

                    // For CWRU Dataset: Split into two panels (10 classes)
                    if (selectedDataset === 'CWRU' && classNames.length === 10) {
                      // Part 1: Outer Ring (indices 7, 8, 9) and Inner Ring (indices 4, 5)
                      // Part 2: Inner Ring (index 6), Ball Fault (indices 1, 2, 3), Normal (index 0)
                      const part1Indices = [7, 8, 9, 4, 5]; // Outer Ring 0.007, 0.014, 0.021, Inner Ring 0.007, 0.014
                      const part2Indices = [6, 1, 2, 3, 0]; // Inner Ring 0.021, Ball Fault 0.007, 0.014, 0.021, Normal

                      return (
                        <div>
                          <h3 className={`text-xl font-semibold mb-4 text-center ${darkMode ? 'text-gray-300' : 'text-gray-700'} font-persian`}>
                            {translations[language].cwruDataset} - Fault Classification
                          </h3>

                          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                            {/* Part 2 */}
                            <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
                              <h4 className={`text-lg font-semibold mb-3 text-center ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                                {translations[language].cwruDataset} - Fault Classification (Part 2)
                              </h4>
                              <div className="space-y-3">
                                {part2Indices.map((idx) => {
                                  const prob = probabilities[idx];
                                  const originalClassName = classNames[idx] || `Class ${idx}`;
                                  const className = formatClassName(originalClassName);
                                  const isMax = idx === maxIndex;

                                  return (
                                    <div key={idx} className="flex items-center gap-3" dir="ltr">
                                      <div className="flex-1">
                                        <div className="flex justify-between items-center mb-1" dir="ltr">
                                          <span className={`text-sm font-semibold ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                                            {className}
                                          </span>
                                          <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                                            {prob.toFixed(4)}
                                          </span>
                                        </div>
                                        <div className={`h-5 rounded-full overflow-hidden ${darkMode ? 'bg-gray-600' : 'bg-gray-300'}`} dir="ltr">
                                          <div
                                            className={`h-full transition-all duration-500 ${getBarColor(originalClassName, idx, isMax)}`}
                                            style={{ width: `${Math.min(prob * 100, 100)}%` }}
                                          ></div>
                                        </div>
                                      </div>
                                    </div>
                                  );
                                })}
                              </div>
                            </div>

                            {/* Part 1 */}
                            <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
                              <h4 className={`text-lg font-semibold mb-3 text-center ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                                {translations[language].cwruDataset} - Fault Classification (Part 1)
                              </h4>
                              <div className="space-y-3">
                                {part1Indices.map((idx) => {
                                  const prob = probabilities[idx];
                                  const originalClassName = classNames[idx] || `Class ${idx}`;
                                  const className = formatClassName(originalClassName);
                                  const isMax = idx === maxIndex;

                                  return (
                                    <div key={idx} className="flex items-center gap-3" dir="ltr">
                                      <div className="flex-1">
                                        <div className="flex justify-between items-center mb-1" dir="ltr">
                                          <span className={`text-sm font-semibold ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                                            {className}
                                          </span>
                                          <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                                            {prob.toFixed(4)}
                                          </span>
                                        </div>
                                        <div className={`h-5 rounded-full overflow-hidden ${darkMode ? 'bg-gray-600' : 'bg-gray-300'}`} dir="ltr">
                                          <div
                                            className={`h-full transition-all duration-500 ${getBarColor(originalClassName, idx, isMax)}`}
                                            style={{ width: `${Math.min(prob * 100, 100)}%` }}
                                          ></div>
                                        </div>
                                      </div>
                                    </div>
                                  );
                                })}
                              </div>
                            </div>
                          </div>

                          {/* Predicted Class Summary */}
                          <div className={`mt-6 p-4 rounded-lg text-center ${darkMode ? 'bg-gray-700' : 'bg-secondary'}`}>
                            <p className={`text-lg font-semibold ${darkMode ? 'text-white' : 'text-gray-800'}`}>
                              {translations[language].predictedClass}: <span className="text-green-500">{formatClassName(classNames[maxIndex] || `Class ${maxIndex}`)}</span> ({translations[language].confidence}: {(maxProb * 100).toFixed(2)}%)
                            </p>
                          </div>
                        </div>
                      );
                    }

                    // For PU Dataset or other datasets: Single panel
                    return (
                      <div>
                        <h3 className={`text-xl font-semibold mb-4 text-center ${darkMode ? 'text-gray-300' : 'text-gray-700'} font-persian`}>
                          {selectedDataset === 'PU' ? translations[language].puDataset : translations[language].cwruDataset} - Fault Classification
                        </h3>

                        <div className="space-y-4">
                          {probabilities.map((prob: number, idx: number) => {
                            const isMax = idx === maxIndex;
                            const originalClassName = classNames[idx] || `Class ${idx}`;
                            const className = formatClassName(originalClassName);

                            return (
                              <div key={idx} className="flex items-center gap-4" dir="ltr">
                                <div className="flex-1">
                                  <div className="flex justify-between items-center mb-1" dir="ltr">
                                    <span className={`font-semibold ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                                      {className}
                                    </span>
                                    <span className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                                      {prob.toFixed(4)}
                                    </span>
                                  </div>
                                  <div className={`h-6 rounded-full overflow-hidden ${darkMode ? 'bg-gray-700' : 'bg-gray-200'}`} dir="ltr">
                                    <div
                                      className={`h-full transition-all duration-500 ${getBarColor(originalClassName, idx, isMax)}`}
                                      style={{ width: `${Math.min(prob * 100, 100)}%` }}
                                    ></div>
                                  </div>
                                </div>
                              </div>
                            );
                          })}

                          {/* Predicted Class Summary */}
                          <div className={`mt-6 p-4 rounded-lg text-center ${darkMode ? 'bg-gray-700' : 'bg-secondary'}`}>
                            <p className={`text-lg font-semibold ${darkMode ? 'text-white' : 'text-gray-800'}`}>
                              {translations[language].predictedClass}: <span className="text-green-500">{formatClassName(classNames[maxIndex] || `Class ${maxIndex}`)}</span> ({translations[language].confidence}: {(maxProb * 100).toFixed(2)}%)
                            </p>
                          </div>
                        </div>
                      </div>
                    );
                  })()}
                </div>
              ) : null}
            </section>
          )}

          {/* Charts Section */}
          {signalData.length > 0 && (
            <section className={`mb-8 p-6 rounded-lg shadow-lg ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
              <h2 className={`text-2xl font-bold mb-4 text-center ${darkMode ? 'text-white' : 'text-primary'} font-persian`}>
                {translations[language].charts}
              </h2>

              {/* Time Domain Signal */}
              <div className="mb-16">
                <h3 className={`text-xl font-semibold mb-2 text-center ${darkMode ? 'text-gray-200' : 'text-gray-700'} font-persian`}>
                  {translations[language].timeDomain}
                </h3>
                <div className="h-64">
                  <Line data={timeDomainChartData} options={chartOptions} />
                </div>
              </div>

              {/* Frequency Domain (FFT) */}
              {fftChartData && (
                <div className="mb-16">
                  <h3 className={`text-xl font-semibold mb-2 text-center ${darkMode ? 'text-gray-200' : 'text-gray-700'} font-persian`}>
                    {translations[language].frequencyDomain}
                  </h3>
                  <div className="h-64">
                    <Line data={fftChartData} options={chartOptions} />
                  </div>
                </div>
              )}

              {/* Short-Time Fourier Transform (STFT) */}
              <STFTResult
                stftData={stftData}
                windowSize={windowSize}
                hopLength={hopLength}
                onHopLengthChange={handleHopLengthChange}
                onRecalculate={handleRecalculateSTFT}
                isLoading={stftLoading}
                darkMode={darkMode}
                translations={translations}
                language={language}
              />

            </section>
          )}
        </div>
      </div>
    );
  }

  // Main Application
  return (
    <div className={`min-h-screen transition-colors duration-300 ${darkMode ? 'dark bg-gray-900' : 'bg-secondary'}`}>
      {/* Top Navigation Header */}
      <header className={`sticky top-0 z-50 w-full border-b ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} shadow-sm transition-colors duration-300`}>
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-between h-16">
            <div className="flex-1"></div>
            <div className="flex items-center justify-center gap-3 flex-1">
              <img
                src="/logo.png"
                alt="Rotary Insight Logo"
                className="h-16 w-16 object-contain"
              />
              <h1 className={`text-3xl font-bold ${darkMode ? 'text-white' : 'text-primary'} font-persian`}>
                Rotary Insight
              </h1>
            </div>
            <div className="flex items-center gap-4 flex-1 justify-end">
              <div className="flex items-center border rounded-lg overflow-hidden">
                <button
                  onClick={() => setLanguage('fa')}
                  className={`px-3 py-1.5 text-sm ${language === 'fa' ? (darkMode ? 'bg-primary text-white' : 'bg-primary text-white') : (darkMode ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' : 'bg-gray-200 text-gray-600 hover:bg-gray-300')} transition-colors`}
                >
                  فا
                </button>
                <button
                  onClick={() => setLanguage('en')}
                  className={`px-3 py-1.5 text-sm border-r ${language === 'en' ? (darkMode ? 'bg-primary text-white' : 'bg-primary text-white') : (darkMode ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' : 'bg-gray-200 text-gray-600 hover:bg-gray-300')} transition-colors ${darkMode ? 'border-gray-600' : 'border-gray-300'}`}
                >
                  en
                </button>
              </div>
              <button
                onClick={() => setDarkMode(!darkMode)}
                className={`p-2 rounded-lg ${darkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-200 hover:bg-gray-300'} transition-colors`}
                title={darkMode ? translations[language].light : translations[language].dark}
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

      <div className="container mx-auto px-4 py-8">

        {/* Error Message */}
        {error && (
          <div className={`mb-6 p-4 rounded-lg text-center ${darkMode ? 'bg-red-900 text-red-200' : 'bg-red-100 text-red-800'}`}>
            {error}
          </div>
        )}

        {/* Configuration Section */}
        <section className={`mb-8 p-6 rounded-lg shadow-lg ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
          <h2 className={`text-2xl font-bold mb-4 text-center ${darkMode ? 'text-white' : 'text-primary'} font-persian`}>
            {translations[language].configuration}
          </h2>

          {/* Window Size Filter */}
          <div className="mb-6">
            <label className={`block mb-2 font-semibold text-center ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
              {translations[language].filterByWindowSize}
            </label>
            <select
              value={windowSizeFilter}
              onChange={(e) => setWindowSizeFilter(e.target.value === 'all' ? 'all' : parseInt(e.target.value))}
              className={`w-full md:w-1/3 mx-auto px-4 py-2 rounded-lg border ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300'} focus:outline-none focus:ring-2 focus:ring-primary`}
            >
              <option value="all">{translations[language].allWindowSizes}</option>
              {availableWindowSizes.map(size => (
                <option key={size} value={size}>{size}</option>
              ))}
            </select>
          </div>
          {loading ? (
            <div className="text-center mt-4">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-primary mb-4"></div>
              <p className={`${darkMode ? 'text-gray-400' : 'text-gray-600'} font-persian`}>
                {translations[language].loading}
              </p>
            </div>
          ) : (
            <div className="space-y-6">
              {Object.keys(modelsByDataset).length === 0 ? (
                <div className="text-center py-8">
                  <p className={`${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                    {translations[language].loadModelsFirst}
                  </p>
                </div>
              ) : (
                Object.keys(modelsByDataset).map((dataset) => (
                  <div key={dataset} className="space-y-3">
                    <h3 className={`text-lg font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'} border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'} pb-2`}>
                      {dataset === 'PU' ? translations[language].puDataset : dataset === 'CWRU' ? translations[language].cwruDataset : dataset}
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {modelsByDataset[dataset].map((modelKey) => {
                        const model = models[modelKey];
                        const isSelected = selectedModel === modelKey;
                        return (
                          <div
                            key={modelKey}
                            onClick={() => handleModelSelect(modelKey)}
                            className={`
                              relative cursor-pointer rounded-xl border-2 p-4 transition-all duration-200
                              ${isSelected
                                ? (darkMode ? 'border-primary bg-primary/20' : 'border-primary bg-primary/10')
                                : (darkMode ? 'border-gray-700 bg-gray-800 hover:border-gray-600' : 'border-gray-200 bg-white hover:border-gray-300')
                              }
                            `}
                          >
                            <div className="flex justify-between items-start mb-2">
                              <h4 className={`font-bold text-sm ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                                {modelKey}
                              </h4>
                              {isSelected && (
                                <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary text-white">
                                  <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                                  </svg>
                                </span>
                              )}
                            </div>

                            <div className="flex items-center gap-2 mb-2">
                              <span className={`text-xs px-2 py-1 rounded-full ${darkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-600'}`}>
                                {translations[language].windowSize}: {model.window_size}
                              </span>
                            </div>

                            <p className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'} line-clamp-2`}>
                              {model.description}
                            </p>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))
              )}
            </div>
          )}
          {loading && (
            <div className="text-center mt-4">
              <p className={`${darkMode ? 'text-gray-400' : 'text-gray-600'} font-persian`}>
                {translations[language].loading}
              </p>
            </div>
          )}
        </section>

        {/* File Upload Section */}
        <section className={`mb-8 p-6 rounded-lg shadow-lg ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
          <h2 className={`text-2xl font-bold mb-4 text-center ${darkMode ? 'text-white' : 'text-primary'} font-persian`}>
            {translations[language].uploadCsv}
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
              {translations[language].dragDrop}
            </p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFileSelect(file);
              }}
              className="hidden"
            />
          </div>
          {csvFile && (
            <div className={`mt-4 p-4 rounded-lg text-center ${darkMode ? 'bg-gray-700' : 'bg-secondary'}`}>
              <p className={`${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                <strong>{translations[language].file}:</strong> {csvFile.name}
              </p>
            </div>
          )}
          {csvFile && selectedModel && (
            <div className="mt-6 text-center">
              <button
                onClick={handleCalculate}
                disabled={loading || !selectedModel || !csvFile}
                className={`px-8 py-3 rounded-lg text-lg ${darkMode ? 'bg-primary hover:bg-primary/80' : 'bg-primary hover:bg-primary/90'} text-white font-semibold disabled:opacity-50 disabled:cursor-not-allowed transition-colors`}
              >
                {translations[language].calculate}
              </button>
            </div>
          )}
        </section>

        {/* Sample Files Section */}
        <section className={`mb-8 p-6 rounded-lg shadow-lg ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
          <h2 className={`text-2xl font-bold mb-4 text-center ${darkMode ? 'text-white' : 'text-primary'} font-persian`}>
            {translations[language].orSelectSample}
          </h2>
          {loadingExamples ? (
            <div className="text-center py-8">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-primary mb-4"></div>
              <p className={`${darkMode ? 'text-gray-400' : 'text-gray-600'} font-persian`}>
                {translations[language].loading}
              </p>
            </div>
          ) : exampleFiles.length === 0 ? (
            <div className="text-center py-8">
              <p className={`${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                {translations[language].noSamplesAvailable}
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {exampleFiles.map((example) => (
                <div
                  key={example.filename}
                  onClick={() => handleLoadExample(example.filename)}
                  className={`
                    cursor-pointer rounded-xl border-2 p-4 transition-all duration-200
                    ${csvFile?.name === example.filename
                      ? (darkMode ? 'border-primary bg-primary/20' : 'border-primary bg-primary/10')
                      : (darkMode ? 'border-gray-700 bg-gray-800 hover:border-gray-600' : 'border-gray-200 bg-white hover:border-gray-300')
                    }
                  `}
                >
                  <div className="flex justify-between items-start mb-2">
                    <h4 className={`font-bold text-sm ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                      Sample {example.sample_index}
                    </h4>
                    {csvFile?.name === example.filename && (
                      <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary text-white">
                        <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                        </svg>
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-2 mb-2">
                    <span className={`text-xs px-2 py-1 rounded-full ${darkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-600'}`}>
                      {example.fault_name.replace(/_/g, ' ')}
                    </span>
                  </div>

                  <p className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    {example.filename}
                  </p>
                </div>
              ))}
            </div>
          )}
        </section>

      </div>
    </div>
  );
}

export default App;
