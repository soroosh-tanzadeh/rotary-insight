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
    heatMap: 'نقشه حرارتی',
    login: 'ورود',
    validating: 'در حال بررسی...',
    file: 'فایل',
    model: 'مدل',
    loading: 'در حال بارگذاری مدل‌ها...',
    select: 'انتخاب کنید',
    loadModelsFirst: 'ابتدا مدل‌ها را بارگذاری کنید',
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
    heatMap: 'Heat Map',
    login: 'Login',
    validating: 'Validating...',
    file: 'File',
    model: 'Model',
    loading: 'Loading models...',
    select: 'Select',
    loadModelsFirst: 'Load models first',
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
  },
};

function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [language, setLanguage] = useState<'fa' | 'en'>('fa');
  const [apiUrl, setApiUrl] = useState('http://localhost:8000');
  const [apiKey, setApiKey] = useState('');
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [tempApiUrl, setTempApiUrl] = useState('http://localhost:8000');
  const [tempApiKey, setTempApiKey] = useState('');
  const [models, setModels] = useState<{ [key: string]: Model }>({});
  const [selectedModel, setSelectedModel] = useState('');
  const [windowSize, setWindowSize] = useState(512);
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [signalData, setSignalData] = useState<number[]>([]);
  const [fftData, setFftData] = useState<{ frequencies: number[]; magnitudes: number[] } | null>(null);
  const [signalStats, setSignalStats] = useState<SignalStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [heatMapData, setHeatMapData] = useState<number[][] | null>(null);
  const [validating, setValidating] = useState(false);
  const [isCalculated, setIsCalculated] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Load saved config from localStorage
    const savedApiUrl = localStorage.getItem('apiUrl');
    const savedApiKey = localStorage.getItem('apiKey');
    const savedDarkMode = localStorage.getItem('darkMode') === 'true';
    const savedAuth = localStorage.getItem('isAuthenticated') === 'true';
    const savedLanguage = localStorage.getItem('language') as 'fa' | 'en' | null;
    
    if (savedApiUrl) {
      setApiUrl(savedApiUrl);
      setTempApiUrl(savedApiUrl);
    }
    if (savedApiKey) {
      setApiKey(savedApiKey);
      setTempApiKey(savedApiKey);
    }
    if (savedDarkMode) setDarkMode(savedDarkMode);
    if (savedLanguage) setLanguage(savedLanguage);
    if (savedAuth && savedApiUrl && savedApiKey) {
      setIsAuthenticated(true);
    }
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
    setSignalStats(null);
    setHeatMapData(null);
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

  const handleModelSelect = (modelName: string) => {
    setSelectedModel(modelName);
    if (models[modelName]) {
      setWindowSize(models[modelName].window_size);
    }
    setIsCalculated(false);
    setShowResults(false);
  };

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
          return;
        }

        setSignalData(data);
        calculateStats(data);
        computeFFT(data);
        setIsCalculated(true);
        setShowResults(true);
        setError('');
      },
      error: (error) => {
        setError(translations[language].failedParseCsv + error.message);
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
    if (windowSize <= 0) {
      setError(translations[language].windowSizeError);
      return;
    }
    
    setError('');
    parseCSV(csvFile);
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
      const fftLength = Math.min(windowSize, data.length);
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

  useEffect(() => {
    if (signalData.length > 0) {
      const computeHeatMap = async () => {
        const windowCount = Math.ceil(signalData.length / windowSize);
        const heatMapResult: number[][] = [];

        for (let i = 0; i < windowCount; i++) {
          const window = signalData.slice(i * windowSize, (i + 1) * windowSize);
          let paddedWindow = [...window];
          if (paddedWindow.length < windowSize) {
            paddedWindow = [...paddedWindow, ...new Array(windowSize - paddedWindow.length).fill(0)];
          }
          
          try {
            // Compute FFT for each window using API
            const response = await fetch(`${apiUrl}/processing/fft`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'X-API-Key': apiKey,
              },
              body: JSON.stringify({
                signal: paddedWindow,
                n: windowSize,
              }),
            });

            if (response.ok) {
              const result = await response.json();
              // Take first 128 frequency bins for visualization
              heatMapResult.push(result.magnitudes.slice(0, 128));
            } else {
              // Fallback: use simple magnitude calculation
              const magnitudes = paddedWindow.map(val => Math.abs(val));
              heatMapResult.push(magnitudes.slice(0, 128));
            }
          } catch {
            // Fallback: use simple magnitude calculation
            const magnitudes = paddedWindow.map(val => Math.abs(val));
            heatMapResult.push(magnitudes.slice(0, 128));
          }
        }

        setHeatMapData(heatMapResult);
      };

      computeHeatMap();
    } else {
      setHeatMapData(null);
    }
  }, [signalData, windowSize, apiUrl, apiKey]);

  const heatMapChartData = heatMapData && heatMapData.length > 0 ? {
    labels: Array.from({ length: heatMapData[0].length }, (_, i) => i),
    datasets: heatMapData.map((row, idx) => {
      const maxValue = Math.max(...heatMapData.flat());
      return {
        label: `Window ${idx + 1}`,
        data: row,
        backgroundColor: row.map((value) => {
          const intensity = maxValue > 0 ? value / maxValue : 0;
          return darkMode 
            ? `rgba(96, 165, 250, ${Math.max(0.1, intensity)})`
            : `rgba(56, 95, 140, ${Math.max(0.1, intensity)})`;
        }),
      };
    }),
  } : null;

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

  // Login/Setup Page
  if (!isAuthenticated) {
    return (
      <div className={`min-h-screen transition-colors duration-300 flex items-center justify-center ${darkMode ? 'dark bg-gray-900' : 'bg-secondary'}`}>
        <div className={`w-full max-w-md p-8 rounded-xl shadow-2xl ${darkMode ? 'bg-gray-800' : 'bg-white'} transition-colors duration-300`}>
          <div className="text-center mb-8">
            <div className="flex flex-col items-center justify-center mb-4">
              <img 
                src="/logo.png" 
                alt="Rotary Insight Logo" 
                className="h-28 w-28 object-contain mb-4"
              />
              <h1 className={`text-5xl font-bold ${darkMode ? 'text-white' : 'text-primary'} font-persian`}>
                Rotary Insight
              </h1>
            </div>
          </div>

          {error && (
            <div className={`mb-6 p-4 rounded-lg text-center ${darkMode ? 'bg-red-900 text-red-200' : 'bg-red-100 text-red-800'}`}>
              {error}
            </div>
          )}

          <div className="space-y-6">
            <div>
              <label className={`block mb-2 font-semibold text-center ${darkMode ? 'text-gray-200' : 'text-gray-700'} font-persian`}>
                API URL
              </label>
              <input
                type="text"
                value={tempApiUrl}
                onChange={(e) => setTempApiUrl(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleLogin()}
                className={`w-full px-4 py-3 rounded-lg border text-center ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300'} focus:outline-none focus:ring-2 focus:ring-primary transition-all`}
                placeholder="http://localhost:8000"
                disabled={validating}
              />
            </div>

            <div>
              <label className={`block mb-2 font-semibold text-center ${darkMode ? 'text-gray-200' : 'text-gray-700'} font-persian`}>
                API Key
              </label>
              <input
                type="password"
                value={tempApiKey}
                onChange={(e) => setTempApiKey(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleLogin()}
                className={`w-full px-4 py-3 rounded-lg border text-center ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300'} focus:outline-none focus:ring-2 focus:ring-primary transition-all`}
                placeholder="Enter API Key"
                disabled={validating}
              />
            </div>

            <button
              onClick={handleLogin}
              disabled={validating}
              className={`w-full px-6 py-3 rounded-lg ${darkMode ? 'bg-primary hover:bg-primary/80' : 'bg-primary hover:bg-primary/90'} text-white font-semibold disabled:opacity-50 disabled:cursor-not-allowed transition-colors`}
            >
              {validating ? 'در حال بررسی...' : 'ورود / Login'}
            </button>
          </div>
        </div>
      </div>
    );
  }

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
                    <path d="M19 12H5M12 19l-7-7 7-7"/>
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
                  {windowSize}
                </p>
              </div>
            </div>
          </div>

          {/* Signal Statistics */}
          {signalStats && (
            <section className={`mb-8 p-6 rounded-lg shadow-lg ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
              <h2 className={`text-2xl font-bold mb-4 text-center ${darkMode ? 'text-white' : 'text-primary'} font-persian`}>
                {translations[language].signalInfo}
              </h2>
              
              {/* Basic Statistics */}
              <h3 className={`text-xl font-semibold mb-3 text-center ${darkMode ? 'text-gray-300' : 'text-gray-700'} font-persian`}>
                {translations[language].basicStats}
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className={`p-4 rounded-lg text-center ${darkMode ? 'bg-gray-700' : 'bg-secondary'}`}>
                  <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>{translations[language].min}</p>
                  <p className={`text-xl font-bold ${darkMode ? 'text-white' : 'text-primary'}`}>
                    {signalStats.min.toFixed(4)}
                  </p>
                  <p className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-600'}`}>
                    Index: {signalStats.extrema.min.index}
                  </p>
                </div>
                <div className={`p-4 rounded-lg text-center ${darkMode ? 'bg-gray-700' : 'bg-secondary'}`}>
                  <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>{translations[language].max}</p>
                  <p className={`text-xl font-bold ${darkMode ? 'text-white' : 'text-primary'}`}>
                    {signalStats.max.toFixed(4)}
                  </p>
                  <p className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-600'}`}>
                    Index: {signalStats.extrema.max.index}
                  </p>
                </div>
                <div className={`p-4 rounded-lg text-center ${darkMode ? 'bg-gray-700' : 'bg-secondary'}`}>
                  <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>{translations[language].mean}</p>
                  <p className={`text-xl font-bold ${darkMode ? 'text-white' : 'text-primary'}`}>
                    {signalStats.mean.toFixed(4)}
                  </p>
                </div>
                <div className={`p-4 rounded-lg text-center ${darkMode ? 'bg-gray-700' : 'bg-secondary'}`}>
                  <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>{translations[language].median}</p>
                  <p className={`text-xl font-bold ${darkMode ? 'text-white' : 'text-primary'}`}>
                    {signalStats.median.toFixed(4)}
                  </p>
                </div>
              </div>

              {/* Spread Statistics */}
              <h3 className={`text-xl font-semibold mb-3 text-center ${darkMode ? 'text-gray-300' : 'text-gray-700'} font-persian`}>
                {translations[language].spreadStats}
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className={`p-4 rounded-lg text-center ${darkMode ? 'bg-gray-700' : 'bg-secondary'}`}>
                  <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>{translations[language].stdDev}</p>
                  <p className={`text-xl font-bold ${darkMode ? 'text-white' : 'text-primary'}`}>
                    {signalStats.std.toFixed(4)}
                  </p>
                </div>
                <div className={`p-4 rounded-lg text-center ${darkMode ? 'bg-gray-700' : 'bg-secondary'}`}>
                  <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>{translations[language].variance}</p>
                  <p className={`text-xl font-bold ${darkMode ? 'text-white' : 'text-primary'}`}>
                    {signalStats.variance.toFixed(4)}
                  </p>
                </div>
                <div className={`p-4 rounded-lg text-center ${darkMode ? 'bg-gray-700' : 'bg-secondary'}`}>
                  <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>{translations[language].range}</p>
                  <p className={`text-xl font-bold ${darkMode ? 'text-white' : 'text-primary'}`}>
                    {signalStats.range.toFixed(4)}
                  </p>
                </div>
                <div className={`p-4 rounded-lg text-center ${darkMode ? 'bg-gray-700' : 'bg-secondary'}`}>
                  <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>{translations[language].peakToPeak}</p>
                  <p className={`text-xl font-bold ${darkMode ? 'text-white' : 'text-primary'}`}>
                    {signalStats.peakToPeak.toFixed(4)}
                  </p>
                </div>
              </div>

              {/* Advanced Statistics */}
              <h3 className={`text-xl font-semibold mb-3 text-center ${darkMode ? 'text-gray-300' : 'text-gray-700'} font-persian`}>
                {translations[language].advancedStats}
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className={`p-4 rounded-lg text-center ${darkMode ? 'bg-gray-700' : 'bg-secondary'}`}>
                  <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>{translations[language].rms}</p>
                  <p className={`text-xl font-bold ${darkMode ? 'text-white' : 'text-primary'}`}>
                    {signalStats.rms.toFixed(4)}
                  </p>
                </div>
                <div className={`p-4 rounded-lg text-center ${darkMode ? 'bg-gray-700' : 'bg-secondary'}`}>
                  <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>{translations[language].energy}</p>
                  <p className={`text-xl font-bold ${darkMode ? 'text-white' : 'text-primary'}`}>
                    {signalStats.energy.toExponential(2)}
                  </p>
                </div>
                <div className={`p-4 rounded-lg text-center ${darkMode ? 'bg-gray-700' : 'bg-secondary'}`}>
                  <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>{translations[language].crestFactor}</p>
                  <p className={`text-xl font-bold ${darkMode ? 'text-white' : 'text-primary'}`}>
                    {signalStats.crestFactor.toFixed(4)}
                  </p>
                </div>
                <div className={`p-4 rounded-lg text-center ${darkMode ? 'bg-gray-700' : 'bg-secondary'}`}>
                  <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>{translations[language].zeroCrossings}</p>
                  <p className={`text-xl font-bold ${darkMode ? 'text-white' : 'text-primary'}`}>
                    {signalStats.zeroCrossings}
                  </p>
                </div>
              </div>
            </section>
          )}

          {/* Charts Section */}
          {signalData.length > 0 && (
            <section className={`mb-8 p-6 rounded-lg shadow-lg ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
              <h2 className={`text-2xl font-bold mb-4 text-center ${darkMode ? 'text-white' : 'text-primary'} font-persian`}>
                {translations[language].charts}
              </h2>
              
              {/* Time Domain Signal */}
              <div className="mb-8">
                <h3 className={`text-xl font-semibold mb-2 text-center ${darkMode ? 'text-gray-200' : 'text-gray-700'} font-persian`}>
                  {translations[language].timeDomain}
                </h3>
                <div className="h-64">
                  <Line data={timeDomainChartData} options={chartOptions} />
                </div>
              </div>

              {/* Frequency Domain (FFT) */}
              {fftChartData && (
                <div className="mb-8">
                  <h3 className={`text-xl font-semibold mb-2 text-center ${darkMode ? 'text-gray-200' : 'text-gray-700'} font-persian`}>
                    {translations[language].frequencyDomain}
                  </h3>
                  <div className="h-64">
                    <Line data={fftChartData} options={chartOptions} />
                  </div>
                </div>
              )}

              {/* Heat Map */}
              {heatMapData && heatMapData.length > 0 && (
                <div>
                  <h3 className={`text-xl font-semibold mb-2 text-center ${darkMode ? 'text-gray-200' : 'text-gray-700'} font-persian`}>
                    {translations[language].heatMap}
                  </h3>
                  <div className="overflow-x-auto">
                    <div className="inline-block min-w-full">
                      <div className="flex flex-col gap-1 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                        {(() => {
                          const maxValue = Math.max(...heatMapData.flat());
                          const minValue = Math.min(...heatMapData.flat());
                          const range = maxValue - minValue;
                          
                          return heatMapData.map((row, rowIdx) => (
                            <div key={rowIdx} className="flex gap-1" style={{ height: `${Math.max(4, 400 / heatMapData.length)}px` }}>
                              {row.map((value, colIdx) => {
                                const normalizedValue = range > 0 ? (value - minValue) / range : 0;
                                
                                // Color gradient: blue (low) to red (high)
                                let r, g, b;
                                if (darkMode) {
                                  // Dark mode: darker blue to bright yellow/red
                                  r = Math.floor(96 + normalizedValue * 159); // 96 to 255
                                  g = Math.floor(165 - normalizedValue * 100); // 165 to 65
                                  b = Math.floor(250 - normalizedValue * 190); // 250 to 60
                                } else {
                                  // Light mode: light blue to dark red
                                  r = Math.floor(59 + normalizedValue * 196); // 59 to 255
                                  g = Math.floor(130 - normalizedValue * 70); // 130 to 60
                                  b = Math.floor(236 - normalizedValue * 176); // 236 to 60
                                }
                                
                                return (
                                  <div
                                    key={colIdx}
                                    className="flex-1 rounded-sm transition-all hover:opacity-80 cursor-pointer"
                                    style={{
                                      backgroundColor: `rgb(${r}, ${g}, ${b})`,
                                      minWidth: `${Math.max(2, 100 / row.length)}px`,
                                    }}
                                    title={`Window ${rowIdx + 1}, Freq ${colIdx}: ${value.toFixed(2)}`}
                                  />
                                );
                              })}
                            </div>
                          ));
                        })()}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center justify-between mt-4">
                    <p className={`text-sm text-center ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                      {translations[language].showing} {heatMapData.length} {translations[language].windows}
                    </p>
                    <div className="flex items-center gap-2">
                      <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                        {translations[language].low}
                      </span>
                      <div className="flex gap-1 w-32 h-4 rounded overflow-hidden">
                        {Array.from({ length: 20 }, (_, i) => {
                          const normalizedValue = i / 19;
                          let r, g, b;
                          if (darkMode) {
                            r = Math.floor(96 + normalizedValue * 159);
                            g = Math.floor(165 - normalizedValue * 100);
                            b = Math.floor(250 - normalizedValue * 190);
                          } else {
                            r = Math.floor(59 + normalizedValue * 196);
                            g = Math.floor(130 - normalizedValue * 70);
                            b = Math.floor(236 - normalizedValue * 176);
                          }
                          return (
                            <div
                              key={i}
                              className="flex-1"
                              style={{ backgroundColor: `rgb(${r}, ${g}, ${b})` }}
                            />
                          );
                        })}
                      </div>
                      <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                        {translations[language].high}
                      </span>
                    </div>
                  </div>
                </div>
              )}
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
                onClick={handleLogout}
                className={`p-2 rounded-lg ${darkMode ? 'bg-red-700 text-white hover:bg-red-600' : 'bg-red-500 text-white hover:bg-red-600'} transition-colors`}
                title={translations[language].logout}
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9"/>
                </svg>
              </button>
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
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div>
              <label className={`block mb-2 font-semibold text-center ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                {translations[language].selectModel}
              </label>
              <select
                value={selectedModel}
                onChange={(e) => handleModelSelect(e.target.value)}
                className={`w-full px-4 py-2 rounded-lg border ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300'} focus:outline-none focus:ring-2 focus:ring-primary`}
                disabled={Object.keys(models).length === 0}
              >
                <option value="">{Object.keys(models).length === 0 ? translations[language].loadModelsFirst : translations[language].select}</option>
                {Object.keys(models).map((key) => (
                  <option key={key} value={key}>
                    {key}
                  </option>
                ))}
              </select>
              {selectedModel && models[selectedModel] && (
                <p className={`mt-2 text-sm text-center ${darkMode ? 'text-gray-400' : 'text-gray-600'} font-persian`}>
                  {language === 'fa' ? MODEL_DESCRIPTIONS[selectedModel] || MODEL_DESCRIPTIONS_EN[selectedModel] : MODEL_DESCRIPTIONS_EN[selectedModel] || MODEL_DESCRIPTIONS[selectedModel]}
                </p>
              )}
            </div>
            <div>
              <label className={`block mb-2 font-semibold text-center ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                {translations[language].windowSize}
              </label>
              <input
                type="number"
                value={windowSize}
                onChange={(e) => {
                  const value = parseInt(e.target.value);
                  if (!isNaN(value) && value > 0) {
                    setWindowSize(value);
                    setIsCalculated(false);
                  }
                }}
                min="1"
                className={`w-full px-4 py-2 rounded-lg border ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300'} focus:outline-none focus:ring-2 focus:ring-primary`}
              />
            </div>
          </div>
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
            className={`border-2 border-dashed rounded-xl p-16 text-center cursor-pointer transition-all duration-300 ${
              isDragging
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

      </div>
    </div>
  );
}

export default App;
