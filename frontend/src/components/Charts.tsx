import React, { useState } from 'react';
import { Line } from 'react-chartjs-2';
import { Language, FFTData, STFTData } from '../types';
import { translations } from '../constants/translations';
import { STFTResult } from '../STFTResult';

interface ChartsProps {
  darkMode: boolean;
  language: Language;
  signalData: number[];
  fftData: FFTData | null;
  stftData: STFTData | null;
  stftLoading: boolean;
  windowSize: number | '';
  hopLength: number | '';
  onHopLengthChange: (value: number | '') => void;
  onRecalculateSTFT: () => void;
  samplingRate: number;
}

type ModalType = 'timeDomain' | 'fft' | 'stft' | null;

export const Charts: React.FC<ChartsProps> = ({
  darkMode,
  language,
  signalData,
  fftData,
  stftData,
  stftLoading,
  windowSize,
  hopLength,
  onHopLengthChange,
  onRecalculateSTFT,
  samplingRate,
}) => {
  const t = translations[language];
  const [modalOpen, setModalOpen] = useState<ModalType>(null);

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

  const chartOptionsFFT = {
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
        title: {
          display: true,
          text: samplingRate ? (language === 'fa' ? 'فرکانس' : 'Frequency') : (language === 'fa' ? 'فرکانس (Hz)' : 'Frequency (Hz)'),
          color: darkMode ? '#E9E9E9' : '#1F2937',
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

  const chartOptionsTimeDomain = {
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
        title: {
          display: true,
          text: samplingRate ? (language === 'fa' ? 'زمان (نمونه)' : 'Time (Samples)') : (language === 'fa' ? 'زمان (ثانیه)' : 'Time (Seconds)'),
          color: darkMode ? '#E9E9E9' : '#1F2937',
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

  if (signalData.length === 0) {
    return null;
  }

  return (
    <section className={`mb-8 p-6 rounded-lg shadow-lg ${darkMode ? 'bg-gray-800' : 'bg-white'} transition-colors duration-300`}>
      <h2 className={`text-2xl font-bold mb-4 text-center ${darkMode ? 'text-white' : 'text-primary'} font-persian`}>
        {t.charts}
      </h2>

      {/* Time Domain Signal */}
      <div className="mb-16">
        <div className="flex items-center justify-center gap-3 mb-2">
          <h3 className={`text-xl font-semibold ${darkMode ? 'text-gray-200' : 'text-gray-700'} font-persian`}>
            {t.timeDomain}
          </h3>
          <button
            onClick={() => setModalOpen('timeDomain')}
            className={`group flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all duration-200 border-2 ${darkMode
              ? 'border-blue-500/50 bg-blue-500/10 text-blue-400 hover:bg-blue-500/20 hover:border-blue-400 hover:scale-105'
              : 'border-blue-300 bg-blue-50 text-blue-600 hover:bg-blue-100 hover:border-blue-400 hover:scale-105'
              }`}
          >
            <svg className="w-4 h-4 transition-transform group-hover:scale-110" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
            </svg>
            {language === 'fa' ? 'بزرگ‌نمایی' : 'Expand'}
          </button>
        </div>
        <div className="h-64">
          <Line data={timeDomainChartData} options={chartOptionsTimeDomain} />
        </div>
      </div>

      {/* Frequency Domain (FFT) */}
      {fftChartData && (
        <div className="mb-16">
          <div className="flex items-center justify-center gap-3 mb-2">
            <h3 className={`text-xl font-semibold ${darkMode ? 'text-gray-200' : 'text-gray-700'} font-persian`}>
              {t.frequencyDomain}
            </h3>
            <button
              onClick={() => setModalOpen('fft')}
              className={`group flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all duration-200 border-2 ${darkMode
                ? 'border-green-500/50 bg-green-500/10 text-green-400 hover:bg-green-500/20 hover:border-green-400 hover:scale-105'
                : 'border-green-300 bg-green-50 text-green-600 hover:bg-green-100 hover:border-green-400 hover:scale-105'
                }`}
            >
              <svg className="w-4 h-4 transition-transform group-hover:scale-110" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
              </svg>
              {language === 'fa' ? 'بزرگ‌نمایی' : 'Expand'}
            </button>
          </div>
          <div className="h-64">
            <Line data={fftChartData} options={chartOptionsFFT} />
          </div>
        </div>
      )}

      {/* Short-Time Fourier Transform (STFT) */}
      <div className="mb-8">
        <div className="flex items-center justify-center gap-3 mb-4">
          <h3 className={`text-xl font-semibold ${darkMode ? 'text-gray-200' : 'text-gray-700'} font-persian`}>
            {t.stft}
          </h3>
          {stftData && (
            <button
              onClick={() => setModalOpen('stft')}
              className={`group flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all duration-200 border-2 ${darkMode
                ? 'border-purple-500/50 bg-purple-500/10 text-purple-400 hover:bg-purple-500/20 hover:border-purple-400 hover:scale-105'
                : 'border-purple-300 bg-purple-50 text-purple-600 hover:bg-purple-100 hover:border-purple-400 hover:scale-105'
                }`}
            >
              <svg className="w-4 h-4 transition-transform group-hover:scale-110" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
              </svg>
              {language === 'fa' ? 'بزرگ‌نمایی' : 'Expand'}
            </button>
          )}
        </div>
        <STFTResult
          stftData={stftData}
          windowSize={windowSize}
          hopLength={hopLength}
          onHopLengthChange={onHopLengthChange}
          onRecalculate={onRecalculateSTFT}
          isLoading={stftLoading}
          darkMode={darkMode}
          translations={translations}
          language={language}
        />
      </div>

      {/* Modal for Chart Review */}
      {modalOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
          onClick={() => setModalOpen(null)}
        >
          <div
            className={`relative w-[95vw] h-[95vh] p-8 rounded-2xl shadow-2xl ${darkMode ? 'bg-gray-800' : 'bg-white'
              }`}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Close Button */}
            <button
              onClick={() => setModalOpen(null)}
              className={`absolute top-4 left-4 z-10 p-3 rounded-full transition-colors ${darkMode
                ? 'bg-gray-700 hover:bg-gray-600 text-gray-300'
                : 'bg-gray-100 hover:bg-gray-200 text-gray-600'
                }`}
            >
              <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>

            {/* Modal Content */}
            <div className="h-[calc(95vh-120px)]">
              {modalOpen === 'timeDomain' && (
                <Line data={timeDomainChartData} options={{ ...chartOptionsTimeDomain, maintainAspectRatio: false }} />
              )}
              {modalOpen === 'fft' && fftChartData && (
                <Line data={fftChartData} options={{ ...chartOptionsFFT, maintainAspectRatio: false }} />
              )}
              {modalOpen === 'stft' && stftData && (
                <div className="h-full flex items-center justify-center">
                  <div className="relative flex items-center gap-4">

                    <div className="flex items-center">
                      {/* STFT Image */}
                      <img
                        src={`data:image/png;base64,${stftData.image_base64}`}
                        alt="STFT"
                        className="max-w-[80vw] object-contain"
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </section>
  );
};

