export interface Model {
  name: string;
  window_size: number;
  dataset_name: string;
  class_names: string[];
  description: string;
}

export interface ExampleFile {
  filename: string;
  sample_index: number;
  fault_name: string;
}

export interface SignalStats {
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

export interface FFTData {
  frequencies: number[];
  magnitudes: number[];
}

export interface STFTData {
  file_name: string;
  file_type?: string;
  image_base64: string;
}

export interface ClassificationResult {
  predictions?: any[];
  prediction?: any;
  probabilities?: number[];
  model_name?: string;
}

export type Language = 'fa' | 'en';
export type Dataset = 'PU' | 'CWRU';

