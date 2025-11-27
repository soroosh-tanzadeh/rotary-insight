# Rotary Insight - Frontend

A modern, responsive web application for bearing fault diagnosis and signal analysis. Built with React and TypeScript, this application provides comprehensive signal processing capabilities with real-time visualization and analysis tools.

## 🚀 Features

### Core Functionality
- **Signal Analysis**: Upload CSV files containing signal data for comprehensive analysis
- **FFT Processing**: Fast Fourier Transform computation for frequency domain analysis
- **STFT Analysis**: Short-Time Fourier Transform with configurable hop length
- **Multiple Model Support**: Load and select from various analysis models (CWRU & PU datasets)
- **Window-based Processing**: Configurable window size for signal segmentation
- **Classification Results**: Machine learning-based fault classification with confidence scores

### Visualization
- **Time Domain Chart**: Visualize raw signal data over time with expand/zoom modal
- **Frequency Domain Chart**: Display FFT results showing frequency components
- **STFT Spectrogram**: Time-frequency representation with color-coded magnitude
- **Interactive Modals**: Full-screen chart viewing with detailed axis labels

### User Experience
- **Bilingual Support**: Full support for Persian (Farsi) and English languages
- **RTL Layout**: Right-to-left layout support for Persian language
- **Dark/Light Mode**: Toggle between dark and light themes with smooth transitions
- **Responsive Design**: Optimized for desktop and mobile devices
- **Modern UI**: Clean, intuitive interface built with Tailwind CSS
- **Sample Files**: Pre-loaded sample files for quick testing
- **Interactive Components**: Accordion-style sample selection, animated buttons

### Security & Configuration
- **API Authentication**: Secure API key-based authentication
- **Configurable API Endpoint**: Customizable backend API URL
- **Session Persistence**: User preferences and authentication state saved in localStorage

## 📋 Prerequisites

Before you begin, ensure you have the following installed:
- **Node.js**: Version 16.x or higher
- **npm**: Version 8.x or higher (comes with Node.js)
- **Backend API**: A running backend API server (see backend documentation)

## 🛠️ Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd rotary-insight/frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

## 🚀 Getting Started

### Development Mode

1. **Start the development server**:
   ```bash
   npm start
   ```

2. **Open your browser**:
   Navigate to `http://localhost:3000` (or the port shown in the terminal)

3. **Configure API Connection**:
   - Enter your backend API URL (default: `http://localhost:8000`)
   - Enter your API Key
   - Click "Login" to authenticate

### Production Build

1. **Build the application**:
   ```bash
   npm run build
   ```

2. **Serve the build**:
   The `build` folder contains the optimized production build. You can serve it using any static file server:
   ```bash
   # Using serve (install globally: npm install -g serve)
   serve -s build
   
   # Or using Python
   python -m http.server 3000 -d build
   
   # Or using Node.js http-server
   npx http-server build -p 3000
   ```

## 📖 Usage Guide

### Initial Setup

1. **Login Page**:
   - Enter your backend API URL
   - Enter your API Key
   - Click "Login" to authenticate and access the main application

### Main Application

1. **Select Model**:
   - Choose an analysis model from the Configuration section
   - Filter models by window size
   - Models are grouped by dataset (CWRU, PU)

2. **Upload or Select Sample File**:
   - Drag and drop your CSV file into the upload area, or click to browse
   - Or expand the "Sample Files" accordion to select from pre-loaded examples
   - Samples are sorted by index for easy navigation

3. **Calculate**:
   - Click the "Calculate" button to start the analysis
   - Intelligent validation shows specific error messages if requirements are not met
   - The application will process the signal and generate visualizations

### Viewing Results

1. **File Information Panel**:
   - View selected file, model, and window size with color-coded status indicators

2. **Charts**:
   - **Time Domain Signal**: View the raw signal with "Expand" button for full-screen modal
   - **Frequency Domain (FFT)**: Analyze frequency components
   - **STFT Spectrogram**: Time-frequency analysis with adjustable hop length

3. **Classification Results**:
   - View predicted fault class with confidence percentages
   - Probability distribution across all classes

4. **Navigation**:
   - Use the back arrow button to return to the main page
   - Switch between languages using the language toggle buttons
   - Toggle dark/light mode using the theme button

## 🏗️ Project Structure

```
frontend/
├── public/
│   ├── index.html          # HTML template
│   ├── logo.png            # Application logo
│   ├── favicon.png         # Favicon
│   └── manifest.json       # PWA manifest
├── src/
│   ├── components/         # Reusable UI components
│   │   ├── index.ts        # Component exports
│   │   ├── Header.tsx      # App header with theme/language toggles
│   │   ├── FileUpload.tsx  # CSV file upload with drag & drop
│   │   ├── ModelSelector.tsx # Model selection with filtering
│   │   ├── SampleFiles.tsx # Accordion-style sample file selector
│   │   ├── Charts.tsx      # Signal visualization charts
│   │   ├── ClassificationResults.tsx # ML classification display
│   │   ├── FileInfo.tsx    # File/model status display
│   │   ├── ErrorMessage.tsx # Error message display
│   │   └── LoadingModal.tsx # Loading spinner overlay
│   ├── constants/
│   │   └── translations.ts # i18n translations (FA/EN)
│   ├── types/
│   │   └── index.ts        # TypeScript interfaces
│   ├── App.tsx             # Main application component
│   ├── App.css             # Application styles
│   ├── STFTResult.tsx      # STFT visualization component
│   ├── index.tsx           # Application entry point
│   └── index.css           # Global styles and Tailwind imports
├── package.json            # Dependencies and scripts
├── tailwind.config.js      # Tailwind CSS configuration
├── tsconfig.json           # TypeScript configuration
└── README.md               # This file
```

## 🛠️ Technologies Used

### Core Framework
- **React 19.2.0**: UI library
- **TypeScript 4.9.5**: Type-safe JavaScript
- **React Scripts 5.0.1**: Build tooling

### UI & Styling
- **Tailwind CSS 3.4.18**: Utility-first CSS framework
- **PostCSS**: CSS processing
- **Autoprefixer**: CSS vendor prefixing

### Data Visualization
- **Chart.js 4.5.1**: Charting library
- **react-chartjs-2 5.3.1**: React wrapper for Chart.js

### Data Processing
- **PapaParse 5.5.3**: CSV parsing library

### Testing
- **@testing-library/react**: React testing utilities
- **@testing-library/jest-dom**: DOM testing matchers
- **@testing-library/user-event**: User interaction simulation

## 📜 Available Scripts

### `npm start`
Runs the app in development mode. Open [http://localhost:3000](http://localhost:3000) to view it in the browser. The page will reload if you make edits.

### `npm run build`
Builds the app for production to the `build` folder. It correctly bundles React in production mode and optimizes the build for the best performance.

### `npm test`
Launches the test runner in interactive watch mode.

### `npm run eject`
**Note: This is a one-way operation. Once you `eject`, you can't go back!**

If you aren't satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

## 🌐 Internationalization

The application supports two languages:
- **Persian (Farsi)**: Right-to-left (RTL) layout
- **English**: Left-to-right (LTR) layout

Language can be switched using the language toggle buttons in the header. The application automatically adjusts:
- Text direction (RTL/LTR)
- Font family (Vazir for Persian, Inter for English)
- All UI text and labels

## 🎨 Theming

The application supports both dark and light themes:
- **Dark Mode**: Dark background with light text
- **Light Mode**: Light background with dark text

Theme preference is saved in localStorage and persists across sessions. Smooth transitions are applied when switching themes.

## 🔐 API Integration

The frontend communicates with a backend API for:
- Model loading (`GET /models/`)
- Example files (`GET /examples/{model_name}`)
- FFT computation (`POST /fft/`)
- STFT computation (`POST /stft/`)
- Classification (`POST /classify/`)

All API requests require an `X-API-Key` header for authentication.

## 📝 Environment Variables

Currently, the application uses hardcoded default API URL (`http://localhost:8000`). You can modify this in the code or add environment variable support:

Create a `.env` file in the `frontend` directory:
```
REACT_APP_API_URL=http://localhost:8000
```

Then use it in your code:
```typescript
const defaultApiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
```

## 🐛 Troubleshooting

### Common Issues

1. **API Connection Failed**:
   - Verify your backend API is running
   - Check the API URL is correct
   - Ensure your API Key is valid

2. **Models Not Loading**:
   - Check your API authentication
   - Verify the `/models/` endpoint is accessible
   - Check browser console for error messages

3. **CSV Parsing Errors**:
   - Ensure your CSV file contains only numeric data
   - Check file encoding (should be UTF-8)
   - Verify the file is a valid CSV format

4. **Build Errors**:
   - Clear `node_modules` and reinstall: `rm -rf node_modules && npm install`
   - Clear build cache: `rm -rf build`
   - Check Node.js version compatibility

## 📄 License

This project is part of the Rotary Insight application. Please refer to the main project repository for license information.

## 👥 Contributing

When contributing to this project:
1. Follow the existing code style
2. Ensure TypeScript types are properly defined
3. Test your changes thoroughly
4. Update documentation as needed