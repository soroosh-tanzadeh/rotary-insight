/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: '#385F8C',
        'primary-light': '#4A7AB0',
        'primary-dark': '#2A4A6B',
        secondary: '#EBEDF0',
        'secondary-dark': '#1F2937',
      },
      fontFamily: {
        'persian': ['Vazir', 'Tahoma', 'sans-serif'],
        'english': ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
      transitionProperty: {
        'colors': 'color, background-color, border-color, text-decoration-color, fill, stroke',
      },
    },
  },
  plugins: [],
}

