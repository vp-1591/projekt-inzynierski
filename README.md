# Detektor Dezinformacji

A modern, minimalist web application for detecting manipulation techniques and disinformation in articles using AI.

## 🚀 Overview

**Detektor Dezinformacji** is a tool designed to help users identify common propaganda and manipulation techniques in text content. Built with a focus on speed, aesthetics, and clarity, it provides an intuitive interface for analyzing Polish-language content.

### Core Features
- **Manipulation Detection**: Identifies techniques like *Ad Hominem*, *False Dilemma*, *Emotional Language*, and more.
- **Aesthetic UI**: A sleek, dark-mode-first design with smooth transitions and glassmorphism elements.
- **AI-Powered**: Designed to integrate with LLMs (e.g., via Ollama) for real-time semantic analysis.
- **Polish Language Support**: Fully localized interface and detection descriptions.

## 🛠️ Technology Stack
- **Frontend**: React 18
- **Build Tool**: Vite
- **Styling**: Vanilla CSS (Modern CSS variables, Flexbox/Grid)
- **Icons/Assets**: Custom AI-generated minimalist icons

## 📂 Project Structure
```text
├── public/                 # Static assets (favicons, logos)
├── src/
│   ├── components/         # Reusable UI components (InputSection, ResultCard)
│   ├── services/           # Business logic and AI integration (disinformationDetector.js)
│   ├── App.jsx             # Main application layout and state logic
│   ├── main.jsx            # Entry point
│   └── index.css           # Global design system and theme
├── index.html              # Main HTML template
└── vite.config.js          # Vite configuration
```

## 🚦 Getting Started

### Prerequisites
- Node.js (v16+)
- npm or yarn

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd projekt-inzynierski
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

## 🧠 AI Integration (v1/v2)
- **v1 (Current)**: Uses a structured mock service to simulate AI responses with realistic delays and detection patterns.
- **v2 (Planned)**: Integration with a local Ollama instance or a remote API to perform live inference on user-provided text.

## 🎨 Design Principles
- **Clarity**: Results are displayed with clear labels and descriptive tooltips.
- **Minimalism**: No unnecessary distractions; the focus is entirely on the user's content and the analysis results.
- **Responsiveness**: Fully functional on both desktop and mobile devices.

---
*Developed as part of an engineering project.*
