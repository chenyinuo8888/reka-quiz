# Reka Quiz - Hackathon Project

A hackathon project based on the Reka AI "Roast My Life" template, adapted for quiz functionality using Reka's Vision API.

## 🎯 Project Overview

This project demonstrates how to use the **Reka Vision API** to create an interactive quiz application that can analyze videos and generate questions or commentary. Built on the foundation of the original "Roast My Life" template, this version is adapted for educational and quiz purposes.

## ✨ Features

- 🔍 Dynamic video list fetched from Reka Vision backend 
- 🤖 Reka Vision chat endpoint integration for quiz generation
- 🧪 Clean, documented Python code (type hints + docstrings)
- 🐳 Docker support for fast containerized runs
- 📱 Responsive UI with a lightweight custom palette
- 🎓 Quiz-focused functionality (adapted from roasting template)

## 🏗️ Project Structure

```
.
├── src/                  # Application source code
│   ├── app.py           # Main Flask application
│   ├── templates/       # HTML templates
│   │   ├── index.html  # Home page
│   │   └── form.html   # Video selection form page
│   └── static/         # Static files
│       ├── css/
│       │   └── style.css    # Stylesheets
│       └── images/     # Image assets
├── workshop/           # Workshop version with additional features
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
└── .env-sample         # Environment variables template
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12 or higher
- pip (Python package manager)
- Reka AI API key ([Get free API key](https://link.reka.ai/free))

### Installation & Setup

1. **Clone this repository**
   ```bash
   git clone <your-repo-url>
   cd reka-quiz
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env-sample .env
   # Edit .env with your API key
   ```

5. **Run the application**
   ```bash
   python src/app.py
   ```

6. **Open your browser**
   Navigate to: `http://localhost:8111`

### Docker Setup

1. **Build the Docker image**
   ```bash
   docker build -t reka-quiz .
   ```

2. **Run with environment variables**
   ```bash
   docker run --env-file .env -p 8111:8111 reka-quiz
   ```

## 🔧 Environment Variables

Create a `.env` file using the `.env-sample` template:

```env
# Primary API key (get free key at https://link.reka.ai/free)
API_KEY=your_api_key_here

# Reka Vision API endpoint
BASE_URL=https://vision-agent.api.reka.ai
```

## 🎮 Usage

1. Open the app and navigate to the Quiz page
2. Select a video from the available list
3. Click "Generate Quiz" to create questions based on the video content
4. Enjoy your AI-generated quiz experience!

## 🛠️ Development

This project is based on the [Reka AI roast_my_life template](https://github.com/reka-ai/api-examples-python/tree/main/roast_my_life) and has been adapted for quiz functionality.

### Key Adaptations

- Modified prompts to generate quiz questions instead of roasts
- Updated UI to reflect quiz functionality
- Enhanced error handling for educational content
- Added quiz-specific features

## 📚 Resources

- [Reka Vision API Documentation](https://docs.reka.ai/vision)
- [Get Free API Key](https://link.reka.ai/free)
- [Original Template Repository](https://github.com/reka-ai/api-examples-python)

## 🤝 Contributing

This is a hackathon project. Feel free to fork, modify, and extend for your own quiz applications!

## 📄 License

Educational / sample use. Adapt freely.

---

**Built with ❤️ for the hackathon using Reka AI's powerful Vision API**