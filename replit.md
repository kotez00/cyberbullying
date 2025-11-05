# Cyberbullying Detection System

## Overview
An AI-powered web application built with Streamlit that detects cyberbullying and hate speech in text using machine learning. The system provides transparent explanations of its decisions using LIME (Local Interpretable Model-Agnostic Explanations) and offers actionable recommendations for responding to detected content.

## Current Features

### 1. Text Analysis & Detection
- **Text Input Interface**: User-friendly text area for submitting content to analyze
- **Preprocessing Pipeline**: Automatically removes emojis, URLs, @mentions, #hashtags, and special characters
- **Dual Classifier System**: 
  - Naive Bayes classifier
  - Random Forest classifier (100 estimators)
  - Users can select which model to use via sidebar

### 2. AI Decision Transparency
- **LIME Explanations**: Highlights top 10 words/phrases that influenced the prediction
- **Visual Word Weights**: Color-coded display showing which words push toward "safe" (green) or "cyberbullying" (red)
- **Confidence Scores**: Displays probability percentages for both classifications
- **Preprocessed Text View**: Shows original vs. cleaned text for transparency

### 3. Action Suggestions
- **Severity Levels**: High (>80%), Medium (50-80%), Low (<50%)
- **Recommended Actions**:
  - Report to platform moderators
  - Block the sender
  - Contact support or trusted adults
- **Resources**: Additional guidance for handling cyberbullying situations

### 4. Custom Dataset Training
- **Upload Interface**: Accepts CSV files with training data
- **Format Requirements**: 
  - Two columns: `text` (message content) and `label` (0 for safe, 1 for cyberbullying)
  - Validates file format and label values
- **Training Options**:
  - Combine with default training data (recommended)
  - Use only custom data
- **Real-time Accuracy**: Displays model accuracy after training
- **Template Download**: Provides sample CSV for users to get started

### 5. UI/UX Design
- **Tabbed Interface**:
  - Tab 1: Detect Cyberbullying (main analysis interface)
  - Tab 2: Train with Dataset (custom dataset upload)
- **Color-Coded Results**:
  - Green boxes for safe content
  - Red boxes for detected cyberbullying
  - Yellow/blue boxes for warnings and information
- **Sidebar Navigation**: Model selection and feature information
- **Privacy Notice**: Emphasizes local processing and data privacy

## Technical Architecture

### Core Files
- **app.py**: Main Streamlit application with tabbed interface
- **text_processor.py**: TextProcessor class for preprocessing
- **train_model.py**: Model training script with custom dataset support

### Models & Data
- **nb_model.pkl**: Trained Naive Bayes classifier
- **rf_model.pkl**: Trained Random Forest classifier  
- **vectorizer.pkl**: TF-IDF vectorizer (1000 max features, 1-2 ngrams)

### Dependencies
- streamlit: Web interface
- scikit-learn: ML models and vectorization
- lime: Model explanations
- nltk: Natural language processing
- pandas: Data handling
- numpy: Numerical operations
- emoji: Emoji detection and removal

## Training Data
Default dataset includes 60 examples (30 cyberbullying, 30 safe). Users can add custom data through the web interface.

## Recent Changes (November 5, 2025)
- Added custom dataset upload functionality
- Implemented tabbed interface for better organization
- Added template CSV download feature
- Enhanced training options (combine with default or use custom only)
- Display model accuracy metrics after training
- Improved session state management for model retraining

## Configuration
- **Port**: 5000 (configured for webview)
- **Workflow**: "Cyberbullying Detector"
- **Command**: `streamlit run app.py --server.port 5000`

## Privacy & Safety
- All analysis performed locally
- Text not stored or transmitted
- Educational tool with ~80-90% accuracy on test data
- Encourages human judgment alongside AI predictions

## Future Enhancements (Planned)
- Multi-language support for detecting cyberbullying in different languages
- Severity levels (mild, moderate, severe) for detected cyberbullying
- Batch analysis capability for checking multiple texts at once
- Data export functionality for analysis results and reports
- User feedback mechanism to improve classifier accuracy over time
