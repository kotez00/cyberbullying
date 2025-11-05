# Cyberbullying Detection System

## Overview

This is a machine learning-powered cyberbullying detection system built with Streamlit. The application analyzes text input to identify potential cyberbullying or hate speech using Natural Language Processing (NLP) and classification algorithms. The system provides real-time predictions with confidence scores and uses LIME (Local Interpretable Model-agnostic Explanations) to explain which words contribute to the classification decision.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application
- **Design Pattern**: Single-page application with custom CSS styling
- **UI Components**: 
  - Text input area for message analysis
  - Color-coded result boxes (safe/warning/danger) based on prediction confidence
  - Interactive explanations using LIME visualizations
- **Styling**: Custom CSS classes for visual feedback (safe-box, warning-box, danger-box)

### Backend Architecture
- **ML Pipeline**: 
  - Text preprocessing using custom TextProcessor class
  - TF-IDF vectorization for feature extraction
  - Classification models (Naive Bayes and/or Random Forest)
  - LIME integration for model explainability
- **Model Storage**: Pickle-based serialization for trained models and vectorizers
- **Processing Flow**:
  1. Raw text input → TextProcessor
  2. Cleaned text → TF-IDF Vectorizer
  3. Feature vector → Classifier
  4. Prediction + confidence → LIME explainer → Results display

### Text Processing Pipeline
- **TextProcessor class** handles multi-stage cleaning:
  - Emoji removal (using emoji library)
  - URL removal (pattern matching)
  - Social media artifacts removal (@mentions, #hashtags)
  - Special character filtering
  - Whitespace normalization
  - Optional stopword removal (NLTK stopwords)
- **Design Decision**: Modular preprocessing allows flexibility in feature engineering and maintains separation of concerns

### Machine Learning Model
- **Training Approach**: Supervised learning with labeled dataset
- **Feature Engineering**: TF-IDF (Term Frequency-Inverse Document Frequency) vectorization converts text to numerical features
- **Classification Algorithm**: MultinomialNB and/or RandomForestClassifier
- **Model Explainability**: LIME provides interpretable predictions by highlighting influential words
- **Training Data**: Embedded sample dataset with binary labels (1=cyberbullying, 0=normal)
- **Rationale**: TF-IDF captures important words while reducing noise; LIME adds transparency to model decisions

### Data Architecture
- **Storage**: File-based pickle serialization (no database)
- **Model Artifacts**: Saved as .pkl files (vectorizer, trained model)
- **Training Data**: Hardcoded in train_model.py for demonstration
- **Design Decision**: Simple file-based storage suitable for demonstration and single-model deployment; would need database for production with multiple models or user data

## External Dependencies

### Python Libraries
- **streamlit**: Web application framework for ML/data science apps
- **scikit-learn**: Machine learning library (TfidfVectorizer, MultinomialNB, RandomForestClassifier, train_test_split)
- **nltk**: Natural Language Toolkit for text processing (stopwords, tokenization)
- **lime**: Model interpretation library (LimeTextExplainer)
- **emoji**: Emoji detection and removal
- **numpy**: Numerical computing support
- **pickle**: Python serialization (standard library)

### NLTK Data
- **stopwords**: English stopword corpus for optional text filtering
- **punkt**: Tokenizer models for sentence splitting
- **Download Strategy**: Automatic quiet download on TextProcessor initialization with fallback to empty set

### No External Services
- No external APIs
- No cloud services
- No database connections
- Fully self-contained application running locally