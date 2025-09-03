# Marketing Campaign ROI Predictor

## Overview

This is a Streamlit-based web application that predicts the Return on Investment (ROI) for marketing campaigns using machine learning. The application loads a pre-trained scikit-learn model to make predictions based on campaign parameters and provides actionable recommendations for campaign optimization. It features an intuitive interface for inputting campaign data and displays predictions with color-coded ROI categories and strategic recommendations.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for rapid web application development
- **Layout**: Wide layout configuration for better data visualization
- **UI Components**: Form inputs, metrics display, and interactive elements for campaign parameter entry
- **Responsive Design**: Streamlit's built-in responsive components for cross-device compatibility

### Backend Architecture
- **Model Loading**: Joblib-based model persistence and loading system
- **Prediction Engine**: Scikit-learn compatible model inference
- **Business Logic**: ROI categorization system with three tiers (High >1.0, Medium 0.3-1.0, Low <0.3)
- **Recommendation System**: Rule-based recommendation engine providing specific actions based on ROI performance

### Data Processing
- **Input Handling**: Pandas DataFrames for structured data manipulation
- **Model Interface**: Type-hinted functions for robust data flow
- **Error Handling**: Comprehensive exception handling for model loading and prediction failures

### Application Structure
- **Single-file Architecture**: Monolithic structure in app.py for simplicity
- **Functional Design**: Pure functions for predictions, categorizations, and recommendations
- **State Management**: Streamlit's built-in session state for user interactions

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for rapid prototyping and deployment
- **Pandas**: Data manipulation and analysis library
- **Joblib**: Model serialization and persistence
- **Scikit-learn**: Machine learning model compatibility (implied by joblib usage)

### Model Requirements
- **Pre-trained Model**: Requires 'campaign_model.joblib' file containing a trained scikit-learn model
- **Model Format**: Joblib-serialized machine learning model compatible with the prediction interface

### Runtime Dependencies
- **Python Environment**: Python 3.x runtime with scientific computing stack
- **File System**: Local file system access for model loading
- **Web Browser**: Modern web browser for Streamlit interface interaction