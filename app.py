import streamlit as st
import pickle
import os
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
from text_processor import TextProcessor
from train_model import train_models

# Page configuration
st.set_page_config(
    page_title="Cyberbullying Detection System",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .safe-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.nb_model = None
    st.session_state.rf_model = None
    st.session_state.vectorizer = None
    st.session_state.processor = TextProcessor()
    st.session_state.explainer = LimeTextExplainer(class_names=['Safe', 'Cyberbullying'])

def load_models():
    """Load trained models or train new ones if they don't exist"""
    if not os.path.exists('nb_model.pkl') or not os.path.exists('rf_model.pkl'):
        with st.spinner('Training models for the first time... This may take a moment.'):
            from train_model import train_models
            train_models()
    
    with open('nb_model.pkl', 'rb') as f:
        st.session_state.nb_model = pickle.load(f)
    
    with open('rf_model.pkl', 'rb') as f:
        st.session_state.rf_model = pickle.load(f)
    
    with open('vectorizer.pkl', 'rb') as f:
        st.session_state.vectorizer = pickle.load(f)
    
    st.session_state.models_loaded = True

def predict_text(text, model_type='nb'):
    """Predict if text contains cyberbullying"""
    processed_text = st.session_state.processor.preprocess(text)
    vectorized = st.session_state.vectorizer.transform([processed_text])
    
    if model_type == 'nb':
        model = st.session_state.nb_model
    else:
        model = st.session_state.rf_model
    
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    
    return prediction, probability, processed_text

def get_lime_explanation(text, model_type='nb'):
    """Generate LIME explanation for the prediction"""
    processed_text = st.session_state.processor.preprocess(text)
    
    if model_type == 'nb':
        model = st.session_state.nb_model
    else:
        model = st.session_state.rf_model
    
    def predictor(texts):
        processed = [st.session_state.processor.preprocess(t) for t in texts]
        vectorized = st.session_state.vectorizer.transform(processed)
        return model.predict_proba(vectorized)
    
    explanation = st.session_state.explainer.explain_instance(
        text, 
        predictor, 
        num_features=10,
        top_labels=1
    )
    
    return explanation

def display_action_suggestions(is_cyberbullying, confidence):
    """Display action suggestions based on detection results"""
    st.markdown("### 📋 Recommended Actions")
    
    if is_cyberbullying:
        severity = "High" if confidence > 0.8 else "Medium" if confidence > 0.5 else "Low"
        
        st.markdown(f"""
        <div class="danger-box">
            <strong>⚠️ Severity Level: {severity}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🚫 Report")
            st.write("Report this content to platform moderators")
            st.info("This helps protect the community")
        
        with col2:
            st.markdown("#### 🔇 Block")
            st.write("Block the sender to prevent further contact")
            st.info("Protect yourself from harassment")
        
        with col3:
            st.markdown("#### 💬 Contact Support")
            st.write("Reach out to support or trusted adults")
            st.info("Get help from professionals")
        
        st.markdown("---")
        st.markdown("#### Additional Resources")
        st.write("- **Talk to someone you trust** about the situation")
        st.write("- **Save evidence** by taking screenshots")
        st.write("- **Don't respond** to cyberbullying messages")
        st.write("- **Report to authorities** if threats are serious")
        
    else:
        st.markdown("""
        <div class="safe-box">
            <strong>✅ This text appears safe</strong><br>
            No immediate action needed. Continue positive communication!
        </div>
        """, unsafe_allow_html=True)

def train_with_dataset():
    """Interface for uploading and training with custom datasets"""
    st.markdown("## 📚 Upload Training Dataset")
    st.write("Upload a CSV file with your custom training data to improve model accuracy.")
    
    st.markdown("""
    <div class="info-box">
        <strong>📋 Dataset Format Requirements:</strong><br>
        • CSV file with two columns: <code>text</code> and <code>label</code><br>
        • <code>text</code>: The message or comment content (string)<br>
        • <code>label</code>: 0 for safe content, 1 for cyberbullying/hate speech (numeric)<br>
        • <strong>Minimum 4 examples</strong> (at least one of each class)<br>
        • <strong>Both classes required</strong> (need examples of 0 and 1)<br>
        • No missing values allowed<br><br>
        Example: <code>text,label</code><br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<code>"Have a great day!",0</code><br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<code>"You are stupid",1</code>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with 'text' and 'label' columns"
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Validate columns
            if 'text' not in df.columns or 'label' not in df.columns:
                st.error("❌ Error: CSV must have 'text' and 'label' columns!")
                return
            
            # Drop rows with missing values
            original_len = len(df)
            df = df.dropna(subset=['text', 'label'])
            if len(df) < original_len:
                st.warning(f"⚠️ Removed {original_len - len(df)} rows with missing values.")
            
            # Coerce labels to integers (handle string "0"/"1")
            try:
                df['label'] = pd.to_numeric(df['label'], errors='coerce').astype('Int64')
            except:
                st.error("❌ Error: Labels must be numeric (0 or 1)!")
                return
            
            # Drop any rows where label conversion failed
            df = df.dropna(subset=['label'])
            
            # Validate labels are 0 or 1
            if not df['label'].isin([0, 1]).all():
                st.error("❌ Error: Labels must be 0 (safe) or 1 (cyberbullying)!")
                return
            
            # Check minimum sample size
            if len(df) < 4:
                st.error(f"❌ Error: Need at least 4 examples for training. You have {len(df)}.")
                return
            
            # Check for both classes
            unique_labels = df['label'].unique()
            if len(unique_labels) < 2:
                missing_class = "safe (0)" if 0 not in unique_labels else "cyberbullying (1)"
                st.error(f"❌ Error: Dataset must contain both classes. Missing: {missing_class}")
                st.info("💡 Your dataset needs examples of both safe content (label=0) and cyberbullying (label=1).")
                return
            
            # Show preview
            st.success(f"✅ File uploaded successfully! Found {len(df)} examples.")
            
            col1, col2 = st.columns(2)
            with col1:
                safe_count = (df['label'] == 0).sum()
                st.metric("Safe Examples", safe_count)
            with col2:
                bully_count = (df['label'] == 1).sum()
                st.metric("Cyberbullying Examples", bully_count)
            
            with st.expander("📄 Preview Dataset (first 10 rows)"):
                st.dataframe(df.head(10))
            
            # Training options
            st.markdown("### ⚙️ Training Options")
            combine_data = st.checkbox(
                "Combine with default training data",
                value=True,
                help="Include the built-in training examples along with your custom data"
            )
            
            if st.button("🚀 Train Models with This Dataset", type="primary"):
                with st.spinner('Training models with your dataset... This may take a moment.'):
                    try:
                        # Train models
                        nb_model, rf_model, vectorizer, accuracy = train_models(
                            custom_data=df,
                            combine_with_default=combine_data
                        )
                        
                        # Update session state
                        st.session_state.nb_model = nb_model
                        st.session_state.rf_model = rf_model
                        st.session_state.vectorizer = vectorizer
                        st.session_state.models_loaded = True
                        
                        # Display results
                        st.success("✅ Models trained successfully!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Training Samples", accuracy['total_samples'])
                        with col2:
                            st.metric("Naive Bayes Accuracy", f"{accuracy['naive_bayes']*100:.1f}%")
                        with col3:
                            st.metric("Random Forest Accuracy", f"{accuracy['random_forest']*100:.1f}%")
                        
                        st.info("💡 Your models have been updated! Switch to the 'Detect Cyberbullying' tab to test them.")
                        
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    # Download template
    st.markdown("---")
    st.markdown("### 📥 Download Template")
    st.write("Need a template to get started? Download this sample CSV file:")
    
    template_data = {
        'text': [
            'Have a wonderful day',
            'Thank you for your help',
            'You are so stupid',
            'Nobody likes you',
            'Great work on the project',
            'Kill yourself',
            'I appreciate your effort',
            'You are worthless'
        ],
        'label': [0, 0, 1, 1, 0, 1, 0, 1]
    }
    template_df = pd.DataFrame(template_data)
    
    csv = template_df.to_csv(index=False)
    st.download_button(
        label="📄 Download CSV Template",
        data=csv,
        file_name="training_template.csv",
        mime="text/csv"
    )

def main():
    # Header
    st.markdown('<div class="main-header">🛡️ Cyberbullying Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered text analysis to identify and prevent cyberbullying</div>', unsafe_allow_html=True)
    
    # Load models
    if not st.session_state.models_loaded:
        load_models()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ℹ️ About")
        st.write("This system uses machine learning to detect potential cyberbullying and hate speech in text.")
        
        st.markdown("## 🤖 AI Models")
        model_choice = st.radio(
            "Select classifier:",
            ["Naive Bayes", "Random Forest"],
            help="Choose which AI model to use for detection"
        )
        
        st.markdown("## 📊 Features")
        st.write("✓ Text preprocessing")
        st.write("✓ AI-powered detection")
        st.write("✓ LIME explanations")
        st.write("✓ Action suggestions")
        st.write("✓ Custom dataset training")
        
        st.markdown("---")
        st.markdown("### 🔒 Privacy")
        st.info("All analysis is done locally. Your text is not stored or shared.")
    
    # Create tabs
    tab1, tab2 = st.tabs(["🔍 Detect Cyberbullying", "📚 Train with Dataset"])
    
    with tab1:
        # Main content
        st.markdown("## 📝 Enter Text to Analyze")
        
        user_input = st.text_area(
            "Paste or type the text you want to check:",
            height=150,
            placeholder="Example: Type or paste a message, comment, or post here...",
            help="Enter any text to check if it contains cyberbullying or hate speech"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            analyze_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.rerun()
        
        if analyze_button and user_input.strip():
            model_type = 'nb' if model_choice == "Naive Bayes" else 'rf'
            
            with st.spinner('Analyzing text...'):
                # Make prediction
                prediction, probabilities, processed_text = predict_text(user_input, model_type)
                is_cyberbullying = prediction == 1
                confidence = probabilities[1] if is_cyberbullying else probabilities[0]
                
                # Display results
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")
                
                # Result box
                if is_cyberbullying:
                    st.markdown(f"""
                    <div class="danger-box">
                        <h3>⚠️ Cyberbullying Detected</h3>
                        <p><strong>Confidence:</strong> {confidence*100:.1f}%</p>
                        <p>This text may contain cyberbullying or hate speech.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-box">
                        <h3>✅ Text Appears Safe</h3>
                        <p><strong>Confidence:</strong> {confidence*100:.1f}%</p>
                        <p>No cyberbullying or hate speech detected.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence scores
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Safe Probability", f"{probabilities[0]*100:.1f}%")
                with col2:
                    st.metric("Cyberbullying Probability", f"{probabilities[1]*100:.1f}%")
                
                # Show preprocessed text
                with st.expander("🔧 View Preprocessed Text"):
                    st.markdown("**Original Text:**")
                    st.write(user_input)
                    st.markdown("**After Preprocessing:**")
                    st.write(processed_text if processed_text else "(empty after preprocessing)")
                    st.caption("Emojis, URLs, mentions, and special characters were removed for analysis")
                
                # LIME Explanation
                st.markdown("---")
                st.markdown("## 🔍 AI Decision Explanation")
                st.write("The highlighted words below show which parts of the text influenced the AI's decision:")
                
                with st.spinner('Generating explanation...'):
                    explanation = get_lime_explanation(user_input, model_type)
                    
                    # Get explanation as list
                    exp_list = explanation.as_list(label=1)
                    
                    # Display word importance
                    st.markdown("### Key Words and Their Impact")
                    
                    if exp_list:
                        for word, weight in exp_list:
                            if weight > 0:
                                color = "#ffcccc"  # Light red for cyberbullying indicators
                                direction = "towards Cyberbullying"
                                icon = "⚠️"
                            else:
                                color = "#ccffcc"  # Light green for safe indicators
                                direction = "towards Safe"
                                icon = "✅"
                            
                            st.markdown(f"""
                            <div style="background-color: {color}; padding: 0.5rem; margin: 0.3rem 0; border-radius: 5px;">
                                {icon} <strong>"{word}"</strong> - Weight: {abs(weight):.3f} ({direction})
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No significant words found in the analysis.")
                    
                    # Display HTML explanation
                    with st.expander("📄 View Detailed Explanation"):
                        html_exp = explanation.as_html(labels=(1,))
                        st.components.v1.html(html_exp, height=400, scrolling=True)
                
                # Action Suggestions
                st.markdown("---")
                display_action_suggestions(is_cyberbullying, confidence)
        
        elif analyze_button:
            st.warning("⚠️ Please enter some text to analyze.")
    
    with tab2:
        train_with_dataset()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>💙 Remember: Online safety is important. Be kind, report harmful content, and seek help when needed.</p>
        <p style="font-size: 0.8rem;">This tool uses AI and may not be 100% accurate. Use your best judgment.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
