import streamlit as st
import pickle
import os
import numpy as np
from lime.lime_text import LimeTextExplainer
from text_processor import TextProcessor

# Page configuration
st.set_page_config(
    page_title="Cyberbullying Detection System",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        margin: 1rem auto;
    }
    
    /* Headers */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: fadeInDown 1s ease-in-out;
    }
    
    .sub-header {
        text-align: center;
        color: #555;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
        animation: fadeIn 1.5s ease-in-out;
    }
    
    /* Animated Boxes */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Alert Boxes with Enhanced Styling */
    .warning-box {
        background: linear-gradient(135deg, #FFF9C4 0%, #FFF59D 100%);
        border-left: 6px solid #FFA000;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(255, 160, 0, 0.2);
        animation: slideInUp 0.6s ease-out;
    }
    
    .danger-box {
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
        border-left: 6px solid #E53935;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(229, 57, 53, 0.2);
        animation: slideInUp 0.6s ease-out;
    }
    
    .safe-box {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-left: 6px solid #43A047;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(67, 160, 71, 0.2);
        animation: slideInUp 0.6s ease-out;
    }
    
    .info-box {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        border-left: 6px solid #1E88E5;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(30, 136, 229, 0.2);
        animation: slideInUp 0.6s ease-out;
    }
    
    /* Action Cards */
    .action-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .action-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
        border-color: #667eea;
    }
    
    /* Word Impact Cards */
    .word-impact-positive {
        background: linear-gradient(135deg, #C8E6C9 0%, #A5D6A7 100%);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 12px;
        border-left: 4px solid #43A047;
        box-shadow: 0 4px 8px rgba(67, 160, 71, 0.15);
        transition: all 0.3s ease;
        animation: slideInUp 0.4s ease-out;
    }
    
    .word-impact-positive:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 12px rgba(67, 160, 71, 0.25);
    }
    
    .word-impact-negative {
        background: linear-gradient(135deg, #FFCDD2 0%, #EF9A9A 100%);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 12px;
        border-left: 4px solid #E53935;
        box-shadow: 0 4px 8px rgba(229, 57, 53, 0.15);
        transition: all 0.3s ease;
        animation: slideInUp 0.4s ease-out;
    }
    
    .word-impact-negative:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 12px rgba(229, 57, 53, 0.25);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: white;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* Button Enhancements */
    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
    }
    
    /* Text Area Styling */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
        font-family: 'Poppins', sans-serif;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #e8ecf1 0%, #dce2e9 100%);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        margin-top: 2rem;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-top-color: #667eea;
    }
    
    /* Section Headers */
    h2, h3 {
        color: #333;
        font-weight: 600;
    }
    
    /* Divider Enhancement */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
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
            st.markdown("""
            <div class="action-card">
                <h4 style="color: #E53935;">🚫 Report</h4>
                <p>Report this content to platform moderators</p>
                <p style="background: #FFEBEE; padding: 0.5rem; border-radius: 8px; font-size: 0.9rem; color: #666;">
                    This helps protect the community
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="action-card">
                <h4 style="color: #F57C00;">🔇 Block</h4>
                <p>Block the sender to prevent further contact</p>
                <p style="background: #FFF3E0; padding: 0.5rem; border-radius: 8px; font-size: 0.9rem; color: #666;">
                    Protect yourself from harassment
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="action-card">
                <h4 style="color: #1E88E5;">💬 Contact Support</h4>
                <p>Reach out to support or trusted adults</p>
                <p style="background: #E3F2FD; padding: 0.5rem; border-radius: 8px; font-size: 0.9rem; color: #666;">
                    Get help from professionals
                </p>
            </div>
            """, unsafe_allow_html=True)
        
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
        
        st.markdown("---")
        st.markdown("### 🔒 Privacy")
        st.info("All analysis is done locally. Your text is not stored or shared.")
    
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
                            css_class = "word-impact-negative"
                            direction = "towards Cyberbullying"
                            icon = "⚠️"
                        else:
                            css_class = "word-impact-positive"
                            direction = "towards Safe"
                            icon = "✅"
                        
                        st.markdown(f"""
                        <div class="{css_class}">
                            <span style="font-size: 1.2rem;">{icon}</span> 
                            <strong style="font-size: 1.1rem;">"{word}"</strong> 
                            <span style="float: right; font-weight: 600;">Impact: {abs(weight):.3f}</span>
                            <br>
                            <small style="color: #666;">{direction}</small>
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
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h3 style="color: white; margin-bottom: 1rem;">💙 Stay Safe Online</h3>
        <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">Remember: Online safety is important. Be kind, report harmful content, and seek help when needed.</p>
        <p style="font-size: 0.9rem; opacity: 0.9;">This tool uses AI and may not be 100% accurate. Use your best judgment.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
