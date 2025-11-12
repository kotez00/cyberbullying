import streamlit as st
import pickle
import os
import threading
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
from text_processor import TextProcessor
from train_model import train_models
import logging
import traceback
import json
import hashlib
    # Also write to stderr so Streamlit Cloud captures it in logs
# Install the global exception hook

# --- Health check: respond before any heavy import or logic ---
import os
import streamlit as st
if os.environ.get("STREAMLIT_HEALTH_CHECK") == "1":
    st.write("ok")
    st.stop()

import pickle
import threading
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
from text_processor import TextProcessor
from train_model import train_models
import logging
import traceback
import sys

# Configure basic error logging to a file so startup errors are captured in deployment logs
logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s %(levelname)s %(message)s')


# Set Streamlit page config (must be at the top, after health check)
st.set_page_config(
    page_title="Cyberbullying Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Custom CSS for better styling
st.markdown("""
<style>
    /* Import a modern system font with a web fallback */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
    }

    /* Page container adjustments */
    .reportview-container .main .block-container {
        max-width: 1150px;
        padding-top: 1.5rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }

    /* Header */
    .main-header {
            font-size: 2.4rem;
            font-weight: 800;
            color: #ffffff;
            text-align: left;
            margin-bottom: 0.25rem;
            letter-spacing: -0.4px;
            text-shadow: 0 2px 12px rgba(2,6,23,0.6);
    }
    .sub-header {
            text-align: left;
            color: #cbd5e1;
            margin-bottom: 1.25rem;
            font-size: 1rem;
            text-shadow: 0 1px 6px rgba(2,6,23,0.45);
    }

    /* Sidebar styling */
    .css-1d391kg { /* sidebar container (may vary by Streamlit version) */
        background-color: #f8fafc !important;
        padding-top: 1rem !important;
    }

    /* Cards and result boxes */
    .danger-box, .safe-box, .warning-box, .info-box {
        padding: 1rem 1rem;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        margin: 0.6rem 0;
    }
        .danger-box { background: linear-gradient(90deg, rgba(255,236,236,1), rgba(255,245,245,1)); border-left: 4px solid #ef4444; }
        .safe-box { background: linear-gradient(90deg, rgba(240,253,244,1), rgba(247,255,250,1)); border-left: 4px solid #10b981; }
        .warning-box { background: linear-gradient(90deg, rgba(255,250,235,1), rgba(255,253,240,1)); border-left: 4px solid #f59e0b; }
        .info-box { background: linear-gradient(90deg, rgba(240,249,255,1), rgba(247,251,255,1)); border-left: 4px solid #06b6d4; }

        /* Ensure readable text color on light cards (avoid very-light/white text being unreadable)
             Streamlit dark themes may leave global text light; enforce dark text inside light cards */
            .danger-box, .safe-box, .warning-box, .info-box {
                color: #0f172a !important;
            }
            /* Brighten card headings for emphasis */
            .danger-box h3, .safe-box h3, .warning-box h3, .info-box h3 {
                color: #052f2f !important;
                font-size: 1.25rem;
            }
        .danger-box h3, .safe-box h3, .warning-box h3, .info-box h3 {
            color: #0f172a !important;
            font-weight: 700;
        }
        .danger-box p, .safe-box p, .warning-box p, .info-box p,
        .danger-box span, .safe-box span, .warning-box span, .info-box span {
            color: #0f172a !important;
        }

        /* Make code-like chips inside cards readable */
        .danger-box code, .safe-box code, .info-box code, .warning-box code {
            background: rgba(15, 23, 42, 0.06) !important;
            color: #065f46 !important;
            padding: 0.12rem 0.4rem !important;
            border-radius: 6px !important;
        }

        /* Improve expander content contrast */
        .streamlit-expanderHeader, .stExpander {
            color: #e2e8f0 !important;
        }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(180deg,#0ea5a9,#0284c7) !important;
        color: white !important;
        border: none !important;
        padding: 0.55rem 1rem !important;
        border-radius: 10px !important;
        box-shadow: 0 6px 18px rgba(2,6,23,0.08) !important;
        font-weight: 600 !important;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
    }

    /* Text area styling */
    textarea {
        border-radius: 8px !important;
        padding: 0.75rem !important;
    }

    /* Metrics and small UI tweaks */
    .stMetric { padding: 0.6rem; }
    .css-1v0mbdj.e16nr0p30 { /* adjust expander header size */
        font-size: 0.95rem;
    }

    /* Smaller helper text */
    .caption { color: #64748b; }
</style>
""", unsafe_allow_html=True)




# Always reset user on app start unless authenticated
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'page' not in st.session_state:
    st.session_state.page = 'Login'
if 'user' not in st.session_state or not st.session_state.authenticated:
    st.session_state.user = None
# --- BACKEND SERVICE CLASSES ---

class AuthService:
    USERS_FILE = 'users.json'

    @staticmethod
    def hash_password(password):
        return hashlib.sha256(password.encode()).hexdigest()

    @staticmethod
    def load_users():
        if not os.path.exists(AuthService.USERS_FILE):
            return {}
        with open(AuthService.USERS_FILE, 'r') as f:
            return json.load(f)

    @staticmethod
    def save_users(users):
        with open(AuthService.USERS_FILE, 'w') as f:
            json.dump(users, f)

    @staticmethod
    def register(username, password):
        username = username.strip().lower()
        users = AuthService.load_users()
        st.write(f"[DEBUG] Registering. Users loaded: {users}")
        if username in users:
            return False, "Username already exists."
        users[username] = AuthService.hash_password(password)
        AuthService.save_users(users)
        st.write(f"[DEBUG] Registered {username}. Users now: {users}")
        return True, "Account created successfully. Please log in."

    @staticmethod
    def login(username, password):
        username = username.strip().lower()
        users = AuthService.load_users()
        st.write(f"[DEBUG] Login. Users loaded: {users}")
        st.write(f"[DEBUG] Login attempt: {username} / {AuthService.hash_password(password)}")
        if username in users and users[username] == AuthService.hash_password(password):
            return True
        return False

    @staticmethod
    def logout():
        st.session_state.authenticated = False
        st.session_state.page = 'Login'

class ModelService:
    @staticmethod
    def initialize_state():
        if 'models_loaded' not in st.session_state:
            st.session_state.models_loaded = False
            st.session_state.nb_model = None
            st.session_state.rf_model = None
            st.session_state.vectorizer = None
            st.session_state.processor = None
            st.session_state.explainer = None
            st.session_state.training_in_progress = False

    @staticmethod
    def load_models():
        """Load trained models or train new ones if they don't exist"""
        try:
            if os.path.exists('nb_model.pkl') and os.path.exists('rf_model.pkl') and os.path.exists('vectorizer.pkl'):
                with open('nb_model.pkl', 'rb') as f:
                    st.session_state.nb_model = pickle.load(f)
                with open('rf_model.pkl', 'rb') as f:
                    st.session_state.rf_model = pickle.load(f)
                with open('vectorizer.pkl', 'rb') as f:
                    st.session_state.vectorizer = pickle.load(f)
                if st.session_state.processor is None:
                    st.session_state.processor = TextProcessor()
                if st.session_state.explainer is None:
                    st.session_state.explainer = LimeTextExplainer(class_names=['Safe', 'Cyberbullying'])
                st.session_state.models_loaded = True
                st.session_state.training_in_progress = False
                return
            # Handle background training and flag file as before
            if os.path.exists('training_done.flag'):
                with open('nb_model.pkl', 'rb') as f:
                    st.session_state.nb_model = pickle.load(f)
                with open('rf_model.pkl', 'rb') as f:
                    st.session_state.rf_model = pickle.load(f)
                with open('vectorizer.pkl', 'rb') as f:
                    st.session_state.vectorizer = pickle.load(f)
                if st.session_state.processor is None:
                    st.session_state.processor = TextProcessor()
                if st.session_state.explainer is None:
                    st.session_state.explainer = LimeTextExplainer(class_names=['Safe', 'Cyberbullying'])
                st.session_state.models_loaded = True
                st.session_state.training_in_progress = False
                try:
                    os.remove('training_done.flag')
                except Exception:
                    pass
                return
            if not st.session_state.training_in_progress:
                st.session_state.training_in_progress = True
                def _train_only():
                    try:
                        train_models()
                    except Exception as e:
                        try:
                            with open('training_error.log', 'w', encoding='utf-8') as f:
                                import traceback as _tb
                                f.write(str(e) + '\n')
                                _tb.print_exc(file=f)
                        except Exception:
                            pass
                    finally:
                        try:
                            with open('training_done.flag', 'w') as f:
                                f.write('done')
                        except Exception:
                            pass
                thread = threading.Thread(target=_train_only, daemon=True)
                thread.start()
        except Exception as e:
            logging.error(f"Model loading error: {e}")
            st.session_state.models_loaded = False
            st.session_state.training_in_progress = False

    @staticmethod
    def predict_text(text, model_type='nb'):
        try:
            processed_text = st.session_state.processor.preprocess(text)
            vectorized = st.session_state.vectorizer.transform([processed_text])
            model = st.session_state.nb_model if model_type == 'nb' else st.session_state.rf_model
            prediction = model.predict(vectorized)[0]
            probability = model.predict_proba(vectorized)[0]
            return prediction, probability, processed_text
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return None, None, None

    @staticmethod
    def get_lime_explanation(text, model_type='nb'):
        try:
            processed_text = st.session_state.processor.preprocess(text)
            model = st.session_state.nb_model if model_type == 'nb' else st.session_state.rf_model
            def predictor(texts):
                processed = [st.session_state.processor.preprocess(t) for t in texts]
                vectorized = st.session_state.vectorizer.transform(processed)
                return model.predict_proba(vectorized)
            explanation = st.session_state.explainer.explain_instance(
                text,
                predictor,
                num_features=6,
                top_labels=1,
                num_samples=500
            )
            return explanation
        except Exception as e:
            logging.error(f"LIME explanation error: {e}")
            return None

# --- Initialize backend state ---
ModelService.initialize_state()


def show_login():
    st.write("")
    st.write("")
    st.write("")
    st.markdown("<h2 style='text-align:center; color:#0ea5e9;'>🔒 Login to CyberGuardAI</h2>", unsafe_allow_html=True)
    st.write("")
    st.subheader("Login", divider="rainbow")
    username = st.text_input("Username", key="login_username", placeholder="Enter your username")
    password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
    st.write("")
    col1, col2 = st.columns(2)
    login_btn = col1.button("Login", use_container_width=True, key="login_btn")
    register_btn = col2.button("Create Account", use_container_width=True, key="register_btn")
    if login_btn:
        if not username.strip() or not password:
            st.error("Username and password required.")
        elif AuthService.login(username, password):
            st.session_state.authenticated = True
            st.session_state.page = 'Home'
            st.session_state.user = username.strip().lower()
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password.")
    if register_btn:
        st.session_state.page = 'Register'
        st.rerun()

def show_register():
    st.write("")
    st.write("")
    st.write("")
    st.markdown("<h2 style='text-align:center; color:#0ea5e9;'>📝 Create Your Account</h2>", unsafe_allow_html=True)
    st.write("")
    st.subheader("Register", divider="rainbow")
    username = st.text_input("New Username", key="register_username", placeholder="Choose a username")
    password = st.text_input("New Password", type="password", key="register_password", placeholder="Choose a password")
    confirm = st.text_input("Confirm Password", type="password", key="register_confirm", placeholder="Confirm your password")
    st.write("")
    col1, col2 = st.columns(2)
    register_btn2 = col1.button("Register", use_container_width=True, key="register_btn2")
    back_to_login_btn = col2.button("Back to Login", use_container_width=True, key="back_to_login_btn")
    if register_btn2:
        if not username or not password:
            st.error("Username and password required.")
        elif password != confirm:
            st.error("Passwords do not match.")
        else:
            success, msg = AuthService.register(username, password)
            if success:
                st.success(msg)
                st.session_state.page = 'Login'
                st.rerun()
            else:
                st.error(msg)
    if back_to_login_btn:
        st.session_state.page = 'Login'
        st.rerun()

def show_logout():
    st.sidebar.write("")
    if st.sidebar.button("Logout", key="logout_btn"):
        st.session_state.authenticated = False
        st.session_state.page = 'Login'
        st.session_state.user = None
        # Clear login form fields
        st.session_state["login_username"] = ""
        st.session_state["login_password"] = ""
        st.rerun()

def show_home():
    st.markdown("""
        <div style='display: flex; align-items: center; justify-content: space-between;'>
            <h1 style='margin-bottom: 0; color: #0ea5e9;'>🛡️ CyberGuardAI Dashboard</h1>
            <span style='font-size: 1.1rem; color: #64748b;'>Welcome, <b>{user}</b></span>
        </div>
        <hr style='margin-top: 0.5rem; margin-bottom: 1.5rem;'>
    """.format(user=st.session_state.get('user', 'User')), unsafe_allow_html=True)

    st.markdown("""
        <div style='margin-bottom: 1.5rem;'>
            <span style='font-size: 1.15rem; color: #334155;'>
                Use the form below to analyze text for cyberbullying or hate speech.<br>
                Results and explanations will appear below.
            </span>
        </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.subheader("Text Analysis")
        text = st.text_area("Paste or type the text you want to check:", height=100)
        col1, col2 = st.columns([1,1])
        with col1:
            analyze_btn = st.button("🔍 Analyze Text", use_container_width=True)
        with col2:
            clear_btn = st.button("🧹 Clear", use_container_width=True)

    if clear_btn:
        st.experimental_rerun()

    if analyze_btn and text.strip():
        ModelService.load_models()
        if not st.session_state.models_loaded:
            st.warning("Model is loading or training. Please wait and try again.")
            return
        pred, prob, processed = ModelService.predict_text(text)
        if pred is None:
            st.error("Prediction failed. Please try again.")
            return
        label = "Cyberbullying" if pred == 1 else "Safe"
        color = "#ef4444" if pred == 1 else "#10b981"
        st.markdown(f"""
            <div style='background: #f8fafc; border-left: 6px solid {color}; padding: 1.2rem 1rem; border-radius: 10px; margin-top: 1.5rem;'>
                <h3 style='margin-bottom: 0.5rem; color: {color};'>
                    {'🚨' if pred == 1 else '✅'} Text Appears <b>{label}</b>
                </h3>
                <b>Confidence:</b> {prob[pred]*100:.1f}%<br>
                <span style='color: #334155;'>
                    {'Potential cyberbullying or hate speech detected.' if pred == 1 else 'No cyberbullying or hate speech detected.'}
                </span>
            </div>
        """, unsafe_allow_html=True)

        with st.expander("Show LIME Explanation", expanded=False):
            explanation = ModelService.get_lime_explanation(text)
            if explanation:
                st.markdown("<b>Top features influencing prediction:</b>", unsafe_allow_html=True)
                try:
                    for word, weight in explanation.as_list(label=pred):
                        st.markdown(f"<span style='color: #0ea5e9;'>{word}</span>: <span style='color: #64748b;'>{weight:.3f}</span>", unsafe_allow_html=True)
                except Exception as e:
                    st.info(f"No explanation available for this prediction. ({e})")
            else:
                st.info("No explanation available.")

# --- Sidebar Navigation ---
if st.session_state.authenticated:
    page = st.sidebar.radio("Navigation", ["Home", "Logout"], index=0 if st.session_state.page=="Home" else 1)
    st.session_state.page = page
    if page == "Logout":
        show_logout()
        st.stop()
else:
    # Allow switching between Login and Register
    if st.session_state.page not in ["Login", "Register"]:
        st.session_state.page = "Login"

# --- Routing Logic ---
if not st.session_state.authenticated:
    if st.session_state.page == "Register":
        show_register()
    else:
        show_login()
    st.stop()
elif st.session_state.page == "Home":
    show_home()

def load_models():
    """Load trained models or train new ones if they don't exist"""
    # If models already exist on disk, load them immediately (fast).
    if os.path.exists('nb_model.pkl') and os.path.exists('rf_model.pkl') and os.path.exists('vectorizer.pkl'):
        try:
            with open('nb_model.pkl', 'rb') as f:
                st.session_state.nb_model = pickle.load(f)
            with open('rf_model.pkl', 'rb') as f:
                st.session_state.rf_model = pickle.load(f)
            with open('vectorizer.pkl', 'rb') as f:
                st.session_state.vectorizer = pickle.load(f)
            # Ensure processor and explainer exist
            if st.session_state.processor is None:
                st.session_state.processor = TextProcessor()
            if st.session_state.explainer is None:
                st.session_state.explainer = LimeTextExplainer(class_names=['Safe', 'Cyberbullying'])
            st.session_state.models_loaded = True
            st.session_state.training_in_progress = False
        except Exception as e:
            st.error(f"Failed to load models: {e}")
            st.session_state.models_loaded = False
            st.session_state.training_in_progress = False
        return

    # If model files are missing, start training in a background thread so Streamlit can finish startup quickly.
    # IMPORTANT: background threads must not modify `st.session_state` directly (Streamlit session state is not thread-safe).
    # We'll start training in a background thread which will write a flag file when done; the main thread will detect the flag
    # and then load the pickles into session state.
    if os.path.exists('training_done.flag'):
        # Training finished previously in background; load results now
        try:
            with open('nb_model.pkl', 'rb') as f:
                st.session_state.nb_model = pickle.load(f)
            with open('rf_model.pkl', 'rb') as f:
                st.session_state.rf_model = pickle.load(f)
            with open('vectorizer.pkl', 'rb') as f:
                st.session_state.vectorizer = pickle.load(f)
            if st.session_state.processor is None:
                st.session_state.processor = TextProcessor()
            if st.session_state.explainer is None:
                st.session_state.explainer = LimeTextExplainer(class_names=['Safe', 'Cyberbullying'])
            st.session_state.models_loaded = True
            st.session_state.training_in_progress = False
        except Exception as e:
            st.error(f"Failed to load models after background training: {e}")
            st.session_state.models_loaded = False
            st.session_state.training_in_progress = False
        finally:
            try:
                os.remove('training_done.flag')
            except Exception:
                pass
        return

    if not st.session_state.training_in_progress:
        st.session_state.training_in_progress = True

        def _train_only():
            try:
                train_models()
            except Exception as e:
                # Persist the traceback to a file for diagnostics
                try:
                    with open('training_error.log', 'w', encoding='utf-8') as f:
                        import traceback as _tb
                        f.write(str(e) + '\n')
                        _tb.print_exc(file=f)
                except Exception:
                    pass
            finally:
                # signal completion by creating a flag file; main thread will pick this up
                try:
                    with open('training_done.flag', 'w') as f:
                        f.write('done')
                except Exception:
                    pass

        thread = threading.Thread(target=_train_only, daemon=True)
        thread.start()

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

    # Use fewer samples by default to improve responsiveness.
    # If users need more precise explanations they can increase this value.
    explanation = st.session_state.explainer.explain_instance(
        text,
        predictor,
        num_features=6,
        top_labels=1,
        num_samples=500
    )

    return explanation

def display_action_suggestions(is_cyberbullying, confidence):
    """Display action suggestions based on detection results"""
    st.markdown("### Recommended Actions")
    
    if is_cyberbullying:
        severity = "High" if confidence > 0.8 else "Medium" if confidence > 0.5 else "Low"
        
        st.markdown(f"""
        <div class="danger-box">
            <strong> Severity Level: {severity}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("####  Report")
            st.write("Report this content to platform moderators")
            st.info("This helps protect the community")
        
        with col2:
            st.markdown("####  Block")
            st.write("Block the sender to prevent further contact")
            st.info("Protect yourself from harassment")
        
        with col3:
            st.markdown("####  Contact Support")
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
            <strong> This text appears safe</strong><br>
            No immediate action needed. Continue positive communication!
        </div>
        """, unsafe_allow_html=True)

def train_with_dataset():
    """Interface for uploading and training with custom datasets"""
    st.markdown("##  Upload Training Dataset")
    st.write("Upload a CSV file with your custom training data to improve model accuracy.")
    
    st.markdown("""
    <div class="info-box">
        <strong> Dataset Format Requirements:</strong><br>
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
                st.error(" Error: CSV must have 'text' and 'label' columns!")
                return
            
            # Drop rows with missing values
            original_len = len(df)
            df = df.dropna(subset=['text', 'label'])
            if len(df) < original_len:
                st.warning(f"Removed {original_len - len(df)} rows with missing values.")
            
            # Coerce labels to integers (handle string "0"/"1")
            try:
                df['label'] = pd.to_numeric(df['label'], errors='coerce').astype('Int64')
            except:
                st.error(" Error: Labels must be numeric (0 or 1)!")
                return
            
            # Drop any rows where label conversion failed
            df = df.dropna(subset=['label'])
            
            # Validate labels are 0 or 1
            if not df['label'].isin([0, 1]).all():
                st.error(" Error: Labels must be 0 (safe) or 1 (cyberbullying)!")
                return
            
            # Check minimum sample size
            if len(df) < 4:
                st.error(f" Error: Need at least 4 examples for training. You have {len(df)}.")
                return
            
            # Check for both classes
            unique_labels = df['label'].unique()
            if len(unique_labels) < 2:
                missing_class = "safe (0)" if 0 not in unique_labels else "cyberbullying (1)"
                st.error(f" Error: Dataset must contain both classes. Missing: {missing_class}")
                st.info("Your dataset needs examples of both safe content (label=0) and cyberbullying (label=1).")
                return
            
            # Show preview
            st.success(f" File uploaded successfully! Found {len(df)} examples.")
            
            col1, col2 = st.columns(2)
            with col1:
                safe_count = (df['label'] == 0).sum()
                st.metric("Safe Examples", safe_count)
            with col2:
                bully_count = (df['label'] == 1).sum()
                st.metric("Cyberbullying Examples", bully_count)
            
            with st.expander(" Preview Dataset (first 10 rows)"):
                st.dataframe(df.head(10))
            
            # Training options
            st.markdown("###  Training Options")
            combine_data = st.checkbox(
                "Combine with default training data",
                value=True,
                help="Include the built-in training examples along with your custom data"
            )
            
            if st.button(" Train Models with This Dataset", type="primary"):
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
                        st.success(" Models trained successfully!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Training Samples", accuracy['total_samples'])
                        with col2:
                            st.metric("Naive Bayes Accuracy", f"{accuracy['naive_bayes']*100:.1f}%")
                        with col3:
                            st.metric("Random Forest Accuracy", f"{accuracy['random_forest']*100:.1f}%")
                        
                        st.info(" Your models have been updated! Switch to the 'Detect Cyberbullying' tab to test them.")
                        
                    except Exception as e:
                        st.error(f" Training failed: {str(e)}")
        
        except Exception as e:
            st.error(f" Error reading file: {str(e)}")
    
    # Download template
    st.markdown("---")
    st.markdown("###  Download Template")
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
        label=" Download CSV Template",
        data=csv,
        file_name="training_template.csv",
        mime="text/csv"
    )

def main():
    # Header
    st.markdown('<div class="main-header"> Cyberbullying Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered text analysis to identify and prevent cyberbullying</div>', unsafe_allow_html=True)
    # Show background training status if models are being prepared
    if st.session_state.get('training_in_progress', False):
        st.info(" Models are being prepared in the background. Some features (analysis/explanations) may be unavailable until training completes.")
    
    # Load models
    if not st.session_state.models_loaded:
        load_models()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ℹ About")
        st.write("This system uses machine learning to detect potential cyberbullying and hate speech in text.")
        
        st.markdown("##  AI Models")
        model_choice = st.radio(
            "Select classifier:",
            ["Naive Bayes", "Random Forest"],
            help="Choose which AI model to use for detection"
        )
        
        st.markdown("##  Features")
        st.write("✓ Text preprocessing")
        st.write("✓ AI-powered detection")
        st.write("✓ LIME explanations")
        st.write("✓ Action suggestions")
        st.write("✓ Custom dataset training")
        
        st.markdown("---")
        st.markdown("###  Privacy")
        st.info("All analysis is done locally. Your text is not stored or shared.")
    
    # Create tabs
    tab1, tab2 = st.tabs([" Detect Cyberbullying", " Train with Dataset"])
    
    with tab1:
        # Main content
        st.markdown("##  Enter Text to Analyze")
        
        user_input = st.text_area(
            "Paste or type the text you want to check:",
            height=150,
            placeholder="Example: Type or paste a message, comment, or post here...",
            help="Enter any text to check if it contains cyberbullying or hate speech"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            analyze_button = st.button(" Analyze Text", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button(" Clear", use_container_width=True)
        
        if clear_button:
            st.rerun()
        
        if analyze_button and user_input.strip():
            # Ensure models are loaded before attempting prediction
            if not st.session_state.models_loaded:
                # Trigger background loading/training if not already running
                load_models()
                if st.session_state.training_in_progress:
                    st.warning("Models are being prepared. Training is running in the background — try again in a moment.")
                else:
                    st.error("Models are not available right now. Please try again shortly or check the logs.")
            else:
                model_type = 'nb' if model_choice == "Naive Bayes" else 'rf'
                with st.spinner('Analyzing text...'):
                    # Make prediction
                    prediction, probabilities, processed_text = predict_text(user_input, model_type)
                is_cyberbullying = prediction == 1
                confidence = probabilities[1] if is_cyberbullying else probabilities[0]
                
                # Display results
                st.markdown("---")
                st.markdown("##  Analysis Results")
                
                # Result box
                if is_cyberbullying:
                    st.markdown(f"""
                    <div class="danger-box">
                        <h3> Cyberbullying Detected</h3>
                        <p><strong>Confidence:</strong> {confidence*100:.1f}%</p>
                        <p>This text may contain cyberbullying or hate speech.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-box">
                        <h3> Text Appears Safe</h3>
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
                with st.expander(" View Preprocessed Text"):
                    st.markdown("**Original Text:**")
                    st.write(user_input)
                    st.markdown("**After Preprocessing:**")
                    st.write(processed_text if processed_text else "(empty after preprocessing)")
                    st.caption("Emojis, URLs, mentions, and special characters were removed for analysis")
                
                # LIME Explanation
                st.markdown("---")
                st.markdown("##  AI Decision Explanation")
                st.write("The highlighted words below show which parts of the text influenced the AI's decision:")
                
                # Explanations can be slow. Allow the user to opt-in to generating them.
                show_explanation = st.checkbox("Generate AI explanation (may be slow)", value=False)

                if show_explanation:
                    with st.spinner('Generating explanation...'):
                        explanation = get_lime_explanation(user_input, model_type)

                        # Determine which label the explanation contains (LIME may only compute the top label)
                        # Prefer the model prediction if available, otherwise fall back to the first available label
                        try:
                            available_labels = list(explanation.local_exp.keys()) if hasattr(explanation, 'local_exp') else []
                        except Exception:
                            available_labels = []

                        if not available_labels and hasattr(explanation, 'available_labels'):
                            try:
                                available_labels = explanation.available_labels()
                            except Exception:
                                available_labels = []

                        if not available_labels:
                            st.info("No explanation available for this input.")
                            exp_list = []
                            label_to_use = None
                        else:
                            label_to_use = prediction if prediction in available_labels else available_labels[0]
                            # Get explanation as list for the selected label
                            exp_list = explanation.as_list(label=label_to_use)
                else:
                    exp_list = []
                    label_to_use = None
                    
                    # Display word importance
                    st.markdown("### Key Words and Their Impact")
                    
                    if exp_list:
                        for word, weight in exp_list:
                            if weight > 0:
                                color = "#ffcccc"  # Light red for cyberbullying indicators
                                direction = "towards Cyberbullying"
                                icon = ""
                            else:
                                color = "#ccffcc"  # Light green for safe indicators
                                direction = "towards Safe"
                                icon = ""
                            
                            st.markdown(f"""
                            <div style="background-color: {color}; padding: 0.5rem; margin: 0.3rem 0; border-radius: 5px;">
                                {icon} <strong>"{word}"</strong> - Weight: {abs(weight):.3f} ({direction})
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No significant words found in the analysis.")
                    
                    # Display HTML explanation
                    with st.expander(" View Detailed Explanation"):
                        if label_to_use is None:
                            st.write("No detailed explanation available.")
                        else:
                            html_exp = explanation.as_html(labels=(label_to_use,))
                            st.components.v1.html(html_exp, height=400, scrolling=True)
                
                # Action Suggestions
                st.markdown("---")
                display_action_suggestions(is_cyberbullying, confidence)
        
        elif analyze_button:
            st.warning(" Please enter some text to analyze.")
    
    with tab2:
        train_with_dataset()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p> Remember: Online safety is important. Be kind, report harmful content, and seek help when needed.</p>
        <p style="font-size: 0.8rem;">This tool uses AI and may not be 100% accurate. Use your best judgment.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
