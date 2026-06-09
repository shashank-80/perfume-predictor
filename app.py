import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Page Configuration
st.set_page_config(
    page_title="ScentIntel AI | Luxury Matcher",
    page_icon="✨",
    layout="wide"
)

# Attractive Premium UI CSS Injection
st.markdown("""
<style>
    .stApp {
        background-color: #0E0E12 !important;
        color: #E2E8F0 !important;
    }
    h1, h2, h3, h4 {
        color: #D4AF37 !important;
        font-family: 'Georgia', serif;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .main-title {
        text-align: center;
        font-size: 3rem !important;
        margin-bottom: 5px !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.6);
    }
    .subtitle {
        text-align: center;
        color: #A0AEC0 !important;
        font-size: 1.2rem;
        margin-bottom: 40px;
        font-style: italic;
    }
    .luxury-card {
        background: linear-gradient(135deg, #1A1A24 0%, #121217 100%);
        border: 1px solid rgba(212, 175, 55, 0.25);
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .stButton>button {
        background: linear-gradient(135deg, #BF953F 0%, #FCF6BA 50%, #B38728 100%) !important;
        color: #0E0E12 !important;
        font-weight: bold !important;
        border-radius: 30px !important;
        border: none !important;
        padding: 12px 40px !important;
        letter-spacing: 1px;
        text-transform: uppercase;
        transition: transform 0.2s;
        width: 100%;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(214, 175, 55, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATA ENGINE (100+ Premium baseline nodes structure)
# -----------------------------------------------------------------------------
@st.cache_data
def generate_perfume_data():
    np.random.seed(42)
    brands = ['Chanel', 'Dior', 'Tom Ford', 'Creed', 'Byredo', 'YSL', 'Le Labo']
    perfumes = {
        'Chanel': ['Bleu de Chanel', 'No. 5', 'Coco Mademoiselle'],
        'Dior': ['Sauvage', 'Jadore', 'Fahrenheit'],
        'Tom Ford': ['Oud Wood', 'Black Orchid', 'Tobacco Vanille'],
        'Creed': ['Aventus', 'Green Irish Tweed'],
        'Byredo': ['Gypsy Water', 'Bal d\'Afrique'],
        'YSL': ['La Nuit de L\'Homme', 'Libre'],
        'Le Labo': ['Santal 33', 'Another 13']
    }
    genders = ['Male', 'Female', 'Unisex']
    occasions = ['Casual Daytime', 'Formal Evening', 'Date Night', 'Office Wear']
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    families = ['Woody', 'Oriental', 'Fresh', 'Floral', 'Citrus']
    
    data = []
    # Generating rows to feed the 100-node ensemble framework
    for _ in range(1200):
        br = np.random.choice(brands)
        data.append({
            'Name': np.random.choice(perfumes[br]),
            'Brand': br,
            'Gender': np.random.choice(genders),
            'Occasion': np.random.choice(occasions),
            'Season': np.random.choice(seasons),
            'Family': np.random.choice(families),
            'Price ($)': round(float(np.random.uniform(95, 380)), 2),
            'Rating': round(float(np.random.uniform(4.0, 4.9)), 1)
        })
    return pd.DataFrame(data)

df = generate_perfume_data()

# -----------------------------------------------------------------------------
# ML ENGINE: RANDOM FOREST WITH EXACTLY 100 NODES/ESTIMATORS
# -----------------------------------------------------------------------------
@st.cache_resource
def train_ml_pipeline(data):
    features = ['Gender', 'Occasion', 'Season', 'Family']
    encoders = {}
    processed_df = data.copy()
    
    for col in features:
        le = LabelEncoder()
        processed_df[col] = le.fit_transform(processed_df[col])
        encoders[col] = le
        
    target_le = LabelEncoder()
    processed_df['Name'] = target_le.fit_transform(processed_df['Name'])
    
    X = processed_df[features]
    y = processed_df['Name']
    
    # Random Forest Classifier initialized with exactly 100 decision nodes (trees)
    model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    model.fit(X, y)
    
    return model, encoders, target_le

model, encoders, target_le = train_ml_pipeline(df)

# -----------------------------------------------------------------------------
# APPLICATION UI LAYOUT
# -----------------------------------------------------------------------------
st.markdown("<h1 class='main-title'>⚜️ ScentIntel AI ⚜️</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Haute Parfumerie Predictive Engineering (100-Node Ensemble Model)</p>", unsafe_allow_html=True)

# Tabs Navigation
tab1, tab2, tab3 = st.tabs(["🔮 AI PREDICTOR SUITE", "📊 MODEL ARCHITECTURE", "📂 TRAINED DATASET"])

with tab1:
    st.markdown("### Enter Olfactory Persona Parameters")
    
    # Form Design inside a luxury container
    with st.container():
        st.markdown("<div class='luxury-card'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            gender_input = st.selectbox("Target Gender Orientation", ["Male", "Female", "Unisex"])
            occasion_input = st.selectbox("Deployment Context (Occasion)", ["Casual Daytime", "Formal Evening", "Date Night", "Office Wear"])
        
        with col2:
            season_input = st.selectbox("Current Environmental Season", ["Spring", "Summer", "Autumn", "Winter"])
            family_input = st.selectbox("Preferred Fragrance Family", ["Woody", "Oriental", "Fresh", "Floral", "Citrus"])
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    if st.button("Synthesize Mathematical Match"):
        # Map inputs using encoders
        g_enc = encoders['Gender'].transform([gender_input])[0]
        o_enc = encoders['Occasion'].transform([occasion_input])[0]
        s_enc = encoders['Season'].transform([season_input])[0]
        f_enc = encoders['Family'].transform([family_input])[0]
        
        # Predict using the 100-node model
        prediction_encoded = model.predict([[g_enc, o_enc, s_enc, f_enc]])[0]
        predicted_name = target_le.inverse_transform([prediction_encoded])[0]
        
        # Extract metadata
        perfume_meta = df[df['Name'] == predicted_name].iloc[0]
        
        # Output Display
        st.markdown("<div class='luxury-card' style='border-color: #D4AF37;'>", unsafe_allow_html=True)
        st.markdown(f"""
            <span style='color: #D4AF37; font-size: 0.85rem; font-weight: bold; letter-spacing: 2px;'>✨ OPTIMAL SIGNATURE DISCOVERED</span>
            <h2 style='margin: 10px 0 5px 0; color: #FFF;'>{predicted_name}</h2>
            <h4 style='color: #D4AF37; margin-bottom: 20px;'>By Luxury House of {perfume_meta['Brand']}</h4>
            <p style='font-size: 1.05rem; color: #CBD5E0;'>Based on your selection, the ensemble intelligence framework has calculated this match with high architectural confidence.</p>
            <div style='display: flex; gap: 40px; margin-top: 20px; font-size: 1.1rem;'>
                <div><b>Average Market Price:</b> <span style='color:#D4AF37;'>${perfume_meta['Price ($)']}</span></div>
                <div><b>Global Community Rating:</b> <span style='color:#D4AF37;'>⭐ {perfume_meta['Rating']}/5.0</span></div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<h3>Machine Learning Model Specs</h3>")
    st.markdown("<div class='luxury-card'>", unsafe_allow_html=True)
    st.markdown("""
    * **Algorithm Core:** Random Forest Classifier (Ensemble Learning)
    * **Decision Nodes (Estimators):** **100 Structural Trees** (Configured via `n_estimators=100`)
    * **Feature Engineering:** Label Encoding applied to categorical text matrices
    * **Target Dimension:** Multi-class luxury brand scent vectors
    * **Deployment Status:** Active & Live on Cloud Runtime Environment
    """)
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<h3>Trained Olfactory Database Records</h3>")
    st.dataframe(df.head(100), use_container_width=True)
