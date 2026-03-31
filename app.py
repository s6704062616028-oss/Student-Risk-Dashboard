import streamlit as st
import pandas as pd
import joblib


# PAGE CONFIG

st.set_page_config(
    page_title="Student Risk Dashboard",
    page_icon="🎓",
    layout="wide"
)


# CUSTOM CSS

st.markdown("""
    <style>
        .stApp { background-color: #ffffff; color: #000000; }
        [data-testid="stSidebar"] { background-color: #111111; }
        [data-testid="stSidebar"] * { color: #ffffff !important; }
        h1, h2, h3, h4 { color: #000000 !important; font-family: 'Segoe UI', sans-serif; }
        label { color: #222222 !important; font-weight: 600; }
        input, select, textarea {
            background-color: #f5f5f5 !important;
            color: #000000 !important;
            border: 1px solid #999999 !important;
            border-radius: 6px !important;
        }
        [data-baseweb="select"] {
            background-color: #f5f5f5 !important;
            border: 1px solid #999999 !important;
            border-radius: 6px !important;
        }
        .stButton > button {
            background-color: #000000 !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.6rem 2rem !important;
            font-size: 1rem !important;
            font-weight: bold !important;
        }
        .stButton > button:hover { background-color: #333333 !important; }
        hr { border: 1px solid #cccccc; }
        .result-box {
            padding: 1.2rem 1.5rem;
            border-radius: 10px;
            font-size: 1.3rem;
            font-weight: bold;
            text-align: center;
            margin-top: 1rem;
        }
        .risk-high { background-color: #111111; color: #ffffff; border: 2px solid #000000; }
        .risk-low  { background-color: #eeeeee; color: #000000; border: 2px solid #aaaaaa; }
        [data-testid="stNumberInput"] input { background-color: #f5f5f5 !important; }
    </style>
""", unsafe_allow_html=True)


# SIDEBAR

page = st.sidebar.selectbox("📄 เลือกหน้า", [
    "🎓 Dropout Prediction",
    "🧠 Depression Prediction",
    "📘 Methodology"
])


# PAGE 1: Dropout Prediction

if page == "🎓 Dropout Prediction":

    st.title("🎓 Student Dropout Risk Prediction")
    st.markdown("---")
    st.header("กรอกข้อมูลนักศึกษา")
    st.markdown(" ")

    model         = joblib.load("model.pkl")
    encoders      = joblib.load("encoders.pkl")
    scaler        = joblib.load("scaler.pkl")
    pca           = joblib.load("pca.pkl")
    feature_names = joblib.load("dropout_features.pkl")

    user_input = {}
    mid        = len(feature_names) // 2
    col1, col2 = st.columns(2, gap="large")

    with col1:
        for f in feature_names[:mid]:
            if f in encoders:
                user_input[f] = st.selectbox(f, encoders[f].classes_, key=f"l_{f}")
            else:
                user_input[f] = st.number_input(f, value=0.0, key=f"l_{f}")

    with col2:
        for f in feature_names[mid:]:
            if f in encoders:
                user_input[f] = st.selectbox(f, encoders[f].classes_, key=f"r_{f}")
            else:
                user_input[f] = st.number_input(f, value=0.0, key=f"r_{f}")

    st.markdown("---")
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        predict = st.button("🔮 Predict Dropout")

    if predict:
        df_in = pd.DataFrame([user_input])
        df_in = df_in[feature_names]

        for col in encoders:
            if col in df_in.columns:
                df_in[col] = encoders[col].transform(df_in[col])

        X_scaled = scaler.transform(df_in)
        X_pca    = pca.transform(X_scaled)
        pred     = model.predict(X_pca)[0]

        if pred == 1:
            st.markdown('<div class="result-box risk-high">⚠️ มีความเสี่ยง Dropout สูง</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box risk-low">✅ ความเสี่ยงต่ำ</div>', unsafe_allow_html=True)


# PAGE 2: Depression Prediction (MLP)

elif page == "🧠 Depression Prediction":

    st.title("🧠 Student Depression Prediction")
    st.markdown("---")
    st.header("กรอกข้อมูลนักศึกษา")
    st.markdown(" ")

    dep_model    = joblib.load("depression_nn.pkl")
    dep_encoders = joblib.load("depression_encoders.pkl")
    dep_scaler   = joblib.load("depression_scaler.pkl")
    dep_features = joblib.load("depression_features.pkl")

    user_input = {}
    mid        = len(dep_features) // 2
    col1, col2 = st.columns(2, gap="large")

    with col1:
        for f in dep_features[:mid]:
            if f in dep_encoders:
                user_input[f] = st.selectbox(f, dep_encoders[f].classes_, key=f"dep_l_{f}")
            else:
                user_input[f] = st.number_input(f, value=0.0, key=f"dep_l_{f}")

    with col2:
        for f in dep_features[mid:]:
            if f in dep_encoders:
                user_input[f] = st.selectbox(f, dep_encoders[f].classes_, key=f"dep_r_{f}")
            else:
                user_input[f] = st.number_input(f, value=0.0, key=f"dep_r_{f}")

    st.markdown("---")
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        predict2 = st.button("🔮 Predict Depression")

    if predict2:
        df_in2 = pd.DataFrame([user_input])
        df_in2 = df_in2[dep_features]

        for col in dep_encoders:
            if col in df_in2.columns:
                df_in2[col] = dep_encoders[col].transform(df_in2[col])

        X_scaled2 = dep_scaler.transform(df_in2)
        prob      = dep_model.predict_proba(X_scaled2)[0][1]
        pred2     = dep_model.predict(X_scaled2)[0]

        if prob > 0.8:
            st.markdown('<div class="result-box risk-high">🔴 เสี่ยงสูงมาก — ควรพบผู้เชี่ยวชาญ</div>', unsafe_allow_html=True)
        elif prob > 0.5:
            st.markdown('<div class="result-box" style="background:#555;color:#fff;border:2px solid #333;">🟠 เสี่ยงปานกลาง</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box risk-low">✅ ไม่พบภาวะซึมเศร้า</div>', unsafe_allow_html=True)

        st.markdown(f"<p style='text-align:center;color:#555;margin-top:0.5rem'>Probability: {prob:.2%}</p>", unsafe_allow_html=True)


# PAGE 3: Methodology

elif page == "📘 Methodology":

    st.title("📘 Methodology & Model Development")
    st.markdown("---")

    st.header("1. การเตรียมข้อมูล (Data Preparation)")
    st.markdown("""
**1.1 แหล่งที่มาของข้อมูล**

🎓 **Student Mental Health & Burnout Dataset**  
ใช้สำหรับทำนายความเสี่ยงการ Dropout โดยใช้ Random Forest Classifier ร่วมกับ PCA

🧠 **Depression Student Dataset**  
ใช้สำหรับทำนายภาวะซึมเศร้า โดยใช้ Neural Network (MLPClassifier)

**1.2 การจัดการข้อมูลสูญหาย**  
- ตัวแปรเชิงตัวเลข → เติมด้วยค่า Median  
- ตัวแปรเชิงหมวดหมู่ → เติมด้วยค่า Mode  

**1.3 การแปลงข้อมูล**  
ใช้ Label Encoding แปลงตัวแปรหมวดหมู่ให้เป็นตัวเลข

**1.4 การแก้ปัญหาข้อมูลไม่สมดุลด้วย SMOTE**  
ใช้ SMOTE สร้างข้อมูล Synthetic ของกลุ่มน้อยเพิ่มขึ้น เพื่อให้โมเดลเรียนรู้ได้อย่างสมดุล
""")

    st.header("2. ทฤษฎีของอัลกอริทึม (Algorithm Theory)")
    st.markdown("""
**2.1 Random Forest Classifier (Dropout Model)**  
สร้าง Decision Tree จำนวน 300 ต้น โดยใช้ Bootstrap Sampling และ Majority Voting  
- n_estimators = 300, max_depth = 12, random_state = 42

**2.2 Neural Network — MLPClassifier (Depression Model)**  
โครงข่ายประสาทเทียมแบบ Multi-layer Perceptron  
- hidden_layer_sizes = (64, 32), activation = ReLU, max_iter = 300
""")

    st.header("3. ขั้นตอนการพัฒนาโมเดล (Pipeline)")
    st.markdown("""
Raw Data → Clean → Encode → Split (80/20) → SMOTE → Scale → (PCA) → Train → Evaluate → Save
""")

    st.header("4. แหล่งอ้างอิง (References)")
    st.markdown("""
- **Kaggle** — Student Mental Health & Burnout Dataset, Depression Student Dataset  
- **ChatGPT** — อธิบาย SMOTE, Random Forest, Neural Network, Pipeline  
- **Claude** — ช่วยพัฒนา Streamlit UI และแก้ไขโค้ด  
- **Scikit-learn** — scikit-learn.org  
- **Streamlit** — docs.streamlit.io  
- **Imbalanced-learn** — imbalanced-learn.org  
""")
