import streamlit as st
import pandas as pd
import joblib

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Student Risk Dashboard",
    page_icon="🎓",
    layout="wide"
)

# ==========================================
# CUSTOM CSS (Black & White Theme)
# ==========================================
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

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
page = st.sidebar.selectbox("📄 เลือกหน้า", [
    "🎓 Dropout Prediction",
    "🧠 Depression Prediction (NN)",
    "📘 Methodology"
])

# ==========================================
# PAGE 1: Dropout Prediction
# ==========================================
if page == "🎓 Dropout Prediction":

    model    = joblib.load("model.pkl")
    encoders = joblib.load("encoders.pkl")
    scaler   = joblib.load("scaler.pkl")
    pca      = joblib.load("pca.pkl")
    feature_names = joblib.load("all_features.pkl")

    st.title("🎓 Student Dropout Risk Prediction")
    st.markdown("---")
    st.header("กรอกข้อมูลนักศึกษา")
    st.markdown(" ")

    user_input = {}
    mid = len(feature_names) // 2
    col1, col2 = st.columns(2)

    with col1:
       for feat in feature_names[:mid]:
           if feat in encoders:
               user_input[feat] = st.selectbox(feat, encoders[feat].classes_, key=f"l_{feat}")
           else:
               user_input[feat] = st.number_input(feat, value=0.0, key=f"l_{feat}")

    with col2:
        for feat in feature_names[mid:]:
           if feat in encoders:
               user_input[feat] = st.selectbox(feat, encoders[feat].classes_, key=f"r_{feat}")
           else:
               user_input[feat] = st.number_input(feat, value=0.0, key=f"r_{feat}")

    st.markdown("---")
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        predict = st.button("🔮 Predict")

    if predict:
        df_in = pd.DataFrame([user_input])
        for col in encoders:
            if col in df_in.columns:
                df_in[col] = encoders[col].transform(df_in[col])

        df_in = pd.DataFrame([user_input])

        # encode
        for col in encoders:
           df_in[col] = encoders[col].transform(df_in[col])

        # scale
        X_scaled = scaler.transform(df_in)

        # PCA
        X_pca = pca.transform(X_scaled)

        # predict
        y_pred = model.predict(X_pca)[0]
        if y_pred == 1:
            st.markdown('<div class="result-box risk-high">⚠️ มีความเสี่ยง Dropout สูง</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box risk-low">✅ ความเสี่ยงต่ำ</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 2: Depression Prediction
# ==========================================
elif page == "🧠 Depression Prediction (NN)":

    model = joblib.load("depression_nn.pkl")
    encoders = joblib.load("encoders.pkl")
    scaler = joblib.load("scaler.pkl")

    features = joblib.load("depression_features.pkl")

    st.title("🧠 Depression Prediction (Neural Network)")

    user_input = {}
    mid = len(features) // 2
    col1, col2 = st.columns(2)

    with col1:
       for f in features[:mid]:
            if f in encoders:
                user_input[f] = st.selectbox(f, encoders[f].classes_, key=f"l_{f}")
            else:
                user_input[f] = st.number_input(f, value=0.0, key=f"l_{f}")

    with col2:
      for f in features[mid:]:
            if f in encoders:
                user_input[f] = st.selectbox(f, encoders[f].classes_, key=f"r_{f}")
            else:
                user_input[f] = st.number_input(f, value=0.0, key=f"r_{f}")

    if st.button("🔮 Predict Depression"):

        df = pd.DataFrame([user_input])

        # encode
        for col in encoders:
            df[col] = encoders[col].transform(df[col])

        # scale
        X = scaler.transform(df)

        # predict
        prob = model.predict_proba(X)[0][1]
        pred = model.predict(X)[0]

        st.write(f"🧠 Probability: {prob:.2f}")

        # 🔥 แสดงระดับความเสี่ยง
        if prob > 0.8:
            st.error("🔴 เสี่ยงสูงมาก")
        elif prob > 0.5:
            st.warning("🟠 เสี่ยงปานกลาง")
        else:
            st.success("🟢 ปกติ")
elif page == "📘 Methodology":

    st.title("📘 Methodology & Model Development")
    st.markdown("---")

    st.header("1. การเตรียมข้อมูล (Data Preparation)")
    st.markdown("""**1.1 แหล่งที่มาของข้อมูล**  
โครงการนี้ใช้ชุดข้อมูล 2 ชุด และใช้โมเดล Machine Learning ที่แตกต่างกันดังนี้

🎓 **Student Mental Health & Burnout Dataset**  
ใช้สำหรับทำนายความเสี่ยงการ Dropout โดยใช้ **Random Forest Classifier** ร่วมกับเทคนิค **PCA (Principal Component Analysis)** เพื่อลดมิติของข้อมูล และช่วยเพิ่มประสิทธิภาพของโมเดล

🧠 **Depression Student Dataset**  
ใช้สำหรับทำนายภาวะซึมเศร้า โดยใช้ **Neural Network (MLPClassifier)** ซึ่งเป็นโมเดล Deep Learning แบบหลายชั้น (Multi-layer Perceptron)

โดยทั้งสองโมเดลถูกนำไปใช้งานจริงใน Streamlit Application ผ่านขั้นตอน:
- Label Encoding สำหรับแปลงข้อมูลหมวดหมู่
- Feature Scaling ด้วย StandardScaler
- Dimensionality Reduction ด้วย PCA (เฉพาะ Dropout Model)
- การทำนายผลด้วย Machine Learning Model

การออกแบบลักษณะนี้ช่วยให้ระบบสามารถเรียนรู้ทั้งรูปแบบเชิงเส้นและไม่เชิงเส้นได้อย่างมีประสิทธิภาพ
""")

    st.header("2. ทฤษฎีของอัลกอริทึมที่พัฒนา (Algorithm Theory)")
    st.markdown("""
**2.1 Random Forest Classifier**  
โครงการนี้เลือกใช้ Random Forest ซึ่งเป็นอัลกอริทึมประเภท Ensemble Learning โดยมีหลักการทำงานดังนี้

**หลักการพื้นฐาน**  
Random Forest สร้าง Decision Tree จำนวนมาก (ในโครงการนี้ใช้ 300 ต้น) โดยแต่ละต้นถูกเทรนด้วยข้อมูลที่สุ่มมาแบบ Bootstrap Sampling และใช้ Feature แบบสุ่มในแต่ละ Node การทำนายขั้นสุดท้ายได้จากการโหวตเสียงข้างมาก (Majority Voting) ของทุก Tree

**พารามิเตอร์ที่ใช้**  
- n_estimators = 300 → จำนวน Decision Tree  
- max_depth = 10–12 → ความลึกสูงสุดของแต่ละ Tree  
- min_samples_split = 5 → จำนวนข้อมูลขั้นต่ำก่อนแบ่ง Node  
- random_state = 42 → กำหนด Seed เพื่อให้ผลลัพธ์คงที่

**ข้อดีของ Random Forest**
- ทนทานต่อ Overfitting เนื่องจากใช้หลาย Tree ร่วมกัน
- รองรับทั้งตัวแปรเชิงตัวเลขและหมวดหมู่
- สามารถคำนวณ Feature Importance เพื่อวิเคราะห์ตัวแปรสำคัญได้
- ทำงานได้ดีแม้ข้อมูลมี Noise
""")
    
    st.header("3. ขั้นตอนการพัฒนาโมเดล (Model Development Pipeline)")
    st.markdown("""
Raw Data → Missing Value → Encoding → Split → SMOTE → Train → Evaluate → Save

**ขั้นที่ 1: โหลดและสำรวจข้อมูล**  
นำเข้าไฟล์ CSV และตรวจสอบโครงสร้างข้อมูล ประเภทตัวแปร และจำนวนข้อมูลที่หายไปในแต่ละคอลัมน์

**ขั้นที่ 2: ทำความสะอาดข้อมูล**  
เติมค่าที่หายไปด้วย Median และ Mode ตามประเภทของตัวแปร

**ขั้นที่ 3: แปลงข้อมูล**  
ใช้ Label Encoder แปลงตัวแปรหมวดหมู่ทุกตัว และบันทึก Encoder เพื่อใช้ในแอปพลิเคชัน

**ขั้นที่ 4: แบ่งข้อมูล Train/Test**  
แบ่งข้อมูลในอัตราส่วน 80:20 โดยใช้ train_test_split พร้อม random_state=42 เพื่อความ Reproducible

**ขั้นที่ 5: ปรับสมดุลด้วย SMOTE**  
ใช้ SMOTE กับข้อมูล Training เท่านั้น เพื่อป้องกัน Data Leakage ไปยัง Test Set

**ขั้นที่ 6: เทรนโมเดล**  
เทรน Random Forest Classifier ด้วยข้อมูล Training ที่ผ่าน SMOTE แล้ว

**ขั้นที่ 7: ประเมินผลโมเดล**  
วัดประสิทธิภาพด้วยเมตริกหลายตัว ได้แก่
- Accuracy
- Precision / Recall / F1-Score
- Confusion Matrix

**ขั้นที่ 8: บันทึกโมเดล**  
บันทึกโมเดลและ Encoder ด้วย joblib เพื่อนำไปใช้ใน Streamlit Application

**ขั้นที่ 9: พัฒนา Web Application**  
สร้าง UI ด้วย Streamlit แบ่งเป็น 2 หน้า รองรับการกรอกข้อมูลแบบ 2 คอลัมน์ และแสดงผลการทำนายแบบ Real-time
""")

    st.header("4. แหล่งอ้างอิง (References)")
    st.markdown("""
- Kaggle: Student Mental Health & Burnout Dataset, Depression Student Dataset
- ChatGPT: อธิบาย SMOTE, Random Forest, Pipeline
- Claude: ช่วยพัฒนา Streamlit UI และแก้ไขโค้ด
- Scikit-learn Documentation (scikit-learn.org)
- Streamlit Documentation (docs.streamlit.io)
- Imbalanced-learn Documentation
""")

