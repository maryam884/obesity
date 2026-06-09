"""
Obesity Level Prediction - Streamlit App
Run: streamlit run app.py
Requires: streamlit, scikit-learn, pandas, numpy, matplotlib, seaborn, plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, silhouette_score
)
from sklearn.preprocessing import label_binarize
import os, warnings
warnings.filterwarnings('ignore')

# Page Config 
st.set_page_config(
    page_title="ObesityIQ 🧠",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }

    .big-title { font-size: 2.6rem; font-weight: 800; margin: 0; }
    .subtitle  { font-size: 1.05rem; color: #888; margin-top: 4px; }

    .fun-card {
        background: white;
        border-radius: 16px;
        padding: 20px 22px;
        text-align: center;
        border: 2px solid #f0f0f0;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06);
    }
    .fun-card .icon  { font-size: 2rem; }
    .fun-card .val   { font-size: 2rem; font-weight: 800; margin: 4px 0 0; }
    .fun-card .label { font-size: 0.82rem; color: #888; margin: 0; }

    .result-box {
        border-radius: 20px;
        padding: 28px 24px;
        text-align: center;
        margin: 16px 0;
    }
    .result-box h1 { font-size: 2.2rem; font-weight: 800; margin: 0 0 6px; }
    .result-box p  { font-size: 1rem; margin: 0; opacity: 0.85; }

    .bmi-bar-wrap {
        background: linear-gradient(to right, #3b82f6, #22c55e, #f59e0b, #ef4444, #7c3aed);
        border-radius: 20px;
        height: 18px;
        position: relative;
        margin: 10px 0 4px;
    }
    .bmi-marker {
        position: absolute;
        top: -6px;
        width: 6px;
        height: 30px;
        background: #111;
        border-radius: 4px;
        transform: translateX(-50%);
    }

    .tip-box {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        border-radius: 10px;
        padding: 14px 16px;
        font-size: 0.92rem;
        color: #166534;
        margin-top: 12px;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label { font-weight: 700; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Data & Model Loading 
@st.cache_data
def load_and_train():
    DATA_PATH = "ObesityDataSet_raw_and_data_sinthetic-2.csv"
    if not os.path.exists(DATA_PATH):
        st.error("❌ Dataset not found. Place 'ObesityDataSet_raw_and_data_sinthetic-2.csv' in the same folder.")
        st.stop()

    df = pd.read_csv(DATA_PATH)
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    df_enc = df.copy()

    cat_cols = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS']
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col])
        le_dict[col] = le

    le_target = LabelEncoder()
    df_enc['y'] = le_target.fit_transform(df_enc['NObeyesdad'])

    features = ['Gender','Age','Height','Weight','family_history_with_overweight',
                'FAVC','FCVC','NCP','CAEC','SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS']
    X = df_enc[features].values
    y = df_enc['y'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    lr = LogisticRegression(max_iter=1000, random_state=42); lr.fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=100, random_state=42); rf.fit(X_train, y_train)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42); gb.fit(X_train, y_train)
    km = KMeans(n_clusters=7, random_state=42, n_init=10); km.fit(X_scaled)
    sil = silhouette_score(X_scaled, km.predict(X_scaled))

    models = {'Logistic Regression': lr, 'Random Forest': rf, 'Gradient Boosting': gb}
    preds  = {k: m.predict(X_test) for k, m in models.items()}
    probs  = {k: m.predict_proba(X_test) for k, m in models.items()}

    y_bin = label_binarize(y_test, classes=list(range(7)))
    metrics = {}
    for name, model in models.items():
        acc = accuracy_score(y_test, preds[name])
        auc = roc_auc_score(y_bin, probs[name], multi_class='ovr', average='macro')
        cm  = confusion_matrix(y_test, preds[name])
        rep = classification_report(y_test, preds[name],
                                    target_names=le_target.classes_, output_dict=True)
        metrics[name] = {'acc': acc, 'auc': auc, 'cm': cm, 'report': rep}

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fi = dict(zip(features, rf.feature_importances_))

    return (df, df_enc, X_scaled, X_train, X_test, y_train, y_test,
            scaler, le_dict, le_target, models, metrics,
            km, sil, X_pca, km.predict(X_scaled), fi, features)

(df, df_enc, X_scaled, X_train, X_test, y_train, y_test,
 scaler, le_dict, le_target, models, metrics,
 km, sil, X_pca, cluster_labels, fi, features) = load_and_train()

CLASS_NAMES = le_target.classes_.tolist()
PALETTE = px.colors.qualitative.Bold

CLASS_EMOJI = {
    'Insufficient_Weight': '🪶',
    'Normal_Weight':        '✅',
    'Obesity_Type_I':       '⚠️',
    'Obesity_Type_II':      '🔶',
    'Obesity_Type_III':     '🔴',
    'Overweight_Level_I':   '🟡',
    'Overweight_Level_II':  '🟠',
}
CLASS_COLOR = {
    'Insufficient_Weight': '#3b82f6',
    'Normal_Weight':        '#22c55e',
    'Overweight_Level_I':   '#facc15',
    'Overweight_Level_II':  '#f97316',
    'Obesity_Type_I':       '#ef4444',
    'Obesity_Type_II':      '#dc2626',
    'Obesity_Type_III':     '#7c3aed',
}
CLASS_TIPS = {
    'Insufficient_Weight': "Your weight is below the healthy range. Consider increasing caloric intake with nutrient-rich foods and consult a dietitian. 🥑",
    'Normal_Weight':        "You're in a healthy weight range — keep it up! Maintain your diet and stay active. 🏃",
    'Overweight_Level_I':   "Slightly above the healthy range. Small lifestyle changes like more walking and less processed food can help. 🚶",
    'Overweight_Level_II':  "Moderately overweight. A balanced diet and 150 mins/week of exercise is a great start. 🥗",
    'Obesity_Type_I':       "Consider consulting a healthcare provider. Gradual changes in diet and activity can make a big difference. 💪",
    'Obesity_Type_II':      "Please consult a doctor. Medical guidance alongside lifestyle changes is recommended. 🏥",
    'Obesity_Type_III':     "Medical consultation is strongly recommended. Comprehensive support can help you on your journey. ❤️",
}

def bmi_category(bmi):
    if bmi < 18.5:   return "Underweight"
    elif bmi < 25:   return "Normal"
    elif bmi < 30:   return "Overweight"
    elif bmi < 35:   return "Obese I"
    elif bmi < 40:   return "Obese II"
    else:            return "Obese III"

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.markdown("## 🥗 ObesityIQ")
st.sidebar.caption("ML-powered health insights")
page = st.sidebar.radio("", [
    "🏠  Home",
    "🔍  Try it yourself",
    "📊  Explore the data",
    "🤖  Model results",
    "🗂️  Clusters",
])
st.sidebar.markdown("---")
st.sidebar.caption("Dataset: UCI Obesity Estimation  \n2,111 people · 16 features · 7 classes")

# 
# HOME
# 
if page == "🏠  Home":
    st.markdown('<p class="big-title">🧠 ObesityIQ</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Can machine learning predict obesity levels from lifestyle habits? Let\'s find out.</p>', unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, val, label in zip(
        [c1,c2,c3,c4],
        ["👥","📋","🏷️","🏆"],
        ["2,111","16","7","96%"],
        ["People surveyed","Lifestyle features","Obesity categories","Best model accuracy"]
    ):
        col.markdown(f"""<div class="fun-card">
            <div class="icon">{icon}</div>
            <div class="val">{val}</div>
            <p class="label">{label}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("### 📖 What's this about?")
    st.markdown("""
    This app uses **three machine learning models** trained on survey data to predict a person's obesity level 
    based on their eating habits, physical activity, and demographics.

    | Model | Accuracy | AUC |
    |---|---|---|
    | 🔵 Logistic Regression | 87.0% | 98.5% |
    | 🌲 Random Forest | 95.3% | 99.7% |
    | 🚀 Gradient Boosting | **96.0%** | **99.7%** |

    We also used **KMeans clustering** (unsupervised) to discover natural groupings in the data.
    """)

    st.markdown("### 🏷️ The 7 Obesity Categories")
    cols = st.columns(7)
    for col, cls in zip(cols, CLASS_NAMES):
        col.markdown(f"""<div class="fun-card">
            <div class="icon">{CLASS_EMOJI[cls]}</div>
            <p class="label" style="font-weight:700;margin-top:6px;">{cls.replace('_',' ')}</p>
        </div>""", unsafe_allow_html=True)

    st.info("👈 Head to **Try it yourself** to get your own prediction!")

# 
# PREDICT
# 
elif page == "🔍  Try it yourself":
    st.markdown('<p class="big-title">🔍 Get your prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Answer a few questions and let the model do its magic ✨</p>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 👤 About you")
        gender = st.selectbox("Gender", ["Female", "Male"])
        age    = st.slider("Age", 14, 61, 25)
        height = st.slider("Height (m)", 1.45, 1.98, 1.70, 0.01, format="%.2f m")
        weight = st.slider("Weight (kg)", 30.0, 170.0, 70.0, 0.5, format="%.1f kg")
        family_hist = st.selectbox("Family history of overweight?", ["yes", "no"])

        bmi = weight / (height ** 2)
        bmi_cat = bmi_category(bmi)
        bmi_pct = min(max((bmi - 10) / (50 - 10), 0), 1) * 100
        st.markdown(f"""
        <div style="background:#f8fafc;border-radius:14px;padding:14px 16px;margin-top:10px;border:1.5px solid #e2e8f0;">
            <b>📏 Your BMI</b><br>
            <span style="font-size:2rem;font-weight:800;">{bmi:.1f}</span>
            <span style="color:#888;font-size:0.9rem;margin-left:6px;">{bmi_cat}</span>
            <div class="bmi-bar-wrap">
                <div class="bmi-marker" style="left:{bmi_pct:.1f}%;"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#aaa;">
                <span>Underweight<br>&lt;18.5</span>
                <span>Normal<br>18.5–25</span>
                <span>Overweight<br>25–30</span>
                <span>Obese<br>&gt;30</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### 🍔 Eating habits")
        favc  = st.selectbox("Eat high-calorie food often?", ["yes", "no"])
        fcvc  = st.slider("Veggies per meal (1=rarely, 3=always)", 1.0, 3.0, 2.0, 0.1)
        ncp   = st.slider("Main meals per day", 1.0, 4.0, 3.0, 0.5)
        caec  = st.selectbox("Eat between meals?", ["no", "Sometimes", "Frequently", "Always"])
        ch2o  = st.slider("Water intake (1=low, 3=high)", 1.0, 3.0, 2.0, 0.1)
        calc  = st.selectbox("Alcohol consumption", ["no", "Sometimes", "Frequently", "Always"])
        scc   = st.selectbox("Do you monitor calories?", ["no", "yes"])

    with col3:
        st.markdown("#### 🏃 Lifestyle")
        faf   = st.slider("Physical activity (0=none, 3=a lot)", 0.0, 3.0, 1.0, 0.1)
        tue   = st.slider("Daily screen time in hours", 0.0, 2.0, 0.5, 0.1)
        smoke = st.selectbox("Do you smoke?", ["no", "yes"])
        mtrans = st.selectbox("Main transport", ["Walking", "Bike", "Motorbike", "Public_Transportation", "Automobile"])
        model_choice = st.selectbox("🤖 Choose model", list(models.keys()))

    st.markdown("---")
    if st.button("✨ Predict my obesity level!", use_container_width=True):
        enc = {
            'Gender': {'Female':0,'Male':1},
            'family_history_with_overweight': {'no':0,'yes':1},
            'FAVC': {'no':0,'yes':1},
            'CAEC': {'Always':0,'Frequently':1,'Sometimes':2,'no':3},
            'SMOKE': {'no':0,'yes':1},
            'SCC': {'no':0,'yes':1},
            'CALC': {'Always':0,'Frequently':1,'Sometimes':2,'no':3},
            'MTRANS': {'Automobile':0,'Bike':1,'Motorbike':2,'Public_Transportation':3,'Walking':4}
        }
        row = np.array([[
            enc['Gender'][gender], age, height, weight,
            enc['family_history_with_overweight'][family_hist],
            enc['FAVC'][favc], fcvc, ncp,
            enc['CAEC'][caec], enc['SMOKE'][smoke],
            ch2o, enc['SCC'][scc], faf, tue,
            enc['CALC'][calc], enc['MTRANS'][mtrans]
        ]])
        row_scaled = scaler.transform(row)
        pred       = models[model_choice].predict(row_scaled)[0]
        proba      = models[model_choice].predict_proba(row_scaled)[0]
        pred_label = CLASS_NAMES[pred]
        confidence = proba[pred] * 100
        color      = CLASS_COLOR.get(pred_label, '#6b7280')
        emoji      = CLASS_EMOJI.get(pred_label, '🔵')

        st.markdown(f"""
        <div class="result-box" style="background:{color}18;border:2.5px solid {color};">
            <h1 style="color:{color};">{emoji} {pred_label.replace('_',' ')}</h1>
            <p style="color:{color};">Predicted by <b>{model_choice}</b> with <b>{confidence:.1f}% confidence</b></p>
        </div>
        """, unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        r1.metric("📏 Your BMI", f"{bmi:.1f}", bmi_cat)
        r2.metric("🎯 Confidence", f"{confidence:.1f}%")
        r3.metric("🔬 Model used", model_choice)

        st.markdown(f'<div class="tip-box">💡 <b>What this means:</b> {CLASS_TIPS[pred_label]}</div>',
                    unsafe_allow_html=True)

        st.markdown("#### 📊 How confident is the model across all categories?")
        prob_df = pd.DataFrame({'Category': [c.replace('_',' ') for c in CLASS_NAMES],
                                'Probability': proba * 100})
        fig = px.bar(prob_df, x='Category', y='Probability',
                     color='Probability', color_continuous_scale='Blues',
                     labels={'Probability':'Confidence (%)'},
                     text=prob_df['Probability'].apply(lambda x: f"{x:.1f}%"))
        fig.update_traces(textposition='outside')
        fig.update_layout(height=350, xaxis_tickangle=-20, showlegend=False,
                          coloraxis_showscale=False,
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# EDA
# ═══════════════════════════════════════════════════════════════
elif page == "📊  Explore the data":
    st.markdown('<p class="big-title">📊 Explore the data</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Get to know the 2,111 people behind this dataset</p>', unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["🏷️ Class distribution", "📏 BMI analysis", "🔥 Correlations", "📦 Feature vs class"])

    with tab1:
        counts = df['NObeyesdad'].value_counts().reset_index()
        counts.columns = ['Class','Count']
        counts['Emoji'] = counts['Class'].map(CLASS_EMOJI)
        counts['Color'] = counts['Class'].map(CLASS_COLOR)
        fig = px.bar(counts, x='Class', y='Count',
                     color='Class', color_discrete_map=CLASS_COLOR,
                     text='Count')
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, height=400, xaxis_tickangle=-20,
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.info("✅ The dataset is fairly **balanced** across all 7 classes — great for training fair models!")

    with tab2:
        st.markdown("#### 📏 BMI Distribution by Obesity Class")
        df['BMI_category'] = df['BMI'].apply(bmi_category)
        fig = px.violin(df, x='NObeyesdad', y='BMI', color='NObeyesdad',
                        color_discrete_map=CLASS_COLOR, box=True, points=False)
        fig.update_layout(showlegend=False, height=420, xaxis_tickangle=-20,
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 🗺️ Weight vs Height coloured by class")
        fig2 = px.scatter(df, x='Height', y='Weight', color='NObeyesdad',
                          color_discrete_map=CLASS_COLOR, opacity=0.6,
                          hover_data=['Age','BMI'])
        fig2.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Average BMI", f"{df['BMI'].mean():.1f}")
        col2.metric("Min BMI", f"{df['BMI'].min():.1f}")
        col3.metric("Max BMI", f"{df['BMI'].max():.1f}")

    with tab3:
        st.markdown("#### 🔥 Feature Correlation Heatmap")
        num_df = df_enc[['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE','y']].copy()
        num_df.columns = ['Age','Height','Weight','Veggies','Meals','Water','Activity','Screen time','Class']
        corr = num_df.corr()
        fig, ax = plt.subplots(figsize=(9,7))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
                    center=0, ax=ax, linewidths=0.5, square=True)
        ax.set_title("Correlation between features", fontsize=13, pad=12)
        fig.patch.set_facecolor('white')
        st.pyplot(fig)
        st.info("💡 **Weight** has the strongest correlation with obesity class (makes sense!). **Physical activity** is negatively correlated.")

    with tab4:
        feat = st.selectbox("Pick a feature to explore", ['Age','Weight','Height','FAF','CH2O','FCVC','TUE','NCP'])
        fig = px.box(df, x='NObeyesdad', y=feat, color='NObeyesdad',
                     color_discrete_map=CLASS_COLOR, points=False)
        fig.update_layout(height=420, xaxis_tickangle=-20, showlegend=False,
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# MODEL RESULTS
# ═══════════════════════════════════════════════════════════════
elif page == "🤖  Model results":
    st.markdown('<p class="big-title">🤖 Model results</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">How well did each model do?</p>', unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Leaderboard", "🟦 Confusion matrices", "📈 ROC curves", "🌲 Feature importance"])

    with tab1:
        st.markdown("#### 🏆 Model Leaderboard")
        rows = []
        for i, (name, m) in enumerate(metrics.items()):
            medal = ["🥇","🥈","🥉"][i] if i < 3 else ""
            rows.append({
                'Model': f"{medal} {name}",
                'Accuracy': f"{m['acc']*100:.2f}%",
                'AUC (macro)': f"{m['auc']*100:.2f}%",
                'Macro F1': f"{m['report']['macro avg']['f1-score']*100:.2f}%",
                'Macro Precision': f"{m['report']['macro avg']['precision']*100:.2f}%",
            })
        st.dataframe(pd.DataFrame(rows).set_index('Model'), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            acc_vals = {k: v['acc']*100 for k,v in metrics.items()}
            fig = px.bar(x=list(acc_vals.keys()), y=list(acc_vals.values()),
                         color=list(acc_vals.keys()), color_discrete_sequence=['#3b82f6','#22c55e','#f59e0b'],
                         labels={'x':'Model','y':'Accuracy (%)'},
                         text=[f"{v:.1f}%" for v in acc_vals.values()])
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, height=320, title="Accuracy",
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                              yaxis=dict(range=[80,101]))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            auc_vals = {k: v['auc']*100 for k,v in metrics.items()}
            fig = px.bar(x=list(auc_vals.keys()), y=list(auc_vals.values()),
                         color=list(auc_vals.keys()), color_discrete_sequence=['#3b82f6','#22c55e','#f59e0b'],
                         labels={'x':'Model','y':'AUC (%)'},
                         text=[f"{v:.1f}%" for v in auc_vals.values()])
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, height=320, title="AUC Score",
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                              yaxis=dict(range=[95,101]))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### 🟦 Confusion Matrices")
        st.caption("Diagonal = correct predictions. Off-diagonal = mistakes.")
        short_names = [c.replace('_',' ').replace('Obesity ','Ob. ').replace('Overweight','OW') for c in CLASS_NAMES]
        cols = st.columns(3)
        for i, (name, m) in enumerate(metrics.items()):
            with cols[i]:
                st.markdown(f"**{name}** · {m['acc']*100:.1f}%")
                fig, ax = plt.subplots(figsize=(4.5, 3.8))
                sns.heatmap(m['cm'], annot=True, fmt='d', cmap='Blues',
                            xticklabels=short_names, yticklabels=short_names,
                            ax=ax, cbar=False, linewidths=0.3)
                ax.set_xlabel("Predicted", fontsize=9)
                ax.set_ylabel("Actual", fontsize=9)
                ax.tick_params(axis='x', labelsize=7, rotation=30)
                ax.tick_params(axis='y', labelsize=7, rotation=0)
                fig.patch.set_facecolor('white')
                plt.tight_layout()
                st.pyplot(fig)

    with tab3:
        sel = st.selectbox("Select model", list(models.keys()))
        y_bin = label_binarize(y_test, classes=list(range(7)))
        probs = models[sel].predict_proba(X_test)
        fig = go.Figure()
        colors = px.colors.qualitative.Bold
        for i, cls in enumerate(CLASS_NAMES):
            fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
            auc_i = roc_auc_score(y_bin[:, i], probs[:, i])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                      name=f"{CLASS_EMOJI[cls]} {cls.replace('_',' ')} ({auc_i:.3f})",
                                      line=dict(color=colors[i], width=2)))
        fig.add_shape(type='line', x0=0, x1=1, y0=0, y1=1,
                      line=dict(dash='dash', color='gray'))
        fig.update_layout(title=f"ROC Curves — {sel} (macro AUC: {metrics[sel]['auc']:.4f})",
                          xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                          height=500, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("#### 🌲 What matters most? (Random Forest)")
        fi_df = pd.DataFrame({'Feature': list(fi.keys()), 'Importance': list(fi.values())})
        fi_df = fi_df.sort_values('Importance', ascending=True)
        fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Blues',
                     text=fi_df['Importance'].apply(lambda x: f"{x*100:.1f}%"))
        fig.update_traces(textposition='outside')
        fig.update_layout(height=500, showlegend=False, coloraxis_showscale=False,
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.success("🏆 **Weight** is by far the most predictive feature (35.2%), followed by Height and Age.")

# ═══════════════════════════════════════════════════════════════
# CLUSTERS
# ═══════════════════════════════════════════════════════════════
elif page == "🗂️  Clusters":
    st.markdown('<p class="big-title">🗂️ Clustering</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">KMeans finds natural groups in the data — without using the labels!</p>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Algorithm", "KMeans")
    col2.metric("Number of clusters", "7")
    col3.metric("Silhouette score", f"{sil:.3f}")

    st.markdown("#### 🗺️ PCA Projection of Clusters")
    st.caption("We compressed 16 features into 2 dimensions using PCA, then coloured each point by its cluster.")
    pca_df = pd.DataFrame({
        'PC1': X_pca[:,0], 'PC2': X_pca[:,1],
        'Cluster': [f"Cluster {c}" for c in cluster_labels],
        'True Class': [CLASS_NAMES[i].replace('_',' ') for i in df_enc['y'].values]
    })
    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                     hover_data=['True Class'], opacity=0.65,
                     color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(height=460, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🤔 Why is the silhouette score low?")
    st.info("""
    A score of **0.152** isn't a failure — it reflects a biological reality. 
    Obesity levels exist on a **continuous spectrum**, so adjacent categories naturally overlap 
    (e.g. Overweight Level I vs II). KMeans still captures meaningful clusters, especially 
    for the extreme categories like **Obesity Type III** and **Insufficient Weight**.
    """)

    st.markdown("#### 🗂️ Cluster vs True Class Breakdown")
    ct = pd.crosstab(cluster_labels, [CLASS_NAMES[i].replace('_',' ') for i in df_enc['y'].values])
    fig2, ax = plt.subplots(figsize=(11,4))
    sns.heatmap(ct, annot=True, fmt='d', cmap='YlOrRd', ax=ax, linewidths=0.3)
    ax.set_xlabel("True Class"); ax.set_ylabel("Cluster")
    fig2.patch.set_facecolor('white')
    plt.tight_layout()
    st.pyplot(fig2)

# ── Footer ────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using Streamlit + scikit-learn")