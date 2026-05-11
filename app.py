import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import pandas as pd
import os

API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000"
)
st.set_page_config(page_title="Voyage Analytics", layout="wide")

# CSS

st.markdown("""
<style>

/*
GLOBAL BACKGROUND */

html, body, [data-testid="stAppViewContainer"]{
    background:
    radial-gradient(circle at top left, rgba(56,189,248,0.12), transparent 25%),
    radial-gradient(circle at top right, rgba(168,85,247,0.10), transparent 25%),
    radial-gradient(circle at bottom left, rgba(16,185,129,0.10), transparent 25%),
    linear-gradient(180deg,#020617 0%,#071226 45%,#020617 100%) !important;
    color:#ffffff !important;
}

/* MAIN AREA */

[data-testid="stAppViewContainer"] > .main{
    background:transparent !important;
}

/* HEADER */

[data-testid="stHeader"]{
    background:transparent !important;
}

/* BLOCK CONTAINER */

.block-container{
    padding-top:1.5rem;
    padding-bottom:2rem;
    background:transparent !important;
}

/* TEXT */

h1,h2,h3,h4,h5,h6,p,label{
    color:white !important;
}

/*
HEADER */

.header-container{
    text-align:center;
    margin-top:20px;
    margin-bottom:10px;
}

.main-title{
    font-size:56px;
    font-weight:900;
    letter-spacing:2px;
    background:linear-gradient(90deg,#38bdf8,#818cf8,#c084fc);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    text-shadow:0 0 30px rgba(56,189,248,0.35);
}

.sub-title{
    font-size:18px;
    color:#cbd5e1 !important;
    margin-top:8px;
}

/* DIVIDER */

.divider{
    height:2px;
    background:linear-gradient(90deg,transparent,#38bdf8,#818cf8,#c084fc,transparent);
    margin:28px 0;
    box-shadow:0 0 18px rgba(56,189,248,0.55);
    border-radius:999px;
}

/*
SIDEBAR */

section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0f172a,#020617) !important;
    border-right:1px solid rgba(255,255,255,0.08);
}

/* SIDEBAR TITLE */

.sidebar-title{
    font-size:18px;
    font-weight:700;
    color:#38bdf8 !important;
    margin-top:20px;
    margin-bottom:10px;
}

/* SIDEBAR BOX */

.sidebar-box{
    background:rgba(255,255,255,0.05);
    padding:16px;
    border-radius:16px;
    margin-bottom:16px;
    border:1px solid rgba(255,255,255,0.08);
}

/*
STATUS */

.status-ok{
    background:rgba(34,197,94,0.18);
    border:1px solid #22c55e;
    padding:14px;
    border-radius:14px;
    text-align:center;
    font-weight:700;
    color:#bbf7d0 !important;
}

.status-bad{
    background:rgba(239,68,68,0.18);
    border:1px solid #ef4444;
    padding:14px;
    border-radius:14px;
    text-align:center;
    font-weight:700;
    color:#fecaca !important;
}

/*
CARDS */

.card{
    background:linear-gradient(
    135deg,
    rgba(255,255,255,0.08),
    rgba(255,255,255,0.03));

    backdrop-filter:blur(18px);
    padding:24px;
    border-radius:20px;
    border:1px solid rgba(255,255,255,0.08);
    transition:0.35s;
    margin-bottom:24px;
    box-shadow:0 0 24px rgba(15,23,42,0.35);
}

.card:hover{
    transform:translateY(-6px);
    border:1px solid rgba(56,189,248,0.65);
    box-shadow:
    0 0 30px rgba(56,189,248,0.25),
    0 0 60px rgba(168,85,247,0.12);
}

/* PRICE */

.price{
    font-size:38px;
    font-weight:900;
    margin-top:10px;
    color:white;
}

/* META */

.meta{
    font-size:14px;
    color:#cbd5e1 !important;
    margin-top:14px;
    line-height:1.8;
}

/*
BUTTONS */

.stButton > button{
    border-radius:14px;
    background:linear-gradient(135deg,#2563eb,#38bdf8);
    border:none;
    padding:12px 28px;
    font-weight:700;
    color:white !important;
    transition:0.3s;
}

.stButton > button:hover{
    transform:translateY(-2px);
    background:linear-gradient(135deg,#1d4ed8,#0ea5e9);
}

/*
INPUTS */

/* TEXT INPUT */

.stTextInput input{
    background:white !important;
    color:black !important;
    border-radius:12px !important;
    font-weight:600 !important;
}

/* NUMBER INPUT */

.stNumberInput input{
    background:white !important;
    color:black !important;
    border-radius:12px !important;
    font-weight:600 !important;
}

/* TEXTAREA */

.stTextArea textarea{
    background:white !important;
    color:black !important;
}

/* SELECTBOX */

.stSelectbox div[data-baseweb="select"]{
    background:white !important;
    border-radius:12px !important;
}

/* SELECTBOX TEXT */

.stSelectbox div[data-baseweb="select"] span{
    color:black !important;
    font-weight:600 !important;
}

/* SELECTBOX ICON */

.stSelectbox svg{
    fill:black !important;
}

/* DROPDOWN OPTIONS */

div[role="listbox"] *{
    color:black !important;
    background:white !important;
}

/*
LABELS */

label{
    font-weight:600 !important;
    color:#e2e8f0 !important;
}

/*
SCROLLBAR */

::-webkit-scrollbar{
    width:10px;
}

::-webkit-scrollbar-track{
    background:#020617;
}

::-webkit-scrollbar-thumb{
    background:linear-gradient(#38bdf8,#818cf8);
    border-radius:20px;
}

</style>
""", unsafe_allow_html=True)



# SAFE API HANDLER
def safe_get(url):
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            return res.json(), None
        return None, "API Error"
    except requests.exceptions.ConnectionError:
        return None, "API not reachable"
    except requests.exceptions.Timeout:
        return None, "Request timed out"
    except Exception as e:
        return None, str(e)

def safe_post(url, payload):
    try:
        res = requests.post(url, json=payload, timeout=200)
        if res.status_code == 200:
            return res.json(), None
        try:
            return None, res.json().get("detail", "API Error")
        except:
            return None, res.text
    except requests.exceptions.ConnectionError:
        return None, "API not reachable"
    except requests.exceptions.Timeout:
        return None, "Request timed out"
    except Exception as e:
        return None, str(e)

def render_metrics(data, task="regression"):

    if not data:
        return

    model_info = data.get("model_info", {})
    metrics = data.get("metrics", {})

    st.markdown("## 📊 Model Insights")
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # MODEL INFO

    st.markdown("### 🔍 Model Information")

    info_cards = [
        ("Model", model_info.get("model_name", "N/A"), "#2563eb"),
        ("Algorithm", model_info.get("algorithm", "N/A"), "#7c3aed"),
        ("Version", model_info.get("version", "N/A"), "#0891b2"),
        ("Platform", "Docker + Kubernetes + MLflow", "#059669")
    ]

    cols = st.columns(4)

    for col, (title, value, color) in zip(cols, info_cards):

        with col:

            st.markdown(f"""
            <div style="
                background:linear-gradient(135deg,rgba(255,255,255,0.07),rgba(255,255,255,0.03));
                border:1px solid {color};
                border-radius:20px;
                padding:24px;
                min-height:150px;
                display:flex;
                flex-direction:column;
                justify-content:center;
                backdrop-filter:blur(18px);
                box-shadow:0 0 20px {color}22;
                margin-bottom:22px;
                transition:0.3s;
            ">
                <div style="
                    color:#94a3b8;
                    font-size:15px;
                    margin-bottom:14px;
                    font-weight:600;
                    text-transform:uppercase;
                    letter-spacing:1px;
                ">
                    {title}
                </div>
                <div style="
                    font-size:18px;;
                    font-weight:900;
                    color:white;
                    overflow:hidden;
                    text-overflow:ellipsis;
                    white-space:nowrap;
                ">
                    {value}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # REGRESSION

    if task == "regression":

        st.markdown("### 📈 Regression Metrics")

        metric_cards = [
            ("MAE", metrics.get("MAE"), "#ef4444"),
            ("RMSE", metrics.get("RMSE"), "#f97316"),
            ("R2", metrics.get("R2"), "#22c55e"),
            ("CV", metrics.get("CV"), "#eab308"),
            ("Score", metrics.get("score"), "#38bdf8")
        ]

    # CLASSIFICATION

    else:

        st.markdown("### 🎯 Classification Metrics")

        metric_cards = [
            ("Accuracy", metrics.get("accuracy"), "#22c55e"),
            ("Precision", metrics.get("precision"), "#38bdf8"),
            ("Recall", metrics.get("recall"), "#f97316"),
            ("F1", metrics.get("f1"), "#a855f7"),
            ("Score", metrics.get("score"), "#eab308")
        ]

    cols = st.columns(5)

    for col, (title, value, color) in zip(cols, metric_cards):

        with col:

            if value is None:

                display_value = "N/A"

            else:

                # CLASSIFICATION → SHOW %
                if task == "classification":

                    display_value = f"{float(value) * 100:.2f}%"

                # REGRESSION → NORMAL VALUE
                else:

                    display_value = f"{float(value):.2e}"
                    
            st.markdown(f"""
            <div style="
                background:linear-gradient(135deg,rgba(255,255,255,0.07),rgba(255,255,255,0.03));
                border:1px solid {color};
                border-radius:20px;
                padding:24px;
                min-height:160px;
                display:flex;
                flex-direction:column;
                justify-content:center;
                align-items:center;
                text-align:center;
                backdrop-filter:blur(18px);
                box-shadow:0 0 22px {color}22;
                margin-bottom:22px;
                transition:0.3s;
            ">
                <div style="
                    color:#94a3b8;
                    font-size:15px;
                    margin-bottom:16px;
                    font-weight:700;
                    text-transform:uppercase;
                    letter-spacing:1px;
                ">
                    {title}
                </div>
                <div style="
                    font-size:34px;
                    font-weight:900;
                    color:{color};
                    word-break:break-word;
                ">
                    {display_value}
                </div>
            </div>
            """, unsafe_allow_html=True)

# HEADER
st.markdown("""
<div class='header-container'>
    <div class='main-title'>Voyage Analytics</div>
    <div class='sub-title'>AI Decision System for Travel Intelligence</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# SIDEBAR

st.sidebar.markdown("<div class='sidebar-title'>Navigation</div>", unsafe_allow_html=True)

menu = st.sidebar.selectbox(
    "Select Module",
    ["Flight Price Prediction", "Gender Prediction", "Recommendation System"]
)

if menu != "Recommendation System":
    mode = st.sidebar.radio("Model Mode", ["auto", "manual"])
else:
    mode = "auto"

model_name = None

if mode == "manual":
    st.sidebar.markdown("<div class='sidebar-title'>Model Selection</div>", unsafe_allow_html=True)

    
    REGRESSION_MODEL_MAP = {
    "Linear Regression": "Linear Regression",

    "Decision Tree (Base)": "Decision Tree",
    "Decision Tree (Tuned)": "Decision Tree",

    "Random Forest (Base)": "Random Forest",
    "Random Forest (Tuned)": "Random Forest",

    "XGBoost (Base)": "XGBoost",
    "XGBoost (Tuned)": "XGBoost"
}

    CLASSIFICATION_MODEL_MAP = {
        "Logistic Regression": "Logistic Regression",

        "Decision Tree (Base)": "Decision Tree",
        "Decision Tree (Tuned)": "Decision Tree",

        "Random Forest (Base)": "Random Forest",
        "Random Forest (Tuned)": "Random Forest",

        "XGBoost (Base)": "XGBoost",
        "XGBoost (Tuned)": "XGBoost"
    }
    if menu == "Flight Price Prediction":

        selected_label = st.sidebar.selectbox(
            "Select Model",
            list(REGRESSION_MODEL_MAP.keys())
        )

        model_name = REGRESSION_MODEL_MAP[selected_label]


    elif menu == "Gender Prediction":

        selected_label = st.sidebar.selectbox(
            "Select Model",
            list(CLASSIFICATION_MODEL_MAP.keys())
        )

        model_name = CLASSIFICATION_MODEL_MAP[selected_label]

if menu == "Recommendation System":
    st.sidebar.markdown("<div class='sidebar-title'>Recommendation</div>", unsafe_allow_html=True)

    rec_options = {
        "Flight Recommender": "flights",
        "Hotel Recommender": "hotels",
        "Package Recommender": "package"
    }

    selected_label = st.sidebar.selectbox("Select Type", list(rec_options.keys()))
    rec_type = rec_options[selected_label]

# SYSTEM HEALTH
st.sidebar.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-title'>System Health</div>", unsafe_allow_html=True)

health, err = safe_get(f"{API_URL}/")

if err:
    st.sidebar.markdown("<div class='status-bad'>Down</div>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("<div class='status-ok'>Connected</div>", unsafe_allow_html=True)

# SYSTEM INFO
st.sidebar.markdown("<div class='sidebar-title'>System Info</div>", unsafe_allow_html=True)

st.sidebar.markdown(""" 
                    <div class='sidebar-box'> 
                    <b>Architecture</b><br> Backend: FastAPI<br> Model Registry: MLflow<br> Interface: Streamlit<br><br> <b>Capabilities</b><br> • Flight Price Prediction<br> • Gender Classification<br> • Travel Recommendation<br><br> """, unsafe_allow_html=True)

#  FLIGHT
if menu == "Flight Price Prediction":

    st.markdown("## Flight Price Prediction")

    loc_data, err = safe_get(f"{API_URL}/metadata/locations")

    if err:
        st.error(err)
        st.stop()

    c1, c2, c3 = st.columns(3)

    source = c1.selectbox(
        "Source",
        loc_data.get("sources", [])
    )

    destination = c2.selectbox(
        "Destination",
        loc_data.get("destinations", [])
    )

    flight_type = c3.selectbox(
        "Flight Type",
        loc_data.get("flight_types", [])
    )

    if st.button("Run Prediction"):

        if not source or not destination:
            st.warning("Please select valid locations")
            st.stop()

        payload = {
            "mode": mode,
            "model_name": model_name,
            "from": source,
            "to": destination,
            "flighttype": flight_type
        }

        with st.spinner("Analyzing..."):

            data, err = safe_post(f"{API_URL}/predict/price", payload)

            if err:
                st.error(err)
                if st.button("🔄 Retry"):
                    st.rerun()
            elif not data or not data.get("results"):
                st.warning("No results found")
            else:
                cols = st.columns(3)
                for i, row in enumerate(data["results"]):

                    with cols[i % 3]:

                        st.markdown(f"""
                        <div style="
                            background:linear-gradient(135deg,rgba(37,99,235,0.22),rgba(14,165,233,0.08));
                            border:1px solid #38bdf8;
                            border-radius:20px;
                            padding:26px;
                            backdrop-filter:blur(18px);
                            box-shadow:0 0 26px rgba(56,189,248,0.25);
                            margin-bottom:22px;
                            transition:0.3s;
                        ">
                            <div style="
                                font-size:24px;
                                font-weight:800;
                                color:#7dd3fc;
                                margin-bottom:12px;
                            ">
                                ✈️ {row.get('agency')}
                            </div>
                            <div style="
                                font-size:42px;
                                font-weight:900;
                                color:white;
                                margin-bottom:18px;
                            ">
                                {row.get('price')}
                            </div>
                            <div style="
                                color:#cbd5e1;
                                line-height:1.9;
                                font-size:15px;
                            ">
                                <b style="color:#38bdf8;">Distance</b><br>
                                {row.get('distance')}
                                <br><br>
                                <b style="color:#38bdf8;">Time</b><br>
                                {row.get('time')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                render_metrics(data, task="regression")

#  GENDER
elif menu == "Gender Prediction":

    st.markdown("## Gender Prediction")

    name = st.text_input("Name")
    age = st.number_input("Age", value=25)

    if st.button("Run Prediction"):

        if not name.strip():
            st.warning("Please enter a name")
            st.stop()

        payload = {
            "mode": mode,
            "model_name": model_name,
            "name": name,
            "age": age
        }

        with st.spinner("Predicting..."):

            data, err = safe_post(f"{API_URL}/predict/gender", payload)

            if err:
                st.error(err)
                if st.button("🔄 Retry"):
                    st.rerun()
            elif not data or not data.get("prediction"):
                st.warning("No prediction returned")
            else:
                prediction = str(data['prediction']).upper()

                color = "#0088ff" if prediction == "MALE" else "#ff0099"

                emoji = "🧑" if prediction == "MALE" else "👩"

                st.markdown(f"""
                <div style="
                    background:linear-gradient(135deg,rgba(255,255,255,0.06),rgba(255,255,255,0.02));
                    border:2px solid {color};
                    border-radius:22px;
                    padding:40px;
                    text-align:center;
                    backdrop-filter:blur(18px);
                    box-shadow:0 0 30px {color}33;
                    margin-top:20px;
                ">
                    <div style="
                        font-size:22px;
                        color:#94a3b8;
                        margin-bottom:14px;
                        font-weight:600;
                        letter-spacing:1px;
                    ">
                        PREDICTED GENDER
                    </div>
                    <div style="
                        font-size:52px;
                        font-weight:900;
                        color:{color};
                        letter-spacing:3px;
                    ">
                        {emoji} {prediction}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            render_metrics(data, task="classification")


# RECOMMENDER
elif menu == "Recommendation System":

    st.markdown("## Recommendation Engine")

    c1, c2, c3 = st.columns(3)

    user_id = c1.number_input("User ID", value=101)
    max_price = c2.number_input("Max Price", value=2000)
    top_n = c3.slider("Results", 1, 20, 5)

    loc_data, err = safe_get(f"{API_URL}/metadata/locations")

    if err:
        st.error(err)

        if st.button("🔄 Retry Recommendation"):
            st.rerun()

        st.stop()

    # DYNAMIC INPUTS

    source = None
    destination = None
    place = None

    # FLIGHTS
    if rec_type == "flights":

        c4, c5 = st.columns(2)

        source = c4.selectbox(
            "Source",
            loc_data.get("sources", [])
        )

        destination = c5.selectbox(
            "Destination",
            loc_data.get("destinations", [])
        )

    # HOTELS
    elif rec_type == "hotels":

        place = st.selectbox(
            "Place",
            loc_data.get("destinations", [])
        )

    # PACKAGE
    elif rec_type == "package":

        c4, c5, c6 = st.columns(3)

        source = c4.selectbox(
            "Source",
            loc_data.get("sources", [])
        )

        destination = c5.selectbox(
            "Destination",
            loc_data.get("destinations", [])
        )

        place = c6.selectbox(
            "Place",
            loc_data.get("destinations", [])
        )

    # RUN BUTTON

    if st.button("Run Recommendation"):

        payload = {
            "rec_type": rec_type,
            "user_id": user_id,
            "source": source,
            "destination": destination,
            "place": place,
            "max_price": max_price,
            "top_n": top_n
        }

        with st.spinner("Generating..."):

            data, err = safe_post(
                f"{API_URL}/recommend",
                payload
            )

            if err:
                st.error(err)

            else:

                recs = data.get("recommendations")

                if not recs:
                    st.warning("No recommendations found")
        
                # SINGLE MODE
        

                elif isinstance(recs, list):

                    cols = st.columns(3)

                    for i, row in enumerate(recs):

                        with cols[i % 3]:

                            # FLIGHTS
                            if rec_type == "flights":

                                 st.markdown(f"""
                                    <div style="background:linear-gradient(135deg,rgba(37,99,235,0.22),rgba(14,165,233,0.08));border:1px solid #38bdf8;border-radius:18px;padding:24px;backdrop-filter:blur(18px);box-shadow:0 0 24px rgba(56,189,248,0.25);margin-bottom:22px;">
                                    <h3 style="color:#7dd3fc;">✈️ {row.get('agency','Unknown')}</h3>
                                    <div style="font-size:34px;font-weight:900;color:white;margin-top:10px;"> {round(row.get('price',0),2)}</div>
                                    <div style="color:#cbd5e1;margin-top:18px;line-height:1.8;font-size:18px;">
                                    <b style="color:#38bdf8;">Route</b><br>
                                    {row.get('from','N/A')} → {row.get('to','N/A')}
                                    <br><br>
                                    <b style="color:#38bdf8;">Flight Type</b><br>
                                    {row.get('flighttype','N/A')}
                                    <br><br>
                                    <b style="color:#38bdf8;">Confidence</b><br>
                                    {round(row.get('confidence',0)*100,1)}%
                                    </div>
                                    </div>
                                    """, unsafe_allow_html=True)

                            # HOTELS
                            elif rec_type == "hotels":

                                st.markdown(f"""
                                    <div style="background:linear-gradient(135deg,rgba(16,185,129,0.22),rgba(5,150,105,0.08));border:1px solid #34d399;border-radius:18px;padding:24px;backdrop-filter:blur(18px);box-shadow:0 0 24px rgba(52,211,153,0.25);margin-bottom:22px;">
                                    <h3 style="color:#6ee7b7;">🏨 {row.get('name','Hotel')}</h3>
                                    <div style="font-size:34px;font-weight:900;color:white;margin-top:10px;"> {round(row.get('price',0),2)}</div>
                                     <div style="
                                        font-size:12px;
                                        color:#d8b4fe;
                                        margin-top:6px;
                                        opacity:0.85;
                                    ">
                                    Per Day Price
                                    </div>
                                    <div style="color:#d1fae5;margin-top:18px;line-height:1.8;font-size:18px;">
                                    <b style="color:#34d399;">Location</b><br>
                                    {row.get('place','N/A')}
                                    <br><br>
                                    <b style="color:#34d399;">Confidence</b><br>
                                    {round(row.get('confidence',0)*100,1)}%
                                    </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            # PACKAGE
                            elif rec_type == "package":

                                 st.markdown(f"""
                                    <div style="background:linear-gradient(135deg,rgba(168,85,247,0.22),rgba(126,34,206,0.08));border:1px solid #c084fc;border-radius:18px;padding:24px;backdrop-filter:blur(18px);box-shadow:0 0 24px rgba(192,132,252,0.25);margin-bottom:22px;">
                                    <h3 style="color:#d8b4fe;">🎒 {row.get('agency','Package')}</h3>
                                    <div style="font-size:34px;font-weight:900;color:white;margin-top:10px;">
                                    {round(row.get('total_price',0),2)}
                                    </div>
                                    <div style="
                                        font-size:12px;
                                        color:#d8b4fe;
                                        margin-top:6px;
                                        opacity:0.85;
                                    ">
                                    Includes 1 Day Hotel Price
                                    </div>
                                                <div style="color:#f3e8ff;margin-top:18px;line-height:1.8;font-size:18px;">
                                    <b style="color:#c084fc;">Flight Price</b><br>
                                    {round(row.get('price',0),2)}
                                    <br><br>
                                    <b style="color:#c084fc;">Hotel</b><br>
                                    {row.get('hotel_name','N/A')}
                                    <br><br>
                                    <b style="color:#c084fc;">Hotel Price</b><br>
                                    {round(row.get('hotel_price',0),2)} Per Day
                                    <br><br>
                                    <b style="color:#c084fc;">Route</b><br>
                                    {row.get('from','N/A')} → {row.get('to','N/A')}
                                    <br><br>
                                    <b style="color:#c084fc;">Confidence</b><br>
                                    {round(row.get('confidence',0)*100,1)}%
                                    </div>
                                    </div>
                                    """, unsafe_allow_html=True)

                            # FALLBACK
                            else:

                                st.markdown(
                                    f"<div class='card'>{row}</div>",
                                    unsafe_allow_html=True
                                )
                                
                    # RECOMMENDATION VISUALIZATION

                    try:

                        rec_df = pd.DataFrame(recs)

                        if not rec_df.empty:

                            st.markdown("## 📊 Recommendation Insights")

                            # FLIGHT VISUALS

                            if rec_type == "flights":

                                if "agency" in rec_df.columns and "price" in rec_df.columns:

                                    st.markdown("### ✈️ Airline Price Comparison")

                                    fig_bar = px.bar(
                                        rec_df,
                                        x="agency",
                                        y="price",
                                        color="price",
                                        text_auto=True,
                                        title="Flight Recommendation Prices"
                                    )

                                    st.plotly_chart(
                                        fig_bar,
                                        use_container_width=True
                                    )

                                if "confidence" in rec_df.columns:

                                    st.markdown("### 🎯 Recommendation Confidence")

                                    fig_conf = px.scatter(
                                        rec_df,
                                        x="agency",
                                        y="confidence",
                                        size="price",
                                        color="confidence",
                                        hover_data=rec_df.columns
                                    )

                                    st.plotly_chart(
                                        fig_conf,
                                        use_container_width=True
                                    )

                            # HOTEL VISUALS

                            elif rec_type == "hotels":

                                if "name" in rec_df.columns and "price" in rec_df.columns:

                                    st.markdown("### 🏨 Hotel Price Comparison")

                                    fig_hotel = px.bar(
                                        rec_df,
                                        x="name",
                                        y="price",
                                        color="price",
                                        text_auto=True
                                    )

                                    st.plotly_chart(
                                        fig_hotel,
                                        use_container_width=True
                                    )

                            # PACKAGE VISUALS

                            elif rec_type == "package":

                                if (
                                    "agency" in rec_df.columns and
                                    "total_price" in rec_df.columns
                                ):

                                    st.markdown("### 🎒 Package Cost Analysis")

                                    fig_package = px.bar(
                                        rec_df,
                                        x="agency",
                                        y="total_price",
                                        color="total_price",
                                        text_auto=True
                                    )

                                    st.plotly_chart(
                                        fig_package,
                                        use_container_width=True
                                    )

                                if (
                                    "hotel_price" in rec_df.columns and
                                    "price" in rec_df.columns
                                ):

                                    st.markdown("### 💰 Flight vs Hotel Cost")

                                    compare_df = pd.DataFrame({
                                        "Category": ["Flight", "Hotel"],
                                        "Cost": [
                                            rec_df["price"].mean(),
                                            rec_df["hotel_price"].mean()
                                        ]
                                    })

                                    fig_compare = px.pie(
                                        compare_df,
                                        names="Category",
                                        values="Cost"
                                    )

                                    st.plotly_chart(
                                        fig_compare,
                                        use_container_width=True
                                    )

                    except Exception as e:

                        st.warning(
                            f"Visualization Error: {e}"
                        )