import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="DrainTurn AI",
    page_icon="🌧️",
    layout="wide"
)

# ---------------- MODEL TRAINING ----------------
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 1000

    data = pd.DataFrame({
        "Rainfall": np.random.uniform(0, 150, n),
        "Water_Level": np.random.uniform(0, 100, n),
        "Flow_Rate": np.random.uniform(0, 100, n),
        "Debris_Level": np.random.uniform(0, 100, n)
    })

    risk_score = (
        data["Rainfall"] * 0.25
        + data["Water_Level"] * 0.30
        + (100 - data["Flow_Rate"]) * 0.20
        + data["Debris_Level"] * 0.25
    )

    data["Risk"] = pd.cut(
        risk_score,
        bins=[-1, 35, 60, 200],
        labels=["Low", "Medium", "High"]
    )

    features = [
        "Rainfall",
        "Water_Level",
        "Flow_Rate",
        "Debris_Level"
    ]

    X = data[features]
    y = data["Risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, data, accuracy, features


model, data, accuracy, features = train_model()

# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.title("🌧️ DrainTurn AI")

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Risk Prediction",
        "Data Analytics",
        "About Project"
    ]
)

# ---------------- DASHBOARD ----------------
if page == "Dashboard":

    st.title("🌧️ DrainTurn AI")
    st.subheader(
        "Predictive Drainage Blockage & Failure Intelligence System"
    )

    st.info(
        "An AI-based prototype for estimating drainage risk "
        "using rainfall, water level, flow rate, and debris level."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Demo Records", len(data))
    col2.metric("ML Algorithm", "Random Forest")
    col3.metric("Test Accuracy", f"{accuracy:.1%}")

    st.markdown("---")
    st.subheader("Project Overview")

    st.write(
        "DrainTurn AI uses a machine-learning classifier to "
        "estimate drainage risk from four input parameters."
    )

    st.write("### Risk Categories")

    col1, col2, col3 = st.columns(3)

    col1.success("🟢 LOW RISK")
    col2.warning("🟠 MEDIUM RISK")
    col3.error("🔴 HIGH RISK")

    st.subheader("Sample Dataset")
    st.dataframe(data.head(10), use_container_width=True)

# ---------------- RISK PREDICTION ----------------
elif page == "Risk Prediction":

    st.title("🔍 Drainage Risk Prediction")

    st.write(
        "Adjust the values below and predict the drainage risk."
    )

    col1, col2 = st.columns(2)

    with col1:
        rainfall = st.slider(
            "Rainfall (mm)",
            0, 150, 50
        )

        water_level = st.slider(
            "Water Level (%)",
            0, 100, 50
        )

    with col2:
        flow_rate = st.slider(
            "Flow Rate (%)",
            0, 100, 50
        )

        debris_level = st.slider(
            "Debris Level (%)",
            0, 100, 50
        )

    if st.button("Predict Drainage Risk", type="primary"):

        input_data = pd.DataFrame(
            [[rainfall, water_level, flow_rate, debris_level]],
            columns=features
        )

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        st.subheader("Prediction Result")

        if prediction == "Low":
            st.success("🟢 LOW RISK")
            st.write("The model estimates a lower risk category.")

        elif prediction == "Medium":
            st.warning("🟠 MEDIUM RISK")
            st.write("The model estimates a moderate risk category.")

        else:
            st.error("🔴 HIGH RISK")
            st.write("The model estimates a higher risk category.")

        st.write(f"**Predicted Category:** {prediction}")

        st.subheader("Prediction Probabilities")

        probability_data = pd.DataFrame({
            "Risk Category": model.classes_,
            "Probability": probabilities
        })

        st.bar_chart(
            probability_data.set_index("Risk Category")
        )

        st.caption(
            "Demonstration only: predictions use a model trained "
            "on synthetic data and are not validated for real drainage "
            "conditions. Do not use them for emergency decisions."
        )

# ---------------- DATA ANALYTICS ----------------
elif page == "Data Analytics":

    st.title("📊 Data Analytics")

    st.subheader("Risk Category Distribution")

    risk_counts = (
        data["Risk"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
    )

    st.bar_chart(risk_counts)

    st.subheader("Input Feature Statistics")

    st.dataframe(
        data[features].describe(),
        use_container_width=True
    )

    st.subheader("Rainfall vs Water Level")

    st.scatter_chart(
        data,
        x="Rainfall",
        y="Water_Level"
    )

    st.subheader("Dataset Preview")

    st.dataframe(data.head(50), use_container_width=True)

    csv_data = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Dataset (CSV)",
        data=csv_data,
        file_name="drainturn_demo_dataset.csv",
        mime="text/csv"
    )

# ---------------- ABOUT PROJECT ----------------
elif page == "About Project":

    st.title("ℹ️ About DrainTurn AI")

    st.write(
        "DrainTurn AI is a machine-learning prototype designed "
        "to estimate drainage blockage and failure risk."
    )

    st.subheader("Input Parameters")

    st.markdown("""
    - **Rainfall:** Rainfall amount in millimetres.
    - **Water Level:** Illustrative water-level percentage.
    - **Flow Rate:** Illustrative drainage-flow percentage.
    - **Debris Level:** Illustrative debris-level percentage.
    """)

    st.subheader("Technology Stack")

    st.markdown("""
    - Python
    - Streamlit
    - Pandas
    - NumPy
    - Scikit-learn
    - Random Forest Classification
    """)

    st.subheader("Future Improvements")

    st.markdown("""
    - Integration with real drainage sensors.
    - Training on verified historical drainage data.
    - Comparison of multiple ML algorithms.
    - Anomaly detection.
    - Map-based drainage monitoring.
    - Computer vision for debris detection.
    """)

    st.warning(
        "This version uses synthetic demonstration data. "
        "Real-world performance has not been established."
    )

# ---------------- FOOTER ----------------
st.sidebar.markdown("---")
st.sidebar.caption("DrainTurn AI | ML Project Prototype")
