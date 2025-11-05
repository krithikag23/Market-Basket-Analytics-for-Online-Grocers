# ============================================================
# 🧠 Unlocking Growth: Market Basket Analytics Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

# ------------------------------------------------------------
# 🎛️ Sidebar Settings
# ------------------------------------------------------------

st.set_page_config(page_title="Market Basket Analytics", layout="wide")
st.sidebar.title("⚙️ Controls")

st.title("🛒 Market Basket Analytics for Online Grocers")
st.caption("Discover product affinities, customer patterns, and business insights interactively!")

# ------------------------------------------------------------
# 📂 Data Loading Section
# ------------------------------------------------------------

st.sidebar.header("📁 Data Source")

use_colab_output = st.sidebar.checkbox("Use my Colab Model Output", value=True)

if use_colab_output:
    try:
        rules = pd.read_csv("rules_output.csv")
        merged = pd.read_csv("transactions_cleaned.csv")
        st.success("✅ Using Colab model outputs successfully!")
    except Exception as e:
        st.error(f"⚠️ Could not load Colab output files: {e}")
        st.stop()
else:
    uploaded = st.sidebar.file_uploader("Upload merged CSV (optional)", type=["csv"])
    if uploaded is not None:
        merged = pd.read_csv(uploaded)
        st.success("✅ Dataset uploaded successfully!")
    else:
        st.info("No file uploaded — using demo dataset.")
        merged = pd.DataFrame({
            "order_id": np.random.randint(1, 1500, 5000),
            "product_name": np.random.choice(
                ["Lime", "Lemon", "Banana", "Milk", "Bread", "Eggs", "Butter"], 5000),
            "add_to_cart_order": np.random.randint(1, 5, 5000)
        })
        # Generate demo rules
        demo_rules = {
            "antecedents_str": ["Lime", "Milk", "Bread"],
            "consequents_str": ["Lemon", "Bread", "Butter"],
            "support": [0.05, 0.03, 0.04],
            "confidence": [0.2, 0.18, 0.25],
            "lift": [3.1, 2.7, 3.5]
        }
        rules = pd.DataFrame(demo_rules)
        st.warning("⚠️ Demo mode active (random grocery data).")
        

# ------------------------------------------------------------
# 🧮 Clean and Validate Data
# ------------------------------------------------------------
if "antecedents_str" not in rules.columns:
    if "antecedents" in rules.columns:
        rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(eval(x)) if isinstance(x, str) else str(x))
    else:
        st.error("❌ 'antecedents_str' column not found in rules_output.csv.")
        st.stop()

if "consequents_str" not in rules.columns:
    if "consequents" in rules.columns:
        rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(eval(x)) if isinstance(x, str) else str(x))
    else:
        st.error("❌ 'consequents_str' column not found in rules_output.csv.")
        st.stop()

# ------------------------------------------------------------
# 📊 Dashboard Metrics
# ------------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Orders Analyzed", f"{merged['order_id'].nunique():,}")
col2.metric("Products", f"{merged['product_name'].nunique():,}")
col3.metric("Rules Found", f"{rules.shape[0]:,}")

# ------------------------------------------------------------
# 📈 Visualization — Top Rules
# ------------------------------------------------------------
st.subheader("📈 Top Product Affinity Rules")
fig = px.bar(
    rules.head(10),
    x="lift",
    y=rules.head(10)['consequents_str'] + " ⇐ " + rules.head(10)['antecedents_str'],
    orientation='h',
    color='confidence',
    color_continuous_scale='Aggrnyl',
    title="Top 10 Product Association Rules"
)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# 🌀 Support vs Confidence
# ------------------------------------------------------------
st.subheader("🌀 Support vs Confidence Relationship")
fig2 = px.scatter(
    rules,
    x="support",
    y="confidence",
    size="lift",
    color="lift",
    hover_data=['antecedents_str', 'consequents_str'],
    title="Support vs Confidence Plot"
)
st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------
# 📄 Detailed Data Table
# ------------------------------------------------------------
with st.expander("🔍 View All Association Rules"):
    st.dataframe(rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']])

# ------------------------------------------------------------
# ⚡ Real-Time Simulation (Option A + B Combined)
# ------------------------------------------------------------
st.subheader("⚡ Real-Time Simulation")
st.caption("Simulates new orders arriving every second — showing live recommendations and chart updates.")

if st.button("Start Real-Time Simulation"):
    messages = []
    rec_counts = {}  # track how often each product is recommended
    container = st.container()
    chart_placeholder = st.empty()

    for i in range(10):
        new_rule = rules.sample(1)
        antecedent = new_rule['antecedents_str'].iloc[0]
        consequent = new_rule['consequents_str'].iloc[0]

        # Option A: keep all live messages visible
        msg = f"🆕 Order {1500+i}: Bought {antecedent} → Recommended: {consequent}"
        messages.append(msg)
        container.write("\n".join(messages))

        # Option B: live updating chart
        rec_counts[consequent] = rec_counts.get(consequent, 0) + 1
        rec_df = pd.DataFrame({"Product": list(rec_counts.keys()), "Count": list(rec_counts.values())})
        chart = px.bar(rec_df, x="Product", y="Count", title="Live Recommended Product Frequency")
        chart_placeholder.plotly_chart(chart, use_container_width=True)

        time.sleep(1)

    container.success("✅ Simulation Complete! New orders processed in real-time.")

# ------------------------------------------------------------
# 📘 Footer
# ------------------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit — Krithika Ganesan’s BIA Project")
