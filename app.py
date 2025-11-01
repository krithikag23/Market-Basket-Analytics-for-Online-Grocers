# ============================================================
# 🧠 Unlocking Growth: Market Basket Analytics Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
import time, random

# ------------------------------------------------------------
# 🎛️ Sidebar Settings
# ------------------------------------------------------------
st.set_page_config(page_title="Market Basket Analytics", layout="wide")
st.sidebar.title("⚙️ Controls")

st.title("🛒 Market Basket Analytics for Online Grocers")
st.caption("Discover product affinities, customer patterns, and business insights interactively!")

# ------------------------------------------------------------
# 📂 Load Dataset
# ------------------------------------------------------------
uploaded = st.sidebar.file_uploader("Upload merged CSV (optional)", type=["csv"])

if uploaded is not None:
    merged = pd.read_csv(uploaded)
else:
    st.info("No file uploaded — loading demo dataset...")
    merged = pd.DataFrame({
        "order_id": np.random.randint(1, 1500, 5000),
        "product_name": np.random.choice(
            ["Lime", "Lemon", "Banana", "Milk", "Bread", "Eggs", "Butter"], 5000),
        "add_to_cart_order": np.random.randint(1,5,5000)
    })
    st.warning("Demo mode active (replace with your Instacart subset).")

# ------------------------------------------------------------
# 🧮 Basket creation + Apriori Algorithm
# ------------------------------------------------------------
# Keep top N products for efficiency
popular = merged['product_name'].value_counts().head(50).index
merged = merged[merged['product_name'].isin(popular)]

# Basket pivot
basket = (merged.groupby(['order_id','product_name'])['add_to_cart_order']
          .count().unstack().fillna(0))
basket = basket.astype(bool)

# User control for min support
min_support = st.sidebar.slider("Minimum Support", 0.001, 0.05, 0.01, 0.001)

# Run Apriori + Rules
frequent_items = apriori(basket, min_support=min_support, use_colnames=True)
rules = association_rules(frequent_items, metric="lift", min_threshold=1)
rules.sort_values("lift", ascending=False, inplace=True)

# ------------------------------------------------------------
# 🧹 Fix: Convert frozensets → strings (for Plotly & Streamlit)
# ------------------------------------------------------------
rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

# ------------------------------------------------------------
# 📊 Dashboard Visuals
# ------------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Orders Analyzed", f"{merged['order_id'].nunique():,}")
col2.metric("Products", f"{merged['product_name'].nunique():,}")
col3.metric("Rules Found", f"{rules.shape[0]:,}")

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
# 🌀 Support vs Confidence (Fixed JSON serialization)
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
# 📄 Optional: Show Data Table
# ------------------------------------------------------------
with st.expander("🔍 View Detailed Association Rules Data"):
    st.dataframe(rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']])

# ------------------------------------------------------------
# ⚡ Real-time Simulation Section
# ------------------------------------------------------------
st.subheader("⚡ Real-Time Simulation (Optional)")
st.caption("Click below to simulate incoming grocery orders and live recommendations:")

if st.button("Simulate Incoming Orders"):
    placeholder = st.empty()
    for i in range(10):
        new_rule = rules.sample(1)
        placeholder.info(
            f"🆕 Order {1500+i}: Bought {new_rule['antecedents_str'].iloc[0]} "
            f"→ Recommended: {new_rule['consequents_str'].iloc[0]}"
        )
        time.sleep(1)
    placeholder.success("✅ Simulation Complete!")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit — Krithika Ganesan’s BIA Project")
