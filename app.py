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
# Sidebar Settings
# ------------------------------------------------------------
st.set_page_config(page_title="Market Basket Analytics", layout="wide")
st.sidebar.title("⚙️ Controls")

st.title("🛒 Market Basket Analytics for Online Grocers")
st.caption("Discover product affinities, customer patterns, and business insights interactively!")

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------
uploaded = st.sidebar.file_uploader("Upload merged CSV (optional)", type=["csv"])

if uploaded is not None:
    merged = pd.read_csv(uploaded)
else:
    st.info("No file uploaded — loading demo dataset...")
    merged = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv")  # placeholder
    st.warning("Demo mode active (replace with Instacart subset).")

# ------------------------------------------------------------
# Demo / Mock dataset for local test
if "product_name" not in merged.columns:
    merged = pd.DataFrame({
        "order_id": np.random.randint(1, 1500, 5000),
        "product_name": np.random.choice(
            ["Lime", "Lemon", "Banana", "Milk", "Bread", "Eggs", "Butter"], 5000),
        "add_to_cart_order": np.random.randint(1,5,5000)
    })

# ------------------------------------------------------------
# Basket creation + Apriori
# ------------------------------------------------------------
popular = merged['product_name'].value_counts().head(50).index
merged = merged[merged['product_name'].isin(popular)]

basket = (merged.groupby(['order_id','product_name'])['add_to_cart_order']
          .count().unstack().fillna(0))
basket = basket.astype(bool)

min_support = st.sidebar.slider("Minimum Support", 0.001, 0.05, 0.01, 0.001)

frequent_items = apriori(basket, min_support=min_support, use_colnames=True)
rules = association_rules(frequent_items, metric="lift", min_threshold=1)
rules.sort_values("lift", ascending=False, inplace=True)

# ------------------------------------------------------------
# Dashboard Visuals
# ------------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Orders Analyzed", f"{merged['order_id'].nunique():,}")
col2.metric("Products", f"{merged['product_name'].nunique():,}")
col3.metric("Rules Found", f"{rules.shape[0]:,}")

st.subheader("📊 Top Product Affinity Rules")
fig = px.bar(rules.head(10),
             x="lift",
             y=rules.head(10)['consequents'].astype(str) + " ⇐ " + rules.head(10)['antecedents'].astype(str),
             orientation='h', color='confidence', color_continuous_scale='Aggrnyl')
st.plotly_chart(fig, use_container_width=True)

st.subheader("🌀 Support vs Confidence")
fig2 = px.scatter(rules, x="support", y="confidence", size="lift", color="lift",
                  hover_data=['antecedents','consequents'])
st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------
# Optional: Real-time Simulation
# ------------------------------------------------------------
st.subheader("⚡ Real-Time Simulation (Optional)")
if st.button("Simulate Incoming Orders"):
    placeholder = st.empty()
    for i in range(10):
        new_rule = rules.sample(1)
        placeholder.info(
            f"🆕 Order {1500+i}: Bought {list(new_rule['antecedents'].iloc[0])} "
            f"→ Suggested {list(new_rule['consequents'].iloc[0])}"
        )
        time.sleep(1)
    placeholder.success("✅ Simulation Complete!")

st.caption("Built with ❤️ in Streamlit — Krithika Ganesan’s BIA Project")
