from __future__ import annotations

import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

from generate_data import generate_all


DATA_DIR = Path("data")


@st.cache_data
def load_data():
    if not (DATA_DIR / "borrowers.csv").exists():
        generate_all(DATA_DIR)

    borrowers = pd.read_csv(DATA_DIR / "borrowers.csv")
    financials = pd.read_csv(DATA_DIR / "monthly_financials.csv")
    relationships = pd.read_csv(DATA_DIR / "relationships.csv")
    economic = pd.read_csv(DATA_DIR / "economic_conditions.csv")
    return borrowers, financials, relationships, economic


def calculate_repayment_ratio(row):
    if row["repayment_due"] in (0, None, np.nan):
        return 0.0
    return row["repayment_amount"] / row["repayment_due"]


def compute_group_stats(financials, borrowers):
    joined = financials.merge(borrowers[["borrower_id", "group_id"]], on="borrower_id", how="left")
    latest = joined.sort_values(["borrower_id", "month"]).groupby("borrower_id").tail(1).copy()
    latest["repayment_ratio"] = latest.apply(calculate_repayment_ratio, axis=1)
    latest["risk_flag"] = (latest["distress_next_3m"] == 1) | (latest["financial_stress_score"] > 0.65)

    stats = latest.groupby("group_id").agg(
        group_average_repayment_ratio=("repayment_ratio", "mean"),
        group_missed_payment_rate=("missed_payment", "mean"),
        group_income_trend=("income_change_pct", "mean"),
        group_distress_rate=("risk_flag", "mean"),
        group_risk_acceleration=("financial_stress_score", "mean"),
        number_of_high_risk_borrowers=("risk_flag", "sum"),
    ).reset_index()
    return stats


def build_graph_features(borrowers, financials, relationships, economic):
    latest = financials.sort_values(["borrower_id", "month"]).groupby("borrower_id").tail(1).copy()
    latest["repayment_ratio"] = latest.apply(calculate_repayment_ratio, axis=1)

    merged = latest.merge(borrowers, on="borrower_id", how="left")
    economic_latest = economic.sort_values(["location_id", "month"]).groupby("location_id").tail(1).reset_index(drop=True)
    merged = merged.merge(economic_latest[["location_id", "market_shock", "weather_shock", "unemployment_rate", "price_inflation", "economic_stress_score"]], on="location_id", how="left")

    # Past distress count for last 3 months per borrower
    history = financials.sort_values(["borrower_id", "month"])
    history["distress_flag"] = (history["missed_payment"] == 1) | (history["days_past_due"] > 30)
    previous_distress = (
        history.groupby("borrower_id")["distress_flag"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    history_for_merge = history.copy()
    history_for_merge["previous_distress_count"] = previous_distress.values
    latest_history = history_for_merge.sort_values(["borrower_id", "month"]).groupby("borrower_id").tail(1)
    merged = merged.merge(latest_history[["borrower_id", "previous_distress_count"]], on="borrower_id", how="left")

    # Add group stats
    group_stats = compute_group_stats(financials, borrowers)
    merged = merged.merge(group_stats, on="group_id", how="left")

    # Graph centrality and neighbor exposure
    G = nx.Graph()
    for row in relationships.itertuples(index=False):
        if row.active == 1:
            G.add_edge(str(row.source_borrower_id), str(row.target_borrower_id), weight=float(row.relationship_strength), relationship_type=row.relationship_type)

    borrower_ids = merged["borrower_id"].astype(str).tolist()
    centrality = nx.degree_centrality(G)
    graph_features = []

    for borrower in borrower_ids:
        neighbors = list(G.neighbors(borrower)) if G.has_node(borrower) else []
        neighbor_df = merged[merged["borrower_id"].astype(str).isin(neighbors)].copy() if neighbors else pd.DataFrame()

        degree = len(neighbors)
        distressed_neighbors = int((neighbor_df["distress_next_3m"] == 1).sum()) if not neighbor_df.empty else 0
        avg_neighbor_distress = float(neighbor_df["distress_next_3m"].mean()) if not neighbor_df.empty else 0.0
        weighted_neighbor_stress = 0.0
        strongest_neighbor_risk = 0.0
        if not neighbor_df.empty:
            for _, nrow in neighbor_df.iterrows():
                edge_weight = G.get_edge_data(borrower, str(nrow["borrower_id"]), default={"weight": 0.0}).get("weight", 0.0)
                weighted_neighbor_stress += float(nrow["financial_stress_score"]) * float(edge_weight)
                strongest_neighbor_risk = max(strongest_neighbor_risk, float(nrow["financial_stress_score"]))

        shared_guarantees = 0
        if neighbors:
            shared_guarantees = int(
                relationships[(relationships["source_borrower_id"].astype(str).isin([borrower])) & (relationships["target_borrower_id"].astype(str).isin(neighbors)) & (relationships["relationship_type"] == "shared_guarantee")].shape[0]
            )

        recent_miss = 0.0
        if not neighbor_df.empty:
            recent_miss = float((neighbor_df["missed_payment"] == 1).mean())

        if not borrower_ids:
            net_centrality = 0.0
        else:
            net_centrality = centrality.get(borrower, 0.0)

        graph_records = {
            "borrower_id": borrower,
            "number_of_connected_borrowers": degree,
            "number_of_distressed_neighbors": distressed_neighbors,
            "average_neighbor_distress": avg_neighbor_distress,
            "weighted_neighbor_stress": weighted_neighbor_stress,
            "strongest_neighbor_risk": strongest_neighbor_risk,
            "number_of_shared_guarantees": shared_guarantees,
            "network_stress_change": max(0.0, avg_neighbor_distress - 0.2),
            "percentage_of_neighbors_with_recent_missed_payments": recent_miss,
            "network_centrality": net_centrality,
        }
        graph_features.append(graph_records)

    graph_df = pd.DataFrame(graph_features)
    merged = merged.merge(graph_df, on="borrower_id", how="left")
    merged["repayment_ratio"] = merged.apply(calculate_repayment_ratio, axis=1)
    merged["previous_distress_count"] = merged["previous_distress_count"].fillna(0)
    merged["local_economic_stress"] = merged["economic_stress_score"].fillna(0.0)
    merged["market_shock"] = merged["market_shock"].fillna(0.0)
    merged["weather_shock"] = merged["weather_shock"].fillna(0.0)
    merged["inflation"] = merged["price_inflation"].fillna(0.0)
    merged["local_unemployment"] = merged["unemployment_rate"].fillna(0.0)
    merged["occupation_level_shock"] = merged["financial_stress_score"].mul(merged["economic_stress_score"]).fillna(0.0)
    merged["target"] = merged["distress_next_3m"].fillna(0)
    return merged


def train_model(feature_df):
    feature_cols = [
        "income",
        "income_change_pct",
        "repayment_ratio",
        "days_past_due",
        "missed_payment",
        "loan_balance",
        "debt_to_income",
        "previous_distress_count",
        "financial_stress_score",
        "number_of_connected_borrowers",
        "number_of_distressed_neighbors",
        "average_neighbor_distress",
        "weighted_neighbor_stress",
        "strongest_neighbor_risk",
        "number_of_shared_guarantees",
        "network_stress_change",
        "percentage_of_neighbors_with_recent_missed_payments",
        "network_centrality",
        "group_average_repayment_ratio",
        "group_missed_payment_rate",
        "group_income_trend",
        "group_distress_rate",
        "group_risk_acceleration",
        "number_of_high_risk_borrowers",
        "local_economic_stress",
        "market_shock",
        "weather_shock",
        "inflation",
        "local_unemployment",
        "occupation_level_shock",
    ]

    X = feature_df[feature_cols].fillna(0.0)
    y = feature_df["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    if xgb is not None:
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=160,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        )
    else:
        model = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=8)

    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    st.session_state["model_auc"] = roc_auc_score(y_test, preds)
    st.session_state["model_accuracy"] = accuracy_score(y_test, (preds >= 0.5).astype(int))
    return model, feature_cols


def compute_risk_components(row):
    individual = float(np.clip(
        0.30 * row["financial_stress_score"]
        + 0.20 * min(row["debt_to_income"], 1.8) / 1.8
        + 0.20 * min(row["days_past_due"] / 45, 1.0)
        + 0.15 * row["missed_payment"]
        + 0.15 * max(0, (1.0 - row["repayment_ratio"]))
        , 0, 1
    ))

    network = float(np.clip(
        0.25 * row["average_neighbor_distress"]
        + 0.25 * row["weighted_neighbor_stress"]
        + 0.20 * row["number_of_distressed_neighbors" ] / max(1, row["number_of_connected_borrowers"])
        + 0.15 * row["percentage_of_neighbors_with_recent_missed_payments"]
        + 0.15 * row["network_centrality"],
        0, 1
    ))

    local = float(np.clip(
        0.35 * row["local_economic_stress"]
        + 0.25 * row["market_shock"]
        + 0.15 * row["weather_shock"]
        + 0.15 * row["local_unemployment"]
        + 0.10 * row["inflation"],
        0, 1
    ))

    total = float(np.clip(0.55 * individual + 0.30 * network + 0.15 * local, 0, 1))
    return individual, network, local, total


def classify_risk(individual, network, local, group_rate):
    if individual > 0.72:
        return "Individual Distress"
    if network > 0.68 and individual < 0.6:
        return "Network Exposed"
    if group_rate > 0.42:
        return "Group-Wide Stress"
    if local > 0.74:
        return "Local Economic Stress"
    return "Low Risk"


def build_simulated_features(selected_borrower, income_decline, missed_payment, stress_boost, feature_df, model, feature_cols):
    sim_df = feature_df.copy()
    idx = sim_df[sim_df["borrower_id"] == selected_borrower].index[0]

    sim_df.loc[idx, "income"] = max(500, sim_df.loc[idx, "income"] * (1 - income_decline / 100))
    sim_df.loc[idx, "income_change_pct"] = min(-12, sim_df.loc[idx, "income_change_pct"] - income_decline)
    sim_df.loc[idx, "financial_stress_score"] = np.clip(sim_df.loc[idx, "financial_stress_score"] + stress_boost, 0, 1)
    sim_df.loc[idx, "missed_payment"] = int(missed_payment)
    sim_df.loc[idx, "days_past_due"] = int(sim_df.loc[idx, "days_past_due"] + (30 if missed_payment else 5))
    sim_df.loc[idx, "repayment_ratio"] = 0.6 if missed_payment else max(0.4, sim_df.loc[idx, "repayment_ratio"] - 0.10)
    sim_df.loc[idx, "group_distress_rate"] = sim_df.loc[idx, "group_distress_rate"] + 0.08
    sim_df.loc[idx, "group_risk_acceleration"] = sim_df.loc[idx, "group_risk_acceleration"] + 0.12

    neighbors = set(
        str(row["source_borrower_id"]) for _, row in pd.read_csv(DATA_DIR / "relationships.csv").iterrows()
        if str(row["target_borrower_id"]) == str(selected_borrower) and int(row["active"]) == 1
    )
    neighbors |= set(
        str(row["target_borrower_id"]) for _, row in pd.read_csv(DATA_DIR / "relationships.csv").iterrows()
        if str(row["source_borrower_id"]) == str(selected_borrower) and int(row["active"]) == 1
    )

    for neighbor in neighbors:
        if neighbor in sim_df["borrower_id"].astype(str).values:
            n_idx = sim_df[sim_df["borrower_id"].astype(str) == neighbor].index[0]
            sim_df.loc[n_idx, "financial_stress_score"] = np.clip(sim_df.loc[n_idx, "financial_stress_score"] + 0.08, 0, 1)
            sim_df.loc[n_idx, "number_of_distressed_neighbors"] = sim_df.loc[n_idx, "number_of_distressed_neighbors"] + 1
            sim_df.loc[n_idx, "average_neighbor_distress"] = np.clip(sim_df.loc[n_idx, "average_neighbor_distress"] + 0.10, 0, 1)

    probs = model.predict_proba(sim_df[feature_cols].fillna(0.0))[:, 1]
    return sim_df, probs


def render_network(feature_df, selected_borrower, probs):
    graph = nx.Graph()
    relationships = pd.read_csv(DATA_DIR / "relationships.csv")
    for _, row in relationships.iterrows():
        if int(row["active"]) == 1:
            graph.add_edge(str(row["source_borrower_id"]), str(row["target_borrower_id"]), weight=float(row["relationship_strength"]))

    risk_map = {}
    for i, row in feature_df.iterrows():
        borrower = str(row["borrower_id"])
        risk = float(probs[feature_df.index.get_loc(i)])
        risk_map[borrower] = risk

    color_lookup = []
    for node in graph.nodes:
        score = risk_map.get(node, 0.0)
        if score < 0.35:
            color = "#2ecc71"
        elif score < 0.6:
            color = "#f1c40f"
        elif score < 0.8:
            color = "#f39c12"
        else:
            color = "#e74c3c"
        color_lookup.append((node, color))

    pos = nx.spring_layout(graph, seed=42, k=0.7)
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    for node in graph.nodes:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br>Risk: {risk_map.get(node, 0.0):.0%}")
        score = risk_map.get(node, 0.0)
        if score < 0.35:
            node_colors.append("#2ecc71")
        elif score < 0.6:
            node_colors.append("#f1c40f")
        elif score < 0.8:
            node_colors.append("#f39c12")
        else:
            node_colors.append("#e74c3c")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="#7f8c8d", width=1), hoverinfo="none"))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text", text=[n for n in graph.nodes], marker=dict(size=18, color=node_colors, line=dict(color="#2c3e50", width=1)), hovertext=node_text, textposition="top center"))
    fig.update_layout(title="KAPUSU Borrower Network", showlegend=False, xaxis_visible=False, yaxis_visible=False, paper_bgcolor="white", plot_bgcolor="white", width=900, height=560)
    return fig, risk_map


def main():
    st.set_page_config(page_title="KAPUSU", page_icon="📈", layout="wide")
    st.title("KAPUSU")
    st.caption("Network-Aware Financial Stress Detection & Early Intervention")

    borrowers, financials, relationships, economic = load_data()
    feature_df = build_graph_features(borrowers, financials, relationships, economic)
    model, feature_cols = train_model(feature_df)

    borrower_options = sorted(feature_df["borrower_id"].astype(str).unique().tolist())
    selected_borrower = st.sidebar.selectbox("Select borrower to simulate", borrower_options, index=borrower_options.index("A") if "A" in borrower_options else 0)
    income_decline = st.sidebar.slider("Income shock (%)", 0, 60, 35)
    missed_payment = st.sidebar.checkbox("Missed repayment", value=True)
    stress_boost = st.sidebar.slider("Financial stress uplift", 0.0, 0.5, 0.22, step=0.01)

    sim_df, probs = build_simulated_features(selected_borrower, income_decline, missed_payment, stress_boost, feature_df, model, feature_cols)
    row = sim_df[sim_df["borrower_id"].astype(str) == str(selected_borrower)].iloc[0]
    probabilities = model.predict_proba(sim_df[feature_cols].fillna(0.0))[:, 1]
    risk_score = float(probabilities[sim_df[sim_df["borrower_id"].astype(str) == str(selected_borrower)].index[0]])

    individual, network, local, total = compute_risk_components(row)
    category = classify_risk(individual, network, local, float(row["group_distress_rate"]))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted risk", f"{risk_score * 100:.0f}%")
    col2.metric("Category", category)
    col3.metric("Network exposure", f"{network * 100:.0f}%")
    col4.metric("Local stress", f"{local * 100:.0f}%")

    fig, risk_map = render_network(sim_df, selected_borrower, probabilities)
    st.plotly_chart(fig, use_container_width=True)

    value_cols = [
        "income",
        "income_change_pct",
        "repayment_ratio",
        "days_past_due",
        "missed_payment",
        "loan_balance",
        "debt_to_income",
        "financial_stress_score",
        "number_of_connected_borrowers",
        "number_of_distressed_neighbors",
        "average_neighbor_distress",
        "weighted_neighbor_stress",
        "strongest_neighbor_risk",
        "group_distress_rate",
        "local_economic_stress",
    ]

    st.subheader(f"Borrower {selected_borrower}: risk decompositions")
    st.markdown(
        "Model-based decomposition. This is an evidence-based risk estimate, not a causal proof that a connected borrower caused the distress."
    )
    component_df = pd.DataFrame(
        {
            "Component": ["Individual Risk", "Network Exposure", "Local Economic Risk", "Total Risk"],
            "Score": [individual, network, local, total],
        }
    )
    st.bar_chart(component_df.set_index("Component") * 100)

    st.write("Risk decomposition details:")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Borrower": selected_borrower,
                    "Predicted Risk": f"{risk_score * 100:.0f}%",
                    "Individual Risk": f"{individual * 100:.0f}%",
                    "Network Exposure": f"{network * 100:.0f}%",
                    "Local Economic Risk": f"{local * 100:.0f}%",
                    "Category": category,
                }
            ]
        ),
        use_container_width=True,
    )

    st.subheader("Borrower profile")
    selected_profile = feature_df[feature_df["borrower_id"].astype(str) == str(selected_borrower)].iloc[0]
    profile_df = pd.DataFrame({
        "Metric": value_cols,
        "Value": [float(selected_profile.get(col, 0.0)) for col in value_cols],
    })
    st.dataframe(profile_df, use_container_width=True)

    st.subheader("Top risk borrowers")
    top_risk = sim_df.copy()
    top_risk["predicted_risk"] = probabilities
    top_risk = top_risk.sort_values("predicted_risk", ascending=False).head(10)
    st.dataframe(top_risk[["borrower_id", "predicted_risk", "financial_stress_score", "group_distress_rate", "local_economic_stress"]], use_container_width=True)

    st.caption(f"Model performance: AUC {st.session_state.get('model_auc', 0.0):.3f}; Accuracy {st.session_state.get('model_accuracy', 0.0):.3f}")


if __name__ == "__main__":
    main()

