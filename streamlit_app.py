import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import io
from datetime import date, datetime
import re
from io import BytesIO
import matplotlib.pyplot as plt
import prince

st.set_page_config(page_title="Amulet Challenge Analysis", layout="wide")
st.title("🪬 Amulet Challenge Analysis")

# ─────────────────────────────────────────
# REFERENCE DATA (hard-coded from R script)
# ─────────────────────────────────────────

modern_sets = pd.DataFrame({
    "set": [
        "SNC","DMU", "BRO", "ONE",
        "MOM", "WOE", "LCI", "LTR",
        "MKM", "OTJ", "MH3", "BLB", "DSK",
        "DFT", "TDM", "FIN", "EOE", "SPM", "TLA",
        "ECL", "TMT", "SOS"
    ],
    "release_date": pd.to_datetime([
        "2022-04-29","2022-09-09", "2022-11-18", "2023-02-10",
        "2023-04-21", "2023-09-08", "2023-11-17", "2023-06-23",
        "2024-02-09", "2024-04-19", "2024-06-14", "2024-08-02", "2024-09-27",
        "2025-02-14", "2025-04-11", "2025-06-13", "2025-08-01", "2025-09-26", "2025-11-21",
        "2026-01-23", "2026-03-06", "2026-04-24"
    ])
}).sort_values("release_date").reset_index(drop=True)

ban_events = pd.DataFrame({
    "event": [
        "Pre-Yorion Ban",
        "Pre-Preordain Unban",
        "Pre-Fury/Bean Ban",
        "Pre-Outburst Ban",
        "Pre-MH3",
        "Pre-Nadu/Grief Ban",
        "Pre-GSZ Unban/Ring Ban",
        "Pre-Breach Ban",
        "Current"
    ],
    "date": pd.to_datetime([
        "2022-10-10","2023-08-07", "2023-12-04", "2024-03-11", "2024-06-14",
        "2024-08-26", "2024-12-16", "2025-03-31", "2026-04-30"
    ])
})

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────


def assign_current_set(d):
    """Return the most recent set released on or before date d."""
    valid = modern_sets[modern_sets["release_date"] <= d]
    if valid.empty:
        return "Unknown"
    return valid.iloc[-1]["set"]


def assign_ban_era(d, events_df):
    """findInterval equivalent: return the era label for date d."""
    idx = (events_df["date"] <= d).sum()
    if idx >= len(events_df):
        idx = len(events_df) - 1
    return events_df.iloc[idx]["event"]


# ─────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────

file_path = "amulet_chal.xlsx"

df = pd.read_excel(file_path, sheet_name="Sheet1")

output = BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="Sheet1")

file_bytes = output.getvalue()

# ─────────────────────────────────────────
# LOAD MAIN SHEET
# ─────────────────────────────────────────

with st.spinner("Processing main deck sheet…"):
    xl = pd.ExcelFile(BytesIO(file_bytes))
    amulet_df = xl.parse(0)

    amulet_df = amulet_df.drop_duplicates(keep="first")

    meta_cols = [c for c in ["Name", "Place", "Date"] if c in amulet_df.columns]
    card_cols = [c for c in amulet_df.columns if c not in meta_cols]

    for col in card_cols:
        amulet_df[col] = pd.to_numeric(amulet_df[col], errors="coerce").fillna(0).astype(int)
    if "Place" in amulet_df.columns:
        amulet_df["Place"] = pd.to_numeric(amulet_df["Place"], errors="coerce").fillna(0).astype(int)
    if "Date" in amulet_df.columns:
        amulet_df["Date"] = pd.to_datetime(amulet_df["Date"]).dt.strftime("%m-%d-%Y")

    env_cols = ["row_number", "Name", "Place", "Date"]
    amulet_env = amulet_df[[c for c in env_cols if c in amulet_df.columns]].copy()
    amulet_int = amulet_df[[c for c in amulet_df.columns if c not in ["Place", "Date", "Name", "row_number"]]].copy()

    amulet_env["current_set"] = amulet_env["Date"].apply(assign_current_set)
    amulet_env["next_ban"]    = amulet_env["Date"].apply(lambda d: assign_ban_era(d, ban_events))

    amulet_comb = pd.concat(
        [amulet_env.drop(columns=["row_number"], errors="ignore").reset_index(drop=True),
         amulet_int.reset_index(drop=True)],
        axis=1
    )

numeric = amulet_int.select_dtypes(include="number")
col_sums = numeric.sum()
keep_cols = col_sums[col_sums > 12].index
amulet_filtered = amulet_int[keep_cols]

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────

tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🃏 Deck Data",
    "📈 Median by Era",
    "🗺️ CCA – Era & Set",
    "🎴 CCA – Card Inclusion",
    "Card Similarity"
])

# ── Tab 2: Deck Data ─────────────────────
with tab2:
    st.subheader("Amulet Deck – Main Sheet")
    st.dataframe(amulet_comb, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Average Card Frequency**")
        means = amulet_int.mean(numeric_only=True).sort_values(ascending=False)
        st.dataframe(means.rename("Mean").reset_index().rename(columns={"index": "Card"}),
                     use_container_width=True)
    with col2:
        st.markdown("**Top 8 Count**")
        name_counts = (
            amulet_df["Name"]
            .value_counts(dropna=False)
            .sort_values(ascending=False)
        )
        st.dataframe(name_counts, use_container_width=True)

# ── Tab 3: Median by Era ─────────────────
with tab3:
    st.subheader("Mean Card Counts by Ban Era")
    num_cols = amulet_comb.select_dtypes(include="number").columns.tolist()
    if "Place" in num_cols:
        num_cols.remove("Place")
    num_cols = [c for c in num_cols if amulet_comb[c].sum() > 25]
    num_cols = [c for c in num_cols if amulet_comb[c].sum() < 1800]
    mean_deck = (
        amulet_comb.groupby("next_ban")[num_cols]
        .mean()
        .reset_index()
    )
    st.markdown("**Heatmap of Mean Counts**")
    heat_data = mean_deck.set_index("next_ban")[num_cols]
    era_order = [
        "Pre-Yorion Ban",
        "Pre-Preordain Unban",
        "Pre-Fury/Bean Ban",
        "Pre-Outburst Ban",
        "Pre-MH3",
        "Pre-Nadu/Grief Ban",
        "Pre-GSZ Unban/Ring Ban",
        "Pre-Breach Ban",
        "Current",
    ]
    heat_data = heat_data.reindex([e for e in era_order if e in heat_data.index])
    fig_heat = px.imshow(
        heat_data,
        aspect="auto",
        color_continuous_scale="Greens",
        labels={"x": "Card", "y": "Era", "color": "Mean"},
        title="Mean Card Counts by Ban Era"
    )
    fig_heat.update_layout(height=750)
    st.plotly_chart(fig_heat, use_container_width=True)

# ─────────────────────────────────────────
# CCA COMPUTATION HELPER
# ─────────────────────────────────────────

ERA_ORDER = [
    "Pre-Yorion Ban",
    "Pre-Preordain Unban",
    "Pre-Fury/Bean Ban",
    "Pre-Outburst Ban",
    "Pre-MH3",
    "Pre-Nadu/Grief Ban",
    "Pre-GSZ Unban/Ring Ban",
    "Pre-Breach Ban",
    "Current",
]


def run_cca_computation():
    """
    Canonical Correspondence Analysis using prince.CA on the species matrix,
    with environmental variables (ban era, set, place) used to build
    centroid overlays via weighted averaging — mirroring vegan::cca() in R.
    """
    with st.spinner("Running Correspondence Analysis (CCA)…"):
        # ── 1. Filter to rows with at least one card ──────────────────────
        X = amulet_int.values.astype(float)
        row_sums = X.sum(axis=1)
        valid_mask = row_sums > 0
        card_df = amulet_int[valid_mask].reset_index(drop=True)
        env_df  = amulet_comb[valid_mask].reset_index(drop=True)

        # ── 2. Keep only cards with > 12 total occurrences ────────────────
        keep = card_df.sum() > 12
        card_df = card_df.loc[:, keep]

        # ── 3. Fit CA (correspondence analysis) ───────────────────────────
        ca = prince.CA(n_components=2, random_state=42)
        ca.fit(card_df)

        site_scores    = ca.row_coordinates(card_df)     # samples
        species_scores = ca.column_coordinates(card_df)  # cards

        site_scores.columns    = ["CA1", "CA2"]
        species_scores.columns = ["CA1", "CA2"]

        # Attach env metadata to site scores
        ord_data = env_df.copy()
        ord_data["CA1"] = site_scores["CA1"].values
        ord_data["CA2"] = site_scores["CA2"].values

        # ── 4. Compute environmental centroids (weighted averages) ────────
        env_centroids = {}
        for env_var in ["next_ban", "current_set"]:
            centroids = (
                ord_data.groupby(env_var)[["CA1", "CA2"]]
                .mean()
                .reset_index()
                .rename(columns={env_var: "label"})
            )
            centroids["env_var"] = env_var
            env_centroids[env_var] = centroids

        # ── 5. Eigenvalue-based inertia explained ─────────────────────────
        eigenvalues = ca.eigenvalues_
        total_inertia = sum(eigenvalues) if eigenvalues is not None else None

        st.session_state["cca_result"]         = ord_data
        st.session_state["cca_species"]        = species_scores.reset_index().rename(columns={"index": "card"})
        st.session_state["cca_env_centroids"]  = env_centroids
        st.session_state["cca_eigenvalues"]    = eigenvalues
        st.session_state["cca_total_inertia"]  = total_inertia

        # ── 6. Excel export ───────────────────────────────────────────────
        cca_output = BytesIO()
        with pd.ExcelWriter(cca_output, engine="openpyxl") as writer:
            export_cols = [c for c in ["Name", "Date", "next_ban", "current_set", "Place", "CA1", "CA2"]
                           if c in ord_data.columns]
            ord_data[export_cols].to_excel(writer, index=False, sheet_name="CCA_Site_Scores")
            species_scores.reset_index().rename(columns={"index": "card"}).to_excel(
                writer, index=False, sheet_name="CCA_Species_Scores"
            )
        st.session_state["cca_excel"] = cca_output.getvalue()


# ── Auto-run CCA on load ──────────────────
if "cca_result" not in st.session_state:
    run_cca_computation()

# ── Tab 4: CCA – Era & Set ────────────────
with tab4:
    st.subheader("CCA Ordination – Era & Set")
 
    if "cca_result" in st.session_state:
        ord_data       = st.session_state["cca_result"]
        species_scores = st.session_state["cca_species"]
        env_centroids  = st.session_state["cca_env_centroids"]
        eigenvalues    = st.session_state.get("cca_eigenvalues")
        total_inertia  = st.session_state.get("cca_total_inertia")
 
        # Inertia metrics
        if eigenvalues is not None and total_inertia:
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("CA1 Inertia", f"{eigenvalues[0]:.4f}")
            col_m2.metric("CA2 Inertia", f"{eigenvalues[1]:.4f}")
            col_m3.metric("Total Inertia", f"{total_inertia:.4f}")
 
        color_by = st.selectbox(
            "Color sites by:",
            ["next_ban", "current_set", "Place"],
            key="cca_color_tab4"
        )
 
        show_centroids = st.checkbox("Show environmental centroids", value=True, key="show_centroids_tab4")
        show_species   = st.checkbox("Show top card vectors", value=False, key="show_species_tab4")
 
        # ── Site scatter ──────────────────────────────────────────────────
        hover_cols = [c for c in ["Name", "Date", "next_ban", "current_set"] if c in ord_data.columns]
        fig = px.scatter(
            ord_data, x="CA1", y="CA2",
            color=color_by,
            hover_data=hover_cols,
            title=f"CCA – sites colored by {color_by}",
            template="plotly_white",
            opacity=0.75
        )
        fig.update_traces(marker=dict(size=8))
 
        # ── Environmental centroid overlay ────────────────────────────────
        if show_centroids and color_by in env_centroids:
            cents = env_centroids[color_by]
            fig.add_trace(go.Scatter(
                x=cents["CA1"], y=cents["CA2"],
                mode="markers+text",
                text=cents["label"],
                textposition="top center",
                marker=dict(size=14, symbol="diamond", color="black", line=dict(width=1, color="white")),
                name=f"{color_by} centroids",
                showlegend=True
            ))
 
        # ── Top card species scores overlay ───────────────────────────────
        if show_species:
            top_n = st.slider("Number of top cards to display", 5, 30, 10, key="cca_top_n")
            sp = species_scores.copy()
            sp["dist"] = np.sqrt(sp["CA1"]**2 + sp["CA2"]**2)
            sp_top = sp.nlargest(top_n, "dist")
            fig.add_trace(go.Scatter(
                x=sp_top["CA1"], y=sp_top["CA2"],
                mode="markers+text",
                text=sp_top["card"],
                textposition="top right",
                marker=dict(size=10, symbol="triangle-up", color="crimson"),
                name="Card scores",
                showlegend=True
            ))
 
        # ── Axis labels with % inertia if available ───────────────────────
        if eigenvalues is not None and total_inertia:
            fig.update_xaxes(title_text=f"CA1 ({eigenvalues[0]/total_inertia*100:.1f}% inertia)")
            fig.update_yaxes(title_text=f"CA2 ({eigenvalues[1]/total_inertia*100:.1f}% inertia)")
 
        fig.update_layout(height=800)
        st.plotly_chart(fig, use_container_width=True)
 
 
        # ── Most Dissimilar Site per Ban Era ─────────────────────────────
        st.markdown("---")
        st.markdown("### 🔀 Most Dissimilar Deck per Ban Era")
        st.caption(
            "Within each ban era, the deck with the highest mean CA distance "
            "to all other decks in that era — i.e. the biggest outlier. "
        )
 
        name_col  = "Name"  if "Name"  in ord_data.columns else None
        date_col  = "Date"  if "Date"  in ord_data.columns else None
        place_col = "Place" if "Place" in ord_data.columns else None
 
        def site_label(idx):
            parts = []
            if name_col: parts.append(str(ord_data.loc[idx, name_col]))
            if date_col: parts.append(f"({ord_data.loc[idx, date_col]})")
            return " ".join(parts) if parts else f"Site {idx}"
 
        def site_place(idx):
            return ord_data.loc[idx, place_col] if place_col else "—"
 
        rows = []
        for era in ERA_ORDER:
            era_idx = ord_data.index[ord_data["next_ban"] == era].tolist()
            if len(era_idx) < 2:
                continue
 
            era_coords = ord_data.loc[era_idx, ["CA1", "CA2"]].values
 
            # For each site, compute mean distance to all others in the era
            best_mean_dist, best_idx = -1, None
            for ii, idx in enumerate(era_idx):
                others = np.delete(era_coords, ii, axis=0)
                dists = np.sqrt(
                    (era_coords[ii, 0] - others[:, 0])**2 +
                    (era_coords[ii, 1] - others[:, 1])**2
                )
                mean_dist = dists.mean()
                if mean_dist > best_mean_dist:
                    best_mean_dist, best_idx = mean_dist, idx
 
            rows.append({
                "Era":              era,
                "Outlier Deck":     site_label(best_idx),
                "Mean CA Distance": f"{best_mean_dist:.4f}",
                "Decks in Era":            len(era_idx),
            })
 
        dissim_df = pd.DataFrame(rows)
        st.dataframe(dissim_df, use_container_width=True, hide_index=True)
 
    else:
        st.info("CCA computation failed. Check your data.")

# ── Tab 5: CCA – Card Inclusion ───────────
with tab5:
    st.subheader("CCA Ordination – Card Inclusion")

    if "cca_result" in st.session_state:
        ord_data       = st.session_state["cca_result"]
        eigenvalues    = st.session_state.get("cca_eigenvalues")
        total_inertia  = st.session_state.get("cca_total_inertia")

        card_options = sorted(amulet_int.columns.tolist())
        selected_card = st.selectbox(
            "Color sites by card count:",
            card_options,
            key="cca_card_select"
        )

        hover_cols = [c for c in ["Name", "Date", "next_ban", "current_set"] if c in ord_data.columns]
        fig2 = px.scatter(
            ord_data, x="CA1", y="CA2",
            color=selected_card if selected_card in ord_data.columns else None,
            color_continuous_scale="thermal",
            hover_data=hover_cols,
            title=f"CCA – sites colored by copies of {selected_card}",
            template="plotly_white",
            opacity=0.8
        )
        fig2.update_traces(marker=dict(size=8))

        if eigenvalues is not None and total_inertia:
            fig2.update_xaxes(title_text=f"CA1 ({eigenvalues[0]/total_inertia*100:.1f}% inertia)")
            fig2.update_yaxes(title_text=f"CA2 ({eigenvalues[1]/total_inertia*100:.1f}% inertia)")

        fig2.update_layout(height=800)
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("CCA computation failed. Check your data.")

# ── Tab 6: Card Similarity ────────────────
with tab6:
    st.subheader("Card Similarity")
    st.title("Maindeck Correspondence Analysis")

    ca1 = prince.CA(n_components=2, random_state=42)
    ca1 = ca1.fit(amulet_filtered)

    species_scores = ca1.column_coordinates(amulet_filtered)
    site_scores    = ca1.row_coordinates(amulet_filtered)

    st.subheader("Card Ordination Plot")

    plot_df = species_scores.copy().reset_index()
    plot_df.columns = ["species", "Dim1", "Dim2"]

    fig = px.scatter(
        plot_df,
        x="Dim1",
        y="Dim2",
        text="species",
        hover_name="species"
    )
    fig.update_traces(mode="text", textposition="top center")
    fig.update_layout(
        title="Interactive Species Ordination",
        xaxis_title="Dim 1",
        yaxis_title="Dim 2",
        template="simple_white"
    )
    st.plotly_chart(fig, use_container_width=True)
