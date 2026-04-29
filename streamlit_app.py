import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
import io
from datetime import date, datetime
import re

st.set_page_config(page_title="Amulet Challenge Analysis", layout="wide")
st.title("🪬 Amulet Challenge Analysis")

# ─────────────────────────────────────────
# REFERENCE DATA (hard-coded from R script)
# ─────────────────────────────────────────

modern_sets = pd.DataFrame({
    "set": [
        "DMU", "BRO", "ONE",
        "MOM", "WOE", "LCI", "LTR",
        "MKM", "OTJ", "MH3", "BLB", "DSK",
        "DFT", "TDM", "FIN", "EOE", "SPM", "TLA",
        "ECL", "TMT", "SOS"
    ],
    "release_date": pd.to_datetime([
        "2022-09-09", "2022-11-18", "2023-02-10",
        "2023-04-21", "2023-09-08", "2023-11-17", "2023-06-23",
        "2024-02-09", "2024-04-19", "2024-06-14", "2024-08-02", "2024-09-27",
        "2025-02-14", "2025-04-11", "2025-06-13", "2025-08-01", "2025-09-26", "2025-11-21",
        "2026-01-23", "2026-03-06", "2026-04-24"
    ])
}).sort_values("release_date").reset_index(drop=True)

ban_events = pd.DataFrame({
    "event": [
        "Pre-Preordain Unban",
        "Pre-Fury/Bean Ban",
        "Pre-Outburst Ban",
        "Pre-Nadu/Grief Ban",
        "Pre-GSZ Unban/Ring Ban",
        "Pre-Breach Ban",
        "Current"
    ],
    "date": pd.to_datetime([
        "2023-08-07", "2023-12-04", "2024-03-11",
        "2024-08-26", "2024-12-16", "2025-03-31", "2026-04-30"
    ])
})

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────

def parse_date(val):
    """Replicate R's multi-format date parsing logic."""
    if pd.isna(val) or str(val).strip() in ("", "Date"):
        return pd.NaT
    s = str(val).strip()
    # Excel serial number
    if re.match(r"^\d+$", s):
        try:
            return pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(s))
        except Exception:
            return pd.NaT
    # dd/mm/yy or mm/dd/yy style
    if re.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", s):
        parts = s.split("/")
        day_part, mid_part = int(parts[0]), int(parts[1])
        if day_part > 12:
            return pd.to_datetime(s, dayfirst=True, errors="coerce")
        if mid_part > 12:
            return pd.to_datetime(s, dayfirst=False, errors="coerce")
        return pd.to_datetime(s, dayfirst=True, errors="coerce")
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


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


def bray_curtis_distance(X):
    """Compute Bray-Curtis dissimilarity matrix."""
    n = len(X)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            num = np.sum(np.abs(X[i] - X[j]))
            den = np.sum(X[i]) + np.sum(X[j])
            d = num / den if den > 0 else 0
            D[i, j] = D[j, i] = d
    return D


# ─────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────

st.sidebar.header("📁 Data")
uploaded_file = st.sidebar.file_uploader("Upload amulet_chal.xlsx", type=["xlsx"])

if uploaded_file is None:
    st.info("👈 Please upload **amulet_chal.xlsx** in the sidebar to begin.")
    st.stop()

# Read into bytes so the buffer can be reused across multiple pd.ExcelFile calls
file_bytes = io.BytesIO(uploaded_file.read())

# ─────────────────────────────────────────
# LOAD MAIN SHEET (sheet 1 = index 0)
# ─────────────────────────────────────────

with st.spinner("Processing main deck sheet…"):
    xl = pd.ExcelFile(file_bytes)
    amulet_df = xl.parse(0)

    # Convert card columns to int, Place to int
    drop_cols = [c for c in ["Amulet of Vigor", "Primeval Titan", "Arboreal Grazer",
                              "Urzas Saga", "Vesuva"] if c in amulet_df.columns]
    amulet_df.drop(columns=drop_cols, inplace=True)

    meta_cols = [c for c in ["Name", "Place", "Date"] if c in amulet_df.columns]
    card_cols = [c for c in amulet_df.columns if c not in meta_cols]

    for col in card_cols:
        amulet_df[col] = pd.to_numeric(amulet_df[col], errors="coerce").fillna(0).astype(int)
    if "Place" in amulet_df.columns:
        amulet_df["Place"] = pd.to_numeric(amulet_df["Place"], errors="coerce").fillna(0).astype(int)

    # Parse dates
    if "Date" in amulet_df.columns:
        amulet_df["Date"] = amulet_df["Date"].apply(parse_date)
        amulet_df.dropna(subset=["Date"], inplace=True)

    amulet_df.reset_index(drop=True, inplace=True)
    amulet_df.insert(0, "row_number", amulet_df.index + 1)

    # Split into env and int
    env_cols = ["row_number", "Name", "Place", "Date"]
    amulet_env = amulet_df[[c for c in env_cols if c in amulet_df.columns]].copy()
    amulet_int = amulet_df[[c for c in amulet_df.columns if c not in ["Place", "Date", "Name", "row_number"]]].copy()

    # Assign set & ban era
    amulet_env["current_set"] = amulet_env["Date"].apply(assign_current_set)
    amulet_env["next_ban"]    = amulet_env["Date"].apply(lambda d: assign_ban_era(d, ban_events))

    # Combine
    amulet_comb = pd.concat(
        [amulet_env.drop(columns=["row_number"], errors="ignore").reset_index(drop=True),
         amulet_int.reset_index(drop=True)],
        axis=1
    )

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────

tab2, tab3, tab4, tab5 = st.tabs([
    "🃏 Deck Data",
    "📈 Median by Era",
    "🗺️ NMDS – Era & Set",
    "🎴 NMDS – Card Inclusion",
])

# ── Tab 2: Deck Data ─────────────────────
with tab2:
    st.subheader("Amulet Deck – Main Sheet")
    st.dataframe(amulet_comb, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Column Means (card counts)**")
        means = amulet_int.mean(numeric_only=True).sort_values(ascending=False)
        st.dataframe(means.rename("Mean").reset_index().rename(columns={"index": "Card"}),
                     use_container_width=True)
    with col2:
        st.markdown("**NA check**")
        na_counts = amulet_df.isna().sum()
        st.write(f"Total NAs: {na_counts.sum()}")
        if na_counts.sum() > 0:
            st.dataframe(na_counts[na_counts > 0], use_container_width=True)
        else:
            st.success("No NAs found ✅")

# ── Tab 3: Median by Era ─────────────────
with tab3:
    st.subheader("Median Card Counts by Ban Era")
    num_cols = amulet_comb.select_dtypes(include="number").columns.tolist()
    if "Place" in num_cols:
        num_cols.remove("Place")
    median_deck = (
        amulet_comb.groupby("next_ban")[num_cols]
        .median()
        .reset_index()
    )
    st.dataframe(median_deck, use_container_width=True)

    # Heatmap
    st.markdown("**Heatmap of Median Counts**")
    heat_data = median_deck.set_index("next_ban")[num_cols]
    fig_heat = px.imshow(
        heat_data,
        aspect="auto",
        color_continuous_scale="Viridis",
        labels={"x": "Card", "y": "Era", "color": "Median"},
        title="Median Card Counts by Ban Era"
    )
    fig_heat.update_layout(height=400)
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Shared NMDS compute helper ────────────
def run_nmds_computation():
    with st.spinner("Computing Bray-Curtis distances and MDS (this may take a moment)…"):
        X = amulet_int.values.astype(float)
        row_sums = X.sum(axis=1)
        valid_mask = row_sums > 0
        X_valid = X[valid_mask]
        D = bray_curtis_distance(X_valid)
        mds = MDS(n_components=2, dissimilarity="precomputed",
                  random_state=42, max_iter=500, n_init=1)
        coords = mds.fit_transform(D)
        ord_data = amulet_comb[valid_mask].copy().reset_index(drop=True)
        ord_data["NMDS1"] = coords[:, 0]
        ord_data["NMDS2"] = coords[:, 1]
        st.session_state["nmds_result"] = ord_data
        st.session_state["stress"] = mds.stress_

def draw_ellipses(fig, ord_data, color_by, show_labels):
    if ord_data[color_by].nunique() <= 15:
        for grp, gdf in ord_data.groupby(color_by):
            if len(gdf) < 3:
                continue
            cx, cy = gdf["NMDS1"].mean(), gdf["NMDS2"].mean()
            sx, sy = gdf["NMDS1"].std(), gdf["NMDS2"].std()
            theta = np.linspace(0, 2 * np.pi, 100)
            ex = cx + 2 * sx * np.cos(theta)
            ey = cy + 2 * sy * np.sin(theta)
            fig.add_trace(go.Scatter(
                x=ex, y=ey, mode="lines",
                name=str(grp) + " ellipse",
                showlegend=False,
                line=dict(dash="dash", width=1)
            ))
            if show_labels:
                fig.add_annotation(x=cx, y=cy, text=str(grp),
                                   showarrow=False, font=dict(size=10))
    return fig

# ── Tab 4: NMDS – Era & Set ───────────────
with tab4:
    st.subheader("NMDS Ordination – Era & Set (Bray-Curtis)")

    if st.button("▶ Run NMDS", type="primary", key="run_nmds_tab4"):
        run_nmds_computation()

    if "nmds_result" in st.session_state:
        ord_data = st.session_state["nmds_result"]
        stress = st.session_state.get("stress")
        if stress is not None:
            st.metric("Stress", f"{stress:.4f}")

        color_by = st.selectbox(
            "Color points by:",
            ["next_ban", "current_set", "Date"],
            key="nmds_color_tab4"
        )
        show_ellipse = st.checkbox("Show group ellipses", value=True, key="ellipse_tab4")
        show_labels  = st.checkbox("Show group centroid labels", value=True, key="labels_tab4")

        fig = px.scatter(
            ord_data, x="NMDS1", y="NMDS2",
            color=color_by,
            hover_data=["Name", "Date"] if "Name" in ord_data.columns else None,
            title=f"NMDS – colored by {color_by}",
            template="plotly_white"
        )
        fig.update_traces(marker=dict(size=8))
        if show_ellipse:
            fig = draw_ellipses(fig, ord_data, color_by, show_labels)
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**NMDS Scores (sorted by NMDS1)**")
        st.dataframe(
            ord_data[["Name", "Date", "next_ban", "current_set", "NMDS1", "NMDS2"]]
            .sort_values("NMDS1"),
            use_container_width=True
        )
    else:
        st.info("Click **Run NMDS** to compute the ordination.")

# ── Tab 5: NMDS – Card Inclusion ──────────
with tab5:
    st.subheader("NMDS Ordination – Card Inclusion (Bray-Curtis)")

    if st.button("▶ Run NMDS", type="primary", key="run_nmds_tab5"):
        run_nmds_computation()

    if "nmds_result" in st.session_state:
        ord_data = st.session_state["nmds_result"]
        stress = st.session_state.get("stress")
        if stress is not None:
            st.metric("Stress", f"{stress:.4f}")

        card_options = sorted(amulet_int.columns.tolist())
        selected_card = st.selectbox(
            "Color points by card count:",
            card_options,
            key="nmds_card_select"
        )

        color_col = ord_data[selected_card] if selected_card in ord_data.columns else None

        fig2 = px.scatter(
            ord_data, x="NMDS1", y="NMDS2",
            color=selected_card,
            color_continuous_scale="Viridis",
            hover_data=["Name", "Date", "next_ban","current_set"] if "Name" in ord_data.columns else None,
            title=f"NMDS – colored by copies of {selected_card}",
            template="plotly_white"
        )
        fig2.update_traces(marker=dict(size=8))
        fig2.update_layout(height=600)
        st.plotly_chart(fig2, use_container_width=True)

        # Summary: mean NMDS position by copy count
        st.markdown(f"**Mean NMDS position by {selected_card} copies**")
        summary = (
            ord_data.groupby(selected_card)[["NMDS1", "NMDS2"]]
            .agg(["mean", "count"])
            .reset_index()
        )
        summary.columns = [selected_card, "NMDS1 mean", "NMDS1 n", "NMDS2 mean", "NMDS2 n"]
        st.dataframe(summary[[selected_card, "NMDS1 mean", "NMDS2 mean", "NMDS1 n"]]
                     .rename(columns={"NMDS1 n": "n"}),
                     use_container_width=True)
    else:
        st.info("Click **Run NMDS** to compute the ordination.")
