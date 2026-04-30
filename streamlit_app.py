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
        "2022-04-29", "2022-09-09", "2022-11-18", "2023-02-10",
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
        "2022-10-11", "2023-08-07", "2023-12-04", "2024-03-11", "2024-06-14",
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
# CARD TYPE REFERENCE LISTS
# ─────────────────────────────────────────
 
lands = [
    "Academy Ruins","Bojuka Bog","Boros Garrison","Boseiju, Who Endures","Breeding Pool",
    "Castle Garenbrig","Cavern of Souls","Cephalid Coliseum","Commercial District",
    "Crumbling Vestige","Dryad Arbor","Echoing Deeps","Forest","Gemstone Caverns",
    "Ghost Quarter","Golgari Rot Farm","Gruul Turf","Halimar Depths","Hanweir Battlements",
    "Hedge Maze","Island","Kessig Wolf Run","Khalni Garden","Littjara Mirrorlake",
    "Lotus Field","Lush Portico","Mirrorpool","Oran-Rief, the Vastwood","Otawara, Soaring City",
    "Radiant Fountain","Selesnya Sanctuary","Shifting Woodland","Simic Growth Chamber",
    "Slayers Stronghold","Snow-Covered Forest","Stomping Ground","Sunhome, Fortress of the Legion",
    "Sunken Citadel","Takenuma, Abandoned Mire","Temple Garden","The Mycosynth Gardens",
    "Tolaria West","Urzas Cave","Urzas Saga","Valakut, the Molten Pinnacle","Vesuva",
    "Waterlogged Grove","Hall of Storm Giants","Kabira Crossroads","Ketria Triome",
    "Kher Keep","Misty Rainforest","Plains","Port of Karfell","Skyline Cascade"
]
 
creatures = [
    "Aftermath Analyst","Arboreal Grazer","Azusa, Lost but Seeking","Badgermole Cub",
    "Bonny Pall, Clearcutter","Cultivator Colossus","Dryad of the Ilysian Grove",
    "Elesh Norn, Mother of Machines","Famished Worldsire","Fecund Greenshell",
    "Formidable Speaker","Generous Ent","Gretchen Titchwillow","Hydroid Krasis",
    "Icetill Explorer","Insidious Fungus","Lumra, Bellow of the Woods",
    "Phyrexian Metamorph","Primeval Titan","Sakura-Tribe Elder",
    "Sakura-Tribe Scout","Six","Springheart Nantuko","Street Wraith","The Wandering Minstrel",
    "Thragtusk","Altered Ego","Avabruck Caretaker","Blossoming Tortoise","Bonecrusher Giant",
    "Cityscape Leveler","Collector Ouphe","Colossal Skyturtle","Dosan the Falling Leaf",
    "Dragonlord Dromoka","Elder Gargaroth","Elvish Reclaimer","Emrakul, the Aeons Torn",
    "Emrakul, the Promised End","Endurance","Eumidian Terrabotanist","Foundation Breaker",
    "Gaddock Teeg","Hanweir Garrison","Haywire Mite",
    "Hexdrinker","Inferno Titan","Itzquinth, Firstborn of Gishath","Kogla and Yidaro",
    "Kozilek, Butcher of Truth","Kura, the Boundless Sky","Kutzil, Malamet Exemplar",
    "Magus of the Moon","Outland Liberator","Questing Beast","Reclamation Sage",
    "Roxanne, Starfall Savant","Skylasher",
    "Soulless Jailer","Sylvan Safekeeper","Terastodon","The Tarrasque",
    "Thief of Existence","Thornscape Battlemage","Tireless Tracker",
    "Trumpeting Carnosaur",
    "Volatile Stormdrake","Walking Ballista","Wurmcoil Engine",
    "Yasharn, Implacable Earth"
]
 
spells = [
    "Amulet of Vigor","Ancient Stirrings","Bridgeworks Battle","Dismember","Expedition Map",
    "Explore","Fetchland","Green Suns Twilight","Green Suns Zenith","Insidious Fungus",
    "Pact of Negation","Preordain","Relic of Progenitus","Scapeshift","Shadowspear",
    "Smugglers Surprise","Spelunking","Stock Up","Summoners Pact","The One Ring",
    "Turntimber Symbiosis","Vexing Bauble","Aether Spellbomb","Ashiok, Dream Render",
    "Back to Nature","Beast Within","Blast Zone","Boil","Chalice of the Void","Choke",
    "Consign to Memory","Creeping Corrosion","Crush the Weak","Culling Ritual",
    "Cursed Totem","Damping Sphere","Deafening Silence","Defense Grid","Disruptor Flute",
    "Earthquake","Echoing Truth","Engineered Explosives","Ensnaring Bridge","Explore",
    "Fire Magic","Firespout","Force of Vigor","Gaeas Blessing","Ghost Vacuum",
    "Grafdiggers Cage","Hurkyls Recall","Into the Flood Maw","Liquimetal Coating",
    "Lithomantic Barrage","Mana Leak","Mystical Dispute","Null Elemental Blast",
    "Oblivion Stone","Orims Chant","Pick Your Poison","Pithing Needle","Pongify",
    "Propaganda","Pyroclasm","Seal of Primordium","Seal of Removal","Silence",
    "Soul-Guide Lantern","Spell Pierce","Stone of Erech","Storms Wrath","Strix Serenade",
    "Surgical Extraction","Swan Song","Tear Asunder","Test of Talents","The Stone Brain",
    "Tormods Crypt","Trinisphere","Turn the Earth","Unlicensed Hearse","Vampires Vengeance",
    "Veil of Summer","Void Mirror","Worldsouls Rage","Karn, the Great Creator","Malevolent Rumble",
    "Grist, the Hunger Tide","Skysovereign, Consul Flagship","Ugin, the Spirit Dragon",
    "Unidentified Hovership","Wrenn and Six","Yggdrasil, Rebirth Engine"
]

# Normalise to lowercase for matching (strip SB suffix before lookup)
_lands_lower    = {c.lower() for c in lands}
_creatures_lower = {c.lower() for c in creatures}
_spells_lower   = {c.lower() for c in spells}
 
def get_card_type(name):
    """Return card type category, accounting for (SB) suffix."""
    base = str(name).replace(" (SB)", "").replace("(SB)", "").strip().lower()
    if base in _lands_lower:
        return "Land"
    if base in _creatures_lower:
        return "Creature"
    if base in _spells_lower:
        return "Spell"
    return "Unknown"


def sort_by_type(df, card_col):
    """Sort a card dataframe: Creatures → Spells → Lands → Sideboard → Unknown."""
    type_order = {"Creature": 0, "Spell": 1, "Land": 2, "Sideboard": 3, "Unknown": 4}
    df = df.copy()
    df["_type"] = df[card_col].apply(
        lambda s: "Sideboard" if "(SB)" in str(s) else get_card_type(s)
    )
    df["_type_order"] = df["_type"].map(type_order).fillna(4)
    df = df.sort_values(["_type_order", card_col]).drop(columns=["_type", "_type_order"])
    return df


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
# CCA SCORES — loaded from repository file
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


def load_cca_scores():
    """Load pre-computed CCA scores from cca_scores.xlsx in the repository."""
    with st.spinner("Loading CCA scores…"):
        cca_file = "cca_scores.xlsx"
        xl = pd.ExcelFile(cca_file)

        # ── Site scores ───────────────────────────────────────────────────
        ord_data = xl.parse("CCA_Site_Scores")
        if "Date" in ord_data.columns:
            ord_data["Date"] = pd.to_datetime(ord_data["Date"], errors="coerce").dt.strftime("%m-%d-%Y")

        # Merge Place from amulet_comb if not present in file
        if "Place" not in ord_data.columns and "Place" in amulet_comb.columns:
            place_lookup = amulet_comb[["Name", "Date", "Place"]].drop_duplicates()
            ord_data = ord_data.merge(place_lookup, on=["Name", "Date"], how="left")

        # ── Species scores ────────────────────────────────────────────────
        species_scores = xl.parse("CCA_Species_Scores")
        if "card" not in species_scores.columns:
            species_scores = species_scores.rename(columns={species_scores.columns[0]: "card"})

        # ── Environmental centroids (weighted averages of site scores) ────
        env_centroids = {}
        for env_var in ["next_ban", "current_set"]:
            if env_var in ord_data.columns:
                centroids = (
                    ord_data.groupby(env_var)[["CA1", "CA2"]]
                    .mean()
                    .reset_index()
                    .rename(columns={env_var: "label"})
                )
                env_centroids[env_var] = centroids

        st.session_state["cca_result"]        = ord_data
        st.session_state["cca_species"]       = species_scores
        st.session_state["cca_env_centroids"] = env_centroids
        st.session_state["cca_eigenvalues"]   = None  # not stored in file
        st.session_state["cca_total_inertia"] = None

        # ── Provide the file itself as the download ───────────────────────
        with open(cca_file, "rb") as f:
            st.session_state["cca_excel"] = f.read()


# ── Load CCA scores on app start ──────────
if "cca_result" not in st.session_state:
    load_cca_scores()
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
            ["next_ban", "current_set"],
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
 
        # ── Most Dissimilar Site per Ban Era ──────────────────────────── 
        name_col  = "Name"  if "Name"  in ord_data.columns else None
        date_col  = "Date"  if "Date"  in ord_data.columns else None
 
        def site_label(idx):
            parts = []
            if name_col: parts.append(str(ord_data.loc[idx, name_col]))
            if date_col: parts.append(f"({ord_data.loc[idx, date_col]})")
            return " ".join(parts) if parts else f"Site {idx}"
 
 
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
                "Era N":            len(era_idx),
            })
 
        dissim_df = pd.DataFrame(rows)
 
        # Store best_idx per era for expander lookup
        era_best_idx = {}
        for era in ERA_ORDER:
            era_idx = ord_data.index[ord_data["next_ban"] == era].tolist()
            if len(era_idx) < 2:
                continue
            era_coords = ord_data.loc[era_idx, ["CA1", "CA2"]].values
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
            era_best_idx[era] = best_idx
 
        # ── Click-to-expand decklist per era ──────────────────────────────
        st.markdown("#### 🃏 Outlier Decklists")
        for row in dissim_df.itertuples():
            era  = row.Era
            best_idx = era_best_idx.get(era)
            if best_idx is None:
                continue
 
            label = f"{row._2}  —  {era}"  # Outlier Deck column
            with st.expander(label):
                # Pull the full card row from amulet_comb matched by Name+Date
                outlier_name = ord_data.loc[best_idx, "Name"] if "Name" in ord_data.columns else None
                outlier_date = ord_data.loc[best_idx, "Date"] if "Date" in ord_data.columns else None
 
                if outlier_name and outlier_date:
                    match = amulet_comb[
                        (amulet_comb["Name"] == outlier_name) &
                        (amulet_comb["Date"].astype(str).str.contains(outlier_date[:7], na=False))
                    ]
                else:
                    match = pd.DataFrame()
 
                if match.empty:
                    st.info("Decklist not found in source data.")
                else:
                    deck_row = match.iloc[0]
                    card_cols_deck = [c for c in amulet_int.columns if c in deck_row.index]
                    decklist = (
                        pd.Series({c: deck_row[c] for c in card_cols_deck})
                        .astype(int)
                    )
                    decklist = decklist[decklist > 0].sort_values(ascending=False)
 
                    # ── Median decklist for this era ──────────────────────
                    era_rows = amulet_comb[amulet_comb["next_ban"] == era]
                    era_cards = era_rows[[c for c in amulet_int.columns if c in era_rows.columns]]
                    median_deck = era_cards.median().round(2)
                    median_deck = median_deck[median_deck > 0].sort_values(ascending=False)
 
                    # ── Layout: meta | outlier decklist | median decklist ──
                    col_meta, col_outlier, col_median = st.columns([1, 1.5, 1.5])
 
                    with col_meta:
                        st.markdown(f"**{outlier_name}** — {outlier_date}")
                        if "Place" in ord_data.columns:
                            st.markdown(f"Place: **{ord_data.loc[best_idx, 'Place']}**")
                        st.markdown(f"Mean CA Distance: **{row._3}**")
                        st.markdown(f"Era N: **{row._4}**")
 
                    # Cards in outlier not present in median (median == 0)
                    median_cards = set(median_deck[median_deck > 0].index)
 
                    with col_outlier:
                        st.markdown("**Outlier Decklist** *(green = not in era median)*")
                        deck_df = decklist.reset_index()
                        deck_df.columns = ["Card", "Copies"]
                        deck_df = sort_by_type(deck_df, "Card")
 
                        def highlight_unique(row):
                            color = "background-color: #00BA34" if row["Card"] not in median_cards else ""
                            return [color, color]
 
                        styled = deck_df.style.apply(highlight_unique, axis=1)
                        st.dataframe(styled, use_container_width=True, hide_index=True, height=350)
 
                    with col_median:
                        st.markdown(f"**Median Decklist ({era})**")
                        median_df = median_deck.reset_index()
                        median_df.columns = ["Card", "Median Copies"]
                        median_df = sort_by_type(median_df, "Card")
                        st.dataframe(median_df, use_container_width=True, hide_index=True, height=350)
 
    else:
        st.info("CCA computation failed. Check your data.")

# ── Tab 5: CCA – Card Inclusion ───────────
with tab5:
    st.subheader("CCA Ordination – Card Inclusion")

    if "cca_result" in st.session_state:
        ord_data       = st.session_state["cca_result"]
        eigenvalues    = st.session_state.get("cca_eigenvalues")
        total_inertia  = st.session_state.get("cca_total_inertia")

        card_options = amulet_int.sum().sort_values(ascending=False).index.tolist()
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
 
    # ── Filters ──────────────────────────────────────────────────────────
    all_species = species_scores.index.tolist()
    sb_species  = [s for s in all_species if "(SB)" in str(s)]
    mb_species  = [s for s in all_species if "(SB)" not in str(s)]
 
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        sb_filter = st.radio(
            "Show cards:",
            options=["All", "Maindeck only", "Sideboard (SB) only"],
            horizontal=True,
            key="sb_filter"
        )
    with filter_col2:
        color_mode = st.radio(
            "Color by:",
            options=["Card type", "Maindeck / Sideboard"],
            horizontal=True,
            key="color_mode"
        )
 
    if sb_filter == "Maindeck only":
        filtered_species = mb_species
    elif sb_filter == "Sideboard (SB) only":
        filtered_species = sb_species
    else:
        filtered_species = all_species
 
    plot_df = species_scores.loc[filtered_species].copy().reset_index()
    plot_df.columns = ["species", "Dim1", "Dim2"]
    plot_df["card_type"] = plot_df["species"].apply(get_card_type)
    plot_df["deck_slot"] = plot_df["species"].apply(
        lambda s: "Sideboard" if "(SB)" in str(s) else "Maindeck"
    )
 
    if color_mode == "Card type":
        color_col = "card_type"
        color_map = {
            "Land":     "#2ca02c",
            "Creature": "#1f77b4",
            "Spell":    "#ff7f0e",
            "Unknown":  "#7f7f7f",
        }
    else:
        color_col = "deck_slot"
        color_map = {"Maindeck": "#1f77b4", "Sideboard": "#d62728"}
 
    fig = px.scatter(
        plot_df,
        x="Dim1",
        y="Dim2",
        text="species",
        hover_name="species",
        hover_data={"card_type": True, "deck_slot": True, "Dim1": False, "Dim2": False},
        color=color_col,
        color_discrete_map=color_map,
    )
    fig.update_traces(mode="markers+text", textposition="top center", marker=dict(size=8))
    fig.update_layout(
        title="Interactive Species Ordination",
        xaxis_title="Dim 1",
        yaxis_title="Dim 2",
        template="simple_white",
        height=1000
    )
    st.plotly_chart(fig, use_container_width=True)

