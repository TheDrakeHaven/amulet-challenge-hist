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
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from scipy.spatial.distance import cdist

st.set_page_config(page_title="Amulet Challenge Analysis", layout="wide")
st.title("🪬 Amulet Challenge Analysis")

# ─────────────────────────────────────────
# REFERENCE DATA (hard-coded from R script)
# ─────────────────────────────────────────

modern_sets = pd.DataFrame({
    "set": [
        # 2013
        "THS","BNG",

        # 2014
        "JOU","M15","KTK","FRF",

        # 2015
        "DTK","ORI","BFZ","OGW",

        # 2016
        "SOI","EMN","KLD","AER",

        # 2017
        "AKH","HOU","XLN","RIX",

        # 2018
        "DOM","M19","GRN","RNA",

        # 2019
        "WAR","MH1","M20","ELD",

        # 2020
        "THB","IKO","M21","ZNR",

        # 2021
        "KHM","STX","MH2","AFR","MID","VOW",

        # 2022
        "NEO","SNC","DMU","BRO",

        # 2023
        "ONE","MOM","LTR","WOE","LCI",

        # 2024
        "MKM","OTJ","MH3","BLB","DSK",

        # 2025
        "DFT","TDM","FIN","EOE","SPM","TLA",

        # 2026
        "ECL","TMT","SOS"
    ],
    "release_date": pd.to_datetime([
        # 2013
        "2013-09-27","2014-02-07",

        # 2014
        "2014-05-02","2014-07-18","2014-09-26","2015-01-23",

        # 2015
        "2015-03-27","2015-07-17","2015-10-02","2016-01-22",

        # 2016
        "2016-04-08","2016-07-22","2016-09-30","2017-01-20",

        # 2017
        "2017-04-28","2017-07-14","2017-09-29","2018-01-19",

        # 2018
        "2018-04-27","2018-07-13","2018-10-05","2019-01-25",

        # 2019
        "2019-05-03","2019-06-14","2019-07-12","2019-10-04",

        # 2020
        "2020-01-24","2020-05-15","2020-07-03","2020-09-25",

        # 2021
        "2021-02-05","2021-04-23","2021-06-18","2021-07-23","2021-09-24","2021-11-19",

        # 2022
        "2022-02-18","2022-04-29","2022-09-09","2022-11-18",

        # 2023
        "2023-02-10","2023-04-21","2023-06-23","2023-09-08","2023-11-17",

        # 2024
        "2024-02-09","2024-04-19","2024-06-14","2024-08-02","2024-09-27",

        # 2025
        "2025-02-14","2025-04-11","2025-06-13","2025-08-01","2025-09-26","2025-11-21",

        # 2026
        "2026-01-23","2026-03-06","2026-04-24"
    ])
}).sort_values("release_date").reset_index(drop=True)

ban_events = pd.DataFrame({
    "event": [
        # 2013–2015
        "Pre-Splinter Twin/Summer Bloom Ban",

        # 2016–2019
        "Pre-MH1 Release",
        "Pre-Lattice/Oko Ban",

        # 2020–2026
        "Pre-Astrolabe Ban",
        "Pre-Field/Uro Ban",
        "Pre-MH2 Release",
        "Pre-Lurrus Ban",
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
        # 2013–2015 (covers all pre-Twin data back to 2013)
        "2016-01-18",  # Splinter Twin + Summer Bloom ban

        # 2017–2019
        "2019-06-14",  # MH1 release
        "2020-01-13",  # Mox Opal + Oko + Lattice ban

        # 2020–2026
        "2020-07-13",
        "2021-02-15",
        "2021-06-03",
        "2022-03-07",
        "2022-10-11",
        "2023-08-07",
        "2023-12-04",
        "2024-03-11",
        "2024-06-14",
        "2024-08-26",
        "2024-12-16",
        "2025-03-31",
        "2026-04-30"
    ])
})


# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────


def assign_current_set(d):
    """Return the most recent set released on or before date d."""
    ts = pd.to_datetime(d, errors="coerce")
    if pd.isna(ts):
        return "Unknown"
    valid = modern_sets[modern_sets["release_date"] <= ts]
    if valid.empty:
        return "Unknown"
    return valid.iloc[-1]["set"]


def assign_ban_era(d, events_df):
    """findInterval equivalent: return the era label for date d."""
    ts = pd.to_datetime(d, errors="coerce")
    if pd.isna(ts):
        return events_df.iloc[0]["event"]
    idx = (events_df["date"] <= ts).sum()
    if idx >= len(events_df):
        idx = len(events_df) - 1
    return events_df.iloc[idx]["event"]


# ─────────────────────────────────────────
# CARD TYPE REFERENCE LISTS
# ─────────────────────────────────────────

lands = [
    "Academy Ruins",
    "Adventurer's Inn",
    "Aether Hub",
    "Blast Zone",
    "Blooming Marsh",
    "Bojuka Bog",
    "Boros Garrison",
    "Boseiju, Who Endures",
    "Boseiju, Who Shelters All",
    "Botanical Sanctum",
    "Breeding Pool",
    "Bristling Backwoods",
    "Castle Garenbrig",
    "Cavern of Souls",
    "Cephalid Coliseum",
    "City of Brass",
    "Commercial District",
    "Crumbling Vestige",
    "Dimir Aqueduct",
    "Dryad Arbor",
    "Echoing Deeps",
    "Field of Ruin",
    "Field of the Dead",
    "Flagstones of Trokair",
    "Forest",
    "Gemstone Caverns",
    "Gemstone Mine",
    "Ghost Quarter",
    "Glimmerpost",
    "Golgari Rot Farm",
    "Grove of the Burnwillows",
    "Gruul Turf",
    "Halimar Depths",
    "Hall of Storm Giants",
    "Hanweir Battlements",
    "Hedge Maze",
    "Horizon Canopy",
    "Inkmoth Nexus",
    "Ipnu Rivulet",
    "Island",
    "Jetmir's Garden",
    "Kabira Crossroads",
    "Kessig Wolf Run",
    "Ketria Triome",
    "Khalni Garden",
    "Kher Keep",
    "Lair of the Hydra",
    "Littjara Mirrorlake",
    "Lotus Field",
    "Lush Oasis",
    "Lush Portico",
    "Mana Confluence",
    "Mirrorpool",
    "Misty Rainforest",
    "Mortuary Mire",
    "Mountain",
    "Northampton Farm",
    "Oran-Rief, the Vastwood",
    "Otawara, Soaring City",
    "Overgrown Arch",
    "Overgrown Tomb",
    "Pit of Offerings",
    "Plains",
    "Port of Karfell",
    "Prismatic Vista",
    "Radiant Fountain",
    "Razorverge Thicket",
    "Restless Vinestalk",
    "Selesnya Sanctuary",
    "Shifting Woodland",
    "Simic Growth Chamber",
    "Skyline Cascade",
    "Slayers' Stronghold",
    "Snow-Covered Forest",
    "Snow-Covered Island",
    "Stomping Ground",
    "Sunhome, Fortress of the Legion",
    "Sunken Citadel",
    "Sunscorched Desert",
    "Takenuma, Abandoned Mire",
    "Tectonic Edge",
    "Teetering Peaks",
    "Temple Garden",
    "Temple of Mystery",
    "Tendo Ice Bridge",
    "The Mycosynth Gardens",
    "Tolaria West",
    "Treasure Vault",
    "Underground Mortuary",
    "Urza's Cave",
    "Urza's Saga",
    "Valakut, the Molten Pinnacle",
    "Verdant Catacombs",
    "Vesuva",
    "Waterlogged Grove",
    "Windswept Heath",
    "Wooded Foothills",
    "Yavimaya Coast",
    "Yavimaya, Cradle of Growth",
    "Zagoth Triome",
]
 
creatures = [
    "Acidic Slime",
    "Aftermath Analyst",
    "Altered Ego",
    "Arasta of the Endless Web",
    "Arboreal Grazer",
    "Atraxa, Grand Unifier",
    "Avabruck Caretaker",
    "Aven Mindcensor",
    "Azusa, Lost but Seeking",
    "Badgermole Cub",
    "Beanstalk Giant",
    "Blossoming Tortoise",
    "Bonecrusher Giant",
    "Bonny Pall, Clearcutter",
    "Boromir, Warden of the Tower",
    "Brazen Borrower",
    "Bristly Bill, Spine Sower",
    "Carnage Tyrant",
    "Chameleon Colossus",
    "Cityscape Leveler",
    "Collector Ouphe",
    "Colossal Skyturtle",
    "Conduit of Worlds",
    "Courser of Kruphix",
    "Cultivator Colossus",
    "Devoted Druid",
    "Disciple of Freyalise",
    "Dosan the Falling Leaf",
    "Dragonlord Atarka",
    "Dragonlord Dromoka",
    "Drannith Magistrate",
    "Dryad of the Ilysian Grove",
    "Eidolon of Rhetoric",
    "Elder Gargaroth",
    "Elderscale Wurm",
    "Elesh Norn, Mother of Machines",
    "Elvish Reclaimer",
    "Elvish Rejuvenator",
    "Emrakul, the Aeons Torn",
    "Emrakul, the Promised End",
    "Endurance",
    "Eumidian Terrabotanist",
    "Faerie Macabre",
    "Famished Worldsire",
    "Fecund Greenshell",
    "Fierce Empath",
    "Formidable Speaker",
    "Foundation Breaker",
    "Froghemoth",
    "Gaddock Teeg",
    "Gaea's Revenge",
    "Generous Ent",
    "Golos, Tireless Pilgrim",
    "Gretchen Titchwillow",
    "Grist, the Hunger Tide",
    "Hanweir Garrison",
    "Haywire Mite",
    "Hexdrinker",
    "Hornet Nest",
    "Hornet Queen",
    "Huntmaster of the Fells",
    "Hydroid Krasis",
    "Icetill Explorer",
    "Inferno Titan",
    "Insidious Fungus",
    "Itzquinth, Firstborn of Gishath",
    "Keen-Eyed Curator",
    "Klothys, God of Destiny",
    "Kogla and Yidaro",
    "Kor Skyfisher",
    "Kozilek, Butcher of Truth",
    "Kura, the Boundless Sky",
    "Kutzil, Malamet Exemplar",
    "Lotus Cobra",
    "Lumra, Bellow of the Woods",
    "Magebane Lizard",
    "Magus of the Moon",
    "Manglehorn",
    "Meddling Mage",
    "Melira, Sylvok Outcast",
    "Nadu, Winged Wisdom",
    "Nullhide Ferox",
    "Obstinate Baloth",
    "Omnath, Locus of Creation",
    "Oracle of Mul Daya",
    "Oran-Rief Hydra",
    "Orvar, the All-Form",
    "Outland Liberator",
    "Phyrexian Metamorph",
    "Plague Engineer",
    "Primeval Titan",
    "Questing Beast",
    "Ramunap Excavator",
    "Reclamation Sage",
    "Roxanne, Starfall Savant",
    "Ruric Thar, the Unbowed",
    "Sakura-Tribe Elder",
    "Sakura-Tribe Scout",
    "Scavenging Ooze",
    "Sigarda, Heron's Grace",
    "Sigarda, Host of Herons",
    "Simian Spirit Guide",
    "Skyclave Apparition",
    "Skylasher",
    "Skyshroud Ranger",
    "Soulless Jailer",
    "Spellskite",
    "Springheart Nantuko",
    "Stormkeld Vanguard",
    "Street Wraith",
    "Subtlety",
    "Sundering Titan",
    "Sylvan Safekeeper",
    "Tarmogoyf",
    "Terastodon",
    "The Tarrasque",
    "The Wandering Minstrel",
    "Thief of Existence",
    "Thornscape Battlemage",
    "Thragtusk",
    "Thrun, Breaker of Silence",
    "Thrun, the Last Troll",
    "Tireless Tracker",
    "Titan of Industry",
    "Titania, Protector of Argoth",
    "Trinket Mage",
    "Trumpeting Carnosaur",
    "Tyrranax Rex",
    "Ulamog, the Infinite Gyre",
    "Uro, Titan of Nature's Wrath",
    "Vampire Hexmage",
    "Vizier of Remedies",
    "Volatile Stormdrake",
    "Walking Ballista",
    "Wall of Blossoms",
    "Wayward Swordtooth",
    "Woodland Bellower",
    "World Breaker",
    "Worldspine Wurm",
    "Wurmcoil Engine",
    "Yasharn, Implacable Earth",
    "Zacama, Primal Calamity",
]
 
spells = [
    "Abrade",
    "Abrupt Decay",
    "Abundant Harvest",
    "Adventurous Impulse",
    "Aether Gust",
    "Aether Spellbomb",
    "Alpine Moon",
    "Amulet of Vigor",
    "Ancient Grudge",
    "Ancient Stirrings",
    "Anger of the Gods",
    "Anticipate",
    "Ashiok, Dream Render",
    "Assassin's Trophy",
    "Avoid Fate",
    "Back to Nature",
    "Basic Conjuration",
    "Batterskull",
    "Beast Within",
    "Blasphemous Act",
    "Blessed Respite",
    "Boil",
    "Bonfire of the Damned",
    "Bridgeworks Battle",
    "Broken Bond",
    "Broken Wings",
    "Cartographer's Survey",
    "Celestial Purge",
    "Chalice of the Void",
    "Chandra, Awakened Inferno",
    "Choke",
    "Chromatic Lantern",
    "Cleansing Wildfire",
    "Coalition Relic",
    "Consign to Memory",
    "Consulate Crackdown",
    "Containment Breach",
    "Creeping Corrosion",
    "Crucible of Worlds",
    "Crumble to Dust",
    "Crush the Weak",
    "Culling Ritual",
    "Cursed Totem",
    "Damping Matrix",
    "Damping Sphere",
    "Dead // Gone",
    "Deafening Silence",
    "Declaration in Stone",
    "Defense Grid",
    "Deicide",
    "Disdainful Stroke",
    "Dismember",
    "Dispel",
    "Disruptor Flute",
    "Dramatic Entrance",
    "Dream's Grip",
    "Earthquake",
    "Echoing Truth",
    "Eladamri's Call",
    "Elixir of Immortality",
    "Engineered Explosives",
    "Ensnaring Bridge",
    "Environmental Sciences",
    "Escape to the Wilds",
    "Esika's Chariot",
    "Expedition Map",
    "Explore",
    "Fade from History",
    "Feldon's Cane",
    "Field Trip",
    "Finale of Devastation",
    "Fire Magic",
    "Firebolt",
    "Firespout",
    "Flare of Cultivation",
    "Force of Vigor",
    "Fry",
    "Gaea's Blessing",
    "Get Lost",
    "Ghost Vacuum",
    "Grafdigger's Cage",
    "Green Sun's Twilight",
    "Green Sun's Zenith",
    "Guttural Response",
    "Hallowed Moonlight",
    "Hive Mind",
    "Hour of Devastation",
    "Hurkyl's Recall",
    "Into the Flood Maw",
    "Introduction to Annihilation",
    "Invasion of Ikoria",
    "Jace, the Mind Sculptor",
    "Journey of Discovery",
    "Karn, the Great Creator",
    "Kozilek's Return",
    "Lantern of the Lost",
    "Leyline of Sanctity",
    "Leyline of the Void",
    "Life from the Loam",
    "Lightning Bolt",
    "Liquimetal Coating",
    "Lithomantic Barrage",
    "Lotus Bloom",
    "Malevolent Rumble",
    "Mana Leak",
    "Mascot Exhibition",
    "Memoricide",
    "Mycosynth Lattice",
    "Mystic Reflection",
    "Mystic Repeal",
    "Mystical Dispute",
    "Natural State",
    "Nature's Claim",
    "Negate",
    "Nihil Spellbomb",
    "Nissa, Steward of Elements",
    "Nix",
    "Null Elemental Blast",
    "Oath of Nissa",
    "Oblivion Stone",
    "Oko, Thief of Crowns",
    "Once Upon a Time",
    "Pact of Negation",
    "Path to Exile",
    "Pentad Prism",
    "Persist",
    "Pest Summoning",
    "Pick Your Poison",
    "Pithing Needle",
    "Pongify",
    "Preordain",
    "Primal Command",
    "Prismatic Ending",
    "Propaganda",
    "Pyrite Spellbomb",
    "Pyroclasm",
    "Radiant Flames",
    "Rapid Hybridization",
    "Ratchet Bomb",
    "Ravenous Trap",
    "Relic of Progenitus",
    "Remand",
    "Rending Volley",
    "Rest in Peace",
    "Run Afoul",
    "Scapeshift",
    "Seal of Primordium",
    "Seal of Removal",
    "Search for Tomorrow",
    "Serum Visions",
    "Shadowspear",
    "Shatterstorm",
    "Sheoldred's Edict",
    "Shuko",
    "Silence",
    "Six",
    "Skysovereign, Consul Flagship",
    "Slaughter Pact",
    "Sleight of Hand",
    "Smuggler's Surprise",
    "Sorcerous Spyglass",
    "Soul-Guide Lantern",
    "Spell Pierce",
    "Spelunking",
    "Spine of Ish Sah",
    "Stock Up",
    "Stone of Erech",
    "Stony Silence",
    "Storm's Wrath",
    "Strix Serenade",
    "Stubborn Denial",
    "Sudden Shock",
    "Summer Bloom",
    "Summoner's Pact",
    "Summoning Trap",
    "Sunhome Enforcer",
    "Surge of Salvation",
    "Surgical Extraction",
    "Swan Song",
    "Sylvan Scrying",
    "Tamiyo's Safekeeping",
    "Tear Asunder",
    "Test of Talents",
    "The One Ring",
    "The Stone Brain",
    "Thorn of Amethyst",
    "Thoughtseize",
    "Through the Breach",
    "Thundering Falls",
    "Torch Breath",
    "Tormod's Crypt",
    "Torpor Orb",
    "Toxic Deluge",
    "Tragic Arrogance",
    "Trinisphere",
    "Turn the Earth",
    "Turntimber Symbiosis",
    "Ugin, the Spirit Dragon",
    "Unholy Heat",
    "Unidentified Hovership",
    "Unlicensed Hearse",
    "Vampires' Vengeance",
    "Veil of Summer",
    "Vexing Bauble",
    "Void Mirror",
    "Wargate",
    "Weather the Storm",
    "Whipflare",
    "Witchbane Orb",
    "Worldsoul's Rage",
    "Worship",
    "Wrenn and Seven",
    "Wrenn and Six",
    "Yggdrasil, Rebirth Engine",
]

# Normalise to lowercase for matching (strip SB suffix before lookup)
_lands_lower    = {c.lower() for c in lands}
_creatures_lower = {c.lower() for c in creatures}
_spells_lower   = {c.lower() for c in spells}

def get_card_type(name):
    """Return card type category, accounting for sb_ prefix and (SB) suffix."""
    base = str(name)
    if base.startswith("sb_"):
        base = base[3:]
    base = base.replace(" (SB)", "").replace("(SB)", "").strip().lower()
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
        lambda s: "Sideboard" if str(s).startswith("sb_") or "(SB)" in str(s) else get_card_type(s)
    )
    df["_type_order"] = df["_type"].map(type_order).fillna(4)
    df = df.sort_values(["_type_order", card_col]).drop(columns=["_type", "_type_order"])
    return df


def _scryfall_image_url(card_name):
    """Build a Scryfall fuzzy-name image URL, stripping sb_ prefix and (SB) suffix."""
    from urllib.parse import quote
    clean = str(card_name)
    if clean.startswith("sb_"):
        clean = clean[3:]
    clean = clean.replace(" (SB)", "").replace("(SB)", "").strip()
    if " // " in clean:
        clean = clean.split(" // ")[0].strip()
    return f"https://api.scryfall.com/cards/named?fuzzy={quote(clean)}&format=image&version=normal"


def render_decklist_html(
    deck_df,
    card_col,
    copies_col,
    highlight_set=None,
    table_id="deck",
    max_height=350,
):
    """Render a decklist as an HTML table whose rows show the card image on hover.

    Hovering a row pops up the Scryfall card image, fixed-positioned in the top-right
    of the viewport so it isn't clipped by Streamlit's column containers.

    Parameters
    ----------
    highlight_set : set or None
        Cards in this set are NOT highlighted. Cards NOT in this set get the green
        "unique to outlier" background (matches the previous st.dataframe styling).
    table_id : str
        Unique id used to scope CSS so multiple tables on the same page don't collide.
    """
    rows_html = []
    for _, r in deck_df.iterrows():
        card   = str(r[card_col])
        copies = r[copies_col]
        if card.startswith("sb_"):
            display_card = card[3:] + " (SB)"
        else:
            display_card = card
        img_url = _scryfall_image_url(card)
        is_unique = highlight_set is not None and card not in highlight_set
        row_class = "deck-row deck-row-unique" if is_unique else "deck-row"
        rows_html.append(
            f'<tr class="{row_class}">'
            f'<td class="deck-card-cell">{display_card}'
            f'<img class="deck-card-preview" src="{img_url}" loading="lazy" alt="" />'
            f'</td>'
            f'<td class="deck-copies-cell">{copies}</td>'
            f'</tr>'
        )

    html = f"""
    <style>
        #{table_id}-wrap .deck-table {{
            border-collapse: collapse;
            width: 100%;
            font-size: 14px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        #{table_id}-wrap .deck-table th,
        #{table_id}-wrap .deck-table td {{
            padding: 4px 8px;
            border-bottom: 1px solid #eee;
            text-align: left;
        }}
        #{table_id}-wrap .deck-table th {{
            background: #f5f5f5;
            position: sticky;
            top: 0;
            z-index: 1;
        }}
        #{table_id}-wrap .deck-row-unique td {{ background-color: #00BA34; color: #fff; }}
        #{table_id}-wrap .deck-card-cell {{ position: relative; cursor: pointer; }}
        #{table_id}-wrap .deck-card-preview {{
            display: none;
            position: fixed;
            top: 90px;
            right: 30px;
            width: 260px;
            border-radius: 12px;
            box-shadow: 0 6px 24px rgba(0,0,0,0.4);
            z-index: 99999;
            pointer-events: none;
            background: #111;
        }}
        #{table_id}-wrap .deck-row:hover .deck-card-preview {{ display: block; }}
        #{table_id}-wrap .deck-scroll {{
            max-height: {max_height}px;
            overflow-y: auto;
            border: 1px solid #eee;
            border-radius: 4px;
        }}
    </style>
    <div id="{table_id}-wrap">
        <div class="deck-scroll">
            <table class="deck-table">
                <thead>
                    <tr><th>{card_col}</th><th>{copies_col}</th></tr>
                </thead>
                <tbody>{''.join(rows_html)}</tbody>
            </table>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────

# ─────────────────────────────────────────
# LOAD MAIN SHEET
# ─────────────────────────────────────────

with st.spinner("Processing main deck sheet…"):
    amulet_df = pd.read_csv("merged_amulet.csv")

    amulet_df = amulet_df.drop_duplicates(keep="first")
    amulet_df = amulet_df.drop(columns=["Maindeck_Total", "Sideboard_Total"], errors="ignore")

    # Meta columns — flexible, works with or without Place/Event_Type
    _all_meta = ["row_number", "Name", "Place", "Event", "Event_Type", "Date",
                 "Maindeck_Total", "Sideboard_Total"]
    meta_cols = [c for c in _all_meta if c in amulet_df.columns]
    card_cols  = [c for c in amulet_df.columns if c not in meta_cols]

    for col in card_cols:
        amulet_df[col] = pd.to_numeric(amulet_df[col], errors="coerce").fillna(0).astype(int)
    if "Place" in amulet_df.columns:
        amulet_df["Place"] = pd.to_numeric(amulet_df["Place"], errors="coerce").fillna(0).astype(int)
    if "Date" in amulet_df.columns:
        amulet_df["Date"] = pd.to_datetime(amulet_df["Date"], errors="coerce").dt.strftime("%m-%d-%Y")

    _env_candidates = ["row_number", "Name", "Place", "Event", "Event_Type", "Date"]
    env_cols   = [c for c in _env_candidates if c in amulet_df.columns]
    amulet_env = amulet_df[env_cols].copy()
    amulet_int = amulet_df[[c for c in amulet_df.columns if c not in meta_cols]].copy()

    amulet_env["current_set"] = amulet_env["Date"].apply(assign_current_set)
    amulet_env["current_era"] = amulet_env["Date"].apply(lambda d: assign_ban_era(d, ban_events))

    amulet_comb = pd.concat(
        [amulet_env.drop(columns=["row_number"], errors="ignore").reset_index(drop=True),
         amulet_int.reset_index(drop=True)],
        axis=1
    )

_date_min = pd.to_datetime(amulet_env["Date"], errors="coerce").min()
_date_max = pd.to_datetime(amulet_env["Date"], errors="coerce").max()
st.caption(
    f"📊 {len(amulet_df):,} decklists · "
    f"{_date_min.strftime('%b %Y')} → {_date_max.strftime('%b %Y')} · "
    f"{amulet_env['current_era'].nunique()} eras"
)

numeric = amulet_int.select_dtypes(include="number")
amulet_filtered = amulet_int[numeric.columns]

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────

tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "🃏 Deck Data",
    "📈 Median by Era",
    "🗺️ PCA – Era & Set",
    "🎴 PCA – Card Inclusion",
    "🃏 Card Similarity (PCA)",
    "🔍 Era-Specific Cards",
    "🌐 NMDS – Era & Set",
    "🎴 NMDS – Card Inclusion",
])

# ── Tab 2: Deck Data ─────────────────────
with tab2:
    subtab_data, subtab_totals = st.tabs(["📋 All Decks", "🔢 Maindeck Card Totals"])

    with subtab_data:
        st.subheader("Amulet Deck – Main Sheet")
        st.dataframe(amulet_comb, width='stretch')

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Average Card Frequency**")
            means = amulet_int.mean(numeric_only=True).sort_values(ascending=False)
            st.dataframe(means.rename("Mean").reset_index().rename(columns={"index": "Card"}),
                         width='stretch')
        with col2:
            st.markdown("**Top 8 Count**")
            name_counts = (
                amulet_df["Name"]
                .value_counts(dropna=False)
                .sort_values(ascending=False)
            )
            st.dataframe(name_counts, width='stretch')

    with subtab_totals:
        st.subheader("Total Maindeck Card Copies")
        st.markdown(
            "Total copies of each mainboard card across all decklists, "
            "broken down by card type. Sideboard cards excluded."
        )

        # Mainboard columns only (no sb_ prefix)
        mb_cols = [c for c in amulet_int.columns if not c.startswith("sb_")]
        mb_totals = (
            amulet_int[mb_cols]
            .sum()
            .astype(int)
            .reset_index()
            .rename(columns={"index": "Card", 0: "Total Copies"})
        )
        mb_totals["Card Type"] = mb_totals["Card"].apply(get_card_type)
        mb_totals = mb_totals[mb_totals["Total Copies"] > 0].sort_values(
            "Total Copies", ascending=False
        ).reset_index(drop=True)

        # ── Summary metrics ───────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Unique Cards", len(mb_totals))
        m2.metric("Total Copies", f"{mb_totals['Total Copies'].sum():,}")
        m3.metric("Avg per Deck",
                  f"{mb_totals['Total Copies'].sum() / max(len(amulet_df), 1):.1f}")
        m4.metric("Decklists", len(amulet_df))

        # ── Type filter ───────────────────────────────────────────────────
        type_filter = st.multiselect(
            "Filter by card type:",
            options=["Creature", "Spell", "Land", "Unknown"],
            default=[],
            key="mb_totals_type_filter",
        )
        display_totals = (
            mb_totals[mb_totals["Card Type"].isin(type_filter)]
            if type_filter else mb_totals
        )

        # ── Bar chart ─────────────────────────────────────────────────────
        top_n = st.slider("Show top N cards in chart", 10, 60, 30, key="mb_totals_n")
        chart_data = display_totals.head(top_n)
        type_colors = {
            "Land":     "#2ca02c",
            "Creature": "#1f77b4",
            "Spell":    "#ff7f0e",
            "Unknown":  "#7f7f7f",
        }
        fig_bar = px.bar(
            chart_data,
            x="Card",
            y="Total Copies",
            color="Card Type",
            color_discrete_map=type_colors,
            title=f"Top {top_n} Maindeck Cards by Total Copies",
            template="plotly_white",
        )
        fig_bar.update_layout(
            xaxis_tickangle=-45,
            height=500,
            legend_title_text="Card Type",
        )
        st.plotly_chart(fig_bar, width='stretch')

        # ── Full table ────────────────────────────────────────────────────
        st.dataframe(
            display_totals,
            width='stretch',
            hide_index=True,
        )

# ── Tab 3: Median by Era ─────────────────
with tab3:
    st.subheader("Mean Card Counts by Ban Era")
    num_cols = amulet_comb.select_dtypes(include="number").columns.tolist()
    if "Place" in num_cols:
        num_cols.remove("Place")
    num_cols = [c for c in num_cols if amulet_comb[c].sum() > 25]
    num_cols = [c for c in num_cols if amulet_comb[c].sum() < 1800]
    mean_deck = (
        amulet_comb.groupby("current_era")[num_cols]
        .mean()
        .reset_index()
    )
    st.markdown("**Heatmap of Mean Counts**")
    heat_data = mean_deck.set_index("current_era")[num_cols]
    era_order = [
        "Pre-Splinter Twin/Summer Bloom Ban",
        "Pre-MH1 Release",
        "Pre-Lattice/Oko Ban",
        "Pre-Astrolabe Ban",
        "Pre-Field/Uro Ban",
        "Pre-MH2 Release",
        "Pre-Lurrus Ban",
        "Pre-Yorion Ban",
        "Pre-Preordain Unban",
        "Pre-Fury/Bean Ban",
        "Pre-Outburst Ban",
        "Pre-MH3",
        "Pre-Nadu/Grief Ban",
        "Pre-GSZ Unban/Ring Ban",
        "Pre-Breach Ban",
        "Current"
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
    st.plotly_chart(fig_heat, width='stretch')

# ─────────────────────────────────────────
# PCA COMPUTATION
# ─────────────────────────────────────────

ERA_ORDER = [
        "Pre-Splinter Twin/Summer Bloom Ban",
        "Pre-MH1 Release",
        "Pre-Lattice/Oko Ban",
        "Pre-Astrolabe Ban",
        "Pre-Field/Uro Ban",
        "Pre-MH2 Release",
        "Pre-Lurrus Ban",
        "Pre-Yorion Ban",
        "Pre-Preordain Unban",
        "Pre-Fury/Bean Ban",
        "Pre-Outburst Ban",
        "Pre-MH3",
        "Pre-Nadu/Grief Ban",
        "Pre-GSZ Unban/Ring Ban",
        "Pre-Breach Ban",
        "Current"
]


def run_pca_computation():
    """Run PCA via sklearn and store results in session_state."""
    with st.spinner("Running PCA…"):
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(amulet_filtered.values.astype(float))

            pca = PCA(n_components=2, random_state=42)
            site_arr = pca.fit_transform(X_scaled)

            site_coords = pd.DataFrame(site_arr, columns=["CA1", "CA2"])
            ord_data = pd.concat(
                [amulet_env.drop(columns=["row_number"], errors="ignore").reset_index(drop=True),
                 site_coords],
                axis=1
            )
            if "Date" in ord_data.columns:
                ord_data["Date"] = pd.to_datetime(
                    ord_data["Date"], errors="coerce"
                ).dt.strftime("%m-%d-%Y")

            loadings = pd.DataFrame(
                pca.components_.T,
                index=amulet_filtered.columns,
                columns=["CA1", "CA2"]
            ).reset_index()
            loadings.rename(columns={"index": "card"}, inplace=True)

            env_centroids = {}
            for env_var in ["current_era", "current_set"]:
                if env_var in ord_data.columns:
                    centroids = (
                        ord_data.groupby(env_var)[["CA1", "CA2"]]
                        .mean()
                        .reset_index()
                        .rename(columns={env_var: "label"})
                    )
                    env_centroids[env_var] = centroids

            var_ratio     = pca.explained_variance_ratio_
            total_inertia = float(var_ratio.sum())

            st.session_state["cca_result"]        = ord_data
            st.session_state["cca_species"]       = loadings
            st.session_state["cca_env_centroids"] = env_centroids
            st.session_state["cca_eigenvalues"]   = var_ratio
            st.session_state["cca_total_inertia"] = total_inertia

            excel_buf = BytesIO()
            with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                ord_data.to_excel(writer, sheet_name="PCA_Site_Scores",   index=False)
                loadings.to_excel(writer, sheet_name="PCA_Card_Loadings", index=False)
            st.session_state["cca_excel"] = excel_buf.getvalue()

        except Exception as e:
            st.error(f"PCA computation failed: {e}")
def run_nmds_computation():
    """Run NMDS (non-metric MDS on Bray-Curtis dissimilarity) and cache in session_state."""
    with st.spinner("Running NMDS (Bray-Curtis, non-metric, 5 random starts)…"):
        try:
            X = amulet_filtered.values.astype(float)

            # Bray-Curtis dissimilarity
            bc_dist = cdist(X, X, metric="braycurtis")
            bc_dist = np.nan_to_num(bc_dist, nan=0.0)

            # Non-metric MDS — fixed seed for reproducibility
            # n_init=5 runs 5 internal starts; random_state=42 makes it deterministic
            mds = MDS(
                n_components=2, metric_mds=False, dissimilarity="precomputed",
                random_state=123, n_init=20, max_iter=5000,
                normalized_stress=True, eps=1e-6, init="random",
            )
            coords = mds.fit_transform(bc_dist)
            stress  = float(mds.stress_)

            site_coords = pd.DataFrame(coords, columns=["NMDS1", "NMDS2"])
            ord_data = pd.concat(
                [amulet_env.drop(columns=["row_number"], errors="ignore").reset_index(drop=True),
                 site_coords], axis=1
            )
            if "Date" in ord_data.columns:
                ord_data["Date"] = (
                    pd.to_datetime(ord_data["Date"], errors="coerce")
                    .dt.strftime("%m-%d-%Y")
                )

            # WA species scores
            col_totals = X.sum(axis=0)
            col_totals[col_totals == 0] = 1
            species = pd.DataFrame({
                "card":  amulet_filtered.columns,
                "NMDS1": (X * coords[:, 0:1]).sum(axis=0) / col_totals,
                "NMDS2": (X * coords[:, 1:2]).sum(axis=0) / col_totals,
            })

            # Environmental centroids
            env_centroids = {}
            for env_var in ["current_era", "current_set"]:
                if env_var in ord_data.columns:
                    env_centroids[env_var] = (
                        ord_data.groupby(env_var)[["NMDS1", "NMDS2"]]
                        .mean().reset_index().rename(columns={env_var: "label"})
                    )

            st.session_state["nmds_result"]        = ord_data
            st.session_state["nmds_species"]       = species
            st.session_state["nmds_env_centroids"] = env_centroids
            st.session_state["nmds_stress"]        = stress
            st.session_state["nmds_dist"]          = bc_dist

        except Exception as e:
            st.error(f"NMDS computation failed: {e}")

# ── NMDS data resolution (GitHub → session_state → compute) ──────────────────
GITHUB_NMDS_URL = (
    "https://raw.githubusercontent.com/"
    "TheDrakeHaven/amulet-challenge-hist/main/nmds_results.xlsx"
)

def _load_nmds_excel(source):
    """Load Site_Scores + Card_WA_Scores from an Excel file-like or URL bytes."""
    xls = pd.ExcelFile(source)
    od  = pd.read_excel(xls, sheet_name="Site_Scores")
    sp  = pd.read_excel(xls, sheet_name="Card_WA_Scores")
    st_val = None
    if "Metadata" in xls.sheet_names:
        try:
            meta = pd.read_excel(xls, sheet_name="Metadata")
            if "stress" in meta.columns and len(meta) > 0:
                v = meta["stress"].iloc[0]
                if pd.notna(v):
                    st_val = float(v)
        except Exception:
            pass
    cents = {}
    for ev in ["current_era", "current_set"]:
        if ev in od.columns:
            try:
                cents[ev] = (
                    od.groupby(ev)[["NMDS1", "NMDS2"]]
                    .mean().reset_index().rename(columns={ev: "label"})
                )
            except Exception:
                pass
    return od, sp, st_val, cents

def _resolve_nmds():
    """
    Return (ord_nmds, species_nmds, stress_nmds, env_centroids_nmds) from the
    best available source: uploaded file → GitHub cache → computed.
    Returns (None, None, None, {}) if nothing is available.
    """
    # 1. Already fetched from GitHub this session
    gh = st.session_state.get("nmds_github")
    if gh is not None:
        return gh["ord"], gh["species"], gh["stress"], gh["centroids"]

    # 2. Already computed this session
    if "nmds_result" in st.session_state:
        return (
            st.session_state["nmds_result"],
            st.session_state["nmds_species"],
            st.session_state["nmds_stress"],
            st.session_state["nmds_env_centroids"],
        )

    # 3. Try GitHub
    try:
        import requests as _req
        resp = _req.get(GITHUB_NMDS_URL, timeout=15)
        resp.raise_for_status()
        from io import BytesIO as _BIO
        od, sp, st_val, cents = _load_nmds_excel(_BIO(resp.content))
        st.session_state["nmds_github"] = {
            "ord": od, "species": sp, "stress": st_val, "centroids": cents
        }
        return od, sp, st_val, cents
    except Exception:
        pass

    return None, None, None, {}

# ── Run PCA on app start (fast — no NMDS here, tab 8 handles it lazily) ──
if "cca_result" not in st.session_state:
    run_pca_computation()
# ── Tab 4: CCA – Era & Set ────────────────
with tab4:
    st.subheader("PCA Ordination – Era & Set")

    if "cca_result" in st.session_state:
        ord_data       = st.session_state["cca_result"]
        species_scores = st.session_state["cca_species"]
        env_centroids  = st.session_state["cca_env_centroids"]
        eigenvalues    = st.session_state.get("cca_eigenvalues")
        total_inertia  = st.session_state.get("cca_total_inertia")

        # Inertia metrics
        if eigenvalues is not None and total_inertia:
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("PC1 Variance", f"{eigenvalues[0]:.4f}")
            col_m2.metric("PC2 Variance", f"{eigenvalues[1]:.4f}")
            col_m3.metric("Total Variance", f"{total_inertia:.4f}")

        color_by = st.selectbox(
            "Color sites by:",
            ["current_era", "current_set"],
            key="cca_color_tab4"
        )

        show_centroids = st.checkbox("Show environmental centroids", value=True, key="show_centroids_tab4")
        show_species   = st.checkbox("Show top card vectors", value=False, key="show_species_tab4")

        # ── Site scatter ──────────────────────────────────────────────────
        hover_cols = [c for c in ["Name", "Date", "current_era", "current_set"] if c in ord_data.columns]
        fig = px.scatter(
            ord_data, x="CA1", y="CA2",
            color=color_by,
            hover_data=hover_cols,
            title=f"PCA – sites colored by {color_by}",
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
            fig.update_xaxes(title_text=f"PC1 ({eigenvalues[0]*100:.1f}% variance)")
            fig.update_yaxes(title_text=f"PC2 ({eigenvalues[1]*100:.1f}% variance)")

        fig.update_layout(height=800)
        st.plotly_chart(fig, width='stretch')

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
            era_idx = ord_data.index[ord_data["current_era"] == era].tolist()
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
            era_idx = ord_data.index[ord_data["current_era"] == era].tolist()
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
        st.markdown("#### 🃏 Outlier Decklists By Era")
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
                    era_rows = amulet_comb[amulet_comb["current_era"] == era]
                    era_cards = era_rows[[c for c in amulet_int.columns if c in era_rows.columns]]
                    median_deck = era_cards.median().round(2)
                    median_deck = median_deck[median_deck > 0].sort_values(ascending=False)

                    # ── Layout: meta | outlier decklist | median decklist ──
                    col_meta, col_outlier, col_median = st.columns([1, 1.5, 1.5])

                    with col_meta:
                        st.markdown(f"**{outlier_name}** — {outlier_date}")
                        if "Place" in ord_data.columns:
                            st.markdown(f"Place: **{ord_data.loc[best_idx, 'Place']}**")
                        st.markdown(f"Mean PC Distance: **{row._3}**")
                        st.markdown(f"Era N: **{row._4}**")

                    # Cards in outlier not present in median (median == 0)
                    median_cards = set(median_deck[median_deck > 0].index)

                    # Sanitise era string for use in CSS ids
                    era_id = re.sub(r"[^a-zA-Z0-9]+", "-", era).strip("-").lower() or "era"

                    with col_outlier:
                        st.markdown("**Outlier Decklist** *(hover a row to see the card · green = not in era median)*")
                        deck_df = decklist.reset_index()
                        deck_df.columns = ["Card", "Copies"]
                        deck_df = sort_by_type(deck_df, "Card")
                        render_decklist_html(
                            deck_df,
                            card_col="Card",
                            copies_col="Copies",
                            highlight_set=median_cards,
                            table_id=f"outlier-{era_id}",
                            max_height=350,
                        )

                    with col_median:
                        st.markdown(f"**Median Decklist ({era})** *(hover a row to see the card)*")
                        median_df = median_deck.reset_index()
                        median_df.columns = ["Card", "Median Copies"]
                        median_df = sort_by_type(median_df, "Card")
                        render_decklist_html(
                            median_df,
                            card_col="Card",
                            copies_col="Median Copies",
                            highlight_set=None,
                            table_id=f"median-{era_id}",
                            max_height=350,
                        )

    else:
        st.info("PCA computation failed. Check your data.")

# ── Tab 5: CCA – Card Inclusion ───────────
with tab5:
    st.subheader("PCA Ordination – Card Inclusion")

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

        # ── Merge the selected card's copies into ord_data ────────────────
        # The CCA scores file doesn't carry per-card columns, so we pull the
        # selected card's count out of amulet_comb and join it onto the site
        # scores by (Name, Date) before plotting.
        plot_data = ord_data.copy()
        if selected_card in amulet_comb.columns:
            merge_keys = [k for k in ["Name", "Date"]
                          if k in plot_data.columns and k in amulet_comb.columns]
            if merge_keys:
                card_lookup = (
                    amulet_comb[merge_keys + [selected_card]]
                    .drop_duplicates(subset=merge_keys)
                )
                # Drop any pre-existing column with the same name to avoid
                # _x / _y suffix collisions on merge.
                plot_data = plot_data.drop(columns=[selected_card], errors="ignore")
                plot_data = plot_data.merge(card_lookup, on=merge_keys, how="left")
                plot_data[selected_card] = (
                    pd.to_numeric(plot_data[selected_card], errors="coerce")
                      .fillna(0)
                      .astype(int)
                )

        hover_cols = [c for c in ["Name", "Date", "current_era", "current_set"]
                      if c in plot_data.columns]
        if selected_card in plot_data.columns and selected_card not in hover_cols:
            hover_cols.append(selected_card)

        fig2 = px.scatter(
            plot_data, x="CA1", y="CA2",
            color=selected_card if selected_card in plot_data.columns else None,
            color_continuous_scale="thermal",
            hover_data=hover_cols,
            title=f"PCA – sites colored by copies of {selected_card}",
            template="plotly_white",
            opacity=0.8
        )
        fig2.update_traces(marker=dict(size=8))

        if eigenvalues is not None and total_inertia:
            fig2.update_xaxes(title_text=f"PC1 ({eigenvalues[0]*100:.1f}% variance)")
            fig2.update_yaxes(title_text=f"PC2 ({eigenvalues[1]*100:.1f}% variance)")

        fig2.update_layout(height=800)
        st.plotly_chart(fig2, width='stretch')

    else:
        st.info("PCA computation failed. Check your data.")


# ── Tab 6: Card Similarity ────────────────
with tab6:
    st.subheader("Card Similarity")
    st.title("Maindeck PCA – Card Loadings")

    _scaler6 = StandardScaler()
    _X6      = _scaler6.fit_transform(amulet_filtered.values.astype(float))
    _pca6    = PCA(n_components=2, random_state=42)
    _pca6.fit(_X6)
    _var6    = _pca6.explained_variance_ratio_

    species_scores = pd.DataFrame(
        _pca6.components_.T,
        index=amulet_filtered.columns,
        columns=["Dim1", "Dim2"]
    )

    st.subheader("Card Loadings Plot")

    # Maindeck only, cards with ≥30 total copies
    card_totals_mb = amulet_comb[
        [c for c in amulet_filtered.columns if not c.startswith("sb_")]
    ].sum(axis=0)
    mb_species = [
        s for s in species_scores.index.tolist()
        if not str(s).startswith("sb_") and "(SB)" not in str(s)
        and card_totals_mb.get(s, 0) >= 30
    ]

    color_mode = st.radio(
        "Color by:",
        options=["Card type", "Maindeck / Sideboard"],
        horizontal=True,
        key="color_mode"
    )

    filtered_species = mb_species

    plot_df = species_scores.loc[filtered_species].copy().reset_index()
    plot_df.columns = ["species", "Dim1", "Dim2"]
    plot_df["card_type"] = plot_df["species"].apply(get_card_type)
    plot_df["deck_slot"] = plot_df["species"].apply(
        lambda s: "Sideboard" if str(s).startswith("sb_") or "(SB)" in str(s) else "Maindeck"
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

    plot_df["_color"] = plot_df["card_type"] if color_mode == "Card type" else plot_df["deck_slot"]

    fig = px.scatter(
        plot_df,
        x="Dim1",
        y="Dim2",
        text="species",
        hover_name="species",
        hover_data={"card_type": True, "deck_slot": True, "Dim1": False, "Dim2": False,
                    "_color": False},
        color="_color",
        color_discrete_map=color_map,
        category_orders={"_color": list(color_map.keys())},
    )
    fig.update_traces(mode="markers+text", textposition="top center", marker=dict(size=8))
    fig.update_layout(
        title="Card Loadings (PCA)",
        xaxis_title=f"PC1 ({_var6[0]*100:.1f}% variance)",
        yaxis_title=f"PC2 ({_var6[1]*100:.1f}% variance)",
        template="simple_white",
        height=1000,
        legend_title_text="Card Type" if color_mode == "Card type" else "Deck Slot",
    )
    st.plotly_chart(fig, width='stretch')

# ── Tab 7: Era-Specific Cards ─────────────
with tab7:
    st.subheader("Era-Specific Cards")
    st.markdown(
        "Cards that appear predominantly in one ban era. "
        "Concentration score = share of a card's total appearances that fall within a single era. "
        "Use the slider to control the minimum number of appearances required."
    )

    era_order_display = [e for e in ERA_ORDER if e in amulet_comb["current_era"].values]
    card_cols_era = [c for c in amulet_int.columns]

    # Build era × card count matrix (raw) and era deck sizes
    era_card_raw = (
        amulet_comb.groupby("current_era")[card_cols_era]
        .sum()
        .reindex(era_order_display)
        .fillna(0)
    )
    era_sizes = (
        amulet_comb.groupby("current_era")
        .size()
        .reindex(era_order_display)
        .fillna(1)  # avoid division by zero
    )

    # Normalize: mean copies per deck in each era
    era_card = era_card_raw.div(era_sizes, axis=0)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        min_appearances = st.slider(
            "Minimum total raw appearances across all eras",
            min_value=1, max_value=50, value=5, step=1,
            key="era_specific_min"
        )
    with col_s2:
        min_concentration = st.slider(
            "Minimum concentration score (0–1)",
            min_value=0.50, max_value=1.0, value=0.75, step=0.05,
            key="era_specific_conc"
        )

    # Eligibility still based on raw counts so rare cards are filtered correctly
    card_totals_raw = era_card_raw.sum(axis=0)
    eligible_cards = card_totals_raw[card_totals_raw >= min_appearances].index.tolist()

    rows = []
    for card in eligible_cards:
        col_data = era_card[card]          # normalized (per-deck rate)
        total = col_data.sum()
        if total == 0:
            continue
        dominant_era = col_data.idxmax()
        dominant_rate = col_data.max()
        concentration = dominant_rate / total
        if concentration >= min_concentration:
            rows.append({
                "Card":                  card,
                "Card Type":             get_card_type(card),
                "Dominant Era":          dominant_era,
                "Era Size (decks)":      int(era_sizes[dominant_era]),
                "Rate in Era":           round(dominant_rate, 3),
                "Raw Era Appearances":   int(era_card_raw.loc[dominant_era, card]),
                "Total Raw Appearances": int(card_totals_raw[card]),
                "Concentration":         round(concentration, 3),
            })

    if not rows:
        st.info("No cards meet the current filters. Try lowering the thresholds.")
    else:
        result_df = (
            pd.DataFrame(rows)
            .sort_values(["Dominant Era", "Concentration"], ascending=[True, False])
            .reset_index(drop=True)
        )

        # ── Summary chart ─────────────────────────────────────────────────
        era_counts = result_df["Dominant Era"].value_counts().reindex(era_order_display).dropna()
        fig_bar = px.bar(
            era_counts.reset_index(),
            x="Dominant Era",
            y="count",
            title="Number of Era-Specific Cards per Era",
            labels={"count": "# Cards", "Dominant Era": "Era"},
            color="Dominant Era",
            template="plotly_white",
        )
        fig_bar.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_bar, width='stretch')

        # ── Filter by era ──────────────────────────────────────────────────
        era_filter = st.multiselect(
            "Filter by era (leave blank for all):",
            options=era_order_display,
            default=[],
            key="era_specific_filter"
        )
        if era_filter:
            result_df = result_df[result_df["Dominant Era"].isin(era_filter)]

        # ── Type filter ───────────────────────────────────────────────────
        type_filter = st.multiselect(
            "Filter by card type:",
            options=["Creature", "Spell", "Land", "Unknown"],
            default=[],
            key="era_specific_type"
        )
        if type_filter:
            result_df = result_df[result_df["Card Type"].isin(type_filter)]

        st.dataframe(result_df, width='stretch', hide_index=True)

        # ── Heatmap of era-specific cards ─────────────────────────────────
        st.markdown("**Heatmap of Era-Specific Cards** *(color = mean copies per deck)*")
        heat_cards = result_df["Card"].tolist()
        if heat_cards:
            heat = era_card[heat_cards].T  # normalized rates
            heat.index.name = "Card"
            fig_heat = px.imshow(
                heat,
                aspect="auto",
                color_continuous_scale="Greens",
                labels={"x": "Era", "y": "Card", "color": "Copies/Deck"},
                title="Era-Specific Cards — Mean Copies per Deck by Era",
            )
            fig_heat.update_layout(height=max(400, len(heat_cards) * 18))
            st.plotly_chart(fig_heat, width='stretch')


# ── Tab 8: NMDS – Era & Set ───────────────
with tab8:
    st.subheader("NMDS – Era & Set")
    st.markdown(
        "Sites (decklists) in **non-metric MDS** space (Bray-Curtis dissimilarity). "
        "Axes have no ecological units — only relative distances matter."
    )

    ord_nmds, species_nmds, stress_nmds, env_cents_nmds = _resolve_nmds()

    if ord_nmds is not None:
        # Ensure current_era assigned from amulet_env if missing
        if "current_era" not in ord_nmds.columns and "Date" in ord_nmds.columns:
            ord_nmds = ord_nmds.copy()
            ord_nmds["current_era"] = ord_nmds["Date"].apply(
                lambda d: assign_ban_era(d, ban_events)
            )
            env_cents_nmds["current_era"] = (
                ord_nmds.groupby("current_era")[["NMDS1", "NMDS2"]]
                .mean().reset_index().rename(columns={"current_era": "label"})
            )

        stress_label = f"  |  stress = {stress_nmds:.4f}" if stress_nmds is not None else ""

        col_n1, col_n2 = st.columns(2)
        with col_n1:
            color_by_nmds = st.selectbox(
                "Color sites by:",
                [c for c in ["current_era", "current_set"] if c in ord_nmds.columns],
                key="nmds_color_tab8"
            )
        with col_n2:
            show_centroids_nmds = st.checkbox(
                "Show era centroids", value=True, key="nmds_centroids_tab8"
            )

        show_species_nmds = st.checkbox(
            "Show top card vectors", value=False, key="nmds_species_tab8"
        )

        hover_nmds = [c for c in ["Name", "Date", "current_era", "current_set"]
                      if c in ord_nmds.columns]

        fig_n = px.scatter(
            ord_nmds, x="NMDS1", y="NMDS2",
            color=color_by_nmds,
            hover_data=hover_nmds,
            title=f"NMDS – sites colored by {color_by_nmds}{stress_label}",
            template="plotly_dark",
            opacity=0.65,
        )
        fig_n.update_traces(marker=dict(size=6, line=dict(width=0)))
        fig_n.update_xaxes(title_text="NMDS1 (no units)", showgrid=True,
                           gridcolor="#333", zeroline=False)
        fig_n.update_yaxes(title_text="NMDS2 (no units)", showgrid=True,
                           gridcolor="#333", zeroline=False)
        fig_n.update_layout(
            plot_bgcolor="#1a1a2e", paper_bgcolor="#0d0d1a",
            legend=dict(orientation="v", yanchor="top", y=1,
                        xanchor="left", x=1.01, font=dict(size=11),
                        itemsizing="constant"),
            height=800,
        )

        if show_centroids_nmds and color_by_nmds in env_cents_nmds:
            cents = env_cents_nmds[color_by_nmds]
            fig_n.add_trace(go.Scatter(
                x=cents["NMDS1"], y=cents["NMDS2"],
                mode="markers",
                marker=dict(size=18, symbol="diamond",
                            color="black", line=dict(width=2, color="white")),
                name="Centroids", showlegend=True,
                hovertext=cents["label"], hoverinfo="text",
            ))
            for _, crow in cents.iterrows():
                fig_n.add_annotation(
                    x=crow["NMDS1"], y=crow["NMDS2"],
                    text=f"<b>{crow['label']}</b>",
                    showarrow=True, arrowhead=2, arrowsize=1,
                    arrowwidth=1.5, arrowcolor="#555",
                    ax=0, ay=-38,
                    font=dict(size=9, color="#222"),
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#aaa", borderwidth=1, borderpad=3,
                )

        if show_species_nmds:
            top_n_n = st.slider("Top N card vectors", 5, 30, 10, key="nmds_top_n_tab8")
            sp = species_nmds.copy()
            sp["dist"] = np.sqrt(sp["NMDS1"]**2 + sp["NMDS2"]**2)
            sp_top = sp.nlargest(top_n_n, "dist")
            fig_n.add_trace(go.Scatter(
                x=sp_top["NMDS1"], y=sp_top["NMDS2"],
                mode="markers+text", text=sp_top["card"],
                textposition="top right",
                textfont=dict(size=10, color="crimson"),
                marker=dict(size=10, symbol="triangle-up", color="crimson",
                            line=dict(width=1, color="white")),
                name="Card WA scores", showlegend=True,
            ))

        st.plotly_chart(fig_n, width='stretch')

        # ── Most Dissimilar Site per Era ──────────────────────────────────
        name_col = "Name" if "Name" in ord_nmds.columns else None
        date_col = "Date" if "Date" in ord_nmds.columns else None

        def nmds_site_label(idx):
            parts = []
            if name_col: parts.append(str(ord_nmds.loc[idx, name_col]))
            if date_col: parts.append(f"({ord_nmds.loc[idx, date_col]})")
            return " ".join(parts) if parts else f"Site {idx}"

        if "current_era" in ord_nmds.columns:
            nmds_rows, nmds_best_idx = [], {}
            for era in ERA_ORDER:
                era_idx = ord_nmds.index[ord_nmds["current_era"] == era].tolist()
                if len(era_idx) < 2:
                    continue
                era_coords = ord_nmds.loc[era_idx, ["NMDS1", "NMDS2"]].values
                best_mean_dist, best_idx = -1, None
                for ii, idx in enumerate(era_idx):
                    others = np.delete(era_coords, ii, axis=0)
                    dists  = np.sqrt(
                        (era_coords[ii, 0] - others[:, 0])**2 +
                        (era_coords[ii, 1] - others[:, 1])**2
                    )
                    mean_dist = dists.mean()
                    if mean_dist > best_mean_dist:
                        best_mean_dist, best_idx = mean_dist, idx
                nmds_rows.append({
                    "Era":               era,
                    "Outlier Deck":      nmds_site_label(best_idx),
                    "Mean NMDS Distance": f"{best_mean_dist:.4f}",
                    "Era N":             len(era_idx),
                })
                nmds_best_idx[era] = best_idx

            nmds_dissim_df = pd.DataFrame(nmds_rows)

            st.markdown("#### 🃏 Outlier Decklists By Era")
            for row in nmds_dissim_df.itertuples():
                era      = row.Era
                best_idx = nmds_best_idx.get(era)
                if best_idx is None:
                    continue
                label = f"{row._2}  —  {era}"
                with st.expander(label):
                    outlier_name = ord_nmds.loc[best_idx, "Name"] if name_col else None
                    outlier_date = ord_nmds.loc[best_idx, "Date"] if date_col else None
                    if outlier_name and outlier_date:
                        match = amulet_comb[
                            (amulet_comb["Name"] == outlier_name) &
                            (amulet_comb["Date"].astype(str).str.contains(
                                str(outlier_date)[:7], na=False))
                        ]
                    else:
                        match = pd.DataFrame()

                    if match.empty:
                        st.info("Decklist not found in source data.")
                    else:
                        deck_row   = match.iloc[0]
                        card_cols_deck = [c for c in amulet_int.columns if c in deck_row.index]
                        decklist   = pd.Series({c: deck_row[c] for c in card_cols_deck}).astype(int)
                        decklist   = decklist[decklist > 0].sort_values(ascending=False)

                        era_rows2  = amulet_comb[amulet_comb["current_era"] == era]
                        era_cards2 = era_rows2[[c for c in amulet_int.columns if c in era_rows2.columns]]
                        median_deck2 = era_cards2.median().round(2)
                        median_deck2 = median_deck2[median_deck2 > 0].sort_values(ascending=False)

                        col_meta2, col_out2, col_med2 = st.columns([1, 1.5, 1.5])
                        with col_meta2:
                            st.markdown(f"**{outlier_name}** — {outlier_date}")
                            st.markdown(f"Mean NMDS Distance: **{row._3}**")
                            st.markdown(f"Era N: **{row._4}**")
                        median_cards2 = set(median_deck2[median_deck2 > 0].index)
                        era_id2 = re.sub(r"[^a-zA-Z0-9]+", "-", era).strip("-").lower() or "era"
                        with col_out2:
                            st.markdown("**Outlier Decklist** *(green = not in era median)*")
                            deck_df2 = decklist.reset_index()
                            deck_df2.columns = ["Card", "Copies"]
                            deck_df2 = sort_by_type(deck_df2, "Card")
                            render_decklist_html(deck_df2, "Card", "Copies",
                                                 median_cards2, f"nmds-outlier-{era_id2}", 350)
                        with col_med2:
                            st.markdown(f"**Median Decklist ({era})**")
                            med_df2 = median_deck2.reset_index()
                            med_df2.columns = ["Card", "Median Copies"]
                            med_df2 = sort_by_type(med_df2, "Card")
                            render_decklist_html(med_df2, "Card", "Median Copies",
                                                 None, f"nmds-median-{era_id2}", 350)
    else:
        st.info("Loading NMDS scores from GitHub… refresh if this persists.")


# ── Tab 9: NMDS – Card Inclusion ─────────
with tab9:
    st.subheader("NMDS – Card Inclusion")
    st.markdown(
        "Sites colored by the number of copies of a selected card. "
        "Reveals where in Bray-Curtis space a card concentrates."
    )

    ord_nmds9, species_nmds9, _, _ = _resolve_nmds()

    if ord_nmds9 is not None:
        card_options9 = amulet_int.sum().sort_values(ascending=False).index.tolist()
        selected_card9 = st.selectbox(
            "Color sites by card count:",
            card_options9,
            key="nmds_card_select9"
        )

        plot9 = ord_nmds9.copy()
        if selected_card9 in amulet_comb.columns:
            merge_keys9 = [k for k in ["Name", "Date"]
                           if k in plot9.columns and k in amulet_comb.columns]
            if merge_keys9:
                card_lookup9 = (
                    amulet_comb[merge_keys9 + [selected_card9]]
                    .drop_duplicates(subset=merge_keys9)
                )
                plot9 = plot9.drop(columns=[selected_card9], errors="ignore")
                plot9 = plot9.merge(card_lookup9, on=merge_keys9, how="left")
                plot9[selected_card9] = (
                    pd.to_numeric(plot9[selected_card9], errors="coerce")
                      .fillna(0).astype(int)
                )

        hover9 = [c for c in ["Name", "Date", "current_era", "current_set"]
                  if c in plot9.columns]
        if selected_card9 in plot9.columns and selected_card9 not in hover9:
            hover9.append(selected_card9)

        fig9 = px.scatter(
            plot9, x="NMDS1", y="NMDS2",
            color=selected_card9 if selected_card9 in plot9.columns else None,
            color_continuous_scale="thermal",
            hover_data=[c for c in hover9 if c in plot9.columns],
            title=f"NMDS – sites colored by copies of {selected_card9}",
            template="plotly_dark",
            opacity=0.8,
        )
        fig9.update_traces(marker=dict(size=8))
        fig9.update_xaxes(title_text="NMDS1 (no units)", showgrid=True,
                          gridcolor="#333", zeroline=False)
        fig9.update_yaxes(title_text="NMDS2 (no units)", showgrid=True,
                          gridcolor="#333", zeroline=False)
        fig9.update_layout(
            plot_bgcolor="#1a1a2e", paper_bgcolor="#0d0d1a", height=800
        )
        st.plotly_chart(fig9, width='stretch')

    else:
        st.info("Loading NMDS scores from GitHub… refresh if this persists.")
