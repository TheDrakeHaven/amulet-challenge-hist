import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import date, datetime
import re
from io import BytesIO
import matplotlib.pyplot as plt
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


def assign_ban_era(d, events_df):
    """findInterval equivalent: return the era label for date d."""
    ts = pd.to_datetime(d, errors="coerce")
    if pd.isna(ts):
        return events_df.iloc[0]["event"]
    idx = (events_df["date"] <= ts).sum()
    if idx >= len(events_df):
        idx = len(events_df) - 1
    return events_df.iloc[idx]["event"]


def assign_current_set(d):
    """Return the most recent set released on or before date d."""
    ts = pd.to_datetime(d, errors="coerce")
    if pd.isna(ts):
        return "Unknown"
    valid = modern_sets[modern_sets["release_date"] <= ts]
    if valid.empty:
        return "Unknown"
    return valid.iloc[-1]["set"]



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
    amulet_df = pd.read_csv("merged_amulet.csv", encoding="utf-8", encoding_errors="replace")

    amulet_df = amulet_df.drop_duplicates(keep="first")
    amulet_df = amulet_df.drop(columns=["Maindeck_Total", "Sideboard_Total"], errors="ignore")

    # Meta columns — flexible, works with or without Place/Event_Type
    _all_meta = ["row_number", "Name", "Place", "Event", "current_era", "Event_Type", "Date",
                 "NMDS1", "NMDS2", "Maindeck_Total", "Sideboard_Total"]
    meta_cols = [c for c in _all_meta if c in amulet_df.columns]
    card_cols  = [c for c in amulet_df.columns if c not in meta_cols]

    for col in card_cols:
        amulet_df[col] = pd.to_numeric(amulet_df[col], errors="coerce").fillna(0).astype(int)
    if "Place" in amulet_df.columns:
        amulet_df["Place"] = pd.to_numeric(amulet_df["Place"], errors="coerce").fillna(0).astype(int)
    if "Date" in amulet_df.columns:
        amulet_df["Date"] = pd.to_datetime(amulet_df["Date"], errors="coerce").dt.strftime("%m-%d-%Y")

    _env_candidates = ["row_number", "Name", "Place", "Event", "current_era", "Event_Type", "Date",
                       "NMDS1", "NMDS2"]
    env_cols   = [c for c in _env_candidates if c in amulet_df.columns]
    amulet_env = amulet_df[env_cols].copy()
    amulet_int = amulet_df[[c for c in amulet_df.columns if c not in meta_cols]].copy()

    amulet_env["current_set"] = amulet_env["Date"].apply(assign_current_set)
    # Derive current_era from Date if not present in CSV (backwards compatibility)
    if "current_era" not in amulet_env.columns:
        amulet_env["current_era"] = amulet_env["Date"].apply(
            lambda d: assign_ban_era(d, ban_events)
        )

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


# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────

tab2, tab3, tab4, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "🃏 Deck Data",
    "📈 Median by Era",
    "📋 Era Decklists",
    "🛡️ Era-Specific Sideboard",
    "🔍 Era-Specific Cards",
    "🌐 NMDS – Era & Set",
    "🎴 NMDS – Card Inclusion",
    "🃏 NMDS – Card Similarity",
])

# ── Tab 2: Deck Data ─────────────────────
with tab2:
    _type_colors = {
        "Land":     "#2ca02c",
        "Creature": "#1f77b4",
        "Spell":    "#ff7f0e",
        "Unknown":  "#7f7f7f",
    }

    subtab_data, subtab_totals, subtab_sb_totals = st.tabs([
        "📋 All Decks", "🔢 Maindeck Card Totals", "🔢 Sideboard Card Totals"
    ])

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
            .rename_axis("Card")
            .reset_index(name="Total Copies")
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
        fig_bar = px.bar(
            chart_data,
            x="Card",
            y="Total Copies",
            color="Card Type",
            color_discrete_map=_type_colors,
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

    with subtab_sb_totals:
        st.subheader("Total Sideboard Card Copies")
        st.markdown(
            "Total copies of each sideboard card across all decklists, "
            "broken down by card type. Maindeck cards excluded."
        )

        # Sideboard columns only (sb_ prefix), strip prefix for display
        sb_cols = [c for c in amulet_int.columns if c.startswith("sb_")]
        sb_totals = (
            amulet_int[sb_cols]
            .sum()
            .astype(int)
            .rename_axis("Card")
            .reset_index(name="Total Copies")
        )
        # Strip sb_ prefix for display
        sb_totals["Card"] = sb_totals["Card"].str.replace("^sb_", "", regex=True)
        sb_totals["Card Type"] = sb_totals["Card"].apply(get_card_type)
        sb_totals = sb_totals[sb_totals["Total Copies"] > 0].sort_values(
            "Total Copies", ascending=False
        ).reset_index(drop=True)

        # ── Summary metrics ───────────────────────────────────────────────
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Unique Cards", len(sb_totals))
        s2.metric("Total Copies", f"{sb_totals['Total Copies'].sum():,}")
        s3.metric("Avg per Deck",
                  f"{sb_totals['Total Copies'].sum() / max(len(amulet_df), 1):.1f}")
        s4.metric("Decklists", len(amulet_df))

        # ── Type filter ───────────────────────────────────────────────────
        sb_type_filter = st.multiselect(
            "Filter by card type:",
            options=["Creature", "Spell", "Land", "Unknown"],
            default=[],
            key="sb_totals_type_filter",
        )
        display_sb_totals = (
            sb_totals[sb_totals["Card Type"].isin(sb_type_filter)]
            if sb_type_filter else sb_totals
        )

        # ── Bar chart ─────────────────────────────────────────────────────
        top_n_sb = st.slider("Show top N cards in chart", 10, 60, 30, key="sb_totals_n")
        chart_sb = display_sb_totals.head(top_n_sb)
        fig_sb_bar = px.bar(
            chart_sb,
            x="Card",
            y="Total Copies",
            color="Card Type",
            color_discrete_map=_type_colors,
            title=f"Top {top_n_sb} Sideboard Cards by Total Copies",
            template="plotly_white",
        )
        fig_sb_bar.update_layout(
            xaxis_tickangle=-45,
            height=500,
            legend_title_text="Card Type",
        )
        st.plotly_chart(fig_sb_bar, width='stretch')

        # ── Full table ────────────────────────────────────────────────────
        st.dataframe(
            display_sb_totals,
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

# ── Tab 4: Era Decklists ──────────────────
with tab4:
    st.subheader("Expected Era Decklist")
    st.markdown(
        "Predicted 60-card maindeck and 15-card sideboard for each era, "
        "derived using mode card counts as the base, with remaining slots filled by mean counts for non-mode cards. "
        "Cards with the largest fractional remainders receive the extra copies."
    )

    era_list_t4 = [e for e in ERA_ORDER if e in amulet_comb["current_era"].values]
    selected_era_t4 = st.selectbox("Select era:", era_list_t4,
                                    index=len(era_list_t4) - 1, key="era_deck_select")

    era_rows_t4 = amulet_comb[amulet_comb["current_era"] == selected_era_t4]

    # Separate maindeck and sideboard columns
    all_card_cols = [c for c in amulet_int.columns]
    md_cols = [c for c in all_card_cols if not c.startswith("sb_")]
    sb_cols_t4 = [c for c in all_card_cols if c.startswith("sb_")]

    def predict_decklist(era_rows, cols, target):
        """
        Build a predicted decklist of exactly `target` cards.
        1. Start with the mode for each card (most common non-zero count).
        2. If mode sum < target, fill remaining slots using mean counts
           for cards not already in the mode deck, scaled to the remainder.
        3. If mode sum > target, scale down proportionally.
        """
        mode_vals = era_rows[cols].mode().iloc[0]
        mode_vals = mode_vals[mode_vals > 0]

        mode_sum = int(mode_vals.sum())

        if mode_sum == target:
            return mode_vals.sort_values(ascending=False).apply(int)

        elif mode_sum > target:
            # Scale down proportionally from mode
            scaled = mode_vals / mode_vals.sum() * target
            floored = scaled.apply(lambda x: int(x))
            remainder = target - floored.sum()
            fractions = (scaled - floored).sort_values(ascending=False)
            for card in fractions.index[:int(remainder)]:
                floored[card] += 1
            return floored[floored > 0].sort_values(ascending=False)

        else:
            # Start with mode deck, fill remainder using mean of non-mode cards
            remainder = target - mode_sum
            mean_vals = era_rows[cols].mean()
            # Exclude cards already in mode deck
            fill_cards = mean_vals.drop(index=mode_vals.index, errors="ignore")
            fill_cards = fill_cards[fill_cards > 0]
            if fill_cards.sum() > 0 and remainder > 0:
                scaled_fill = fill_cards / fill_cards.sum() * remainder
                floored_fill = scaled_fill.apply(lambda x: int(x))
                rem2 = remainder - floored_fill.sum()
                fractions2 = (scaled_fill - floored_fill).sort_values(ascending=False)
                for card in fractions2.index[:int(rem2)]:
                    floored_fill[card] += 1
                floored_fill = floored_fill[floored_fill > 0]
            else:
                floored_fill = pd.Series(dtype=float)
            combined = pd.concat([mode_vals.apply(int), floored_fill.apply(int)])
            return combined[combined > 0].sort_values(ascending=False)

    md_scaled = predict_decklist(era_rows_t4, md_cols, 60)
    sb_scaled = predict_decklist(era_rows_t4, sb_cols_t4, 15)

    col_t4a, col_t4b = st.columns(2)

    with col_t4a:
        st.markdown(f"**Predicted Maindeck** ({len(era_rows_t4)} decklists in era)")
        md_df = md_scaled.reset_index()
        md_df.columns = ["Card", "Copies"]
        md_df = sort_by_type(md_df, "Card")
        render_decklist_html(md_df, "Card", "Copies", None, f"era-md-{selected_era_t4[:10]}", 600)

    with col_t4b:
        st.markdown(f"**Predicted Sideboard**")
        sb_df = sb_scaled.reset_index()
        sb_df.columns = ["Card", "Copies"]
        sb_df = sort_by_type(sb_df, "Card")
        render_decklist_html(sb_df, "Card", "Copies", None, f"era-sb-{selected_era_t4[:10]}", 600)


# ─────────────────────────────────────────
# PCA COMPUTATION
# ─────────────────────────────────────────


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
    """Load Card_WA_Scores and stress from an Excel file-like or URL bytes.
    Site scores are now read directly from the main CSV (amulet_comb)."""
    xls = pd.ExcelFile(source)
    sp  = pd.read_excel(xls, sheet_name="Card_WA_Scores")
    # Normalise species df: ensure lowercase "card" column exists
    sp.columns = [c if c not in ("Card",) else "card" for c in sp.columns]
    if "card" not in sp.columns:
        sp = sp.reset_index()
        sp = sp.rename(columns={sp.columns[0]: "card"})
    # Ensure NMDS1/NMDS2 columns exist
    for ax in ["NMDS1", "NMDS2"]:
        if ax not in sp.columns and ax.lower() in sp.columns:
            sp = sp.rename(columns={ax.lower(): ax})
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
    return sp, st_val

def _build_ord_from_comb():
    """Build site scores and centroids from amulet_comb (which now carries NMDS1/NMDS2)."""
    if "NMDS1" not in amulet_comb.columns or "NMDS2" not in amulet_comb.columns:
        return None, {}
    od = amulet_comb[["Name", "Event", "Date", "NMDS1", "NMDS2",
                       "current_era", "current_set"]].copy()
    od = od.dropna(subset=["NMDS1", "NMDS2"])
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
    return od, cents

def _resolve_nmds():
    """
    Return (ord_nmds, species_nmds, stress_nmds, env_centroids_nmds).
    Site scores come from amulet_comb; species scores and stress come from GitHub Excel.
    """
    # Site scores always come from amulet_comb
    od, cents = _build_ord_from_comb()

    # 1. Already fetched species/stress from GitHub this session
    gh = st.session_state.get("nmds_github")
    if gh is not None:
        return od, gh["species"], gh["stress"], cents

    # 2. Already computed this session
    if "nmds_result" in st.session_state:
        return od, st.session_state["nmds_species"], st.session_state["nmds_stress"], cents

    # 3. Try GitHub for species scores + stress
    try:
        import requests as _req
        resp = _req.get(GITHUB_NMDS_URL, timeout=15)
        resp.raise_for_status()
        from io import BytesIO as _BIO
        sp, st_val = _load_nmds_excel(_BIO(resp.content))
        st.session_state["nmds_github"] = {
            "species": sp, "stress": st_val
        }
        return od, sp, st_val, cents
    except Exception:
        pass

    return od, None, None, cents

# ── Run PCA on app start (fast — no NMDS here, tab 8 handles it lazily) ──
# ── Tab 6: Era-Specific Sideboard ────────
with tab6:
    st.subheader("Era-Specific Sideboard Cards")
    st.markdown(
        "Sideboard cards that appear predominantly in one ban era. "
        "Concentration score = share of a card's total sideboard appearances that fall within a single era. "
        "Use the slider to control the minimum number of appearances required."
    )

    era_order_sb = [e for e in ERA_ORDER if e in amulet_comb["current_era"].values]
    sb_cols = [c for c in amulet_int.columns if c.startswith("sb_")]

    # Build era × sb card count matrix and era deck sizes
    era_sb_raw = (
        amulet_comb.groupby("current_era")[sb_cols]
        .sum()
        .reindex(era_order_sb)
        .fillna(0)
    )
    era_sizes_sb = (
        amulet_comb.groupby("current_era")
        .size()
        .reindex(era_order_sb)
        .fillna(1)
    )

    # Normalize: mean copies per deck in each era
    era_sb = era_sb_raw.div(era_sizes_sb, axis=0)

    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        min_appearances_sb = st.slider(
            "Minimum total raw appearances across all eras",
            min_value=1, max_value=50, value=10, step=1,
            key="era_sb_min"
        )
    with col_sb2:
        min_concentration_sb = st.slider(
            "Minimum concentration score (0–1)",
            min_value=0.50, max_value=1.0, value=0.75, step=0.05,
            key="era_sb_conc"
        )

    sb_totals_raw = era_sb_raw.sum(axis=0)
    eligible_sb = sb_totals_raw[sb_totals_raw >= min_appearances_sb].index.tolist()

    # % of decks in each era that contained the sideboard card
    _sb_presence_bool = amulet_comb[sb_cols].gt(0)
    _sb_presence_bool["current_era"] = amulet_comb["current_era"].values
    era_sb_presence = (
        _sb_presence_bool.groupby("current_era")[sb_cols]
        .sum()
        .reindex(era_order_sb)
        .fillna(0)
    )

    sb_rows = []
    for card in eligible_sb:
        col_data = era_sb[card]
        dominant_era = col_data.idxmax()
        dominant_rate = col_data.max()
        raw_dominant_sb = era_sb_raw.loc[dominant_era, card]
        raw_total_sb = sb_totals_raw[card]
        if raw_total_sb == 0:
            continue
        concentration = raw_dominant_sb / raw_total_sb
        era_size_sb = int(era_sizes_sb[dominant_era])
        pct_decks_sb = round(era_sb_presence.loc[dominant_era, card] / era_size_sb * 100, 1) if era_size_sb > 0 else 0.0
        if concentration >= min_concentration_sb and dominant_rate >= 0.1:
            display_name = card[3:] if card.startswith("sb_") else card
            sb_rows.append({
                "Card":                  display_name,
                "Dominant Era":          dominant_era,
                "Era Size (decks)":      era_size_sb,
                "% Decks w/ Card":       pct_decks_sb,
                "Rate in Era":           round(dominant_rate, 3),
                "Raw Era Appearances":   int(raw_dominant_sb),
                "Total Raw Appearances": int(raw_total_sb),
                "Concentration":         round(concentration, 3),
            })

    if not sb_rows:
        st.info("No sideboard cards meet the current filters. Try lowering the thresholds.")
    else:
        sb_result_df = pd.DataFrame(sb_rows)
        sb_result_df["_era_order"] = sb_result_df["Dominant Era"].map(
            {e: i for i, e in enumerate(era_order_sb)}
        )
        sb_result_df = (
            sb_result_df.sort_values(["_era_order", "% Decks w/ Card", "Concentration"], ascending=[True, False, False])
            .drop(columns=["_era_order"])
            .reset_index(drop=True)
        )

        # ── Summary chart ──────────────────────────────────────────────────
        sb_era_counts = sb_result_df["Dominant Era"].value_counts().reindex(era_order_sb).dropna()
        fig_sb_bar = px.bar(
            sb_era_counts.reset_index(),
            x="Dominant Era",
            y="count",
            title="Number of Era-Specific Sideboard Cards per Era",
            labels={"count": "# Cards", "Dominant Era": "Era"},
            color="Dominant Era",
            template="plotly_white",
        )
        fig_sb_bar.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_sb_bar, width='stretch')

        # ── Filter by era ──────────────────────────────────────────────────
        sb_era_filter = st.multiselect(
            "Filter by era (leave blank for all):",
            options=era_order_sb,
            default=[],
            key="era_sb_filter"
        )
        if sb_era_filter:
            sb_result_df = sb_result_df[sb_result_df["Dominant Era"].isin(sb_era_filter)]

        st.dataframe(sb_result_df, width='stretch', hide_index=True)

        # ── Heatmap ────────────────────────────────────────────────────────
        st.markdown("**Heatmap of Era-Specific Sideboard Cards** *(color = mean copies per deck)*")
        sb_heat_cards_display = sb_result_df["Card"].tolist()
        sb_heat_cards_cols = ["sb_" + c for c in sb_heat_cards_display]
        sb_heat_cards_cols = [c for c in sb_heat_cards_cols if c in era_sb.columns]
        if sb_heat_cards_cols:
            sb_heat = era_sb[sb_heat_cards_cols].copy()
            sb_heat.columns = [c[3:] for c in sb_heat.columns]  # strip sb_ for display
            sb_heat = sb_heat.T
            sb_heat.index.name = "Card"
            fig_sb_heat = px.imshow(
                sb_heat,
                aspect="auto",
                color_continuous_scale="Blues",
                labels={"x": "Era", "y": "Card", "color": "Copies/Deck"},
                title="Era-Specific Sideboard Cards — Mean Copies per Deck by Era",
            )
            fig_sb_heat.update_layout(height=max(400, len(sb_heat_cards_cols) * 18))
            st.plotly_chart(fig_sb_heat, width='stretch')


# ── Tab 7: Era-Specific Cards ─────────────
with tab7:
    st.subheader("Era-Specific Cards")
    st.markdown(
        "Cards that appear predominantly in one ban era. "
        "Concentration score = share of a card's total appearances that fall within a single era. "
        "Use the slider to control the minimum number of appearances required."
    )

    era_order_display = [e for e in ERA_ORDER if e in amulet_comb["current_era"].values]
    card_cols_era = [c for c in amulet_int.columns if not c.startswith("sb_")]

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
            min_value=1, max_value=50, value=10, step=1,
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

    # % of decks in each era that contained the card (presence = at least 1 copy)
    _presence_bool = amulet_comb[card_cols_era].gt(0)
    _presence_bool["current_era"] = amulet_comb["current_era"].values
    era_card_presence = (
        _presence_bool.groupby("current_era")[card_cols_era]
        .sum()
        .reindex(era_order_display)
        .fillna(0)
    )

    rows = []
    for card in eligible_cards:
        col_data = era_card[card]          # normalized (per-deck rate)
        dominant_era = col_data.idxmax()
        dominant_rate = col_data.max()
        raw_dominant = era_card_raw.loc[dominant_era, card]
        raw_total = card_totals_raw[card]
        if raw_total == 0:
            continue
        concentration = raw_dominant / raw_total
        era_size = int(era_sizes[dominant_era])
        pct_decks = round(era_card_presence.loc[dominant_era, card] / era_size * 100, 1) if era_size > 0 else 0.0
        if concentration >= min_concentration and dominant_rate >= 0.1:
            rows.append({
                "Card":                  card,
                "Card Type":             get_card_type(card),
                "Dominant Era":          dominant_era,
                "Era Size (decks)":      era_size,
                "% Decks w/ Card":       pct_decks,
                "Rate in Era":           round(dominant_rate, 3),
                "Raw Era Appearances":   int(raw_dominant),
                "Total Raw Appearances": int(raw_total),
                "Concentration":         round(concentration, 3),
            })

    if not rows:
        st.info("No cards meet the current filters. Try lowering the thresholds.")
    else:
        result_df = pd.DataFrame(rows)
        result_df["_era_order"] = result_df["Dominant Era"].map(
            {e: i for i, e in enumerate(era_order_display)}
        )
        result_df = (
            result_df.sort_values(["_era_order", "% Decks w/ Card", "Concentration"], ascending=[True, False, False])
            .drop(columns=["_era_order"])
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



def _match_amulet_row(name_v, date_v):
    """Match a row in amulet_comb by Name + Date, handling format differences."""
    mask_name = amulet_comb["Name"] == name_v
    try:
        ts = pd.to_datetime(date_v, errors="coerce")
        if pd.notna(ts):
            mask_date = pd.to_datetime(
                amulet_comb["Date"], errors="coerce"
            ).dt.strftime("%Y-%m-%d") == ts.strftime("%Y-%m-%d")
        else:
            mask_date = amulet_comb["Date"].astype(str).str.contains(
                str(date_v)[:7], na=False
            )
    except Exception:
        mask_date = pd.Series(True, index=amulet_comb.index)
    return amulet_comb[mask_name & mask_date]

def _render_nmds_decklist(row_idx, source_ord, label_prefix=""):
    """Render a decklist for the given NMDS site row index."""
    name_v = source_ord.iloc[row_idx]["Name"] if "Name" in source_ord.columns else None
    date_v = source_ord.iloc[row_idx]["Date"] if "Date" in source_ord.columns else None
    if name_v and date_v:
        match = _match_amulet_row(name_v, date_v)
    else:
        match = pd.DataFrame()
    if match.empty:
        st.info("Decklist not found in source data.")
        return
    deck_row = match.iloc[0]
    card_cols_d = [c for c in amulet_int.columns if c in deck_row.index]
    decklist_s = pd.Series({c: deck_row[c] for c in card_cols_d}).astype(int)
    decklist_s = decklist_s[decklist_s > 0].sort_values(ascending=False)

    era_v = source_ord.iloc[row_idx]["current_era"] if "current_era" in source_ord.columns else None
    _date_display = pd.to_datetime(date_v, errors="coerce")
    _date_display = _date_display.strftime("%m/%d/%Y") if pd.notna(_date_display) else str(date_v)[:10]
    st.markdown(f"**{label_prefix}{name_v}** — {_date_display}" +
                (f"  |  *{era_v}*" if era_v else ""))

    era_rows_d = amulet_comb[amulet_comb["current_era"] == era_v] if era_v else pd.DataFrame()
    if not era_rows_d.empty:
        era_cards_d = era_rows_d[[c for c in amulet_int.columns if c in era_rows_d.columns]]
        mean_d = era_cards_d.mean().round(2)
        mean_d = mean_d[mean_d >= 0.2].sort_values(ascending=False)
        mean_set_d = set(mean_d.index)
    else:
        mean_d, mean_set_d = pd.Series(dtype=float), set()

    col_dl, col_med = st.columns(2)
    era_id_d = re.sub(r"[^a-zA-Z0-9]+", "-", str(era_v or "")).strip("-").lower() or "era"
    with col_dl:
        st.markdown("**Decklist** *(green = not in era mean)*")
        dl_df = decklist_s.reset_index()
        dl_df.columns = ["Card", "Copies"]
        dl_df = sort_by_type(dl_df, "Card")
        render_decklist_html(dl_df, "Card", "Copies",
                             mean_set_d, f"nmds-click-{era_id_d}", 400)
    with col_med:
        if not mean_d.empty:
            st.markdown(f"**Era Mean** ({era_v})")
            med_df_d = mean_d.reset_index()
            med_df_d.columns = ["Card", "Mean Copies"]
            med_df_d = sort_by_type(med_df_d, "Card")
            render_decklist_html(med_df_d, "Card", "Mean Copies",
                                 None, f"nmds-clickmed-{era_id_d}", 400)


# ── Tab 8: NMDS – Era & Set ───────────────
with tab8:
    st.subheader("NMDS – Era & Set")
    st.markdown(
        "Sites (decklists) in **non-metric MDS** space (Bray-Curtis dissimilarity). "
        "Axes have no ecological units — only relative distances matter."
    )

    ord_nmds, species_nmds, stress_nmds, env_cents_nmds = _resolve_nmds()

    if ord_nmds is not None:
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

        # Format Date for display (remove HH:MM:SS)
        if "Date" in ord_nmds.columns:
            ord_nmds = ord_nmds.copy()
            ord_nmds["Date"] = pd.to_datetime(
                ord_nmds["Date"], errors="coerce"
            ).dt.strftime("%m/%d/%Y")

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
            if species_nmds is None:
                st.warning("Species scores not available — re-run NMDS and commit nmds_results.xlsx.")
            else:
                top_n_n = st.slider("Top N card vectors", 5, 30, 10, key="nmds_top_n_tab8")
                sp = species_nmds.copy()
                if "card" not in sp.columns:
                    sp = sp.reset_index()
                    # rename first column to "card" whatever it's called
                    sp = sp.rename(columns={sp.columns[0]: "card"})
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

        sel8 = st.plotly_chart(fig_n, width='stretch', on_select="rerun", key="nmds_plot8")

        # ── Click-to-decklist ─────────────────────────────────────────────
        _raw_sel8 = (sel8 or {}).get("selection", {})
        selected_pts8 = _raw_sel8.get("points", [])

        if selected_pts8:
            pt8 = selected_pts8[0]
            cd  = pt8.get("customdata", [])
            # customdata order matches hover_nmds: [Name, Date, current_era, ...]
            if len(cd) >= 2:
                _click_name = cd[0]
                _click_date = cd[1]
                _click_match = _match_amulet_row(_click_name, _click_date)
                if not _click_match.empty:
                    # Build a one-row ord_nmds-like frame so _render can use it
                    _click_ord = pd.DataFrame([{
                        "Name": _click_name,
                        "Date": _click_date,
                        "current_era": cd[2] if len(cd) > 2 else None,
                    }])
                    st.markdown("---")
                    st.markdown("### 🃏 Selected Deck")
                    _render_nmds_decklist(0, _click_ord)
                else:
                    st.info("Decklist not found in source data.")

        # ── Most Dissimilar Site per Era ──────────────────────────────────
        name_col = "Name" if "Name" in ord_nmds.columns else None
        date_col = "Date" if "Date" in ord_nmds.columns else None

        def nmds_site_label(idx):
            parts = []
            if name_col: parts.append(str(ord_nmds.loc[idx, name_col]))
            if date_col:
                _dl = pd.to_datetime(ord_nmds.loc[idx, date_col], errors="coerce")
                parts.append(f"({_dl.strftime('%m/%d/%Y') if pd.notna(_dl) else str(ord_nmds.loc[idx, date_col])[:10]})")
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
                        match = _match_amulet_row(outlier_name, outlier_date)
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
                        mean_deck2 = era_cards2.mean().round(2)
                        mean_deck2 = mean_deck2[mean_deck2 >= 0.2].sort_values(ascending=False)

                        col_meta2, col_out2, col_med2 = st.columns([1, 1.5, 1.5])
                        with col_meta2:
                            _od_disp = pd.to_datetime(outlier_date, errors="coerce")
                            _od_disp = _od_disp.strftime("%m/%d/%Y") if pd.notna(_od_disp) else str(outlier_date)[:10]
                            st.markdown(f"**{outlier_name}** — {_od_disp}")
                            st.markdown(f"Mean NMDS Distance: **{row._3}**")
                            st.markdown(f"Era N: **{row._4}**")
                        mean_cards2 = set(mean_deck2.index)
                        era_id2 = re.sub(r"[^a-zA-Z0-9]+", "-", era).strip("-").lower() or "era"
                        with col_out2:
                            st.markdown("**Outlier Decklist** *(green = not in era mean)*")
                            deck_df2 = decklist.reset_index()
                            deck_df2.columns = ["Card", "Copies"]
                            deck_df2 = sort_by_type(deck_df2, "Card")
                            render_decklist_html(deck_df2, "Card", "Copies",
                                                 mean_cards2, f"nmds-outlier-{era_id2}", 350)
                        with col_med2:
                            st.markdown(f"**Mean Decklist ({era})**")
                            med_df2 = mean_deck2.reset_index()
                            med_df2.columns = ["Card", "Mean Copies"]
                            med_df2 = sort_by_type(med_df2, "Card")
                            render_decklist_html(med_df2, "Card", "Mean Copies",
                                                 None, f"nmds-median-{era_id2}", 350)
    else:
        st.info("NMDS site scores not found in data. Check that merged_amulet.csv contains NMDS1 and NMDS2 columns.")


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

        col9a, col9b = st.columns([1, 2])
        with col9a:
            selected_card9 = st.selectbox(
                "Color sites by card count:",
                card_options9,
                key="nmds_card_select9"
            )
        with col9b:
            era_order9 = ban_events["event"].tolist()
            if "current_era" in ord_nmds9.columns:
                present_eras9 = set(ord_nmds9["current_era"].dropna().unique())
            else:
                present_eras9 = set()
            available_eras9 = [e for e in era_order9 if e in present_eras9]
            selected_eras9 = st.multiselect(
                "Filter by era:",
                options=available_eras9,
                default=available_eras9,
                key="nmds_era_filter9",
                help="Select one or more ban eras to show. Defaults to all eras.",
            )

        plot9 = ord_nmds9.copy()
        # Apply era filter
        if selected_eras9 and "current_era" in plot9.columns:
            plot9 = plot9[plot9["current_era"].isin(selected_eras9)]
        if selected_card9 in amulet_comb.columns:
            if "Name" in plot9.columns and "Name" in amulet_comb.columns:
                # Normalise both Date columns to YYYY-MM-DD string for merging
                plot9["_date_key"] = pd.to_datetime(
                    plot9["Date"], errors="coerce"
                ).dt.strftime("%Y-%m-%d")
                amulet_comb_tmp = amulet_comb.copy()
                amulet_comb_tmp["_date_key"] = pd.to_datetime(
                    amulet_comb_tmp["Date"], errors="coerce"
                ).dt.strftime("%Y-%m-%d")
                card_lookup9 = (
                    amulet_comb_tmp[["Name", "_date_key", selected_card9]]
                    .drop_duplicates(subset=["Name", "_date_key"])
                )
                plot9 = plot9.drop(columns=[selected_card9], errors="ignore")
                plot9 = plot9.merge(card_lookup9, on=["Name", "_date_key"], how="left")
                plot9 = plot9.drop(columns=["_date_key"], errors="ignore")
                plot9[selected_card9] = (
                    pd.to_numeric(plot9[selected_card9], errors="coerce")
                      .fillna(0).astype(int)
                )

        if "Date" in plot9.columns:
            plot9 = plot9.copy()
            plot9["Date"] = pd.to_datetime(
                plot9["Date"], errors="coerce"
            ).dt.strftime("%m-%d-%Y").fillna(plot9["Date"].astype(str))

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
        sel9 = st.plotly_chart(fig9, width='stretch', on_select="rerun", key="nmds_plot9")

        # ── Click-to-decklist ─────────────────────────────────────────────
        selected_pts9 = (sel9 or {}).get("selection", {}).get("points", [])
        if selected_pts9:
            pt9 = selected_pts9[0]
            cd9 = pt9.get("customdata", [])
            if len(cd9) >= 2:
                _click_name9 = cd9[0]
                _click_date9 = cd9[1]
                _click_match9 = _match_amulet_row(_click_name9, _click_date9)
                if not _click_match9.empty:
                    _click_ord9 = pd.DataFrame([{
                        "Name": _click_name9,
                        "Date": _click_date9,
                        "current_era": cd9[2] if len(cd9) > 2 else None,
                    }])
                    st.markdown("---")
                    st.markdown("### 🃏 Selected Deck")
                    _render_nmds_decklist(0, _click_ord9)
                else:
                    st.info("Decklist not found in source data.")

    else:
        st.info("NMDS site scores not found in data. Check that merged_amulet.csv contains NMDS1 and NMDS2 columns.")


# ── Tab 10: NMDS – Card Similarity ────────
with tab10:
    st.subheader("NMDS – Card Similarity (WA Species Scores)")
    st.markdown(
        "Cards plotted by their **weighted-average (WA) position** in NMDS space. "
        "Cards that co-occur in similar decklists cluster together. "
        "Unlike PCA loadings, WA scores reflect ecological co-occurrence in "
        "Bray-Curtis space rather than linear correlation."
    )

    _, species_nmds10, _, _ = _resolve_nmds()

    if species_nmds10 is not None:
        wa = species_nmds10.copy()
        if "card" not in wa.columns:
            wa = wa.reset_index()
            wa = wa.rename(columns={wa.columns[0]: "card"})

        # ── Filters ──────────────────────────────────────────────────────
        fc1, fc2 = st.columns(2)
        with fc1:
            sb_filter10 = st.radio(
                "Show cards:",
                options=["Maindeck only", "Sideboard (SB) only", "All"],
                horizontal=True,
                key="nmds_sim_sb",
            )
        with fc2:
            color_mode10 = st.radio(
                "Color by:",
                options=["Card type", "Maindeck / Sideboard"],
                horizontal=True,
                key="nmds_sim_color",
            )

        # ── Copy filter: only show cards with ≥30 maindeck copies ────────
        if "card" in wa.columns:
            card_totals_mb10 = amulet_comb[
                [c for c in amulet_int.columns if not c.startswith("sb_")]
            ].sum(axis=0)

        def is_sb(name):
            return str(name).startswith("sb_") or "(SB)" in str(name)

        if sb_filter10 == "Maindeck only":
            # Maindeck cards with ≥30 copies
            wa = wa[wa["card"].apply(lambda c: not is_sb(c))]
            if "card" in wa.columns:
                wa = wa[wa["card"].apply(
                    lambda c: card_totals_mb10.get(c, 0) >= 30
                )]
        elif sb_filter10 == "Sideboard (SB) only":
            wa = wa[wa["card"].apply(is_sb)]
        # "All" keeps everything

        wa = wa.copy()
        wa["card_type"] = wa["card"].apply(get_card_type)
        wa["deck_slot"] = wa["card"].apply(
            lambda s: "Sideboard" if is_sb(s) else "Maindeck"
        )

        if color_mode10 == "Card type":
            color_map10 = {
                "Land":     "#2ca02c",
                "Creature": "#1f77b4",
                "Spell":    "#ff7f0e",
                "Unknown":  "#7f7f7f",
            }
            wa["_color"] = wa["card_type"]
        else:
            color_map10 = {"Maindeck": "#00d4ff", "Sideboard": "#d62728"}
            wa["_color"] = wa["deck_slot"]

        fig10 = px.scatter(
            wa,
            x="NMDS1", y="NMDS2",
            text="card",
            hover_name="card",
            hover_data={"card_type": True, "deck_slot": True,
                        "NMDS1": False, "NMDS2": False, "_color": False},
            color="_color",
            color_discrete_map=color_map10,
            category_orders={"_color": list(color_map10.keys())},
            template="plotly_dark",
        )
        fig10.update_traces(
            mode="markers+text",
            textposition="top center",
            textfont=dict(size=9, color="rgba(255,255,255,0.75)"),
            marker=dict(size=7, line=dict(width=0)),
        )
        fig10.update_xaxes(title_text="NMDS1 (no units)", showgrid=True,
                           gridcolor="#333", zeroline=False)
        fig10.update_yaxes(title_text="NMDS2 (no units)", showgrid=True,
                           gridcolor="#333", zeroline=False)
        fig10.update_layout(
            title="Card WA Scores in NMDS Space",
            plot_bgcolor="#1a1a2e",
            paper_bgcolor="#0d0d1a",
            legend_title_text="Card Type" if color_mode10 == "Card type" else "Deck Slot",
            legend=dict(font=dict(size=11), itemsizing="constant"),
            height=1000,
        )
        st.plotly_chart(fig10, width='stretch')

    else:
        st.info("NMDS site scores not found in data. Check that merged_amulet.csv contains NMDS1 and NMDS2 columns.")
