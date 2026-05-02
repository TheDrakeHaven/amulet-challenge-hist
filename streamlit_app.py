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
        # 2014
        "KTK","FRF",

        # 2015
        "BFZ","OGW",

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
        # 2014
        "2014-09-26","2015-01-23",

        # 2015
        "2015-10-02","2016-01-22",

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
        # 2015–2016
        "Pre-Splinter Twin/Summer Bloom Ban",
        "Pre-Gitaxian Probe/GGT Ban",

        # 2017–2019
        "Pre-MH1 Release",
        "Pre-Hogaak Ban",
        "Pre-Opal/Oko Ban",

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
        # 2015–2016
        "2016-01-18",  # Splinter Twin + Summer Bloom ban
        "2017-04-24",  # Probe + GGT ban

        # 2017–2019
        "2019-06-14",  # MH1 release
        "2019-08-26",  # Hogaak ban
        "2020-01-13",  # Opal + Oko ban

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
    # Original
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
    "Kher Keep","Misty Rainforest","Plains","Port of Karfell","Skyline Cascade",
    # New from merged_amulet
    "Dimir Aqueduct","Field of the Dead","Flagstones of Trokair","Gemstone Mine",
    "Grove of the Burnwillows","Hallowed Fountain","Lair of the Hydra","Mortuary Mire",
    "Northampton Farm","Overgrown Arch","Pit of Offerings","Prismatic Vista",
    "Sacred Foundry","Slayers' Stronghold","Snow-Covered Plains","Tectonic Edge",
    "Treasure Vault","Urza's Cave","Urza's Saga","Verdant Catacombs",
    "Windswept Heath","Wooded Foothills","Yavimaya, Cradle of Growth","Zagoth Triome",

    # New from 2016-2020 era
    "Aether Hub",
    "Botanical Sanctum",
    "Razorverge Thicket",
    "Temple of Mystery",

    # New from expanded dataset
    "Blooming Marsh",
    "Cascading Cataracts",
    "Copperline Gorge",
    "Endless Sands",
    "Field of Ruin",
    "Flooded Strand",
    "Gavony Township",
    "Horizon Canopy",
    "Indatha Triome",
    "Ipnu Rivulet",
    "Izzet Boilerworks",
    "Jetmir's Garden",
    "Lumbering Falls",
    "Minamo, School at Water's Edge",
    "Mirran Safehouse",
    "Mountain",
    "Nurturing Peatland",
    "Overgrown Tomb",
    "Polluted Delta",
    "Raffine's Tower",
    "Restless Vinestalk",
    "Scalding Tarn",
    "Sea Gate Restoration",
    "Sheltered Thicket",
    "Snow-Covered Island",
    "Snow-Covered Mountain",
    "Snow-Covered Swamp",
    "Spara's Headquarters",
    "Spirebluff Canal",
    "Steam Vents",
    "Sunscorched Desert",
    "Swamp",
    "Tanglepool Bridge",
    "Teetering Peaks",
    "Temple of Mystery",
    "Thespian's Stage",
    "Underground Mortuary",
    "Urborg, Tomb of Yawgmoth",
    "Watery Grave",
    "Witch's Cottage",
    "Xander's Lounge",
    "Yavimaya Coast",
    "Mana Confluence",

    # New from 2015-2026 expanded dataset
    "Adventurer's Inn",
    "Agadeem's Awakening",
    "Ancient Ziggurat",
    "Arid Mesa",
    "Barren Moor",
    "Blackcleave Cliffs",
    "Blood Crypt",
    "Bloodstained Mire",
    "Bristling Backwoods",
    "Castle Locthwain",
    "Cinder Glade",
    "City of Brass",
    "Cragcrown Pathway",
    "Darkslick Shores",
    "Darksteel Citadel",
    "Drowned Catacomb",
    "Dwarven Mine",
    "Eiganjo Castle",
    "Eiganjo, Seat of the Empire",
    "Eldrazi Temple",
    "Emeria's Call",
    "Fiery Islet",
    "Forgotten Cave",
    "Glimmerpost",
    "Glimmervoid",
    "Godless Shrine",
    "Inkmoth Nexus",
    "Inspiring Vantage",
    "Llanowar Reborn",
    "Llanowar Wastes",
    "Lonely Sandbar",
    "Lush Oasis",
    "Lórien Revealed",
    "Malakir Rebirth",
    "Marsh Flats",
    "Mikokoro, Center of the Sea",
    "Mistvault Bridge",
    "Moorland Haunt",
    "Mount Doom",
    "Mutavault",
    "Mystic Sanctuary",
    "Oboro, Palace in the Clouds",
    "Pendelhaven",
    "Primal Beyond",
    "Raugrin Triome",
    "Sanctum of Ugin",
    "Savai Triome",
    "Scavenger Grounds",
    "Sea Gate Wreckage",
    "Seachrome Coast",
    "Shelldock Isle",
    "Silent Clearing",
    "Sulfur Falls",
    "Sunbaked Canyon",
    "Sunken Ruins",
    "Tendo Ice Bridge",
    "Treetop Village",
    "Twilight Mire",
    "Unclaimed Territory",
    "Urza's Mine",
    "Urza's Power Plant",
    "Urza's Tower",
    "Wanderwine Hub",
    "Wastes",
]
 
creatures = [
    # Original
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
    "Yasharn, Implacable Earth",
    # New from merged_amulet
    "Anointed Peacekeeper","Arasta of the Endless Web","Aven Mindcensor",
    "Beanstalk Giant","Brazen Borrower","Bristly Bill, Spine Sower",
    "Chameleon Colossus","Courser of Kruphix","Dragonlord Atarka","Drannith Magistrate",
    "Elvish Rejuvenator","Esika's Chariot","Fierce Empath","Golos, Tireless Pilgrim",
    "Hornet Queen","Jace, the Mind Sculptor","Kazandu Mammoth","Keen-Eyed Curator",
    "Klothys, God of Destiny","Linvala, Keeper of Silence","Lodestone Golem",
    "Mindslaver","Nadu, Winged Wisdom","Nullhide Ferox","Obstinate Baloth",
    "Oracle of Mul Daya","Oran-Rief Hydra","Orvar, the All-Form","Plague Engineer",
    "Ramunap Excavator","Ruric Thar, the Unbowed","Simian Spirit Guide","Spellskite",
    "Springbloom Druid","Sundering Titan","Surrak Dragonclaw","Tarmogoyf",
    "Titan of Industry","Torpor Orb","Tyrranax Rex","Ulamog, the Infinite Gyre",
    "Uro, Titan of Nature's Wrath","Woodland Bellower","World Breaker",

    # New from 2016-2020 era
    "Hornet Nest",
    "Manglehorn",
    "Melira, Sylvok Outcast",
    "Oko, Thief of Crowns",
    "Sigarda, Heron's Grace",
    "Sigarda, Host of Herons",
    "Skyshroud Ranger",
    "Trinket Mage",
    "Wargate",
    "Wayward Swordtooth",
    "Zacama, Primal Calamity",

    # New from expanded dataset
    "Acidic Slime",
    "Aeve, Progenitor Ooze",
    "Affectionate Indrik",
    "Allosaurus Rider",
    "Archon of Cruelty",
    "Atraxa, Grand Unifier",
    "Biomancer's Familiar",
    "Boromir, Warden of the Tower",
    "Burning-Tree Shaman",
    "Carnage Tyrant",
    "Cataclysmic Gearhulk",
    "Chandra, Awakened Inferno",
    "Cloudthresher",
    "Colossal Dreadmaw",
    "Conduit of Worlds",
    "Delighted Halfling",
    "Devoted Druid",
    "Disciple of Bolas",
    "Disciple of Freyalise",
    "Duskwatch Recruiter",
    "Elesh Norn",
    "Eternal Witness",
    "Fable of the Mirror-Breaker",
    "Felidar Retreat",
    "Filigree Sages",
    "Flamekin Harbinger",
    "Froghemoth",
    "Fulminator Mage",
    "Gaea's Revenge",
    "Genesis Hydra",
    "Goblin Cratermaker",
    "Griselbrand",
    "Heroic Intervention",
    "Huntmaster of the Fells",
    "Jegantha, the Wellspring",
    "Kaheera, the Orphanguard",
    "Kinnan, Bonder Prodigy",
    "Kiora, the Crashing Wave",
    "Knight of Autumn",
    "Koma, Cosmos Serpent",
    "Lotus Cobra",
    "Muldrotha, the Gravetide",
    "Mulldrifter",
    "Mystic Snake",
    "Nissa, Resurgent Animist",
    "Nissa, Steward of Elements",
    "Nissa, Vastwood Seer",
    "Nissa, Vital Force",
    "Oko, Thief of Crowns",
    "Oko, the Ringleader",
    "Omnath, Locus of the Roil",
    "Orcish Bowmasters",
    "Panglacial Wurm",
    "Patron of the Moon",
    "Phantasmal Image",
    "Quagnoth",
    "Risen Reef",
    "Saheeli Rai",
    "Scampering Scorcher",
    "Scute Swarm",
    "Serra's Emissary",
    "Shalai, Voice of Plenty",
    "Sheoldred, the Apocalypse",
    "Shifting Ceratops",
    "Sigarda, Heron's Grace",
    "Skyshroud Ranger",
    "Solemn Simulacrum",
    "Solitude",
    "Sporeweb Weaver",
    "Stormkeld Vanguard",
    "Sylvok Replica",
    "Talisman of Impulse",
    "Tatyova, Benthic Druid",
    "Thassa's Oracle",
    "Thrun, Breaker of Silence",
    "Thrun, the Last Troll",
    "Titania, Protector of Argoth",
    "Tolsimir, Friend to Wolves",
    "Trinket Mage",
    "Ulamog, the Ceaseless Hunger",
    "Vampire Hexmage",
    "Vivien Reid",
    "Vizier of Remedies",
    "Vizier of Tumbling Sands",
    "Vorinclex, Monstrous Raider",
    "Wall of Blossoms",
    "Wayward Swordtooth",
    "Woodfall Primus",
    "Worldspine Wurm",
    "Yorion, Sky Nomad",
    "Zacama, Primal Calamity",

    # New from 2015-2026 expanded dataset
    "Arbor Elf",
    "Arcbound Ravager",
    "Arcbound Worker",
    "Archon of Emeria",
    "Auriok Champion",
    "Autochthon Wurm",
    "Avacyn's Pilgrim",
    "Birds of Paradise",
    "Blighted Agent",
    "Blood Artist",
    "Bloodghast",
    "Boulderbranch Golem",
    "Brain Maggot",
    "Burning-Tree Emissary",
    "Burrenton Forge-Tender",
    "Cavalier of Thorns",
    "Champion of the Parish",
    "Chancellor of the Tangle",
    "Cragganwick Cremator",
    "Curator of Mysteries",
    "Cursecatcher",
    "Dark Confidant",
    "Dauthi Voidwalker",
    "Death's Shadow",
    "Delver of Secrets",
    "Deputy of Detention",
    "Drogskol Captain",
    "Eidolon of Rhetoric",
    "Eidolon of the Great Revel",
    "Elderscale Wurm",
    "Elspeth, Knight-Errant",
    "Emry, Lurker of the Loch",
    "Esper Sentinel",
    "Fallaji Archaeologist",
    "Flickerwisp",
    "Frogmite",
    "Fury",
    "Gallia of the Endless Dance",
    "Giant Killer",
    "Gilded Goose",
    "Gingerbrute",
    "Giver of Runes",
    "Gladecover Scout",
    "Glistener Elf",
    "Gnarlwood Dryad",
    "Goblin Guide",
    "Golgari Thug",
    "Grief",
    "Grim Lavamancer",
    "Hangarback Walker",
    "Hapatra, Vizier of Poisons",
    "Harbinger of the Tides",
    "Healer of the Glade",
    "Hedron Crab",
    "Hidden Herbalists",
    "Hidetsugu Consumes All",
    "Ice-Fang Coatl",
    "Ignoble Hierarch",
    "Incandescent Soulstoke",
    "Jace, the Perfected Mind",
    "Kaldra Compleat",
    "Karn Liberated",
    "Kataki, War's Wage",
    "Kessig Malcontents",
    "Kitchen Finks",
    "Kitesail Freebooter",
    "Kor Firewalker",
    "Kor Outfitter",
    "Kor Skyfisher",
    "Kor Spiritdancer",
    "Kozilek, the Great Distortion",
    "Kroxa and Kunoros",
    "Kroxa, Titan of Death's Hunger",
    "Laboratory Maniac",
    "Lavinia, Azorius Renegade",
    "Ledger Shredder",
    "Leonin Arbiter",
    "Lion Sash",
    "Lord of Atlantis",
    "Lukka, Coppercoat Outcast",
    "Lurrus of the Dream-Den",
    "Magebane Lizard",
    "Mantis Rider",
    "Master of the Pearl Trident",
    "Matter Reshaper",
    "Mausoleum Wanderer",
    "Meddling Mage",
    "Melira, the Living Cure",
    "Memnite",
    "Merfolk Trickster",
    "Merrow Reejerey",
    "Metallic Mimic",
    "Militia Bugler",
    "Mirran Crusader",
    "Monastery Mentor",
    "Monastery Swiftspear",
    "Murktide Regent",
    "Narcomoeba",
    "Narnam Renegade",
    "Narset, Parter of Veils",
    "Noble Hierarch",
    "Obosh, the Preypiercer",
    "Obsidian Charmaw",
    "Omnath, Locus of Creation",
    "Omnath, Locus of Rage",
    "Ornithopter",
    "Ox of Agonas",
    "Patchwork Automaton",
    "Planebound Accomplice",
    "Prized Amalgam",
    "Puresteel Paladin",
    "Ragavan, Nimble Pilferer",
    "Rattlechains",
    "Reality Smasher",
    "Reckless Bushwhacker",
    "Reflector Mage",
    "Sanctifier en-Vec",
    "Sauron's Ransom",
    "Scavenging Ooze",
    "Scion of Draco",
    "Scrapwork Mutt",
    "Seasoned Pyromancer",
    "Selfless Spirit",
    "Shardless Agent",
    "Silvergill Adept",
    "Silversmote Ghoul",
    "Skrelv, Defector Mite",
    "Skyclave Apparition",
    "Slippery Bogle",
    "Smokebraider",
    "Snapcaster Mage",
    "Sojourner's Companion",
    "Soul-Scar Mage",
    "Spell Queller",
    "Spike Feeder",
    "Sprite Dragon",
    "Steel Overseer",
    "Stinkweed Imp",
    "Stonecoil Serpent",
    "Stoneforge Mystic",
    "Strangleroot Geist",
    "Subtlety",
    "Supreme Phantom",
    "Syr Ginger, the Meal Ender",
    "Teferi, Time Raveler",
    "Territorial Kavu",
    "Thalia's Lieutenant",
    "Thalia, Guardian of Thraben",
    "Thing in the Ice",
    "Thought Monitor",
    "Thought-Knot Seer",
    "Thunderkin Awakener",
    "Tourach, Dread Cantor",
    "Troll of Khazad-dûm",
    "Unsettled Mariner",
    "Urza, Lord High Artificer",
    "Valki, God of Lies",
    "Vesperlark",
    "Voice of Resurgence",
    "Waker of Waves",
    "Wall of Roots",
    "Wavesifter",
    "Wild Nacatl",
    "Winding Constrictor",
    "Wispmare",
    "Yargle and Multani",
    "Yawgmoth, Thran Physician",
    "Young Wolf",
    "Zabaz, the Glimmerwasp",
    "Zirda, the Dawnwaker",
    "Zulaport Cutthroat",
]
 
spells = [
    # Original
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
    "Unidentified Hovership","Wrenn and Six","Wrenn and Seven","Yggdrasil, Rebirth Engine",
    # New from merged_amulet
    "Abrade","Abrupt Decay","Abundant Harvest","Adventurous Impulse","Aether Gust",
    "Avoid Fate","Basic Conjuration","Batterskull","Broken Wings","Cartographer's Survey",
    "Celestial Purge","Chromatic Sphere","Containment Breach","Crucible of Worlds",
    "Damping Matrix","Dead // Gone","Deicide","Eladamri's Call","Elixir of Immortality",
    "Environmental Sciences","Feldon's Cane","Field Trip","Finale of Devastation",
    "Gaea's Blessing","Grafdigger's Cage","Green Sun's Twilight","Green Sun's Zenith",
    "Gut Shot","Guttural Response","Hallowed Moonlight","Hibernation",
    "Introduction to Annihilation","Journey of Discovery","Lightning Bolt",
    "Mascot Exhibition","Mystic Reflection","Mystic Repeal","Nature's Claim","Negate",
    "Path to Exile","Pest Summoning","Primal Command","Prismatic Ending",
    "Rapid Hybridization","Rest in Peace","Run Afoul","Shuko","Storm's Wrath",
    "Sudden Shock","Summoner's Pact","Summoning Trap","Sylvan Scrying",
    "Tamiyo's Safekeeping","Thorn of Amethyst","Thoughtseize","Through the Breach",
    "Timely Reinforcements","Tormod's Crypt","Unravel the Aether","Vampires' Vengeance",
    "Warping Wail","Weather the Storm","Wilt","Worldsoul's Rage",

    # New from 2016-2020 era
    "Ancient Grudge",
    "Coalition Relic",
    "Disdainful Stroke",
    "Dissenter's Deliverance",
    "Hive Mind",
    "Hurkyl's Recall",
    "Kozilek's Return",
    "Leyline of Sanctity",
    "Leyline of the Void",
    "Lotus Bloom",
    "Mycosynth Lattice",
    "Once Upon a Time",
    "Ravenous Trap",
    "Serum Visions",
    "Slaughter Pact",
    "Stony Silence",

    # New from expanded dataset
    "Academic Probation",
    "Ad Nauseam",
    "Aetherflux Reservoir",
    "All Is Dust",
    "Anger of the Gods",
    "Archdruid's Charm",
    "Argentum Masticore",
    "Assassin's Trophy",
    "Awaken the Woods",
    "Bala Ged Recovery",
    "Banefire",
    "Basilisk Collar",
    "Beseech the Mirror",
    "Bitter Reunion",
    "Blessed Respite",
    "Bolas's Citadel",
    "Bone Shards",
    "Bonfire of the Damned",
    "Broken Bond",
    "Burn Down the House",
    "Ceremonious Rejection",
    "Chromatic Star",
    "Cloudstone Curio",
    "Codex Shredder",
    "Collective Brutality",
    "Commandeer",
    "Conjurer's Bauble",
    "Consider",
    "Consulate Crackdown",
    "Crumble to Dust",
    "Cut // Ribbons",
    "Damnation",
    "Day of Judgment",
    "Deafening Clarion",
    "Delay",
    "Dispel",
    "Display of Dominance",
    "Divide by Zero",
    "Door to Nothingness",
    "Dramatic Entrance",
    "Dream's Grip",
    "Eldritch Evolution",
    "Eliminate",
    "Empty the Warrens",
    "Ephemerate",
    "Escape to the Wilds",
    "Faerie Macabre",
    "Faith's Reward",
    "Fatal Push",
    "Firebolt",
    "Flare of Cultivation",
    "Fleetwheel Cruiser",
    "Fog",
    "Force of Negation",
    "Fry",
    "Get Lost",
    "Goblin Charbelcher",
    "God-Pharaoh's Statue",
    "Grapeshot",
    "Ground Seal",
    "Hidden Strings",
    "Hour of Devastation",
    "Hour of Promise",
    "Ideas Unbound",
    "Inquisition of Kozilek",
    "Invasion of Ikoria",
    "Invasion of Ixalan",
    "Invasion of Ravnica",
    "Karn's Sylex",
    "Khalni Ambush",
    "Krosan Grip",
    "Lantern of the Lost",
    "Leyline Binding",
    "Life from the Loam",
    "Manamorphose",
    "Manifold Key",
    "Memoricide",
    "Mindbreak Trap",
    "Mine Collapse",
    "Mishra's Bauble",
    "Molder Slug",
    "Moonsilver Key",
    "Natural State",
    "Nevinyrral's Disk",
    "Oath of Nissa",
    "Otherworldly Gaze",
    "Part the Waterveil",
    "Pentad Prism",
    "Persist",
    "Porphyry Nodes",
    "Pulse of Murasa",
    "Pyrite Spellbomb",
    "Radiant Flames",
    "Ratchet Bomb",
    "Reckoner Bankbuster",
    "Recross the Paths",
    "Rending Volley",
    "Reprieve",
    "Reshape",
    "Retreat to Coralhelm",
    "Retreat to Hagra",
    "Root Maze",
    "Root Snare",
    "Sculpting Steel",
    "Shared Summons",
    "Shatterskull Smashing",
    "Shatterstorm",
    "Silundi Vision",
    "Sleight of Hand",
    "Smuggler's Surprise",
    "Song of Creation",
    "Sorcerous Spyglass",
    "Spine of Ish Sah",
    "Splendid Reclamation",
    "Spoils of the Vault",
    "Stern Scolding",
    "Stony Silence",
    "Storm the Festival",
    "Suspend",
    "Tangled Florahedron",
    "The Enigma Jewel",
    "The Filigree Sylex",
    "The Huntsman's Redemption",
    "The Underworld Cookbook",
    "The World Tree",
    "Thought Distortion",
    "Timeless Lotus",
    "Titania's Command",
    "Tome Scour",
    "Tooth and Nail",
    "Toxic Deluge",
    "Tragic Arrogance",
    "Tranquil Thicket",
    "Twiddle",
    "Underworld Breach",
    "Unholy Heat",
    "Unmarked Grave",
    "Unmoored Ego",
    "Up the Beanstalk",
    "Urban Evolution",
    "Valakut Exploration",
    "Void Snare",
    "Welding Jar",
    "Wheel of Sun and Moon",
    "Winds of Abandon",
    "Wish",
    "Wishclaw Talisman",
    "Witchbane Orb",
    "Wizard's Rockets",
    "Zuran Orb",

    # New from 2015-2026 expanded dataset
    "Abundant Growth",
    "Aether Vial",
    "Agatha's Soul Cauldron",
    "All That Glitters",
    "Alpine Moon",
    "Animation Module",
    "Anticipate",
    "Approach of the Second Sun",
    "Architects of Will",
    "Archive Trap",
    "Archmage's Charm",
    "Arcum's Astrolabe",
    "Ardent Plea",
    "Atarka's Command",
    "Atraxa's Fall",
    "Baleful Mastery",
    "Become Immense",
    "Blacksmith's Skill",
    "Blasphemous Act",
    "Blood Moon",
    "Blossoming Calm",
    "Blossoming Defense",
    "Boros Charm",
    "Boseiju, Who Shelters All",
    "Break the Ice",
    "Bring to Light",
    "Brotherhood's End",
    "Cartouche of Solidarity",
    "Cast into the Fire",
    "Cathartic Reunion",
    "Caustic Caterpillar",
    "Change the Equation",
    "Chord of Calling",
    "Chromatic Lantern",
    "Cleansing Wildfire",
    "Cling to Dust",
    "Collected Company",
    "Colossus Hammer",
    "Conflagrate",
    "Counterspell",
    "Cranial Plating",
    "Crashing Footfalls",
    "Creeping Chill",
    "Crime // Punishment",
    "Crypt Incursion",
    "Cryptic Command",
    "Dakmor Salvage",
    "Darkblast",
    "Daybreak Coronet",
    "Dead / Gone",
    "Declaration in Stone",
    "Deflecting Palm",
    "Deprive",
    "Distortion Strike",
    "Dovin's Veto",
    "Dragon's Claw",
    "Dragon's Rage Channeler",
    "Dress Down",
    "Drown in the Loch",
    "Duress",
    "Elven Chorus",
    "Ethereal Armor",
    "Expressive Iteration",
    "Exquisite Firecraft",
    "Extirpate",
    "Fade from History",
    "Farseek",
    "Feed the Swarm",
    "Feign Death",
    "Fire // Ice",
    "Flame Slash",
    "Flame of Anor",
    "Flusterstorm",
    "Forge Anew",
    "Fractured Sanity",
    "Galvanic Blast",
    "Ghor-Clan Rampager",
    "Glimpse of Tomorrow",
    "Go for the Throat",
    "Goryo's Vengeance",
    "Groundswell",
    "Growth Spiral",
    "Gryff's Boon",
    "Hardened Scales",
    "Hyena Umbra",
    "Indomitable Creativity",
    "Inevitable Betrayal",
    "Ingot Chewer",
    "Invasive Surgery",
    "Izzet Charm",
    "Kolaghan's Command",
    "Lava Dart",
    "Lava Spike",
    "Legion's End",
    "Leyline of Abundance",
    "Life Goes On",
    "Lightning Axe",
    "Lightning Helix",
    "Lightning Skelemental",
    "Living End",
    "Maelstrom Pulse",
    "Mana Tithe",
    "March of Otherworldly Light",
    "Memory Deluge",
    "Merchant of the Vale",
    "Metallic Rebuke",
    "Might of Old Krosa",
    "Mind Stone",
    "Mox Amber",
    "Murderous Cut",
    "Mutagenic Growth",
    "Mystic Forge",
    "Necromentia",
    "Neoform",
    "Nettlecyst",
    "Nihil Spellbomb",
    "Nissa, Voice of Zendikar",
    "Nissa, Who Shakes the World",
    "Nix",
    "Not Dead After All",
    "Nourishing Shoal",
    "Noxious Revival",
    "Oliphaunt",
    "Opt",
    "Ozolith, the Shattered Spire",
    "Paradise Mantle",
    "Peek",
    "Pile On",
    "Pillage",
    "Portable Hole",
    "Postmortem Lunge",
    "Prismari Command",
    "Profane Tutor",
    "Prophetic Prism",
    "Prosperous Innkeeper",
    "Rancor",
    "Ranger-Captain of Eos",
    "Remand",
    "Rift Bolt",
    "Roiling Vortex",
    "Ruin Crab",
    "Scale Up",
    "Scrabbling Claws",
    "Seal of Fire",
    "Search for Tomorrow",
    "Searing Blaze",
    "Settle the Wreckage",
    "Shadow Prophecy",
    "Shark Typhoon",
    "Shattering Spree",
    "Sheoldred's Edict",
    "Shriekhorn",
    "Sigarda's Aid",
    "Skewer the Critics",
    "Skullcrack",
    "Smash to Smithereens",
    "Spatial Contortion",
    "Spell Snare",
    "Spider Umbra",
    "Spirit Link",
    "Spirit Mantle",
    "Spreading Seas",
    "Springleaf Drum",
    "Steelshaper's Gift",
    "Stern Dismissal",
    "Strike It Rich",
    "Striped Riverwinder",
    "Stubborn Denial",
    "Summer Bloom",
    "Sunhome Enforcer",
    "Suppression Field",
    "Supreme Verdict",
    "Surge of Salvation",
    "Sword of Fire and Ice",
    "Tainted Indulgence",
    "Talisman of Resilience",
    "Tarfire",
    "Tasha's Hideous Laughter",
    "Teferi, Hero of Dominaria",
    "Temur Battle Rage",
    "Terminate",
    "The Ozolith",
    "Thought Scour",
    "Thoughtcast",
    "Thrilling Discovery",
    "Throne of Geth",
    "Thundering Falls",
    "Time Warp",
    "Torch Breath",
    "Traverse the Ulvenwald",
    "Tribal Flames",
    "Ugin, the Ineffable",
    "Umbral Mantle",
    "Undying Evil",
    "Undying Malice",
    "Unearth",
    "Unified Will",
    "Utopia Sprawl",
    "Valakut Awakening",
    "Vapor Snag",
    "Vines of Vastwood",
    "Violent Outburst",
    "Visions of Beyond",
    "Wear // Tear",
    "Whipflare",
    "Wild Cantor",
    "Worship",
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

_date_min = pd.to_datetime(amulet_env["Date"], errors="coerce").min()
_date_max = pd.to_datetime(amulet_env["Date"], errors="coerce").max()
st.caption(
    f"📊 {len(amulet_df):,} decklists · "
    f"{_date_min.strftime('%b %Y')} → {_date_max.strftime('%b %Y')} · "
    f"{amulet_env['next_ban'].nunique()} eras"
)

numeric = amulet_int.select_dtypes(include="number")
col_sums = numeric.sum()
keep_cols = col_sums[col_sums > 12].index
amulet_filtered = amulet_int[keep_cols]

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────

tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🃏 Deck Data",
    "📈 Median by Era",
    "🗺️ PCA – Era & Set",
    "🎴 PCA – Card Inclusion",
    "🃏 Card Similarity (PCA)",
    "🔍 Era-Specific Cards",
    "🌐 NMDS – Ecological Distance"
])

# ── Tab 2: Deck Data ─────────────────────
with tab2:
    subtab_data, subtab_totals = st.tabs(["📋 All Decks", "🔢 Maindeck Card Totals"])

    with subtab_data:
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
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Full table ────────────────────────────────────────────────────
        st.dataframe(
            display_totals,
            use_container_width=True,
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
        amulet_comb.groupby("next_ban")[num_cols]
        .mean()
        .reset_index()
    )
    st.markdown("**Heatmap of Mean Counts**")
    heat_data = mean_deck.set_index("next_ban")[num_cols]
    era_order = [
        "Pre-Splinter Twin/Summer Bloom Ban",
        "Pre-Gitaxian Probe/GGT Ban",
        "Pre-MH1 Release",
        "Pre-Hogaak Ban",
        "Pre-Opal/Oko Ban",
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
    st.plotly_chart(fig_heat, use_container_width=True)

# ─────────────────────────────────────────
# PCA COMPUTATION
# ─────────────────────────────────────────

ERA_ORDER = [
        "Pre-Splinter Twin/Summer Bloom Ban",
        "Pre-Gitaxian Probe/GGT Ban",
        "Pre-MH1 Release",
        "Pre-Hogaak Ban",
        "Pre-Opal/Oko Ban",
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
            for env_var in ["next_ban", "current_set"]:
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

            # Non-metric MDS — 5 random starts, keep lowest stress
            best_stress, best_coords = np.inf, None
            for seed in range(5):
                mds = MDS(
                    n_components=2, metric=False, dissimilarity="precomputed",
                    random_state=seed, n_init=1, max_iter=1000,
                    normalized_stress=True, eps=1e-6,
                )
                c = mds.fit_transform(bc_dist)
                if mds.stress_ < best_stress:
                    best_stress = mds.stress_
                    best_coords = c

            coords = best_coords
            stress = float(best_stress)

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
            for env_var in ["next_ban", "current_set"]:
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
# ── Run CCA on app start ───────────────────
if "cca_result" not in st.session_state:
    run_pca_computation()
if "nmds_result" not in st.session_state:
    run_nmds_computation()
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

        hover_cols = [c for c in ["Name", "Date", "next_ban", "current_set"]
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
        st.plotly_chart(fig2, use_container_width=True)

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

    # ── Filters ──────────────────────────────────────────────────────────
    all_species = species_scores.index.tolist()
    sb_species  = [s for s in all_species if str(s).startswith("sb_") or "(SB)" in str(s)]
    mb_species  = [s for s in all_species if not str(s).startswith("sb_") and "(SB)" not in str(s)]

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
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 7: Era-Specific Cards ─────────────
with tab7:
    st.subheader("Era-Specific Cards")
    st.markdown(
        "Cards that appear predominantly in one ban era. "
        "Concentration score = share of a card's total appearances that fall within a single era. "
        "Use the slider to control the minimum number of appearances required."
    )

    era_order_display = [e for e in ERA_ORDER if e in amulet_comb["next_ban"].values]
    card_cols_era = [c for c in amulet_int.columns]

    # Build era × card count matrix (raw) and era deck sizes
    era_card_raw = (
        amulet_comb.groupby("next_ban")[card_cols_era]
        .sum()
        .reindex(era_order_display)
        .fillna(0)
    )
    era_sizes = (
        amulet_comb.groupby("next_ban")
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
        st.plotly_chart(fig_bar, use_container_width=True)

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

        st.dataframe(result_df, use_container_width=True, hide_index=True)

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
            st.plotly_chart(fig_heat, use_container_width=True)

# ── Tab 8: NMDS – Ecological Distance ────
with tab8:
    st.subheader("NMDS – Ecological Distance (Bray-Curtis)")

    # ── Load from file OR use computed results ────────────────────────────
    uploaded_nmds = st.file_uploader(
        "Load pre-computed NMDS results (.xlsx from a previous run)",
        type=["xlsx"],
        key="nmds_upload",
    )

    if uploaded_nmds is not None:
        # Load from the uploaded Excel instead of computing
        try:
            xls = pd.ExcelFile(uploaded_nmds)
            ord_nmds       = pd.read_excel(xls, sheet_name="Site_Scores")
            species_nmds   = pd.read_excel(xls, sheet_name="Card_WA_Scores")
            stress_nmds    = None   # not stored in the file
            bc_dist        = None   # not stored in the file
            centroids_nmds = {}
            for env_var in ["current_era", "current_set"]:
                if env_var in ord_nmds.columns:
                    centroids_nmds[env_var] = (
                        ord_nmds.groupby(env_var)[["NMDS1", "NMDS2"]]
                        .mean().reset_index()
                        .rename(columns={env_var: "label"})
                    )
            st.success(f"✅ Loaded {len(ord_nmds)} site scores from uploaded file.")
        except Exception as e:
            st.error(f"Failed to load NMDS file: {e}")
            ord_nmds = None

    elif "nmds_result" in st.session_state:
        # Use the computed results
        ord_nmds       = st.session_state["nmds_result"]
        species_nmds   = st.session_state["nmds_species"]
        centroids_nmds = st.session_state["nmds_env_centroids"]
        stress_nmds    = st.session_state["nmds_stress"]
        bc_dist        = st.session_state["nmds_dist"]
    else:
        ord_nmds = None

    if ord_nmds is not None:
        # ... rest of your existing tab8 display code ...
        # Note: guard the stress metric and stressplot since they're None when loaded from file
        if stress_nmds is not None:
            m1.metric("NMDS Stress", f"{stress_nmds:.4f}")
        else:
            m1.metric("NMDS Stress", "N/A (loaded from file)")

        # Guard the stressplot expander too:
        if bc_dist is not None:
            with st.expander("🔍 Stressplot"):
                # ... stressplot code ...
    
    st.markdown(
        "**Non-metric Multidimensional Scaling** on Bray-Curtis dissimilarity. "
        "Unlike PCA/PCoA, NMDS preserves only the **rank order** of pairwise "
        "dissimilarities — axes have no absolute meaning, only relative positions matter. "
        "**Stress** (Kruskal's normalized stress) measures ordination quality."
    )

    if "nmds_result" in st.session_state:
        ord_nmds       = st.session_state["nmds_result"]
        species_nmds   = st.session_state["nmds_species"]
        centroids_nmds = st.session_state["nmds_env_centroids"]
        stress_nmds    = st.session_state["nmds_stress"]
        bc_dist        = st.session_state["nmds_dist"]

        m1, m2, m3 = st.columns(3)
        m1.metric("NMDS Stress", f"{stress_nmds:.4f}",
                  help="< 0.05 excellent · < 0.10 good · < 0.20 fair · ≥ 0.20 poor")
        m2.metric("Decklists", len(ord_nmds))
        m3.metric("Card Variables", amulet_filtered.shape[1])

        if stress_nmds < 0.05:
            st.success("✅ Stress < 0.05 — excellent.")
        elif stress_nmds < 0.10:
            st.success("✅ Stress < 0.10 — good.")
        elif stress_nmds < 0.20:
            st.info("ℹ️ Stress < 0.20 — fair; interpret carefully.")
        else:
            st.warning("⚠️ Stress ≥ 0.20 — poor fit.")

        ctrl1, ctrl2 = st.columns(2)
        with ctrl1:
            color_by_nmds = st.selectbox(
                "Color sites by:", ["next_ban", "current_set"], key="nmds_color")
        with ctrl2:
            show_centroids_nmds = st.checkbox(
                "Show centroids", value=True, key="nmds_centroids")
        show_species_nmds = st.checkbox(
            "Show top card vectors (WA biplot)", value=False, key="nmds_species_chk")

        hover_nmds = [c for c in ["Name", "Date", "next_ban", "current_set"]
                      if c in ord_nmds.columns]
        plot_nmds = ord_nmds.copy()
        plot_nmds[color_by_nmds] = (
            plot_nmds[color_by_nmds].fillna("Unknown")
            if color_by_nmds in plot_nmds.columns else "Unknown"
        )

        fig_nmds = px.scatter(
            plot_nmds, x="NMDS1", y="NMDS2",
            color=color_by_nmds,
            hover_data=[c for c in hover_nmds if c in plot_nmds.columns],
            title=f"NMDS (Bray-Curtis) – {color_by_nmds}  |  stress = {stress_nmds:.4f}",
            template="plotly_white", opacity=0.75,
        )
        fig_nmds.update_traces(marker=dict(size=7))
        fig_nmds.update_xaxes(title_text="NMDS1 (no units)")
        fig_nmds.update_yaxes(title_text="NMDS2 (no units)")

        if show_centroids_nmds and color_by_nmds in centroids_nmds:
            cents = centroids_nmds[color_by_nmds]
            fig_nmds.add_trace(go.Scatter(
                x=cents["NMDS1"], y=cents["NMDS2"],
                mode="markers+text", text=cents["label"], textposition="top center",
                marker=dict(size=14, symbol="diamond", color="black",
                            line=dict(width=1, color="white")),
                name=f"{color_by_nmds} centroids", showlegend=True,
            ))

        if show_species_nmds:
            top_n_nmds = st.slider("Top N card vectors", 5, 30, 10, key="nmds_top_n")
            sp = species_nmds.copy()
            sp["dist"] = np.sqrt(sp["NMDS1"]**2 + sp["NMDS2"]**2)
            sp_top = sp.nlargest(top_n_nmds, "dist")
            fig_nmds.add_trace(go.Scatter(
                x=sp_top["NMDS1"], y=sp_top["NMDS2"],
                mode="markers+text", text=sp_top["card"], textposition="top right",
                marker=dict(size=10, symbol="triangle-up", color="crimson"),
                name="Card WA scores", showlegend=True,
            ))

        fig_nmds.update_layout(height=800)
        st.plotly_chart(fig_nmds, use_container_width=True)

        with st.expander("🔍 Stressplot — rank-order preservation check"):
            st.markdown(
                "Original Bray-Curtis dissimilarity (x) vs NMDS distance (y). "
                "A monotone relationship is sufficient — NMDS only needs to preserve rank order."
            )
            n = bc_dist.shape[0]
            idx_i, idx_j = np.triu_indices(n, k=1)
            orig_d = bc_dist[idx_i, idx_j]
            nv     = ord_nmds[["NMDS1", "NMDS2"]].values
            ord_d  = np.sqrt(
                (nv[idx_i, 0] - nv[idx_j, 0])**2 +
                (nv[idx_i, 1] - nv[idx_j, 1])**2
            )
            if len(orig_d) > 5000:
                rng  = np.random.default_rng(42)
                sel  = rng.choice(len(orig_d), 5000, replace=False)
                orig_d = orig_d[sel]; ord_d = ord_d[sel]
            sort_idx = np.argsort(orig_d)
            mono_y   = np.maximum.accumulate(ord_d[sort_idx])
            fig_sp = px.scatter(
                x=orig_d, y=ord_d, opacity=0.25, template="plotly_white",
                labels={"x": "Bray-Curtis dissimilarity", "y": "NMDS distance"},
                title="Stressplot",
            )
            fig_sp.add_trace(go.Scatter(
                x=orig_d[sort_idx], y=mono_y, mode="lines",
                line=dict(color="red", width=1.5), name="Monotone fit",
            ))
            fig_sp.update_layout(height=450)
            st.plotly_chart(fig_sp, use_container_width=True)

        st.markdown("#### 🃏 Outlier Decklists By Era (NMDS)")
        nv = ord_nmds[["NMDS1", "NMDS2"]].values
        for era in ERA_ORDER:
            era_idx = ord_nmds.index[ord_nmds["next_ban"] == era].tolist()
            if len(era_idx) < 2:
                continue
            era_coords = nv[era_idx]
            best_mean_dist, best_idx = -1, None
            for ii, idx in enumerate(era_idx):
                others = np.delete(era_coords, ii, axis=0)
                dists  = np.sqrt(
                    (era_coords[ii, 0] - others[:, 0])**2 +
                    (era_coords[ii, 1] - others[:, 1])**2
                )
                md = dists.mean()
                if md > best_mean_dist:
                    best_mean_dist, best_idx = md, idx

            outlier_name = ord_nmds.loc[best_idx, "Name"] if "Name" in ord_nmds.columns else ""
            outlier_date = ord_nmds.loc[best_idx, "Date"] if "Date" in ord_nmds.columns else ""
            with st.expander(f"{outlier_name} ({outlier_date})  —  {era}"):
                match = amulet_comb[
                    (amulet_comb["Name"] == outlier_name) &
                    (amulet_comb["Date"].astype(str).str.contains(
                        outlier_date[:7] if outlier_date else "NOMATCH", na=False))
                ]
                if match.empty:
                    st.info("Decklist not found.")
                    continue
                deck_row    = match.iloc[0]
                card_cols_d = [c for c in amulet_int.columns if c in deck_row.index]
                decklist    = pd.Series({c: deck_row[c] for c in card_cols_d}).astype(int)
                decklist    = decklist[decklist > 0].sort_values(ascending=False)
                era_rows    = amulet_comb[amulet_comb["next_ban"] == era]
                era_cards   = era_rows[[c for c in amulet_int.columns if c in era_rows.columns]]
                median_deck = era_cards.median().round(2)
                median_deck = median_deck[median_deck > 0].sort_values(ascending=False)
                median_cards = set(median_deck[median_deck > 0].index)
                era_id_n = re.sub(r"[^a-zA-Z0-9]+", "-", era).strip("-").lower()
                col_m, col_o, col_med = st.columns([1, 1.5, 1.5])
                with col_m:
                    st.markdown(f"**{outlier_name}** — {outlier_date}")
                    st.markdown(f"Mean NMDS distance: **{best_mean_dist:.4f}**")
                    st.markdown(f"Era N: **{len(era_idx)}**")
                with col_o:
                    st.markdown("**Outlier Decklist** *(hover = card · green = not in median)*")
                    deck_df = decklist.reset_index()
                    deck_df.columns = ["Card", "Copies"]
                    deck_df = sort_by_type(deck_df, "Card")
                    render_decklist_html(deck_df, "Card", "Copies",
                        highlight_set=median_cards, table_id=f"nmds-outlier-{era_id_n}")
                with col_med:
                    st.markdown(f"**Median Decklist ({era})**")
                    med_df = median_deck.reset_index()
                    med_df.columns = ["Card", "Median Copies"]
                    med_df = sort_by_type(med_df, "Card")
                    render_decklist_html(med_df, "Card", "Median Copies",
                        highlight_set=None, table_id=f"nmds-median-{era_id_n}")
    else:
        st.info("NMDS computation failed. Check your data.")

    # ── Download NMDS data ────────────────────────────────────────────────────────
with st.expander("⬇️ Download NMDS Data"):
    dl1, dl2, dl3 = st.columns(3)

    # Site scores (deck coordinates)
    with dl1:
        st.markdown("**Site Scores** (deck coordinates)")
        csv_sites = ord_nmds.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download site scores (.csv)",
            data=csv_sites,
            file_name="nmds_site_scores.csv",
            mime="text/csv",
            key="dl_nmds_sites",
        )

    # Species scores (card WA scores)
    with dl2:
        st.markdown("**Card WA Scores** (species scores)")
        csv_species = species_nmds.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download card scores (.csv)",
            data=csv_species,
            file_name="nmds_card_scores.csv",
            mime="text/csv",
            key="dl_nmds_species",
        )

    # Excel with both sheets
    with dl3:
        st.markdown("**Both sheets** (.xlsx)")
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            ord_nmds.to_excel(writer, sheet_name="Site_Scores", index=False)
            species_nmds.to_excel(writer, sheet_name="Card_WA_Scores", index=False)
        st.download_button(
            "Download combined (.xlsx)",
            data=buf.getvalue(),
            file_name="nmds_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_nmds_excel",
        )
