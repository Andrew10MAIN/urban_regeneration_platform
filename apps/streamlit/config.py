"""
Visual constants — colours, colormaps, layer definitions.
Mirrors the Setup section of dashbord_prototyping.ipynb.
"""

from matplotlib.colors import LinearSegmentedColormap

# ── Brand colours (RGBA) ────────────────────────────────────────────────────
PREZ_BLUE   = (2/255,   87/255,  163/255, 1.0)
PREZ_BLUE2  = (2/255,   87/255,  163/255, 0.69)
PREZ_BLUE3  = (2/255,   87/255,  163/255, 0.35)
PREZ_RED    = (227/255, 26/255,  28/255,  1.0)
PREZ_RED2   = (227/255, 26/255,  28/255,  0.69)
PREZ_RED3   = (227/255, 26/255,  28/255,  0.35)
PREZ_YELLOW = (224/255, 184/255, 24/255,  1.0)
PREZ_GREY   = (210/255, 214/255, 216/255, 1.0)
PREZ_GREEN  = (27/255,  158/255, 119/255, 1.0)
PREZ_PURPLE = (117/255, 107/255, 177/255, 1.0)
PREZ_BROWN  = (140/255, 81/255,  10/255,  1.0)
PREZ_WHITE  = (1.0, 1.0, 1.0, 1.0)
PREZ_BLACK  = "#000000"

# Named colour dict (for dropdowns)
COLOURS: dict = {
    "blue":   PREZ_BLUE,
    "red":    PREZ_RED,
    "green":  PREZ_GREEN,
    "yellow": PREZ_YELLOW,
    "grey":   PREZ_GREY,
    "purple": PREZ_PURPLE,
    "brown":  PREZ_BROWN,
    "black":  PREZ_BLACK,
}

# ── Heatmap colormaps ───────────────────────────────────────────────────────
_b = PREZ_BLUE[:3]
_r = PREZ_RED[:3]
_y = (224/255, 184/255, 24/255)
_g = PREZ_GREY[:3]

CMAP_BLUE   = LinearSegmentedColormap.from_list("custom_blue",   [(1,1,1), _b])
CMAP_RED    = LinearSegmentedColormap.from_list("custom_red",    [(1,1,1), _r])
CMAP_YELLOW = LinearSegmentedColormap.from_list("custom_yellow", [(1,1,1), _y])
CMAP_GREY   = LinearSegmentedColormap.from_list("custom_grey",   [(1,1,1), _g])
CMAP_DIV    = LinearSegmentedColormap.from_list(
    "blue_white_red",
    [(0.0, (0/255, 70/255, 180/255)),
     (0.5, (1, 1, 1)),
     (1.0, (180/255, 0/255, 0/255))],
)

# Saturated variants (for price overlay)
CMAP_BLUE2   = LinearSegmentedColormap.from_list("custom_blue2",   [(1.0, 0.65, 0.65), _b])
CMAP_RED2    = LinearSegmentedColormap.from_list("custom_red2",    [(1.0, 0.65, 0.65), _r])
CMAP_YELLOW2 = LinearSegmentedColormap.from_list("custom_yellow2", [(1.0, 0.65, 0.65), _y])

HEATMAP_CMAPS: dict = {
    "blue":          CMAP_BLUE,
    "red":           CMAP_RED,
    "yellow":        CMAP_YELLOW,
    "grey":          CMAP_GREY,
    "blue_white_red": CMAP_DIV,
}

SAT_CMAPS: dict = {
    "blue":   CMAP_BLUE2,
    "red":    CMAP_RED2,
    "yellow": CMAP_YELLOW2,
}

# ── Treatment encoding ──────────────────────────────────────────────────────
TREATMENT_SHORTCUT = {"direct": "d1nq", "indirect": "1nq"}
TREATMENT_NUMBER   = {"direct": 2,      "indirect": 1}

# ── OSM / mined point-layer definitions ────────────────────────────────────
OSM_FCLASS_GROUPS: dict[str, list[str]] = {
    "small catering business": [
        "sports_centre", "playground", "school",
        "pub", "restaurant", "cafe", "bar", "fast_food",
        "bakery", "butcher", "greengrocer", "kiosk",
    ],
    "alcohol outlets": ["pub", "bar", "nightclub"],
    "food services":   ["pub", "restaurant", "cafe", "bar", "fast_food"],
    "pharmacies":      ["pharmacy"],
    "accommodation":   ["hotel", "motel", "hostel"],
    "health services": ["dentist", "clinic"],
}

PENALTIES_TYPES = ["alcohol_consumption", "offense"]

# ── Per-model colours for Policy Recommendations ────────────────────────────
MODEL_COLOUR_CYCLE = ["blue", "red", "green", "yellow", "purple", "brown"]

# ── Parallel Trends group styling ───────────────────────────────────────────
PT_GROUPS: dict = {
    0: (PREZ_GREY,  "-",  "Control group"),
    1: (PREZ_RED,   "-",  "Indirectly treated"),
    2: (PREZ_BLUE,  "-",  "Directly treated"),
}

# ── Regeneration programme bounds ───────────────────────────────────────────
REGEN_START = 2020
REGEN_END   = 2024
