KEYPOINT_MAP = {
    0: "withers",
    1: "back",
    2: "hook up",
    3: "hook down",
    4: "hip",
    5: "tail head",
    6: "pin up",
    7: "pin down",
}


POINT_CONNECTIONS = [
    ("withers", "back"),
    ("withers", "hook up"),
    ("withers", "hook down"),
    ("back", "hip"),
    ("back", "hook up"),
    ("back", "hook down"),
    ("hook up", "hook down"),
    ("hook up", "hip"),
    ("hook down", "hip"),
    ("hip", "tail head"),
    ("hook up", "tail head"),
    ("hook down", "tail head"),
    ("hook up", "pin up"),
    ("hook down", "pin down"),
    ("tail head", "pin up"),
    ("tail head", "pin down"),
    ("pin up", "pin down"),
]

ANGLE_TRIPLETS = [
    ("withers", "back", "hook up"),
    ("withers", "back", "hook down"),
    ("withers", "hook up", "hook down"),
    ("back", "hook up", "hook down"),
    ("back", "hook up", "hip"),
    ("back", "hook down", "hip"),
    ("hook up", "hook down", "hip"),
    ("hook up", "hook down", "tail head"),
    ("hook up", "tail head", "pin up"),
    ("hook down", "tail head", "pin down"),
    ("tail head", "pin up", "pin down"),
]
