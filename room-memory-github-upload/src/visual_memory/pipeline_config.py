"""Shared detection defaults for the CLI and upload worker."""

DEFAULT_YOLOE_MODEL = "yoloe-11l-seg.pt"
ITEM_PROMPTS = {
    "eyeglasses": "glasses", "water bottle": "water bottle", "remote control": "remote",
    "compact camera": "camera", "card holder": "card holder", "keys": "keys",
    "medicine bottle": "pill bottle",
}
SURFACES = {"desk", "bed", "dresser", "rug"}


def room_prompts(include_bottle_candidates=False):
    prompts = [*ITEM_PROMPTS.values(), *sorted(SURFACES)]
    if include_bottle_candidates:
        prompts.append("bottle")
    return prompts
