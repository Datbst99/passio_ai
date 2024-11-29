from vinorm import TTSnorm

def normalize_vietnamese_text(text):
    text = text.lower().replace("%,", "% ,")
    return (
        TTSnorm(text, unknown=False, lower=False, rule=True)
        .replace("..", ".")
        .replace("?.", "?")
        .replace(" .", ".")
        .replace(" ,", ",")
        .replace('"', "")
        .replace("'", "")
        .replace("!", "")
        .replace("ai", "Ây Ai")
        .replace("a.i", "Ây Ai")
        .replace("kol", "cây âu eo")
        .replace("cerave", "xê ra vi")
        .replace("₫", "đồng")
    )