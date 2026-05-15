def get_recommendations(body_type, occasion, weather, preferred_style, preferred_color):
    recommendations = []

    if body_type == "slim":
        fit = "slim fit"
    elif body_type == "athletic":
        fit = "regular fit"
    elif body_type == "plus":
        fit = "relaxed fit"
    else:
        fit = "regular fit"


    if occasion == "formal":
        if preferred_style == "western":
            clothing = "Blazer + Trousers + Formal Shirt"
        elif preferred_style == "ethnic":
            clothing = "Kurta + Churidar"
        else:
            clothing = "Formal Shirt + Trousers"

    elif occasion == "casual":
        if preferred_style == "western":
            clothing = "T-Shirt + Jeans"
        elif preferred_style == "ethnic":
            clothing = "Casual Kurta + Palazzo"
        else:
            clothing = "Polo Shirt + Chinos"

    elif occasion == "party":
        if preferred_style == "western":
            clothing = "Party Dress / Blazer + Slim Jeans"
        elif preferred_style == "ethnic":
            clothing = "Anarkali / Sherwani"
        else:
            clothing = "Stylish Outfit with Accessories"

    elif occasion == "sports":
        clothing = "Track Pants + Dry Fit T-Shirt + Sneakers"

    else:
        clothing = "Comfortable Casual Wear"

    # ─── WEATHER LOGIC ───
    if weather == "hot":
        fabric = "Cotton / Linen"
        avoid  = "Avoid dark colors and heavy fabrics"
    elif weather == "cold":
        fabric = "Wool / Fleece / Layered clothing"
        avoid  = "Avoid thin fabrics"
    elif weather == "rainy":
        fabric = "Quick dry / Synthetic fabric"
        avoid  = "Avoid white and light colors"
    else:
        fabric = "Any comfortable fabric"
        avoid  = ""

    # ─── COLOR LOGIC ───
    if preferred_color == "neutral":
        colors = "White, Black, Grey, Beige"
    elif preferred_color == "bright":
        colors = "Red, Yellow, Blue, Green"
    elif preferred_color == "pastel":
        colors = "Light Pink, Lavender, Mint, Peach"
    elif preferred_color == "dark":
        colors = "Navy, Dark Green, Maroon, Charcoal"
    else:
        colors = "Any color based on preference"

    # ─── FINAL RECOMMENDATION ───
    recommendations.append(f"Outfit     : {clothing}")
    recommendations.append(f"Fit        : {fit}")
    recommendations.append(f"Fabric     : {fabric}")
    recommendations.append(f"Colors     : {colors}")
    if avoid:
        recommendations.append(f"Avoid      : {avoid}")

    return recommendations


# ─── TEST THE LOGIC ───
print("=" * 45)
print("   AI Clothing Recommendation System")
print("=" * 45)

user = {
    "body_type"       : "athletic",   # slim / athletic / plus
    "occasion"        : "casual",     # formal / casual / party / sports
    "weather"         : "hot",        # hot / cold / rainy
    "preferred_style" : "western",    # western / ethnic
    "preferred_color" : "neutral"     # neutral / bright / pastel / dark
}

print(f"\nUser Preferences:")
for key, value in user.items():
    print(f"  {key.replace('_', ' ').title()}: {value}")

print(f"\nRecommendations:")
results = get_recommendations(**user)
for r in results:
    print(f"  {r}")

print("=" * 45)