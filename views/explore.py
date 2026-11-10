import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from src.ui import ROOM_COLORS, SEQ_BLUE, VALUE_COLORS, listings, money

df = listings()

st.title("Explore listings")
st.caption("Use the filters in the sidebar. The map, numbers and table all update together.")

# ---------------------------------------------------------------- filters
with st.sidebar:
    st.header("Filters")
    boroughs = st.multiselect("Borough", sorted(df["neighbourhood_group"].unique()),
                              placeholder="All boroughs")
    nb_options = sorted(df.loc[df["neighbourhood_group"].isin(boroughs) if boroughs else slice(None),
                               "neighbourhood"].unique())
    hoods = st.multiselect("Neighborhood", nb_options, placeholder="All neighborhoods")
    rooms = st.pills("Room type", list(ROOM_COLORS), selection_mode="multi", default=list(ROOM_COLORS))
    price = st.slider("Nightly price ($)", 0, 1000, (0, 500), step=10,
                      help="The top of the slider ($1,000) also includes more expensive listings.")
    max_nights = st.slider("Minimum stay at most (nights)", 1, 60, 60,
                           help="60 includes every listing, also those with longer minimum stays.")
    min_reviews = st.number_input("At least this many reviews", 0, 600, 0, step=5)
    values = st.multiselect(
        "Price compared with the model", list(VALUE_COLORS), placeholder="Any",
        help="Compares each listing's real price with the model's 80% range for a listing like it.",
    )
    query = st.text_input("Title contains", placeholder="e.g. loft, view, brownstone")

mask = pd.Series(True, index=df.index)
if boroughs:
    mask &= df["neighbourhood_group"].isin(boroughs)
if hoods:
    mask &= df["neighbourhood"].isin(hoods)
mask &= df["room_type"].isin(rooms or [])
mask &= df["price"] >= price[0]
if price[1] < 1000:
    mask &= df["price"] <= price[1]
if max_nights < 60:
    mask &= df["minimum_nights"] <= max_nights
mask &= df["number_of_reviews"] >= min_reviews
if values:
    mask &= df["value_label"].isin(values)
if query:
    mask &= df["name"].str.contains(query, case=False, regex=False)
view = df[mask]

if view.empty:
    st.warning("No listings match these filters. Try widening the price range or clearing a filter.")
    st.stop()

# ---------------------------------------------------------------- headline numbers
c1, c2, c3, c4 = st.columns(4)
c1.metric("Matching listings", f"{len(view):,}", f"{len(view) / len(df):.0%} of all", delta_color="off",
          delta_arrow="off")
c2.metric("Median price", money(view["price"].median()))
c3.metric("Median minimum stay", f"{view['minimum_nights'].median():.0f} nights")
below = (view["value_label"] == "Below typical range").mean()
c4.metric("Priced below typical", f"{below:.0%}",
          help="Share of matching listings whose price is under the model's 80% range.")

# ---------------------------------------------------------------- map
color_by = st.segmented_control(
    "Color points by", ["Price", "Room type", "Price vs model"], default="Price",
)


def hex_rgb(h: str) -> list[int]:
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


mapdf = view[["id", "name", "neighbourhood", "room_type", "price", "predicted_price",
              "value_label", "latitude", "longitude", "number_of_reviews"]].copy()
if color_by == "Room type":
    mapdf["color"] = mapdf["room_type"].map(lambda r: hex_rgb(ROOM_COLORS[r]))
    legend = " · ".join(f"<span style='color:{c}'>●</span> {r}" for r, c in ROOM_COLORS.items())
elif color_by == "Price vs model":
    mapdf["color"] = mapdf["value_label"].map(lambda v: hex_rgb(VALUE_COLORS[v]))
    legend = " · ".join(f"<span style='color:{c}'>●</span> {v}" for v, c in VALUE_COLORS.items())
else:
    bins = [0, 50, 75, 100, 150, 200, 300, np.inf]
    labels = ["< $50", "$50–75", "$75–100", "$100–150", "$150–200", "$200–300", "$300+"]
    idx = np.clip(np.digitize(mapdf["price"], bins[1:-1]), 0, len(SEQ_BLUE) - 1)
    mapdf["color"] = [hex_rgb(SEQ_BLUE[i]) for i in idx]
    legend = " ".join(f"<span style='color:{c}'>■</span> {lab}" for c, lab in zip(SEQ_BLUE, labels))
mapdf["price_txt"] = mapdf["price"].map(money)
mapdf["pred_txt"] = mapdf["predicted_price"].map(money)

layer = pdk.Layer(
    "ScatterplotLayer", mapdf, get_position=["longitude", "latitude"], get_fill_color="color",
    get_radius=35, radius_min_pixels=2, radius_max_pixels=12, opacity=0.75, pickable=True,
    stroked=False,
)
vs = pdk.ViewState(latitude=float(mapdf["latitude"].median()), longitude=float(mapdf["longitude"].median()),
                   zoom=10 if len(view) > 2000 else 12, pitch=0)
tooltip = {
    "html": "<b>{name}</b><br/>{neighbourhood} · {room_type}<br/>Price: <b>{price_txt}</b> · "
            "Model estimate: {pred_txt}<br/>{value_label} · {number_of_reviews} reviews",
    "style": {"fontSize": "13px"},
}
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=vs, tooltip=tooltip, map_style=None),
                height=520)
st.markdown(f"<small>{legend}</small>", unsafe_allow_html=True)

# ---------------------------------------------------------------- table + download
st.subheader("Matching listings")
sort_by = st.selectbox(
    "Sort by", ["Best value first (furthest below model estimate)", "Price: low to high",
                "Price: high to low", "Most reviews"],
)
table = view.copy()
if sort_by.startswith("Best value"):
    table = table.sort_values("price_vs_predicted_pct")
elif sort_by == "Price: low to high":
    table = table.sort_values("price")
elif sort_by == "Price: high to low":
    table = table.sort_values("price", ascending=False)
else:
    table = table.sort_values("number_of_reviews", ascending=False)
table["url"] = "https://www.airbnb.com/rooms/" + table["id"].astype(str)
cols = ["name", "neighbourhood_group", "neighbourhood", "room_type", "price", "predicted_price",
        "price_vs_predicted_pct", "value_label", "minimum_nights", "number_of_reviews",
        "availability_365", "url"]
st.dataframe(
    table[cols].head(1000),
    hide_index=True,
    width="stretch",
    column_config={
        "name": st.column_config.TextColumn("Listing", width="large"),
        "neighbourhood_group": "Borough",
        "neighbourhood": "Neighborhood",
        "room_type": "Room type",
        "price": st.column_config.NumberColumn("Price", format="$%d"),
        "predicted_price": st.column_config.NumberColumn("Model estimate", format="$%d"),
        "price_vs_predicted_pct": st.column_config.NumberColumn("vs estimate", format="%+.0f%%"),
        "value_label": "Compared with model",
        "minimum_nights": "Min nights",
        "number_of_reviews": "Reviews",
        "availability_365": "Days available",
        "url": st.column_config.LinkColumn("Link", display_text="Open"),
    },
)
if len(table) > 1000:
    st.caption(f"Showing the first 1,000 of {len(table):,} listings. The download has all of them.")
st.download_button(
    "Download matching listings (CSV)", table[cols].to_csv(index=False).encode(),
    file_name="nyc_airbnb_filtered.csv", mime="text/csv", icon=":material/download:",
)
st.caption("Links point to the listing as it was in 2019; many have since been taken down.")
