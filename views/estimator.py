import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from src.features import BOROUGHS, ROOM_TYPES, haversine_km
from src.ui import ROOM_COLORS, listings, model, money, style_fig

df = listings()
mdl = model()

st.title("Price estimator")
st.caption(
    "Describe a listing and the model estimates its nightly price. Results update as you change any input."
)

centroids = df.groupby("neighbourhood")[["latitude", "longitude"]].median()
nb_borough = df.groupby("neighbourhood")["neighbourhood_group"].first()
nb_counts = df["neighbourhood"].value_counts()

# ---------------------------------------------------------------- inputs
left, right = st.columns([2, 3], gap="large")
with left:
    st.subheader("Your listing")
    title = st.text_input(
        "Listing title (optional)", value="Sunny 2BR apartment with balcony near the park",
        help="The model reads signals like '2BR', 'studio', 'luxury', 'loft', 'view' or 'cozy' from the title.",
    )
    borough = st.selectbox("Borough", BOROUGHS, index=1)
    hoods = sorted(nb_borough[nb_borough == borough].index)
    default_nb = "Williamsburg" if "Williamsburg" in hoods else hoods[0]
    hood = st.selectbox("Neighborhood", hoods, index=hoods.index(default_nb),
                        format_func=lambda n: f"{n} ({nb_counts.get(n, 0):,} listings)")
    room = st.segmented_control("Room type", ROOM_TYPES, default=ROOM_TYPES[0]) or ROOM_TYPES[0]
    c1, c2 = st.columns(2)
    min_nights = c1.number_input("Minimum nights", 1, 365, 2)
    availability = c2.number_input("Days available per year", 0, 365, 180,
                                   help="How many days of the next year the listing can be booked.")
    asking = st.number_input("Your asking price, $ (optional)", 0, 10000, 0, step=5,
                             help="Enter a price to see how it compares with similar listings.")

    with st.expander("Location and review history"):
        lat0, lon0 = centroids.loc[hood]
        lat = st.number_input("Latitude", 40.49, 40.92, float(round(lat0, 5)), step=0.001, format="%.5f",
                              key=f"lat_{hood}")
        lon = st.number_input("Longitude", -74.26, -73.69, float(round(lon0, 5)), step=0.001, format="%.5f",
                              key=f"lon_{hood}")
        st.caption("Defaults to the center of the neighborhood. Paste coordinates from a map for a precise spot.")
        status = st.radio("Review history", ["Established listing", "New listing (no reviews yet)"],
                          horizontal=True)
        if status.startswith("Established"):
            n_reviews = st.number_input("Number of reviews", 1, 1000, 20)
            rpm = st.number_input("Reviews per month", 0.01, 30.0, 1.0, step=0.1)
            days_since = st.number_input("Days since the last review", 0, 3000, 30)
        else:
            n_reviews, rpm, days_since = 0, 0.0, np.nan
        host_count = st.number_input("Listings this host has on Airbnb", 1, 400, 1)

dist_from_center = float(haversine_km(lat, lon, lat0, lon0))


def make_row(**over) -> pd.DataFrame:
    row = dict(
        name=title, neighbourhood_group=borough, neighbourhood=hood, room_type=room,
        latitude=lat, longitude=lon, minimum_nights=min_nights, number_of_reviews=n_reviews,
        reviews_per_month=rpm, calculated_host_listings_count=host_count,
        availability_365=availability, days_since_last_review=days_since,
    )
    row.update(over)
    return pd.DataFrame([row])


row = make_row()
pred = mdl.predict(row).iloc[0]

# ---------------------------------------------------------------- headline
with right:
    st.subheader("Estimated nightly price")
    with st.container(border=True):
        a, b = st.columns([1, 1])
        a.metric("Model estimate", money(pred["predicted_price"]))
        b.metric("Likely range (80%)", f"{money(pred['price_low'])} – {money(pred['price_high'])}",
                 help="About 8 in 10 comparable real listings are priced inside this range.")
        if asking:
            diff = asking / pred["predicted_price"] - 1
            if asking < pred["price_low"]:
                st.info(f":material/trending_down: {money(asking)} is **{abs(diff):.0%} below** the estimate and "
                        "under the typical range. You may be leaving money on the table.")
            elif asking > pred["price_high"]:
                st.warning(f":material/trending_up: {money(asking)} is **{diff:.0%} above** the estimate and over "
                           "the typical range for a listing like this.")
            else:
                st.success(f":material/check_circle: {money(asking)} is within the typical range "
                           f"({diff:+.0%} vs the estimate).")
    if dist_from_center > 2.5:
        st.warning(f"The coordinates are {dist_from_center:.1f} km from the center of {hood}. "
                   "Check that the neighborhood and the coordinates match.")
    if nb_counts.get(hood, 0) < 20:
        st.caption(f":material/info: {hood} has only {nb_counts.get(hood, 0)} listings, so this estimate leans "
                   "more on borough and location than on the neighborhood itself.")

    # ------------------------------------------------------------ explanation
    base, effects = mdl.explain(row)
    top = effects.head(8)  # largest effects first
    rest = effects.iloc[8:]["log_effect"].sum()
    st.markdown(f"**Why this price?** Starting from a typical NYC listing at **{money(base)}**, "
                "each factor moves the estimate up or down:")
    labels = top["label"].tolist() + ["All other factors"]
    pct = top["pct_effect"].tolist() + [(np.exp(rest) - 1) * 100]
    fig = go.Figure(go.Bar(
        y=labels, x=pct, orientation="h",
        marker_color=["#2a78d6" if v >= 0 else "#e34948" for v in pct], marker_line_width=0,
        text=[f"{v:+.0f}%" for v in pct], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:+.1f}%<extra></extra>",
    ))
    fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Effect on price (%)",
                      showlegend=False, margin=dict(l=10, r=40, t=10, b=10))
    st.plotly_chart(style_fig(fig, height=360), width="stretch")
    st.caption("Blue raises the price, red lowers it. Effects multiply: +20% and −10% together give about +8%.")

# ---------------------------------------------------------------- what-if
st.divider()
st.subheader("What if…")
w1, w2 = st.columns(2)
with w1:
    rt = pd.concat([make_row(room_type=r) for r in ROOM_TYPES], ignore_index=True)
    rp = mdl.predict(rt)
    fig = go.Figure(go.Bar(
        x=ROOM_TYPES, y=rp["predicted_price"], marker_color=[ROOM_COLORS[r] for r in ROOM_TYPES],
        marker_line_width=0,
        error_y=dict(type="data", symmetric=False, array=rp["price_high"] - rp["predicted_price"],
                     arrayminus=rp["predicted_price"] - rp["price_low"], color="#898781", thickness=1.5),
        text=rp["predicted_price"].map(money), textposition="inside",
        customdata=np.stack([rp["price_low"], rp["price_high"]], axis=1),
        hovertemplate="%{x}: $%{y:,.0f}<br>80% range $%{customdata[0]:,.0f}–$%{customdata[1]:,.0f}<extra></extra>",
    ))
    fig.update_layout(title="…it were a different room type?", yaxis_title="Estimated price ($)", showlegend=False)
    st.plotly_chart(style_fig(fig, height=340), width="stretch")
with w2:
    nights = [1, 2, 3, 4, 5, 7, 10, 14, 21, 30, 60, 90]
    mn = mdl.predict(pd.concat([make_row(minimum_nights=n) for n in nights], ignore_index=True))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nights + nights[::-1], y=list(mn["price_high"]) + list(mn["price_low"])[::-1],
                             fill="toself", fillcolor="rgba(42,120,214,0.15)", line=dict(width=0),
                             hoverinfo="skip", name="80% range"))
    fig.add_trace(go.Scatter(x=nights, y=mn["predicted_price"], mode="lines+markers", name="Estimate",
                             line=dict(color="#2a78d6", width=2), marker=dict(size=8),
                             hovertemplate="%{x} night minimum: $%{y:,.0f}<extra></extra>"))
    fig.update_layout(title="…it had a different minimum stay?", xaxis_title="Minimum nights (log scale)",
                      yaxis_title="Estimated price ($)", xaxis_type="log", showlegend=False)
    st.plotly_chart(style_fig(fig, height=340), width="stretch")

# ---------------------------------------------------------------- comparables
st.subheader("Similar listings nearby")
comps = df[(df["room_type"] == room) & df["in_training_range"]].copy()
comps["distance_km"] = haversine_km(lat, lon, comps["latitude"].to_numpy(), comps["longitude"].to_numpy())
comps = comps.nsmallest(15, "distance_km")
far = comps["distance_km"].max()
zoom = 16 if far < 0.3 else 15 if far < 0.8 else 14 if far < 1.6 else 13 if far < 3 else 12
m1, m2 = st.columns([3, 2])
with m1:
    comps["color"] = [[42, 120, 214]] * len(comps)
    comps["price_txt"] = comps["price"].map(money)
    you = pd.DataFrame({"latitude": [lat], "longitude": [lon], "color": [[227, 73, 72]],
                        "name": ["Your listing"], "price_txt": [money(pred["predicted_price"]) + " (estimate)"],
                        "room_type": [room]})
    layers = [
        pdk.Layer("ScatterplotLayer", comps, get_position=["longitude", "latitude"], get_fill_color="color",
                  get_radius=25, radius_min_pixels=6, pickable=True, stroked=True, get_line_color=[255, 255, 255],
                  line_width_min_pixels=2),
        pdk.Layer("ScatterplotLayer", you, get_position=["longitude", "latitude"], get_fill_color="color",
                  get_radius=40, radius_min_pixels=10, pickable=True, stroked=True, get_line_color=[255, 255, 255],
                  line_width_min_pixels=2),
    ]
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom),
                             tooltip={"html": "<b>{name}</b><br/>{room_type} · {price_txt}"}, map_style=None),
                    height=380)
    st.markdown("<small><span style='color:#e34948'>●</span> Your listing · "
                "<span style='color:#2a78d6'>●</span> 15 nearest listings of the same room type</small>",
                unsafe_allow_html=True)
with m2:
    st.metric("Median price of these 15", money(comps["price"].median()),
              f"{comps['price'].median() / pred['predicted_price'] - 1:+.0%} vs your estimate", delta_color="off",
              delta_arrow="off")
    st.dataframe(
        comps[["name", "price", "distance_km", "number_of_reviews"]],
        hide_index=True, width="stretch", height=300,
        column_config={"name": st.column_config.TextColumn("Listing", width="medium"),
                       "price": st.column_config.NumberColumn("Price", format="$%d"),
                       "distance_km": st.column_config.NumberColumn("km away", format="%.2f"),
                       "number_of_reviews": "Reviews"},
    )

st.caption(
    "Estimates reflect 2019 Airbnb prices in New York City and do not include cleaning or service fees. "
    "The data has no bedroom, bathroom or amenity fields, so the title is the only way to tell the model about size."
)
