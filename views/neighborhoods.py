import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ui import ACCENT, ROOM_COLORS, listings, money, style_fig

df = listings()
df = df[df["in_training_range"]]  # $10–$1,000: excludes placeholder prices

st.title("Compare neighborhoods")

f1, f2, f3 = st.columns([2, 2, 1])
borough = f1.selectbox("Borough", ["All boroughs", *sorted(df["neighbourhood_group"].unique())])
room = f2.selectbox("Room type", list(ROOM_COLORS))
min_n = f3.number_input("Min. listings per neighborhood", 5, 500, 20, step=5,
                        help="Neighborhoods with fewer listings are hidden, since their medians are unreliable.")

sub = df[df["room_type"] == room]
if borough != "All boroughs":
    sub = sub[sub["neighbourhood_group"] == borough]

stats = (
    sub.groupby(["neighbourhood", "neighbourhood_group"], observed=True)
    .agg(listings=("id", "size"), median_price=("price", "median"),
         p25=("price", lambda s: s.quantile(0.25)), p75=("price", lambda s: s.quantile(0.75)),
         median_min_nights=("minimum_nights", "median"),
         pct_below=("value_label", lambda s: (s == "Below typical range").mean() * 100),
         median_reviews=("number_of_reviews", "median"))
    .reset_index()
)
stats = stats[stats["listings"] >= min_n].sort_values("median_price", ascending=False)
if stats.empty:
    st.warning("No neighborhood has that many listings for this selection. Lower the minimum.")
    st.stop()

# ---------------------------------------------------------------- ranking
n_show = st.slider("Neighborhoods to show", 5, min(60, len(stats)), min(20, len(stats)))
top = stats.head(n_show)
st.subheader(f"Most expensive neighborhoods · {room}")
fig = go.Figure()
fig.add_trace(go.Bar(
    y=top["neighbourhood"], x=top["median_price"], orientation="h", marker_color=ACCENT,
    marker_line_width=0, name="Median price",
    customdata=np.stack([top["p25"], top["p75"], top["listings"], top["neighbourhood_group"]], axis=1),
    hovertemplate="<b>%{y}</b> (%{customdata[3]})<br>Median: $%{x:,.0f}<br>"
                  "Middle half: $%{customdata[0]:,.0f}–$%{customdata[1]:,.0f}<br>"
                  "%{customdata[2]:,} listings<extra></extra>",
    text=top["median_price"].map(money), textposition="outside", cliponaxis=False,
))
fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Median nightly price ($)", showlegend=False)
st.plotly_chart(style_fig(fig, height=max(320, 26 * n_show)), width="stretch")
st.caption(f"{len(stats)} neighborhoods meet the threshold. Hover a bar for the middle-50% price range.")

# ---------------------------------------------------------------- head to head
st.subheader("Head-to-head")
default = stats["neighbourhood"].head(3).tolist()
picked = st.multiselect("Pick up to 3 neighborhoods to compare", stats["neighbourhood"].tolist(),
                        default=default, max_selections=3)
if picked:
    comp = sub[sub["neighbourhood"].isin(picked)]
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig = go.Figure()
    for nb, col in zip(picked, colors):
        p = comp.loc[comp["neighbourhood"] == nb, "price"]
        fig.add_trace(go.Box(x=p, name=nb, marker_color=col, boxpoints=False, line_width=2,
                             hovertemplate=f"{nb}<extra></extra>"))
    fig.update_layout(xaxis_title="Nightly price ($)", showlegend=False, xaxis_range=[0, comp["price"].quantile(0.98)])
    st.plotly_chart(style_fig(fig, height=120 + 70 * len(picked)), width="stretch")
    table = stats.set_index("neighbourhood").loc[picked]
    st.dataframe(
        table[["neighbourhood_group", "listings", "median_price", "p25", "p75", "median_min_nights",
               "median_reviews", "pct_below"]],
        width="stretch",
        column_config={
            "_index": "Neighborhood", "neighbourhood_group": "Borough", "listings": "Listings",
            "median_price": st.column_config.NumberColumn("Median price", format="$%d"),
            "p25": st.column_config.NumberColumn("25th pct", format="$%d"),
            "p75": st.column_config.NumberColumn("75th pct", format="$%d"),
            "median_min_nights": st.column_config.NumberColumn("Median min nights", format="%d"),
            "median_reviews": st.column_config.NumberColumn("Median reviews", format="%d"),
            "pct_below": st.column_config.NumberColumn("% below typical", format="%.0f%%",
                                                       help="Listings priced under the model's 80% range."),
        },
    )

# ---------------------------------------------------------------- price vs popularity
st.subheader("Is cheaper more popular?")
st.caption("Each dot is a neighborhood. Size is the number of listings.")
fig = px.scatter(
    stats, x="median_price", y="median_reviews", size="listings", hover_name="neighbourhood",
    labels={"median_price": "Median nightly price ($)", "median_reviews": "Median number of reviews",
            "listings": "Listings"},
    size_max=30,
)
fig.update_traces(marker=dict(color=ACCENT, opacity=0.7, line=dict(width=1, color="white")),
                  hovertemplate="<b>%{hovertext}</b><br>Median price: $%{x:,.0f}<br>"
                                "Median reviews: %{y}<br>%{marker.size:,} listings<extra></extra>")
st.plotly_chart(style_fig(fig, height=420), width="stretch")
corr = stats[["median_price", "median_reviews"]].corr(method="spearman").iloc[0, 1]
st.caption(f"Rank correlation between price and reviews across these neighborhoods: {corr:+.2f}.")

st.download_button("Download neighborhood table (CSV)", stats.to_csv(index=False).encode(),
                   "nyc_airbnb_neighborhoods.csv", "text/csv", icon=":material/download:")
