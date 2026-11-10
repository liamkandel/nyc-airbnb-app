import plotly.express as px
import streamlit as st

from src.ui import ROOM_COLORS, listings, metrics, money, style_fig

df = listings()
m = metrics()

st.title("NYC Airbnb Price Explorer & Estimator")
st.markdown(
    "Explore **48,895 real Airbnb listings** across New York City's five boroughs, "
    "compare neighborhoods, and get an **instant, explainable price estimate** for any listing you describe."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Listings", f"{len(df):,}")
c2.metric("Neighborhoods", df["neighbourhood"].nunique())
c3.metric("Median nightly price", money(df["price"].median()))
c4.metric(
    "Model typical error",
    f"±{m['model']['median_abs_pct_error']:.0f}%",
    help="Median absolute percentage error on 9,600 listings the model never saw during training.",
)

st.subheader("What you can do here")
a, b, c, d = st.columns(4)
with a:
    with st.container(border=True):
        st.markdown("**:material/map: Explore listings**")
        st.caption("Filter by borough, neighborhood, room type and price, see every listing on a map, and download the result as CSV.")
        st.page_link("views/explore.py", label="Open map", icon=":material/arrow_forward:")
with b:
    with st.container(border=True):
        st.markdown("**:material/bar_chart: Compare neighborhoods**")
        st.caption("Rank neighborhoods by price, compare price distributions and see where listings are under- or over-priced.")
        st.page_link("views/neighborhoods.py", label="Compare", icon=":material/arrow_forward:")
with c:
    with st.container(border=True):
        st.markdown("**:material/calculate: Price estimator**")
        st.caption("Describe a listing and get a predicted nightly price, an 80% range, the reasons behind it and similar listings.")
        st.page_link("views/estimator.py", label="Estimate a price", icon=":material/arrow_forward:")
with d:
    with st.container(border=True):
        st.markdown("**:material/api: REST API**")
        st.caption("Get the same predictions and filtered data from your own code through JSON endpoints.")
        st.page_link("views/api_docs.py", label="API docs", icon=":material/arrow_forward:")

st.subheader("Price at a glance")
left, right = st.columns([3, 2])
with left:
    med = (
        df.groupby(["neighbourhood_group", "room_type"], observed=True)["price"]
        .median().reset_index()
    )
    order = df.groupby("neighbourhood_group")["price"].median().sort_values(ascending=False).index.tolist()
    fig = px.bar(
        med, x="neighbourhood_group", y="price", color="room_type", barmode="group",
        category_orders={"neighbourhood_group": order, "room_type": list(ROOM_COLORS)},
        color_discrete_map=ROOM_COLORS,
        labels={"neighbourhood_group": "", "price": "Median nightly price ($)", "room_type": "Room type"},
        title="Median nightly price by borough and room type",
    )
    fig.update_traces(hovertemplate="%{x}<br>%{fullData.name}: $%{y:,.0f}<extra></extra>",
                      marker_line_width=0)
    st.plotly_chart(style_fig(fig), width="stretch")
with right:
    counts = df["room_type"].value_counts().reindex(list(ROOM_COLORS)).reset_index()
    counts.columns = ["room_type", "listings"]
    fig = px.bar(
        counts, x="listings", y="room_type", orientation="h", color="room_type",
        color_discrete_map=ROOM_COLORS, labels={"room_type": "", "listings": "Listings"},
        title="Listings by room type", text="listings",
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False,
                      hovertemplate="%{y}: %{x:,} listings<extra></extra>")
    fig.update_layout(showlegend=False, xaxis_range=[0, counts["listings"].max() * 1.25],
                      yaxis=dict(categoryorder="array", categoryarray=list(ROOM_COLORS)[::-1]))
    st.plotly_chart(style_fig(fig), width="stretch")

with st.expander("About the data and the model"):
    st.markdown(
        f"""
**Data.** [New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data),
a July 2019 snapshot published by [Inside Airbnb](http://insideairbnb.com/get-the-data/) under a
CC BY 4.0 license. Each row is one listing: location, room type, nightly price, minimum stay,
review activity, how many listings the host has, and how many days a year it is bookable.
Host names were removed before loading.

**Model.** A LightGBM gradient-boosted tree model predicts the log of the nightly price from
{len(m['feature_importance'])} features: borough, neighborhood, room type, exact location and distance to
Midtown, Downtown and Williamsburg, booking rules, review activity, and signals parsed from the listing
title (for example "2BR", "studio", "luxury", "view"). Two extra quantile models give an 80% price range,
calibrated so that about 80% of real prices fall inside it. Listings priced below $10 or above $1,000
({(~df['in_training_range']).sum():,} listings) were left out of training as likely placeholders.

See **Model performance** for test-set accuracy and the comparison with a simple baseline.
"""
    )
