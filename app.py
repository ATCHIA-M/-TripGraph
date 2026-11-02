import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from travel_core import (CRITERIA, INF, all_routes_dfs, bfs_fewest_stops,
                         build_sample_graph, components, dijkstra, kruskal_mst,
                         path_totals, plan_trip)

st.set_page_config(page_title="TripGraph | Travel Planner", page_icon="🧭", layout="wide")
st.title("🧭 TripGraph: Travel Planner using Graph Data Structures")
st.caption("Adjacency-list graph · Min-Heap Dijkstra · BFS · DFS · Kruskal + Union-Find · Held-Karp DP")

# ---------------- build graph + sidebar ----------------
g = build_sample_graph()
cities = g.cities()

st.sidebar.header("Settings")
crit_label = st.sidebar.selectbox("Optimise route by", list(CRITERIA))
crit = CRITERIA[crit_label]
unit = {"km": "km", "hrs": "hrs", "cost": "Rs"}[crit]

road_names = {f"{a} - {b}": frozenset((a, b)) for a, b, _ in g.all_roads()}
closed = st.sidebar.multiselect("Close roads (simulate blockage)", sorted(road_names))
for name in closed:
    g.closed.add(road_names[name])

if g.closed:
    comps = components(g)
    if len(comps) > 1:
        st.sidebar.warning(f"Network split into {len(comps)} disconnected parts.")


# ---------------- map helper ----------------
def draw_map(paths=None, extra_edges=None, title=""):
    fig = go.Figure()
    for a, b, _ in g.roads():                       # base network
        (la1, lo1), (la2, lo2) = g.coords[a], g.coords[b]
        fig.add_trace(go.Scattergeo(lat=[la1, la2], lon=[lo1, lo2], mode="lines",
                                    line=dict(width=1, color="#bbbbbb"), hoverinfo="skip",
                                    showlegend=False))
    for a, b, _ in g.all_roads():                   # closed roads in red dashes
        if not g.is_open(a, b):
            (la1, lo1), (la2, lo2) = g.coords[a], g.coords[b]
            fig.add_trace(go.Scattergeo(lat=[la1, la2], lon=[lo1, lo2], mode="lines",
                                        line=dict(width=2, color="red", dash="dash"),
                                        hoverinfo="skip", showlegend=False))
    for a, b, _ in (extra_edges or []):             # e.g. MST edges
        (la1, lo1), (la2, lo2) = g.coords[a], g.coords[b]
        fig.add_trace(go.Scattergeo(lat=[la1, la2], lon=[lo1, lo2], mode="lines",
                                    line=dict(width=3, color="#1f77b4"), hoverinfo="skip",
                                    showlegend=False))
    colors = ["#e4572e", "#2e9e5b", "#8e44ad"]
    for i, p in enumerate(paths or []):             # highlighted routes
        fig.add_trace(go.Scattergeo(lat=[g.coords[c][0] for c in p], lon=[g.coords[c][1] for c in p],
                                    mode="lines", line=dict(width=4, color=colors[i % 3]),
                                    hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scattergeo(lat=[g.coords[c][0] for c in cities], lon=[g.coords[c][1] for c in cities],
                                mode="markers+text", text=cities, textposition="top center",
                                marker=dict(size=7, color="#222222"), hoverinfo="text", showlegend=False))
    fig.update_geos(scope="asia", fitbounds="locations", showcountries=True, showland=True,
                    landcolor="#f3f1ec", countrycolor="#999999", showocean=True, oceancolor="#e8f1f8")
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=30, b=0), title=title)
    return fig


def show_totals(path):
    t = path_totals(g, path)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Distance", f"{t['km']} km")
    c2.metric("Travel time", f"{t['hrs']} hrs")
    c3.metric("Cost", f"Rs {t['cost']:,}")
    c4.metric("Cities on route", len(path))


tab1, tab2, tab3, tab4 = st.tabs(["🚗 Route Finder", "🔀 Alternative Routes",
                                   "🗺️ Multi-Stop Trip Planner", "🌐 Road Network (MST)"])

# ---------------- TAB 1: Dijkstra + BFS ----------------
with tab1:
    c1, c2 = st.columns(2)
    src = c1.selectbox("From", cities, index=cities.index("Chennai"), key="src1")
    dst = c2.selectbox("To", cities, index=cities.index("Kanyakumari"), key="dst1")
    if src == dst:
        st.info("Choose two different cities.")
    else:
        total, path = dijkstra(g, src, dst, crit)
        if not path:
            st.error("No route available with the current road closures.")
        else:
            st.subheader(f"Best route by {crit_label} (Dijkstra with Min-Heap)")
            st.write(" → ".join(path))
            show_totals(path)
            fewest = bfs_fewest_stops(g, src, dst)
            st.write(f"**Fewest stops (BFS):** {' → '.join(fewest)}  ({len(fewest) - 1} roads)")
            st.plotly_chart(draw_map([path]), use_container_width=True)

# ---------------- TAB 2: DFS alternatives ----------------
with tab2:
    c1, c2, c3 = st.columns(3)
    s2 = c1.selectbox("From", cities, index=cities.index("Chennai"), key="src2")
    d2 = c2.selectbox("To", cities, index=cities.index("Madurai"), key="dst2")
    hops = c3.slider("Max roads per route", 2, 8, 5)
    if s2 != d2:
        routes = all_routes_dfs(g, s2, d2, crit, max_hops=hops)
        if not routes:
            st.error("No route found within this limit.")
        else:
            st.write(f"Found **{len(routes)}** routes using DFS + backtracking, ranked by {crit_label}.")
            rows = []
            for score, p in routes[:10]:
                t = path_totals(g, p)
                rows.append({"Route": " → ".join(p), "Distance (km)": t["km"],
                             "Time (hrs)": t["hrs"], "Cost (Rs)": t["cost"]})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.plotly_chart(draw_map([r[1] for r in routes[:3]],
                                     title="Top 3 routes (red, green, purple)"),
                            use_container_width=True)

# ---------------- TAB 3: Held-Karp trip planner ----------------
with tab3:
    c1, c2 = st.columns([1, 2])
    start = c1.selectbox("Start city", cities, index=cities.index("Chennai"), key="start3")
    round_trip = c1.checkbox("Return to start", value=True)
    stops = c2.multiselect("Cities to visit (max 10)", [c for c in cities if c != start],
                           default=["Madurai", "Kochi", "Mysuru", "Tirupati"])
    if stops:
        try:
            best, order, full_path = plan_trip(g, start, stops, crit, round_trip)
        except ValueError as e:
            st.error(str(e))
            best = INF
        if best == INF:
            st.error("Some selected cities are unreachable with the current closures.")
        else:
            st.subheader(f"Optimal visiting order by {crit_label} (Held-Karp DP)")
            st.write(" → ".join(order))
            st.caption("Full driving route: " + " → ".join(full_path))
            show_totals(full_path)
            st.plotly_chart(draw_map([full_path]), use_container_width=True)

# ---------------- TAB 4: Kruskal MST ----------------
with tab4:
    st.write("Kruskal's algorithm with a Union-Find structure picks the cheapest set of roads "
             "that still connects every city, useful for planning a minimum-cost road network.")
    mst, total = kruskal_mst(g, crit)
    c1, c2, c3 = st.columns(3)
    c1.metric("Roads in MST", len(mst))
    c2.metric("Roads in full network", sum(1 for _ in g.roads()))
    c3.metric(f"Total {crit_label}", f"{total:,}")
    st.plotly_chart(draw_map(extra_edges=mst, title="Minimum spanning tree (blue)"),
                    use_container_width=True)
    st.dataframe(pd.DataFrame(mst, columns=["City A", "City B", crit_label]),
                 use_container_width=True, hide_index=True)