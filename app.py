from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium
import folium


@lru_cache(maxsize=1)
def load_constituency_geojson() -> dict:
    """
    Load India districts/constituencies GeoJSON from an online source.

    This uses a community dataset (india-maps-data) that exposes district
    polygons with fields like `district` and `st_nm`, which the rest of the
    app already expects.
    """
    url = (
        "https://cdn.jsdelivr.net/gh/udit-001/india-maps-data@8d907bc/geojson/india.geojson"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


@lru_cache(maxsize=1)
def load_country_outline_geojson() -> dict:
    """
    Load India country outline from the local maps-master dataset.

    Uses: maps-master/Country/india-osm.geojson
    """
    outline_path = Path("maps-master/Country/india-osm.geojson")
    if not outline_path.exists():
        # Fail soft: just don't render the extra layer if file not found.
        return {}
    import json

    with outline_path.open(encoding="utf-8") as f:
        return json.load(f)


def compute_party_score(row: pd.Series) -> float:
    """
    Weighted Party Score.

    Formula:
        Party Score = 0.40 × E + 0.35 × G + 0.25 × T

    Where:
        E = electoral_strength_score
        G = gov_performance_score
        T = transparency_score
    """
    e = row["electoral_strength_score"]
    g = row["gov_performance_score"]
    t = row["transparency_score"]
    return 0.40 * e + 0.35 * g + 0.25 * t


def compute_electoral_strength(row: pd.Series) -> float:
    """
    Electoral Strength (E) as a function of vote share and seats won.

    Formula (inputs already normalized to 0–1):
        E = 0.6 × V + 0.4 × S

    Where:
        V = vote_share_norm  (normalized vote share, 0–1)
        S = seats_won_norm   (normalized seats won, 0–1)
    """
    v = row["vote_share_norm"]
    s = row["seats_won_norm"]
    return 0.6 * v + 0.4 * s


def compute_gov_performance(row: pd.Series) -> float:
    """
    Government Performance (G) as a function of:
        A = attendance rate in legislature
        B = bills introduced / participated in
        M = manifesto fulfillment (% of promises addressed)

    Formula (inputs already normalized to 0–1):
        G = 0.5 × A + 0.3 × B + 0.2 × M

    All three inputs are expected to be on a 0–100 scale.
    """
    a = row["attendance_rate"]
    b = row["bills_score"]
    m = row["manifesto_fulfillment"]
    return 0.5 * a + 0.3 * b + 0.2 * m


def compute_transparency(row: pd.Series) -> float:
    """
    Transparency & Accountability (T) as a function of:
        C = criminal cases score (inverted)
        D = disclosure compliance

    Steps (inputs normalized 0–1):
        criminal_cases_norm in [0,1] (higher = more cases)
        C = 1 - criminal_cases_norm
        D = disclosure_compliance

    Formula:
        T = 0.6 × C + 0.4 × D
    """
    criminal_cases_norm = row["criminal_cases_norm"]
    disclosure = row["disclosure_compliance"]
    c = 1 - criminal_cases_norm
    d = disclosure
    return 0.6 * c + 0.4 * d


def apply_eci_parties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optionally overlay real party information from Election Commission data.

    If a file data/eci_parties.csv exists, it is expected to contain at least:
        constituency, party
    (optionally state)

    We normalise constituency names (lower/strip) and overwrite df['party']
    wherever we find a match.
    """
    eci_path = Path("data/eci_parties.csv")
    if not eci_path.exists():
        return df

    eci = pd.read_csv(eci_path)
    if "constituency" not in eci.columns or "party" not in eci.columns:
        # Ignore malformed file but keep app running
        return df

    tmp = df.copy()
    tmp["key"] = tmp["constituency"].astype(str).str.strip().str.lower()
    eci["key"] = eci["constituency"].astype(str).str.strip().str.lower()

    tmp = tmp.merge(
        eci[["key", "party"]].rename(columns={"party": "eci_party"}),
        on="key",
        how="left",
    )

    # Override party where ECI data is available
    tmp["party"] = tmp["eci_party"].combine_first(tmp["party"])
    tmp = tmp.drop(columns=["key", "eci_party"])
    return tmp


def build_sample_metrics(geojson: dict) -> pd.DataFrame:
    """
    Create or load a metrics table keyed by region name.

    Behaviour:
    - On first run, it builds a synthetic dataset for every region in the
      GeoJSON and saves it as data/metrics.csv.
    - On subsequent runs, it *only* loads from data/metrics.csv so you can
      edit the file manually to plug in real scores.
    """
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    csv_path = data_dir / "metrics.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Ensure required columns exist; party/state can be backfilled if absent
        required = {
            "constituency",
            "vote_share_norm",
            "seats_won_norm",
            "attendance_rate",
            "bills_score",
            "manifesto_fulfillment",
            "criminal_cases_norm",
            "disclosure_compliance",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"metrics.csv is missing required columns: {', '.join(sorted(missing))}"
            )

        # Backfill optional columns if they are missing so older CSVs keep working
        if "state" not in df.columns:
            df["state"] = "Unknown"
        if "party" not in df.columns:
            df["party"] = "Unknown"
        # Derive composite scores from components
        df["electoral_strength_score"] = df.apply(
            compute_electoral_strength, axis=1
        )
        df["gov_performance_score"] = df.apply(
            compute_gov_performance, axis=1
        )
        df["transparency_score"] = df.apply(
            compute_transparency, axis=1
        )
        df["party_score"] = df.apply(compute_party_score, axis=1)
        df = apply_eci_parties(df)
        return df

    # No CSV yet -> create one from GeoJSON as a starting dataset
    features = geojson.get("features", [])

    def _feature_name(props: dict) -> str:
        """
        Extract a human-friendly name from feature properties.

        Supports multiple schemas:
        - Constituencies: PC_NAME / pc_name / NAME
        - District map (your current india.geojson): district
        """
        return (
            props.get("PC_NAME")
            or props.get("pc_name")
            or props.get("NAME")
            or props.get("district")
            or "Unknown"
        )

    names: list[str] = []
    states: list[str] = []
    for f in features:
        props = f.get("properties", {})
        names.append(_feature_name(props))
        # State name field in this GeoJSON
        states.append(
            props.get("st_nm")
            or props.get("STATE")
            or props.get("state")
            or "Unknown"
        )

    # Simple deterministic pseudo-random scores based on name hash,
    # mapped into 0–1 so they behave like normalized indicators.
    def _score(seed: int, name_: str) -> float:
        raw = hash((seed, name_)) % 1000  # 0–999
        return raw / 999.0

    df = pd.DataFrame({"constituency": names, "state": states})

    # Synthetic but stable example data (0–1 scale)
    df["vote_share_norm"] = df["constituency"].apply(lambda n: _score(1, n))
    df["seats_won_norm"] = df["constituency"].apply(lambda n: _score(2, n))
    df["attendance_rate"] = df["constituency"].apply(lambda n: _score(3, n))
    df["bills_score"] = df["constituency"].apply(lambda n: _score(4, n))
    df["manifesto_fulfillment"] = df["constituency"].apply(
        lambda n: _score(5, n)
    )
    df["criminal_cases_norm"] = df["constituency"].apply(
        lambda n: _score(6, n)
    )
    df["disclosure_compliance"] = df["constituency"].apply(
        lambda n: _score(7, n)
    )

    # Assign synthetic ruling parties so the map and aggregates have categories.
    sample_parties = ["Party A", "Party B", "Party C", "Party D"]
    df["party"] = df["constituency"].apply(
        lambda n: sample_parties[hash(n) % len(sample_parties)]
    )

    # Now compute composite scores from the primitives
    df["electoral_strength_score"] = df.apply(
        compute_electoral_strength, axis=1
    )
    df["gov_performance_score"] = df.apply(
        compute_gov_performance, axis=1
    )
    df["transparency_score"] = df.apply(
        compute_transparency, axis=1
    )
    df["party_score"] = df.apply(compute_party_score, axis=1)

    # Apply ECI party overlay if provided
    df = apply_eci_parties(df)

    # Save seed dataset so you can edit it with real numbers
    df.to_csv(csv_path, index=False)

    return df


def make_map(
    geojson: dict,
    selected_name: str | None,
    metrics_df: pd.DataFrame,
    state_focus: str | None = None,
    party_colors: dict[str, str] | None = None,
    highlighted_parties: set[str] | None = None,
) -> folium.Map:
    """
    Build a folium map of India constituencies.

    If a constituency is selected, the map will zoom to its bounds and
    highlight it.
    """
    # Default center over India
    m = folium.Map(location=[22.5, 79.0], zoom_start=4, tiles="cartodbpositron")

    # Optional: add country outline from maps-master as a reference layer
    country_outline = load_country_outline_geojson()
    if country_outline.get("features"):
        folium.GeoJson(
            country_outline,
            name="India Outline (maps-master)",
            style_function=lambda _f: {
                "fillColor": "transparent",
                "color": "#555555",
                "weight": 1.0,
                "fillOpacity": 0.0,
            },
        ).add_to(m)

    # Determine selected feature if any
    selected_feature = None
    for f in geojson.get("features", []):
        props = f.get("properties", {})
        name = (
            props.get("PC_NAME")
            or props.get("pc_name")
            or props.get("NAME")
            or props.get("district")
            or "Unknown"
        )
        if selected_name and name == selected_name:
            selected_feature = f
            break

    # Build a lookup from constituency/district name -> ruling party
    party_by_constituency = (
        metrics_df.set_index("constituency")["party"].to_dict()
    )

    # Color palette for parties (provided by caller or default fallback)
    if party_colors is None:
        party_colors = {
            "Party A": "#1b9e77",
            "Party B": "#d95f02",
            "Party C": "#7570b3",
            "Party D": "#e7298a",
        }
    default_party_color = "#3186cc"

    # Style functions
    def style_function(feature):
        props = feature.get("properties", {})
        name = (
            props.get("PC_NAME")
            or props.get("pc_name")
            or props.get("NAME")
            or props.get("district")
            or "Unknown"
        )
        party = party_by_constituency.get(name)
        color = party_colors.get(party, default_party_color)

        # Dim non-highlighted parties if a filter is active
        if highlighted_parties and party not in highlighted_parties:
            return {
                "fillColor": "#cccccc",
                "color": "black",
                "weight": 0.25,
                "fillOpacity": 0.1,
            }
        return {
            "fillColor": color,
            "color": "black",
            "weight": 0.5,
            "fillOpacity": 0.3,
        }

    def highlight_function(_feature):
        return {
            "fillColor": "#ffb703",
            "color": "black",
            "weight": 1.5,
            "fillOpacity": 0.7,
        }

    # Decide which property to show in the tooltip based on available fields
    sample_props = (
        geojson.get("features", [{}])[0].get("properties", {}) if geojson.get("features") else {}
    )
    if "PC_NAME" in sample_props:
        tooltip_field = "PC_NAME"
        tooltip_label = "Constituency:"
    elif "district" in sample_props:
        tooltip_field = "district"
        tooltip_label = "District:"
    else:
        # Fallback: first property key, if any
        keys = list(sample_props.keys())
        tooltip_field = keys[0] if keys else None
        tooltip_label = "Name:"

    # Add all shapes
    folium.GeoJson(
        geojson,
        name="Regions",
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=[tooltip_field] if tooltip_field else [],
            aliases=[tooltip_label] if tooltip_field else [],
            localize=True,
        )
        if tooltip_field
        else None,
    ).add_to(m)

    # If one is selected, fit bounds either to the whole state (if requested)
    # or to just the selected constituency geometry.
    from shapely.geometry import shape
    from shapely.ops import unary_union

    if state_focus:
        # Build a union of all geometries in the selected state.
        state_geoms = []
        for f in geojson.get("features", []):
            props = f.get("properties", {})
            state_name = (
                props.get("st_nm")
                or props.get("STATE")
                or props.get("state")
            )
            if state_name == state_focus and f.get("geometry"):
                state_geoms.append(shape(f["geometry"]))
        if state_geoms:
            union_geom = unary_union(state_geoms)
            bounds = union_geom.bounds
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    elif selected_feature is not None and selected_feature.get("geometry"):
        folium.GeoJson(
            selected_feature,
            name="Selected Constituency",
            style_function=lambda _f: {
                "fillColor": "#d00000",
                "color": "#d00000",
                "weight": 2,
                "fillOpacity": 0.5,
            },
            tooltip=None,
        ).add_to(m)

        geom = shape(selected_feature["geometry"])
        bounds = geom.bounds  # minx, miny, maxx, maxy
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    folium.LayerControl().add_to(m)
    return m


def main() -> None:
    st.set_page_config(
        page_title="India Constituency Explorer",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("India Constituency Explorer")
    st.markdown(
        "Interactive map of India's parliamentary constituencies with a composite "
        "**Party Score** built from:\n"
        "- **Electoral Strength Score**\n"
        "- **Government Performance Score**\n"
        "- **Transparency Score**\n"
        "- **Accountability Score**"
    )

    with st.spinner("Loading constituency boundaries..."):
        geojson = load_constituency_geojson()
        metrics_df = build_sample_metrics(geojson)

    # Initialize session state for clicked state/constituency
    if "clicked_state" not in st.session_state:
        st.session_state["clicked_state"] = None
    if "clicked_constituency" not in st.session_state:
        st.session_state["clicked_constituency"] = None
    
    # Sidebar: REQUIRED state selection first
    all_states = sorted(metrics_df["state"].unique())
    
    # If a state was clicked, use that as default selection
    default_state_index = 0
    if st.session_state.get("clicked_state") and st.session_state["clicked_state"] in all_states:
        try:
            default_state_index = all_states.index(st.session_state["clicked_state"]) + 1
        except ValueError:
            pass
    
    state_focus = st.sidebar.selectbox(
        "Select State (Required)",
        options=["(Select a state)"] + all_states,
        index=default_state_index if default_state_index > 0 else 0,
        key="state_selector",
    )
    state_focus_eff = None if state_focus == "(Select a state)" else state_focus
    
    # Update session state if sidebar selection changed
    if state_focus_eff:
        st.session_state["clicked_state"] = state_focus_eff
        # Clear constituency selection if state changes
        if st.session_state.get("previous_state") != state_focus_eff:
            st.session_state["clicked_constituency"] = None
        st.session_state["previous_state"] = state_focus_eff
    else:
        # No state selected - clear constituency
        st.session_state["clicked_constituency"] = None
        if st.session_state.get("previous_state") != state_focus_eff:
            st.session_state["clicked_constituency"] = None
        st.session_state["previous_state"] = state_focus_eff

    # Build dynamic party color mapping and legend
    unique_parties = sorted(metrics_df["party"].unique())
    base_palette = [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#a6761d",
        "#666666",
    ]
    party_colors = {
        p: base_palette[i % len(base_palette)] for i, p in enumerate(unique_parties)
    }

    # Sidebar: party filter
    st.sidebar.markdown("**Filter by ruling party (map highlight)**")
    selected_parties = st.sidebar.multiselect(
        "Highlight parties (optional)",
        options=unique_parties,
        default=[],
    )
    highlighted_parties = set(selected_parties) if selected_parties else None

    # Party color legend
    legend_lines = []
    for p in unique_parties:
        color = party_colors[p]
        legend_lines.append(
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background-color:{color};margin-right:6px;border-radius:2px;"></span>{p}'
        )
    st.sidebar.markdown("**Party colors**", unsafe_allow_html=True)
    st.sidebar.markdown("<br>".join(legend_lines), unsafe_allow_html=True)

    # Left: map, Right: metrics
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.subheader("Constituency Map")

        # Constituency selection - only available if state is selected
        if state_focus_eff:
            # Filter constituencies to only those in the selected state
            state_constituencies = sorted(
                metrics_df.loc[metrics_df["state"] == state_focus_eff, "constituency"].unique()
            )
            
            # If a constituency was clicked and it's in the current state, use it
            default_const_index = 0
            if st.session_state.get("clicked_constituency") and st.session_state["clicked_constituency"] in state_constituencies:
                try:
                    default_const_index = state_constituencies.index(st.session_state["clicked_constituency"]) + 1
                except ValueError:
                    pass
            
            selected_name = st.sidebar.selectbox(
                "Select Constituency",
                options=["(Select a constituency)"] + state_constituencies,
                index=default_const_index if default_const_index > 0 else 0,
                key="constituency_selector",
            )
            selected_name_eff = None if selected_name == "(Select a constituency)" else selected_name
        else:
            # No state selected - disable constituency selection
            st.sidebar.info("👆 Please select a state first")
            selected_name_eff = None
            # Clear any previously selected constituency
            st.session_state["clicked_constituency"] = None

        fmap = make_map(
            geojson,
            selected_name_eff,
            metrics_df,
            state_focus_eff,
            party_colors=party_colors,
            highlighted_parties=highlighted_parties,
        )

        # Capture click events from folium map
        map_data = st_folium(
            fmap,
            width=None,
            height=600,
            returned_objects=["last_active_drawing"],
        )

        # If user clicked on a polygon, extract its constituency and state
        clicked_constituency = None
        clicked_state = None
        lad = map_data.get("last_active_drawing") if map_data else None
        if lad and "properties" in lad:
            props = lad["properties"]
            clicked_constituency = (
                props.get("PC_NAME")
                or props.get("pc_name")
                or props.get("NAME")
                or props.get("district")
            )
            clicked_state = (
                props.get("st_nm")
                or props.get("STATE")
                or props.get("state")
            )
            
            # If state is clicked, set it and clear constituency
            if clicked_state:
                st.session_state["clicked_state"] = clicked_state
                # Only set constituency if it belongs to the clicked state
                if clicked_constituency:
                    # Verify constituency belongs to the clicked state
                    const_state_match = metrics_df.loc[
                        metrics_df["constituency"] == clicked_constituency, "state"
                    ]
                    if not const_state_match.empty and const_state_match.iloc[0] == clicked_state:
                        st.session_state["clicked_constituency"] = clicked_constituency
                    else:
                        st.session_state["clicked_constituency"] = None
                else:
                    st.session_state["clicked_constituency"] = None
            elif clicked_constituency:
                # Constituency clicked - verify it belongs to current state
                const_state_match = metrics_df.loc[
                    metrics_df["constituency"] == clicked_constituency, "state"
                ]
                if not const_state_match.empty:
                    const_state = const_state_match.iloc[0]
                    # Set state first, then constituency
                    st.session_state["clicked_state"] = const_state
                    st.session_state["clicked_constituency"] = clicked_constituency

    # State must be selected first - use state_focus_eff (from sidebar or session state)
    state_for_view = state_focus_eff or st.session_state.get("clicked_state")
    
    # Only allow constituency selection if state is selected
    if state_for_view:
        # Resolve final constituency - only from the selected state
        final_constituency = (
            st.session_state.get("clicked_constituency")
            or selected_name_eff
        )
        
        # Verify constituency belongs to the selected state
        if final_constituency:
            const_state_match = metrics_df.loc[
                metrics_df["constituency"] == final_constituency, "state"
            ]
            if const_state_match.empty or const_state_match.iloc[0] != state_for_view:
                final_constituency = None
                st.session_state["clicked_constituency"] = None
    else:
        # No state selected - no constituency allowed
        final_constituency = None
        st.session_state["clicked_constituency"] = None

    with right:
        # Show state details FIRST, then constituency details
        if state_for_view:
            st.subheader(f"State: {state_for_view}")
            state_df = metrics_df.loc[
                metrics_df["state"] == state_for_view
            ]
            total_const = len(state_df)
            party_counts = state_df["party"].value_counts()
            if not party_counts.empty:
                major_party = party_counts.idxmax()
                major_count = party_counts.max()
                st.write(
                    f"Total constituencies: **{total_const}**  "
                    f"• Major party: **{major_party}** "
                    f"({major_count} constituencies)"
                )
                st.dataframe(
                    party_counts.rename("Constituencies")
                    .reset_index(name="Constituencies")
                    .rename(columns={"index": "Party"})
                    .sort_values("Constituencies", ascending=False),
                    use_container_width=True,
                )

                # Top constituencies by Party Score within the state
                top_n = min(10, total_const)
                if top_n > 0:
                    st.markdown(f"#### Top {top_n} constituencies by Party Score")
                    top_table = (
                        state_df[
                            [
                                "constituency",
                                "party",
                                "party_score",
                                "electoral_strength_score",
                                "gov_performance_score",
                                "transparency_score",
                            ]
                        ]
                        .sort_values("party_score", ascending=False)
                        .head(top_n)
                    )
                    st.dataframe(top_table, use_container_width=True)
            
            st.markdown("---")
        
        # Then show constituency details if one is selected
        st.subheader("Constituency Details")
        if final_constituency:
            row = metrics_df.loc[
                metrics_df["constituency"] == final_constituency
            ].head(1)
            if row.empty:
                st.info(
                    f"No metrics found for **{final_constituency}**. "
                    "You can wire in your own dataset here."
                )
            else:
                r = row.iloc[0]
                st.markdown(f"### {final_constituency}")

                st.metric(
                    "Party Score (Overall)",
                    f"{r['party_score']:.2f}",
                )

                c1, c2 = st.columns(2)
                with c1:
                    st.metric(
                        "Electoral Strength Score (E)",
                        f"{r['electoral_strength_score']:.2f}",
                    )
                    st.metric(
                        "Governance Performance Score (G)",
                        f"{r['gov_performance_score']:.2f}",
                    )
                with c2:
                    st.metric(
                        "Transparency & Accountability Score (T)",
                        f"{r['transparency_score']:.2f}",
                    )
                    st.metric(
                        "Ruling Party (Constituency)",
                        r["party"],
                    )

                with st.expander("Underlying scores table"):
                    st.dataframe(
                        row[
                            [
                                "electoral_strength_score",
                                "gov_performance_score",
                                "transparency_score",
                                "party_score",
                            ]
                        ].T.rename(columns={row.index[0]: "Score"}),
                        use_container_width=True,
                    )
        else:
            if state_for_view:
                st.info(
                    f"👆 Select a constituency from **{state_for_view}** in the sidebar dropdown or click on the map to see detailed metrics."
                )
            else:
                st.warning(
                    "⚠️ **Please select a state first** using the sidebar dropdown or by clicking on the map. "
                    "Constituency selection is only available after selecting a state."
                )

    st.markdown(
        "---\n"
        "You can replace the example scoring logic with your own data source "
        "(CSV, database, or API) by editing the `build_sample_metrics` "
        "function in `app.py`.\n\n"
        "_Disclaimer_: This index reflects selected public-data-style indicators "
        "and does not represent moral, ideological, or electoral endorsement."
    )


if __name__ == "__main__":
    main()

