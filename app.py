import json, urllib.request, urllib.parse
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect
import pandas as pd

from settings import PROCESSED_DIR

ROOT      = Path(__file__).resolve().parent
DATA_FILE = PROCESSED_DIR / "area_classified.parquet"
NEIGH_ML_FILE = PROCESSED_DIR / "neighbourhood_ml_classified.parquet"
NEIGH_FILE    = NEIGH_ML_FILE if NEIGH_ML_FILE.exists() else PROCESSED_DIR / "neighbourhood_classified.parquet"

app = Flask(__name__, template_folder=str(ROOT / "templates"),
            static_folder=str(ROOT / "static"))

# Load data
if DATA_FILE.exists():
    DF = pd.read_parquet(DATA_FILE)
    print(f"[LiveSafe] Loaded {len(DF)} classified areas.")
else:
    DF = None
    print("[LiveSafe] WARNING: area_classified.parquet not found.")

if NEIGH_FILE.exists():
    NDF = pd.read_parquet(NEIGH_FILE)
    print(f"[LiveSafe] Loaded {len(NDF)} classified neighbourhoods.")
else:
    NDF = None
    print("[LiveSafe] INFO: neighbourhood_classified.parquet not found (run download_neighbourhoods.py)")


# Helpers
def profile_badge_color(p):
    return {"Stable Low-Risk Area":"success","Weather-Sensitive Area":"info",
            "Crime-Sensitive Area":"warning","Mixed-Risk Area":"secondary",
            "Emerging Risk Area":"orange","Volatile Risk Area":"orange",
            "Structurally Exposed Area":"info",
            "Persistent Multi-Factor Risk Area":"danger"}.get(p,"secondary")

def intensity_color(l):
    return {"Very Low Recorded Crime":"success","Moderate Crime Level":"info",
            "Elevated Crime Level":"warning",
            "Severe Crime Concentration":"danger"}.get(l,"secondary")

def trend_color(l):
    return {"Improving":"success","Stable":"info","Fluctuating":"warning",
            "Deteriorating":"danger","Insufficient Data":"secondary"}.get(l,"secondary")

def weather_color(l):
    return {"Mild Conditions":"success","Rain-Exposed":"info","Cold-Exposed":"info",
            "Seasonally Volatile":"warning","High Weather Stress":"danger"}.get(l,"secondary")

def neigh_intensity_color(l):
    return {"High Crime":"danger","Moderate Crime":"warning","Lower Crime":"success"}.get(l,"secondary")

def neigh_trend_color(l):
    return {"Worsening":"danger","Stable":"info","Improving":"success"}.get(l,"secondary")

def area_to_dict(row):
    d = row.to_dict()
    for k,v in d.items():
        if isinstance(v, float): d[k] = round(v, 2)
    return d

def build_interpretation(row):
    name=row.get("la_name",""); profile=row.get("overall_profile","")
    ci=row.get("crime_intensity_label",""); ct=row.get("crime_trend_label","")
    cp=row.get("crime_pattern_label",""); we=row.get("weather_exposure_label","")
    return (f"{name} is classified as a <strong>{profile}</strong>. "
            f"Crime intensity is <strong>{ci}</strong>, with a predominantly "
            f"<strong>{cp}</strong> and a <strong>{ct}</strong> trend over 2024. "
            f"Environmental conditions are characterised as <strong>{we}</strong>.")

@app.context_processor
def inject_helpers():
    return dict(profile_badge_color=profile_badge_color,
                intensity_color=intensity_color, trend_color=trend_color,
                weather_color=weather_color, neigh_intensity_color=neigh_intensity_color,
                neigh_trend_color=neigh_trend_color)


# ML helper functions
def _safe_csv(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def _safe_parquet(path):
    p = Path(path)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

def _load_insights_ml_vars():
    """Load all ML result files for the insights page."""

    # Random Forest
    rf_imp_df  = _safe_csv(PROCESSED_DIR / "rf_feature_importance.csv")
    rf_fold_df = _safe_csv(PROCESSED_DIR / "rf_fold_scores.csv")

    rf_importance_json  = None
    rf_fold_scores_json = None
    rf_accuracy         = None

    if not rf_imp_df.empty and "feature" in rf_imp_df.columns:
        rf_imp_df = rf_imp_df.sort_values("importance_pct", ascending=False)
        rf_importance_json = rf_imp_df[["feature", "importance_pct"]].to_json(orient="records")

    if not rf_fold_df.empty and "accuracy" in rf_fold_df.columns:
        rf_fold_scores_json = rf_fold_df[["accuracy"]].to_json(orient="records")
        rf_accuracy = round(float(rf_fold_df["accuracy"].mean()) * 100, 1)

    # Isolation Forest
    anomaly_df = _safe_csv(PROCESSED_DIR / "anomaly_summary.csv")
    anomaly_summary_json = anomaly_df.to_dict("records") if not anomaly_df.empty else None

    # Crime Forecasting
    fmet_df  = _safe_csv(PROCESSED_DIR / "forecast_metrics.csv")
    fproj_df = _safe_csv(PROCESSED_DIR / "forecast_projections.csv")
    forecast_metrics_json     = fmet_df.to_dict("records")  if not fmet_df.empty  else None
    forecast_projections_json = fproj_df.to_dict("records") if not fproj_df.empty else None

    # IMD Validation
    imd_val_df  = _safe_csv(PROCESSED_DIR / "imd_validation.csv")
    imd_comp_df = _safe_csv(PROCESSED_DIR / "imd_comparison.csv")

    imd_validation_json = None
    imd_comparison_json = imd_comp_df.to_dict("records") if not imd_comp_df.empty else None

    if not imd_val_df.empty and "spearman_r" in imd_val_df.columns:
        row = imd_val_df.iloc[0]
        imd_validation_json = {
            "spearman_r": round(float(row["spearman_r"]), 4),
            "spearman_p": round(float(row["spearman_p"]), 4),
        }
        if "kendall_tau" in row and pd.notna(row.get("kendall_tau")):
            imd_validation_json["kendall_tau"] = round(float(row["kendall_tau"]), 4)

    return dict(
        rf_importance_json=rf_importance_json,
        rf_fold_scores_json=rf_fold_scores_json,
        rf_accuracy=rf_accuracy,
        anomaly_summary_json=anomaly_summary_json,
        forecast_metrics_json=forecast_metrics_json,
        forecast_projections_json=forecast_projections_json,
        imd_validation_json=imd_validation_json,
        imd_comparison_json=imd_comparison_json,
    )

def _enrich_neighbourhood_anomalies(neighbourhood, neighbourhood_ranking):
    """Add is_anomaly + anomaly_score keys to neighbourhood dict and ranking list."""
    df = _safe_parquet(PROCESSED_DIR / "neighbourhood_anomalies.parquet")

    # Set safe defaults so templates never KeyError
    if neighbourhood:
        neighbourhood.setdefault("is_anomaly", False)
        neighbourhood.setdefault("anomaly_score", 0.0)
    for n in (neighbourhood_ranking or []):
        n.setdefault("is_anomaly", False)

    if df.empty:
        return

    name_col = next(
        (c for c in ["neighbourhood_name", "name", "nhood_name"] if c in df.columns), None
    )
    if not name_col:
        return

    # Build fast name → (is_anomaly, score) lookup
    lookup = {}
    for _, row in df.iterrows():
        key = row.get(name_col)
        if pd.notna(key):
            lookup[str(key)] = (
                bool(int(row.get("is_anomaly", 0))),
                float(row.get("anomaly_score", 0.0)),
            )

    if neighbourhood:
        name = str(neighbourhood.get("neighbourhood_name") or neighbourhood.get("name", ""))
        is_a, score = lookup.get(name, (False, 0.0))
        neighbourhood["is_anomaly"] = is_a
        neighbourhood["anomaly_score"] = score

    for n in (neighbourhood_ranking or []):
        name = str(n.get("neighbourhood_name") or n.get("name", ""))
        n["is_anomaly"] = lookup.get(name, (False, 0.0))[0]

def _get_area_forecast(la_name):
    """Return (forecast_dict or None, projections_list or None) for a given LA name."""
    if not la_name:
        return None, None

    met_df  = _safe_csv(PROCESSED_DIR / "forecast_metrics.csv")
    proj_df = _safe_csv(PROCESSED_DIR / "forecast_projections.csv")
    area_forecast    = None
    area_projections = None

    if not met_df.empty and "area" in met_df.columns:
        match = met_df[met_df["area"] == la_name]
        if match.empty:
            # Partial match for names like "Bristol, City of"
            match = met_df[met_df["area"].str.contains(
                la_name.split(",")[0], case=False, na=False)]
        if not match.empty:
            area_forecast = {
                k: (None if (isinstance(v, float) and pd.isna(v)) else v)
                for k, v in match.iloc[0].to_dict().items()
            }

    if area_forecast and not proj_df.empty and "area" in proj_df.columns:
        pm = proj_df[proj_df["area"] == area_forecast["area"]]
        if not pm.empty:
            area_projections = pm.sort_values("months_ahead").to_dict("records")

    return area_forecast, area_projections


# Postcode lookup
def lookup_postcode(postcode):
    clean = postcode.strip().replace(" ","").upper()
    url   = f"https://api.postcodes.io/postcodes/{urllib.parse.quote(clean)}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        if data.get("status") == 200:
            r = data["result"]
            return {
                "postcode":  r.get("postcode"),
                "la_code":   r.get("codes",{}).get("admin_district",""),
                "la_name":   r.get("admin_district",""),
                "latitude":  r.get("latitude"),
                "longitude": r.get("longitude"),
                "ward":      r.get("admin_ward",""),
                "region":    r.get("region",""),
            }
    except Exception:
        pass
    return None


def find_area_in_dataset(resolved):
    if DF is None: return pd.DataFrame()
    la_code = resolved.get("la_code","")
    la_name = resolved.get("la_name","").lower().strip()
    match = DF[DF["la_code"] == la_code]
    if not match.empty: return match
    match = DF[DF["la_name"].str.lower() == la_name]
    if not match.empty: return match
    words = [w for w in la_name.split() if len(w) > 3]
    for word in words:
        match = DF[DF["la_name"].str.lower().str.contains(word, na=False)]
        if len(match) == 1: return match
    return pd.DataFrame()


def find_neighbourhood(resolved, la_code):
    if NDF is None: return None
    lat = resolved.get("latitude")
    lon = resolved.get("longitude")
    if not lat or not lon: return None

    subset = NDF[NDF["la_code"] == la_code].copy()
    if subset.empty: return None

    subset["dist"] = (
        (subset["centroid_lat"] - float(lat))**2 +
        (subset["centroid_lon"] - float(lon))**2
    )
    closest = subset.nsmallest(1, "dist").iloc[0]
    return area_to_dict(closest)


def get_la_neighbourhood_ranking(la_code):
    if NDF is None: return []
    subset = NDF[NDF["la_code"] == la_code].copy()
    subset = subset.sort_values("crime_rank")
    return [area_to_dict(r) for _, r in subset.iterrows()]


# Routes
@app.route("/")
def home():
    return render_template("home.html",
                           data_loaded=DF is not None,
                           area_count=len(DF) if DF is not None else 0,
                           neigh_loaded=NDF is not None,
                           neigh_count=len(NDF) if NDF is not None else 0)


@app.route("/area")
def area():
    if DF is None: return render_template("no_data.html")
    areas_list     = DF[["la_code","la_name"]].sort_values("la_name").to_dict("records")
    postcode_query = request.args.get("postcode","").strip()
    la_code        = request.args.get("code","").strip()
    error = result = postcode_info = neighbourhood = None
    neighbourhood_ranking = []

    if postcode_query:
        resolved = lookup_postcode(postcode_query)
        if resolved is None:
            error = (f"Postcode <strong>{postcode_query.upper()}</strong> not found. "
                     f"Please check and try again.")
        else:
            postcode_info = resolved
            match         = find_area_in_dataset(resolved)
            if match.empty:
                covered = ", ".join(sorted(DF["la_name"].tolist()))
                error   = (f"<strong>{resolved['postcode']}</strong> is in "
                           f"<strong>{resolved['la_name']}</strong>, which is not "
                           f"in our current dataset. We cover: {covered}.")
            else:
                result   = area_to_dict(match.iloc[0])
                result["interpretation"] = build_interpretation(result)
                la_code  = result["la_code"]
                neighbourhood         = find_neighbourhood(resolved, la_code)
                neighbourhood_ranking = get_la_neighbourhood_ranking(la_code)

    elif la_code:
        match = DF[DF["la_code"] == la_code]
        if match.empty:
            error = "Area not found."
        else:
            result = area_to_dict(match.iloc[0])
            result["interpretation"] = build_interpretation(result)
            neighbourhood_ranking    = get_la_neighbourhood_ranking(la_code)

    # Enrich with anomaly flags (safe no-op if file doesn't exist)
    _enrich_neighbourhood_anomalies(neighbourhood, neighbourhood_ranking)

    # Load crime forecast for this area
    area_forecast, area_projections = _get_area_forecast(
        result.get("la_name") if result else None
    )

    return render_template("area.html",
                           areas=areas_list, result=result, error=error,
                           postcode_query=postcode_query,
                           postcode_info=postcode_info,
                           neighbourhood=neighbourhood,
                           neighbourhood_ranking=neighbourhood_ranking,
                           neigh_loaded=NDF is not None,
                           area_forecast=area_forecast,
                           area_projections=area_projections)


@app.route("/compare")
def compare():
    if DF is None: return render_template("no_data.html")
    areas_list = DF[["la_code","la_name"]].sort_values("la_name").to_dict("records")
    result_a = result_b = None
    for param, key in [("a","result_a"),("b","result_b")]:
        code = request.args.get(param,"").strip()
        if code:
            m = DF[DF["la_code"] == code]
            if not m.empty:
                r = area_to_dict(m.iloc[0])
                r["interpretation"] = build_interpretation(r)
                if key == "result_a": result_a = r
                else: result_b = r
    return render_template("compare.html", areas=areas_list,
                           result_a=result_a, result_b=result_b)


@app.route("/insights")
def insights():
    return redirect("/trends")

@app.route("/trends")
def trends():
    if DF is None: return render_template("no_data.html")
    rows       = DF.sort_values("avg_monthly_crimes", ascending=False)
    areas_json = json.dumps([area_to_dict(r) for _, r in rows.iterrows()])
    return render_template("trends.html", areas_json=areas_json)

@app.route("/analysis")
def analysis():
    if DF is None: return render_template("no_data.html")
    rows       = DF.sort_values("avg_monthly_crimes", ascending=False)
    areas_json = json.dumps([area_to_dict(r) for _, r in rows.iterrows()])

    clusters_json = None
    cluster_file  = PROCESSED_DIR / "area_clusters.parquet"
    if cluster_file.exists():
        cdf = pd.read_parquet(cluster_file)
        cluster_cols = ["la_name","cluster_id","cluster_name","pca_x","pca_y",
                        "overall_profile","crime_intensity_label",
                        "avg_monthly_crimes","avg_violent_share","crime_trend_slope"]
        cluster_cols = [c for c in cluster_cols if c in cdf.columns]
        clusters_json = json.dumps([area_to_dict(r) for _, r in cdf[cluster_cols].iterrows()])

    ml_vars = _load_insights_ml_vars()
    return render_template("analysis.html",
                           areas_json=areas_json,
                           clusters_json=clusters_json,
                           **ml_vars)

@app.route("/rankings")
def rankings():
    if DF is None: return render_template("no_data.html")
    rows  = DF.sort_values("avg_monthly_crimes", ascending=False)
    areas = [area_to_dict(r) for _, r in rows.iterrows()]
    max_crimes = rows["avg_monthly_crimes"].max() if len(rows) > 0 else 1
    return render_template("rankings.html", areas=areas, max_crimes=max_crimes)

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/methodology")
def methodology():
    return redirect("/about")


@app.route("/api/areas")
def api_areas():
    if DF is None: return jsonify([])
    return jsonify([area_to_dict(r) for _, r in DF.iterrows()])


@app.route("/api/postcode/<path:postcode>")
def api_postcode(postcode):
    resolved = lookup_postcode(postcode)
    if resolved is None:
        return jsonify({"error":"Postcode not found"}), 404
    if DF is None:
        return jsonify({"error":"No data loaded"}), 503
    match = find_area_in_dataset(resolved)
    if match.empty:
        return jsonify({"postcode_info":resolved,"in_dataset":False})
    row = area_to_dict(match.iloc[0])
    row["interpretation"] = build_interpretation(row)
    neigh   = find_neighbourhood(resolved, row["la_code"])
    ranking = get_la_neighbourhood_ranking(row["la_code"])
    return jsonify({"postcode_info":resolved,"in_dataset":True,
                    "area":row,"neighbourhood":neigh,"ranking":ranking})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
