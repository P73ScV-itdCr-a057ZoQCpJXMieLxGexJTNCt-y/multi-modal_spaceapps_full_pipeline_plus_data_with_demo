import json
import boto3
import requests
import pandas as pd
from io import StringIO
import os
import xml.etree.ElementTree as ET

s3 = boto3.client("s3")

# ---------- FEMA ----------
def fetch_fema_disasters(limit=10):
    url = "https://www.fema.gov/api/open/v1/DisasterDeclarationsSummaries"
    params = {"$top": limit}
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json().get("DisasterDeclarationsSummaries", [])
    df = pd.DataFrame(data)
    cols = [
        "disasterNumber",
        "state",
        "incidentType",
        "title",
        "declarationDate",
        "incidentBeginDate",
        "incidentEndDate",
        "declaredCountyArea"
    ]
    df = df[[c for c in cols if c in df.columns]]
    df["source"] = "FEMA"
    return df

# ---------- Kontur ----------
def fetch_kontur_events(limit=10, bbox=None, token=None):
    base_url = "https://apps.kontur.io/events/v1"
    params = {"feed": "kontur-public"}
    if bbox:
        params["geometry"] = (
            f"POLYGON(({bbox[0]} {bbox[1]}, {bbox[2]} {bbox[1]}, "
            f"{bbox[2]} {bbox[3]}, {bbox[0]} {bbox[3]}, {bbox[0]} {bbox[1]}))"
        )
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(base_url, params=params, headers=headers)
    r.raise_for_status()
    events = r.json().get("events", [])
    rows = []
    for ev in events[:limit]:
        props = ev.get("properties", {})
        rows.append({
            "disasterNumber": ev.get("eventId"),
            "state": None,
            "incidentType": ev.get("type"),
            "title": ev.get("title"),
            "declarationDate": props.get("updateTime"),
            "incidentBeginDate": props.get("startTime"),
            "incidentEndDate": None,
            "declaredCountyArea": None,
            "severity": props.get("magnitude"),
            "confidence": ev.get("severity", {}).get("confidence"),
            "source": "Kontur"
        })
    return pd.DataFrame(rows)

# ---------- NASA (MODIS/VIIRS Fires via FIRMS API) ----------
def fetch_nasa_fires(limit=10):
    url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/MODIS_C6_24h?area=world"
    r = requests.get(url)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    if df.empty:
        return pd.DataFrame()
    df = df.head(limit).rename(columns={
        "latitude": "lat",
        "longitude": "lon",
        "acq_date": "incidentBeginDate",
        "acq_time": "declarationDate"
    })
    df_norm = pd.DataFrame({
        "disasterNumber": df.index.astype(str),
        "state": None,
        "incidentType": "Wildfire",
        "title": "NASA MODIS Fire",
        "declarationDate": df["declarationDate"],
        "incidentBeginDate": df["incidentBeginDate"],
        "incidentEndDate": None,
        "declaredCountyArea": None,
        "severity": df.get("brightness", None),
        "confidence": df.get("confidence", None),
        "source": "NASA_FIRMS"
    })
    return df_norm

# ---------- NOAA Severe Weather Alerts (CAP XML feed) ----------
def fetch_noaa_alerts(limit=10):
    url = "https://api.weather.gov/alerts/active"  # returns CAP JSON feed
    headers = {"User-Agent": "HackathonDisasterAgent/1.0"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    features = r.json().get("features", [])
    rows = []
    for feat in features[:limit]:
        props = feat.get("properties", {})
        rows.append({
            "disasterNumber": feat.get("id"),
            "state": props.get("areaDesc"),
            "incidentType": props.get("event"),
            "title": props.get("headline"),
            "declarationDate": props.get("sent"),
            "incidentBeginDate": props.get("effective"),
            "incidentEndDate": props.get("ends"),
            "declaredCountyArea": props.get("areaDesc"),
            "severity": props.get("severity"),
            "confidence": props.get("certainty"),
            "source": "NOAA"
        })
    return pd.DataFrame(rows)

# ---------- Lambda Handler ----------
def lambda_handler(event, context):
    bucket = "your-hackathon-disaster-bucket"
    key = "disasters/unified_feed.csv"

    # Fetch all 4 feeds
    df_fema = fetch_fema_disasters(limit=10)
    token = os.environ.get("KONTUR_API_TOKEN")
    bbox = [-124.48, 32.53, -114.13, 42.01]  # California example
    df_kontur = fetch_kontur_events(limit=10, bbox=bbox, token=token)
    df_nasa = fetch_nasa_fires(limit=10)
    df_noaa = fetch_noaa_alerts(limit=10)

    # Merge feeds
    frames = [df for df in [df_fema, df_kontur, df_nasa, df_noaa] if not df.empty]
    if frames:
        df_all = pd.concat(frames, ignore_index=True, sort=False)
    else:
        df_all = pd.DataFrame()

    # Write to S3
    csv_buf = StringIO()
    df_all.to_csv(csv_buf, index=False)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=csv_buf.getvalue(),
        ContentType="text/csv"
    )

    return {
        "statusCode": 200,
        "body": json.dumps(f"✅ Unified feed (FEMA+Kontur+NASA+NOAA) saved to s3://{bucket}/{key}")
    }
