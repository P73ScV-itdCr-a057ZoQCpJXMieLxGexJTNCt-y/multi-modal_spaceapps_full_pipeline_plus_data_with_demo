import requests
import pandas as pd
import boto3
import os
from datetime import datetime

# -------------------------------
# FEMA + NOAA + USGS fetch stubs
# -------------------------------
def fetch_fema():
    url = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
    r = requests.get(url, timeout=10)
    data = r.json().get("DisasterDeclarationsSummaries", [])
    df = pd.DataFrame(data)
    return df[["disasterNumber", "state", "incidentType", "title"]]

def fetch_noaa():
    # mock for hackathon demo
    return pd.DataFrame([
        {"disasterNumber": "evt-12345", "state": "CA", "incidentType": "Wildfire",
         "title": "Wildfire in Northern CA", "severity": 7.8, "confidence": 0.95}
    ])

def fetch_usgs():
    # mock for hackathon demo
    return pd.DataFrame([
        {"disasterNumber": "usgs-001", "state": "AK", "incidentType": "Earthquake",
         "title": "6.1 Earthquake near Anchorage", "severity": "Severe", "confidence": 0.88}
    ])

# -------------------------------
# FEMA Action Level
# -------------------------------
def classify_action(row):
    severity = str(row.get("severity", "")).lower()
    confidence = float(row.get("confidence", 0))

    if confidence >= 0.9 or severity in ["extreme", "catastrophic"]:
        return "Deploy"
    elif confidence >= 0.7 or severity in ["severe"]:
        return "Pre-Position"
    else:
        return "Monitor"

def build_disasterops():
    fema = fetch_fema()
    noaa = fetch_noaa()
    usgs = fetch_usgs()

    df = pd.concat([fema, noaa, usgs], ignore_index=True)

    df["FEMA_Action_Level"] = df.apply(classify_action, axis=1)
    df["EMS_Activated"] = df["FEMA_Action_Level"].apply(lambda x: "YES" if x in ["Deploy","Pre-Position"] else "NO")

    return df

# -------------------------------
# S3 Upload
# -------------------------------
def upload_to_s3(local_file, bucket, key):
    s3 = boto3.client("s3")
    s3.upload_file(local_file, bucket, key)
    print(f"✅ Uploaded {local_file} → s3://{bucket}/{key}")

if __name__ == "__main__":
    df = build_disasterops()
    filename = f"DisasterOps_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)

    bucket_name = os.environ.get("DISASTEROPS_BUCKET", "my-hackathon-disasterops")
    upload_to_s3(filename, bucket_name, "DisasterOps/latest.csv")
