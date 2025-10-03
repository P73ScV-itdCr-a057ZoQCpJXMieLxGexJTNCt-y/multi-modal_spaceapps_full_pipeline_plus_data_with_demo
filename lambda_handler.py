from fetcher import build_disasterops, upload_to_s3
from datetime import datetime
import os

def lambda_handler(event, context):
    df = build_disasterops()
    filename = f"/tmp/DisasterOps_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)

    bucket_name = os.environ.get("DISASTEROPS_BUCKET", "my-hackathon-disasterops")
    upload_to_s3(filename, bucket_name, "DisasterOps/latest.csv")

    return {"status": "success", "rows": len(df)}
