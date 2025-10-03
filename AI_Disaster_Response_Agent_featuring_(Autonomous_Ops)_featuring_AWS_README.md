# 🌩️ FEMA DisasterOps Hackathon Demo

This repo demonstrates a disaster response pipeline using **FEMA, NOAA, and USGS data** → unified into one dataset → stored in **S3** → visualized in a **Streamlit dashboard**.

---

## 🚀 Features
- Fetch disaster data (FEMA/NOAA/USGS).
- Normalize into `DisasterOps/latest.csv`.
- Classify events into **Deploy / Pre-Position / Monitor**.
- Auto-upload to **Amazon S3**.
- Auto-refreshing **Streamlit dashboard** every 3 minutes.

---

## 📂 Structure
- `fetcher/` → Data ingestion + Lambda integration.
- `dashboard/` → Streamlit app.
- `data/` → Sample dataset.
- `requirements.txt` → Python deps.

---

## 🛠 Setup

1. Clone repo:
   ```bash
   git clone https://github.com/your-org/disasterops-hackathon.git
   cd disasterops-hackathon
