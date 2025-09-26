# 🚀 Enhanced Multi-Modal Space Apps Pipeline
# Advanced version with multi-modal AI, temporal analysis, and cross-validation

import os
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import rasterio
from rasterio.windows import from_bounds
import folium
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import planetary_computer as pc
from pystac_client import Client
from datetime import datetime, timedelta
import json
import base64
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MultiModalHospitalAnalyzer:
    """Enhanced multi-modal analyzer for hospital capacity and environmental monitoring"""
    
    def __init__(self):
        self.yolo_model = None
        self.historical_data = {}
        self.anomaly_threshold = 0.7
        
    def load_models(self):
        """Load multiple AI models for different analysis tasks"""
        print("Loading YOLO models...")
        self.yolo_model = YOLO('yolov8n.pt')  # Main detection model
        # Could add specialized models: yolov8s-seg.pt for segmentation, custom trained models
        
    def stac_client_fetcher(self, lon: float, lat: float) -> Optional[Dict[str, str]]:
        """
        Fetches the latest cloud-free Sentinel-2 image for the given coordinates
        using the Planetary Computer STAC API.

        Returns:
            A dictionary mapping required band names (B4, B8, B10) to local file paths,
            or None if no suitable image is found.
        """
        print(f"-> Searching STAC for imagery near ({lon:.4f}, {lat:.4f})...")
        
        # Define search area (bounding box around the hospital)
        buffer = 0.005  # A small buffer for a 500m x 500m area roughly
        bbox = [lon - buffer, lat - buffer, lon + buffer, lat + buffer]
        
        # Connect to Planetary Computer STAC API
        STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
        catalog = Client.open(STAC_URL, headers=pc.sign_headers())

        # Define search parameters
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=(datetime.now() - timedelta(days=30)).isoformat() + '/',
            query={"eo:cloud_cover": {"lt": 10}}, # Only get images with < 10% cloud cover
            limit=10
        )
        
        items = list(search.get_items())
        
        if not items:
            print("❌ No recent, low-cloud-cover Sentinel-2 imagery found.")
            return None

        # Sort by least cloud cover and get the best item
        best_item = sorted(items, key=lambda item: item.properties.get("eo:cloud_cover", 100))[0]
        
        # Required bands for analysis (B4: Red, B8: NIR, B10: Thermal/SWIR proxy)
        required_bands = ["B4", "B8", "B10"]
        downloaded_paths = {}

        # Download the required assets (Bands)
        for band in required_bands:
            if band in best_item.assets:
                asset_href = best_item.assets[band].href
                
                # Sign the URL for access
                signed_href = pc.sign(asset_href)

                # Use rasterio to read a window of the image directly from the URL
                try:
                    with rasterio.open(signed_href) as src:
                        # Convert bounds to a window for efficient reading
                        window = from_bounds(*bbox, src.transform)
                        data = src.read(1, window=window)
                        
                        # Save the small data array as a local temporary file
                        temp_path = f"temp_band_{band}_{int(lat*1000)}_{int(lon*1000)}.tif"
                        profile = src.profile
                        profile.update(
                            width=data.shape[1], 
                            height=data.shape[0], 
                            transform=src.window_transform(window),
                            dtype=data.dtype,
                            count=1
                        )
                        
                        with rasterio.open(temp_path, 'w', **profile) as dst:
                            dst.write(data, 1)
                            
                        downloaded_paths[band] = temp_path
                except Exception as e:
                    print(f"Error processing band {band}: {e}")
                    # Clean up any partial files and return None
                    for p in downloaded_paths.values(): 
                        if os.path.exists(p): os.remove(p)
                    return None

        return downloaded_paths if len(downloaded_paths) == len(required_bands) else None
        
    def multi_spectral_analysis(self, imagery_paths: Dict[str, str], hospital_coords: Tuple[float, float]) -> Dict:
        """
        Analyze multiple spectral bands for comprehensive environmental assessment.
        Calculates real NDVI and proxies for Urban Heat if imagery_paths is provided.
        """
        print("-> Performing Multi-Spectral Analysis...")
        
        results = {
            'vegetation_health': None,
            'urban_heat_island': None,
            'water_stress': None,
            'air_quality_proxy': None,
            'change_detection': None
        }
        
        # --- Fallback to Simulation if real data fetch failed or is incomplete ---
        if not imagery_paths or "B4" not in imagery_paths or "B8" not in imagery_paths or "B10" not in imagery_paths:
            print("⚠️ Imagery paths incomplete. Falling back to simulated spectral data.")
            return {
                'vegetation_health': np.random.uniform(0.2, 0.8),
                'urban_heat_island': np.random.uniform(25, 35),
                'water_stress': np.random.uniform(0.1, 0.6),
                'air_quality_proxy': np.random.uniform(0.3, 0.9)
            }
        # ------------------------------------------------------------------------
        
        try:
            # 1. Calculate NDVI (Normalized Difference Vegetation Index)
            # NDVI = (NIR - Red) / (NIR + Red) -> (B8 - B4) / (B8 + B4)
            with rasterio.open(imagery_paths["B8"]) as nir_src, rasterio.open(imagery_paths["B4"]) as red_src:
                nir = nir_src.read(1).astype(float)
                red = red_src.read(1).astype(float)
                
                # Avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    ndvi = (nir - red) / (nir + red)
                
                # Use a masked mean to ignore NoData values/NaNs
                ndvi_mean = np.nanmean(ndvi)
                results['vegetation_health'] = float(ndvi_mean)
                
            # 2. Estimate Urban Heat (using B10 as a proxy for surface temperature)
            # Sentinel-2 B10 is SWIR band. We use the mean Digital Number (DN) as a proxy for heat.
            with rasterio.open(imagery_paths["B10"]) as thermal_src:
                thermal_data = thermal_src.read(1).astype(float)
                heat_proxy_mean = np.nanmean(thermal_data)
                results['urban_heat_island'] = float(heat_proxy_mean)
                
            # 3. Water stress and Air Quality Proxy remain simulated for simplicity
            results['water_stress'] = np.random.uniform(0.1, 0.6)
            results['air_quality_proxy'] = np.random.uniform(0.3, 0.9)
            
        except Exception as e:
            print(f"⚠️ Error during multi-spectral analysis: {e}. Falling back to partial/simulated data.")
            # Set all results to simulated data if a fatal error occurred during processing
            results = {
                'vegetation_health': np.random.uniform(0.2, 0.8),
                'urban_heat_island': np.random.uniform(25, 35),
                'water_stress': np.random.uniform(0.1, 0.6),
                'air_quality_proxy': np.random.uniform(0.3, 0.9)
            }

        # Clean up the temporary files after reading
        for path in imagery_paths.values():
            if os.path.exists(path):
                os.remove(path)
                
        return results
    
    def temporal_analysis(self, hospital_id: str, current_data: Dict, days_back: int = 30) -> Dict:
        """
        Analyze temporal patterns and detect anomalies
        
        Args:
            hospital_id: Unique hospital identifier
            current_data: Current observation data
            days_back: Number of days to look back for comparison
            
        Returns:
            Dict with temporal analysis results
        """
        if hospital_id not in self.historical_data:
            self.historical_data[hospital_id] = []
        
        # Add current data with timestamp
        current_data['timestamp'] = datetime.now()
        self.historical_data[hospital_id].append(current_data)
        
        # Keep only recent data
        cutoff_date = datetime.now() - timedelta(days=days_back)
        self.historical_data[hospital_id] = [
            d for d in self.historical_data[hospital_id] 
            if d['timestamp'] > cutoff_date
        ]
        
        history = self.historical_data[hospital_id]
        
        if len(history) < 3:
            return {'trend': 'insufficient_data', 'anomaly_score': 0.0}
        
        # Calculate trends
        vehicle_counts = [d.get('vehicle_count', 0) for d in history]
        fullness_values = [d.get('fullness', 0) for d in history]
        
        # Simple trend analysis
        if len(vehicle_counts) >= 5:
            recent_avg = np.mean(vehicle_counts[-3:])
            historical_avg = np.mean(vehicle_counts[:-3])
            trend = 'increasing' if recent_avg > historical_avg * 1.2 else 'decreasing' if recent_avg < historical_avg * 0.8 else 'stable'
        else:
            trend = 'stable'
        
        # Anomaly detection (simple statistical approach)
        if len(fullness_values) >= 5:
            mean_fullness = np.mean(fullness_values[:-1])
            std_fullness = np.std(fullness_values[:-1])
            current_fullness = fullness_values[-1]
            
            z_score = abs(current_fullness - mean_fullness) / (std_fullness + 1e-10)
            anomaly_score = min(1.0, z_score / 3.0)  # Normalize to 0-1
        else:
            anomaly_score = 0.0
        
        return {
            'trend': trend,
            'anomaly_score': anomaly_score,
            'is_anomaly': anomaly_score > self.anomaly_threshold,
            'historical_avg_fullness': np.mean(fullness_values),
            'data_points': len(history)
        }
    
    def cross_modal_validation(self, imagery_analysis: Dict, vehicle_count: int, 
                             environmental_data: Dict) -> Dict:
        """
        Cross-validate results across different data modalities
        
        Args:
            imagery_analysis: Results from image analysis
            vehicle_count: Detected vehicle count
            environmental_data: Environmental sensor data
            
        Returns:
            Dict with validation results and confidence scores
        """
        validations = {}
        confidence_score = 1.0
        
        # Validate vehicle count against parking lot analysis
        if 'parking_area_pixels' in imagery_analysis:
            expected_capacity = imagery_analysis['parking_area_pixels'] // 400  # Rough estimate
            if vehicle_count > expected_capacity * 1.5:
                validations['vehicle_count'] = 'suspicious_high'
                confidence_score *= 0.7
            elif vehicle_count == 0 and expected_capacity > 10:
                validations['vehicle_count'] = 'suspicious_low'
                confidence_score *= 0.8
            else:
                validations['vehicle_count'] = 'validated'
        
        # Cross-validate with environmental conditions
        if environmental_data.get('weather') == 'severe':
            if vehicle_count > imagery_analysis.get('normal_capacity', 100) * 0.3:
                validations['weather_consistency'] = 'inconsistent'
                confidence_score *= 0.6
            else:
                validations['weather_consistency'] = 'consistent'
        
        # Validate air quality against vegetation health
        if ('vegetation_health' in imagery_analysis and 
            environmental_data.get('air_quality_index')):
            
            veg_health = imagery_analysis['vegetation_health']
            aqi = environmental_data['air_quality_index']
            
            # Poor vegetation should correlate with poor air quality
            if veg_health < 0.3 and aqi < 50:  # Good AQI but poor vegetation
                validations['environmental_consistency'] = 'inconsistent'
                confidence_score *= 0.8
            else:
                validations['environmental_consistency'] = 'consistent'
        
        return {
            'validations': validations,
            'confidence_score': confidence_score,
            'overall_reliability': 'high' if confidence_score > 0.8 else 'medium' if confidence_score > 0.6 else 'low'
        }
    
    def enhanced_image_analysis(self, image_path: str, hospital_coords: Tuple[float, float]) -> Dict:
        """
        Enhanced image analysis with multiple detection approaches
        
        Args:
            image_path: Path to the image file
            hospital_coords: (lat, lon) of hospital
            
        Returns:
            Dict with comprehensive analysis results
        """
        if not self.yolo_model:
            self.load_models()
        
        img = cv2.imread(image_path)
        if img is None:
            return {'error': 'Could not load image'}
        
        results = {}
        
        # YOLO detection
        detections = self.yolo_model(img, conf=0.4)
        
        vehicle_types = ['car', 'truck', 'bus', 'motorcycle']
        person_count = 0
        vehicle_count = 0
        vehicle_details = []
        
        for detection in detections:
            for box in detection.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.yolo_model.names[cls_id]
                confidence = float(box.conf[0])
                
                if cls_name in vehicle_types:
                    vehicle_count += 1
                    vehicle_details.append({
                        'type': cls_name,
                        'confidence': confidence,
                        'bbox': box.xyxy[0].tolist()
                    })
                elif cls_name == 'person':
                    person_count += 1
        
        results['vehicle_count'] = vehicle_count
        results['person_count'] = person_count
        results['vehicle_details'] = vehicle_details
        
        # Parking lot area analysis using image processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect parking spaces (simplified approach)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        parking_areas = []
        total_parking_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 10000:  # Reasonable parking space size
                parking_areas.append(area)
                total_parking_area += area
        
        results['parking_area_pixels'] = total_parking_area
        results['estimated_parking_spaces'] = len(parking_areas)
        
        # Calculate fullness
        estimated_capacity = max(len(parking_areas), vehicle_count)
        if estimated_capacity > 0:
            results['fullness'] = min(1.0, vehicle_count / estimated_capacity)
        else:
            results['fullness'] = 0.0
        
        # Shadow analysis for time estimation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        shadow_mask = cv2.inRange(hsv, (0, 0, 0), (180, 50, 50))
        shadow_ratio = np.sum(shadow_mask > 0) / (img.shape[0] * img.shape[1])
        results['shadow_ratio'] = shadow_ratio
        
        return results
    
    def environmental_data_integration(self, hospital_coords: Tuple[float, float], 
                                     current_time: datetime = None) -> Dict:
        """
        Integrate environmental data from multiple sources
        
        Args:
            hospital_coords: (lat, lon) of hospital
            current_time: Current timestamp
            
        Returns:
            Dict with environmental data
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Placeholder for real API integrations
        environmental_data = {
            'air_quality_index': np.random.randint(20, 150),  # Would come from EPA API
            'weather': np.random.choice(['clear', 'cloudy', 'rainy', 'severe']),
            'temperature': np.random.normal(20, 10),  # Would come from weather API
            'humidity': np.random.uniform(30, 90),
            'wind_speed': np.random.uniform(0, 25),
            'visibility': np.random.uniform(1, 15),  # km
            'timestamp': current_time,
            'source': 'simulated'  # In production: 'EPA_API', 'NOAA', etc.
        }
        
        # Add derived metrics
        if environmental_data['air_quality_index'] > 100:
            environmental_data['air_quality_status'] = 'unhealthy'
        elif environmental_data['air_quality_index'] > 50:
            environmental_data['air_quality_status'] = 'moderate'
        else:
            environmental_data['air_quality_status'] = 'good'
        
        return environmental_data
    
    def generate_comprehensive_report(self, hospital_id: str, all_analysis_results: Dict) -> Dict:
        """
        Generate a comprehensive multi-modal analysis report
        
        Args:
            hospital_id: Hospital identifier
            all_analysis_results: Combined results from all analyses
            
        Returns:
            Dict with comprehensive report
        """
        report = {
            'hospital_id': hospital_id,
            'timestamp': datetime.now().isoformat(),
            'analysis_summary': {},
            'recommendations': [],
            'alerts': [],
            'confidence_metrics': {}
        }
        
        # Summary metrics
        imagery = all_analysis_results.get('imagery', {})
        temporal = all_analysis_results.get('temporal', {})
        environmental = all_analysis_results.get('environmental', {})
        validation = all_analysis_results.get('validation', {})
        
        report['analysis_summary'] = {
            'current_fullness': imagery.get('fullness', 0),
            'vehicle_count': imagery.get('vehicle_count', 0),
            'person_count': imagery.get('person_count', 0),
            'trend': temporal.get('trend', 'unknown'),
            'anomaly_detected': temporal.get('is_anomaly', False),
            'air_quality_status': environmental.get('air_quality_status', 'unknown'),
            'environmental_risk': 'high' if environmental.get('air_quality_index', 0) > 100 else 'low'
        }
        
        # Generate recommendations
        fullness = imagery.get('fullness', 0)
        if fullness > 0.9:
            report['recommendations'].append("Consider directing patients to alternative facilities due to high capacity")
        elif fullness > 0.7:
            report['recommendations'].append("Monitor capacity closely - approaching full capacity")
        
        if temporal.get('is_anomaly', False):
            report['recommendations'].append("Investigate unusual activity patterns detected")
        
        if environmental.get('air_quality_index', 0) > 100:
            report['recommendations'].append("Poor air quality - consider additional patient monitoring")
        
        # Generate alerts
        if temporal.get('anomaly_score', 0) > 0.8:
            report['alerts'].append({
                'type': 'anomaly',
                'severity': 'high',
                'message': 'Unusual activity pattern detected'
            })
        
        if fullness > 0.95:
            report['alerts'].append({
                'type': 'capacity',
                'severity': 'critical',
                'message': 'Hospital at or near full capacity'
            })
        
        if environmental.get('air_quality_index', 0) > 150:
            report['alerts'].append({
                'type': 'environmental',
                'severity': 'high',
                'message': 'Unhealthy air quality levels detected'
            })
        
        # Confidence metrics
        report['confidence_metrics'] = {
            'overall_confidence': validation.get('confidence_score', 0.5),
            'reliability_assessment': validation.get('overall_reliability', 'unknown'),
            'data_quality': 'good' if len(temporal.get('data_points', [])) > 5 else 'limited'
        }
        
        return report

def run_enhanced_pipeline_demo():
    """Run the enhanced multi-modal pipeline demonstration"""
    
    print("🚀 Starting Enhanced Multi-Modal Hospital Analysis Pipeline")
    
    # Initialize analyzer
    analyzer = MultiModalHospitalAnalyzer()
    
    # Load hospital data
    print("📊 Loading hospital dataset...")
    hosp_url = 'https://healthdata.gov/resource/anag-cw7u.csv'
    try:
        df_hosp = pd.read_csv(hosp_url)
        df_hosp = df_hosp.dropna(subset=['latitude','longitude']).head(10)
        gdf_hosp = gpd.GeoDataFrame(
            df_hosp, 
            geometry=gpd.points_from_xy(df_hosp.longitude, df_hosp.latitude), 
            crs='EPSG:4326'
        )
        gdf_hosp['hospital_id'] = gdf_hosp.index
        print(f"✅ Loaded {len(gdf_hosp)} hospitals")
    except Exception as e:
        print(f"⚠️ Could not load hospital data: {e}")
        return
    
    # Process each hospital
    comprehensive_results = {}
    
    for idx, hospital in gdf_hosp.iterrows():
        hospital_id = str(hospital['hospital_id'])
        coords = (hospital['latitude'], hospital['longitude'])
        lon, lat = hospital['longitude'], hospital['latitude'] # Get separate lon/lat variables
        
        print(f"\n🏥 Processing Hospital {hospital_id}: {hospital.get('hospital_name', 'Unknown')}")
        
        # 1. Image Analysis (Placeholder image used for YOLO, will not match spectral data)
        print("  📸 Running enhanced image analysis...")
        # For demo, create a simple test image
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_img_path = f"temp_hospital_{hospital_id}.jpg"
        cv2.imwrite(test_img_path, test_img)
        
        imagery_results = analyzer.enhanced_image_analysis(test_img_path, coords)
        
        # Clean up temp file
        if os.path.exists(test_img_path):
            os.remove(test_img_path)
            
        # 2. Multi-spectral analysis (REAL STAC FETCH)
        print("  🛰️ Performing multi-spectral analysis (fetching real data)...")
        # New Step: Fetch real satellite data paths
        imagery_paths = analyzer.stac_client_fetcher(lon, lat)
        
        # Pass the real file paths to the spectral analysis function
        multispectral_results = analyzer.multi_spectral_analysis(
            imagery_paths=imagery_paths,
            hospital_coords=coords
        )
        
        # 3. Environmental data integration
        print("  🌍 Integrating environmental data...")
        environmental_data = analyzer.environmental_data_integration(coords)
        
        # Temporal analysis (uses image analysis results)
        print("  📈 Analyzing temporal patterns...")
        current_data = {
            'vehicle_count': imagery_results.get('vehicle_count', 0),
            'fullness': imagery_results.get('fullness', 0),
            'person_count': imagery_results.get('person_count', 0)
        }
        temporal_results = analyzer.temporal_analysis(hospital_id, current_data)
        
        # Cross-modal validation
        print("  🔍 Cross-validating results...")
        validation_results = analyzer.cross_modal_validation(
            {**imagery_results, **multispectral_results}, 
            imagery_results.get('vehicle_count', 0), 
            environmental_data
        )
        
        # Generate comprehensive report
        all_results = {
            'imagery': imagery_results,
            'multispectral': multispectral_results,
            'environmental': environmental_data,
            'temporal': temporal_results,
            'validation': validation_results
        }
        
        comprehensive_report = analyzer.generate_comprehensive_report(hospital_id, all_results)
        comprehensive_results[hospital_id] = comprehensive_report
        
        print(f"  ✅ Analysis complete - Confidence: {validation_results['confidence_score']:.2f}")
    
    # Generate summary visualization
    print("\n🗺️ Generating enhanced visualization...")
    
    # Create enhanced map with multi-modal data
    center_lat = gdf_hosp.geometry.y.mean()
    center_lon = gdf_hosp.geometry.x.mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles='CartoDB positron')
    
    for hospital_id, report in comprehensive_results.items():
        hospital_row = gdf_hosp[gdf_hosp['hospital_id'] == int(hospital_id)].iloc[0]
        
        # Enhanced color coding based on multiple factors
        fullness = report['analysis_summary']['current_fullness']
        anomaly = report['analysis_summary']['anomaly_detected']
        air_quality = report['analysis_summary']['environmental_risk']
        confidence = report['confidence_metrics']['overall_confidence']
        
        if anomaly or air_quality == 'high':
            color = 'red'
            icon = 'exclamation-triangle'
        elif fullness > 0.8:
            color = 'orange'
            icon = 'hospital'
        elif confidence < 0.6:
            color = 'gray'
            icon = 'question'
        else:
            color = 'green'
            icon = 'hospital'
        
        # Create detailed popup
        popup_html = f"""
        <div style="width: 300px;">
            <h4>{hospital_row.get('hospital_name', 'Hospital')} {hospital_id}</h4>
            <hr>
            <b>Current Status:</b><br>
            • Capacity: {fullness:.0%}<br>
            • Vehicles: {report['analysis_summary']['vehicle_count']}<br>
            • People: {report['analysis_summary']['person_count']}<br>
            • Trend: {report['analysis_summary']['trend']}<br>
            <br>
            <b>Environmental:</b><br>
            • Air Quality: {report['analysis_summary']['air_quality_status']}<br>
            • Risk Level: {report['analysis_summary']['environmental_risk']}<br>
            • Vegetation Health (NDVI): {report['multispectral'].get('vegetation_health', 'N/A'):.2f}<br>
            • Urban Heat Proxy (DN): {report['multispectral'].get('urban_heat_island', 'N/A'):.2f}<br>
            <br>
            <b>Analysis Quality:</b><br>
            • Confidence: {confidence:.0%}<br>
            • Reliability: {report['confidence_metrics']['reliability_assessment']}<br>
            <br>
            <b>Alerts:</b> {len(report['alerts'])}<br>
            <b>Recommendations:</b> {len(report['recommendations'])}
        </div>
        """
        
        folium.Marker(
            location=[hospital_row.geometry.y, hospital_row.geometry.x],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=color, icon=icon, prefix='fa')
        ).add_to(m)
    
    # Add legend
    legend_html = """
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 200px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>Hospital Status</b></p>
    <p><i class="fa fa-hospital" style="color:green"></i> Normal Operation</p>
    <p><i class="fa fa-hospital" style="color:orange"></i> High Capacity</p>
    <p><i class="fa fa-exclamation-triangle" style="color:red"></i> Alert Condition</p>
    <p><i class="fa fa-question" style="color:gray"></i> Low Confidence</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save results
    map_path = "enhanced_hospital_analysis.html"
    m.save(map_path)
    print(f"✅ Enhanced map saved: {map_path}")
    
    # Save comprehensive results as JSON
    results_path = "comprehensive_analysis_results.json"
    with open(results_path, 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    print(f"✅ Comprehensive results saved: {results_path}")
    
    # Print summary statistics
    print("\n📊 Analysis Summary:")
    total_hospitals = len(comprehensive_results)
    high_capacity = sum(1 for r in comprehensive_results.values() 
                       if r['analysis_summary']['current_fullness'] > 0.8)
    anomalies = sum(1 for r in comprehensive_results.values() 
                   if r['analysis_summary']['anomaly_detected'])
    env_risks = sum(1 for r in comprehensive_results.values() 
                   if r['analysis_summary']['environmental_risk'] == 'high')
    
    print(f"• Total hospitals analyzed: {total_hospitals}")
    print(f"• High capacity alerts: {high_capacity}")
    print(f"• Anomalies detected: {anomalies}")
    print(f"• Environmental risks: {env_risks}")
    
    avg_confidence = np.mean([r['confidence_metrics']['overall_confidence'] 
                             for r in comprehensive_results.values()])
    print(f"• Average confidence: {avg_confidence:.0%}")
    
    print("\n🎉 Enhanced multi-modal pipeline completed successfully!")
    
    return comprehensive_results

if __name__ == "__main__":
    # Run the enhanced pipeline
    results = run_enhanced_pipeline_demo()
}
