# 🚀 Fixed Multi-Modal Space Apps Pipeline
# Advanced version with multi-modal AI, temporal analysis, and cross-validation

import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Optional imports with error handling
try:
    import geopandas as gpd
    from shapely.geometry import Point, box
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    print("Warning: GeoPandas not available. Using basic coordinate handling.")
    GEOSPATIAL_AVAILABLE = False

try:
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    print("Warning: Folium not available. Map generation will be skipped.")
    FOLIUM_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("Warning: OpenCV not available. Using simulated image analysis.")
    CV2_AVAILABLE = False

try:
    import rasterio
    from rasterio.windows import from_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    print("Warning: Rasterio not available. Using simulated satellite analysis.")
    RASTERIO_AVAILABLE = False

class MultiModalHospitalAnalyzer:
    """Enhanced multi-modal analyzer for hospital capacity and environmental monitoring"""
    
    def __init__(self):
        self.yolo_model = None
        self.historical_data = {}
        self.anomaly_threshold = 0.7
        
    def load_models(self):
        """Load AI models (simulated if YOLO not available)"""
        print("Initializing detection models...")
        # In a real implementation, you would load YOLO here
        # For now, we'll simulate detection results
        self.yolo_model = "simulated_yolo"  # Placeholder
        
    def multi_spectral_analysis(self, imagery_paths_or_coords, hospital_coords):
        """
        Analyze multiple spectral bands for comprehensive environmental assessment
        
        Args:
            imagery_paths_or_coords: Dict mapping band names to file paths or coordinates
            hospital_coords: (lat, lon) of hospital
        
        Returns:
            Dict with multi-spectral analysis results
        """
        results = {
            'vegetation_health': None,
            'urban_heat_island': None,
            'water_stress': None,
            'air_quality_proxy': None,
            'change_detection': None
        }
        
        try:
            # Simulate multi-spectral analysis since we don't have real satellite data
            # In production, this would process actual satellite imagery
            
            # Simulate NDVI calculation (vegetation health)
            results['vegetation_health'] = np.random.uniform(0.2, 0.8)
            
            # Simulate thermal analysis for urban heat island effect
            results['urban_heat_island'] = np.random.uniform(25, 35)
            
            # Simulate water stress indicators
            results['water_stress'] = np.random.uniform(0.1, 0.6)
            
            # Simulate air quality proxy from spectral data
            results['air_quality_proxy'] = np.random.uniform(0.3, 0.9)
            
            print(f"  Multi-spectral analysis completed for coordinates {hospital_coords}")
            
        except Exception as e:
            print(f"Multi-spectral analysis error: {e}")
            
        return results
    
    def temporal_analysis(self, hospital_id, current_data, days_back=30):
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
            return {'trend': 'insufficient_data', 'anomaly_score': 0.0, 'data_points': len(history)}
        
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
            'historical_avg_fullness': np.mean(fullness_values) if fullness_values else 0.0,
            'data_points': len(history)
        }
    
    def cross_modal_validation(self, imagery_analysis, vehicle_count, environmental_data):
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
            expected_capacity = max(10, imagery_analysis['parking_area_pixels'] // 400)  # Rough estimate
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
    
    def enhanced_image_analysis(self, image_path_or_coords, hospital_coords):
        """
        Enhanced image analysis with multiple detection approaches
        
        Args:
            image_path_or_coords: Path to image file or coordinates for simulation
            hospital_coords: (lat, lon) of hospital
            
        Returns:
            Dict with comprehensive analysis results
        """
        if not self.yolo_model:
            self.load_models()
        
        results = {}
        
        # Simulate image analysis if OpenCV is not available or if using coordinates
        if not CV2_AVAILABLE or not os.path.exists(str(image_path_or_coords)):
            print(f"  Simulating image analysis for {hospital_coords}")
            
            # Simulate detection results
            vehicle_count = np.random.randint(5, 50)
            person_count = np.random.randint(10, 100)
            
            vehicle_details = []
            for i in range(vehicle_count):
                vehicle_details.append({
                    'type': np.random.choice(['car', 'truck', 'bus', 'motorcycle']),
                    'confidence': np.random.uniform(0.4, 0.95),
                    'bbox': [
                        np.random.randint(0, 400),
                        np.random.randint(0, 300),
                        np.random.randint(400, 640),
                        np.random.randint(300, 480)
                    ]
                })
            
            results['vehicle_count'] = vehicle_count
            results['person_count'] = person_count
            results['vehicle_details'] = vehicle_details
            
            # Simulate parking analysis
            estimated_spaces = np.random.randint(20, 80)
            parking_area = estimated_spaces * 400  # Rough pixel estimate
            
            results['parking_area_pixels'] = parking_area
            results['estimated_parking_spaces'] = estimated_spaces
            
            # Calculate fullness
            estimated_capacity = max(estimated_spaces, vehicle_count)
            results['fullness'] = min(1.0, vehicle_count / estimated_capacity) if estimated_capacity > 0 else 0.0
            
            # Simulate shadow analysis
            results['shadow_ratio'] = np.random.uniform(0.1, 0.4)
            
        else:
            # Real image analysis would go here
            img = cv2.imread(image_path_or_coords)
            if img is None:
                return {'error': 'Could not load image'}
            
            # Actual YOLO detection and image processing would happen here
            # For now, we'll use the simulation above
            pass
        
        return results
    
    def environmental_data_integration(self, hospital_coords, current_time=None):
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
        
        # Simulate environmental data (in production, would use real APIs)
        environmental_data = {
            'air_quality_index': np.random.randint(20, 150),
            'weather': np.random.choice(['clear', 'cloudy', 'rainy', 'severe']),
            'temperature': np.random.normal(20, 10),
            'humidity': np.random.uniform(30, 90),
            'wind_speed': np.random.uniform(0, 25),
            'visibility': np.random.uniform(1, 15),  # km
            'timestamp': current_time,
            'source': 'simulated'
        }
        
        # Add derived metrics
        if environmental_data['air_quality_index'] > 100:
            environmental_data['air_quality_status'] = 'unhealthy'
        elif environmental_data['air_quality_index'] > 50:
            environmental_data['air_quality_status'] = 'moderate'
        else:
            environmental_data['air_quality_status'] = 'good'
        
        return environmental_data
    
    def generate_comprehensive_report(self, hospital_id, all_analysis_results):
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
            'data_quality': 'good' if temporal.get('data_points', 0) > 5 else 'limited'
        }
        
        return report

def create_sample_hospital_data():
    """Create sample hospital data for testing"""
    sample_hospitals = [
        {
            'hospital_id': 0,
            'hospital_name': 'Central Medical Center',
            'latitude': 37.7749,
            'longitude': -122.4194
        },
        {
            'hospital_id': 1,
            'hospital_name': 'Bay Area General Hospital',
            'latitude': 37.7849,
            'longitude': -122.4094
        },
        {
            'hospital_id': 2,
            'hospital_name': 'Pacific Coast Medical',
            'latitude': 37.7649,
            'longitude': -122.4294
        },
        {
            'hospital_id': 3,
            'hospital_name': 'Golden Gate Health Center',
            'latitude': 37.7549,
            'longitude': -122.4394
        },
        {
            'hospital_id': 4,
            'hospital_name': 'Metropolitan Hospital',
            'latitude': 37.7949,
            'longitude': -122.3994
        }
    ]
    
    df = pd.DataFrame(sample_hospitals)
    return df

def run_enhanced_pipeline_demo():
    """Run the enhanced multi-modal pipeline demonstration"""
    
    print("🚀 Starting Enhanced Multi-Modal Hospital Analysis Pipeline")
    
    # Initialize analyzer
    analyzer = MultiModalHospitalAnalyzer()
    
    # Load hospital data
    print("📊 Loading hospital dataset...")
    try:
        # Try to load real data first
        hosp_url = 'https://healthdata.gov/resource/anag-cw7u.csv'
        df_hosp = pd.read_csv(hosp_url, timeout=10)
        df_hosp = df_hosp.dropna(subset=['latitude','longitude']).head(5)
        print(f"✅ Loaded {len(df_hosp)} hospitals from HealthData.gov")
    except Exception as e:
        print(f"⚠️ Could not load hospital data from URL: {e}")
        print("Using sample hospital data instead...")
        df_hosp = create_sample_hospital_data()
        print(f"✅ Created {len(df_hosp)} sample hospitals")
    
    # Ensure we have the required columns
    if 'hospital_id' not in df_hosp.columns:
        df_hosp['hospital_id'] = df_hosp.index
    if 'hospital_name' not in df_hosp.columns:
        df_hosp['hospital_name'] = df_hosp.apply(lambda x: f"Hospital {x['hospital_id']}", axis=1)
    
    # Process each hospital
    comprehensive_results = {}
    
    for idx, hospital in df_hosp.iterrows():
        hospital_id = str(hospital['hospital_id'])
        coords = (hospital['latitude'], hospital['longitude'])
        
        print(f"\n🏥 Processing Hospital {hospital_id}: {hospital.get('hospital_name', 'Unknown')}")
        
        # Enhanced image analysis (simulated)
        print("  📸 Running enhanced image analysis...")
        imagery_results = analyzer.enhanced_image_analysis(coords, coords)
        
        # Multi-spectral analysis (simulated)
        print("  🛰️ Performing multi-spectral analysis...")
        multispectral_results = analyzer.multi_spectral_analysis({}, coords)
        
        # Environmental data integration
        print("  🌍 Integrating environmental data...")
        environmental_data = analyzer.environmental_data_integration(coords)
        
        # Temporal analysis
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
    
    # Generate visualization
    print("\n🗺️ Generating visualization...")
    
    if FOLIUM_AVAILABLE:
        # Create enhanced map with multi-modal data
        center_lat = df_hosp['latitude'].mean()
        center_lon = df_hosp['longitude'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles='CartoDB positron')
        
        for hospital_id, report in comprehensive_results.items():
            hospital_row = df_hosp[df_hosp['hospital_id'] == int(hospital_id)].iloc[0]
            
            # Enhanced color coding based on multiple factors
            fullness = report['analysis_summary']['current_fullness']
            anomaly = report['analysis_summary']['anomaly_detected']
            air_quality = report['analysis_summary']['environmental_risk']
            confidence = report['confidence_metrics']['overall_confidence']
            
            if anomaly or air_quality == 'high':
                color = 'red'
            elif fullness > 0.8:
                color = 'orange'
            elif confidence < 0.6:
                color = 'gray'
            else:
                color = 'green'
            
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
                location=[hospital_row['latitude'], hospital_row['longitude']],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(m)
        
        # Save map
        map_path = "enhanced_hospital_analysis.html"
        m.save(map_path)
        print(f"✅ Enhanced map saved: {map_path}")
    else:
        print("⚠️ Folium not available - skipping map generation")
    
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
