# Examples

This directory contains practical examples demonstrating how to use the `uncertain-rs`
library for real-world scenarios involving uncertainty quantification and error propagation.

- [Examples](#examples)
  - [Key Concepts Demonstrated](#key-concepts-demonstrated)
    - [Uncertainty Types](#uncertainty-types)
    - [Operations](#operations)
    - [Real-World Applications](#real-world-applications)
  - [Learning Path](#learning-path)
  - [Running Examples](#running-examples)
  - [Examples Overview](#examples-overview)
    - [GPS Navigation (`gps_navigation.rs`)](#gps-navigation-gps_navigationrs)
    - [Medical Diagnosis (`medical_diagnosis.rs`)](#medical-diagnosis-medical_diagnosisrs)
    - [Climate Modeling (`climate_modeling.rs`)](#climate-modeling-climate_modelingrs)
    - [Sensor Processing (`sensor_processing.rs`)](#sensor-processing-sensor_processingrs)

## Key Concepts Demonstrated

### Uncertainty Types

- **Normal distributions**: For measurements with known mean and standard deviation
- **Uniform distributions**: For ranges with equal probability
- **Point values**: For exact known values
- **Custom distributions**: Using closure-based sampling

### Operations

- **Arithmetic**: Addition, subtraction, multiplication, division with uncertainty propagation
- **Comparisons**: Greater than, less than, equality with probabilistic results
- **Transformations**: Custom functions applied to uncertain values
- **Aggregation**: Combining multiple uncertain values

### Real-World Applications

- **Risk Assessment**: Calculating probabilities of exceeding thresholds
- **Decision Making**: Using confidence intervals to make informed choices
- **Error Handling**: Graceful degradation when data quality is poor
- **Validation**: Cross-checking between different measurement sources

## Learning Path

1. **Start with GPS Navigation**: Simple uncertainty propagation and decision making
2. **Try Medical Diagnosis**: Evidence-based reasoning and risk assessment
3. **Explore Climate Modeling**: Complex multi-factor uncertainty analysis
4. **Study Sensor Processing**: Comprehensive error handling and system reliability

Each example builds on concepts from the previous ones, demonstrating increasingly
sophisticated applications of uncertainty quantification.

## Running Examples

Run any example using cargo:

```bash
cargo run --example <example_name>
```

## Examples Overview

### GPS Navigation (`gps_navigation.rs`)

**Purpose**: Shows how GPS measurement uncertainty affects route planning,
arrival time predictions, and travel decisions.

**Key Features**:

- GPS position uncertainty due to satellite errors
- Distance calculation with error propagation
- Travel time analysis with traffic variability
- Route comparison with confidence intervals
- Fuel consumption analysis with gauge uncertainty

**Sample Output**:

```bash
🚗 GPS Navigation with Uncertainty Analysis
===========================================

📍 Distance Analysis:
   Mean distance: 0.976 miles
   Std deviation: 0.0068 miles
   95% confidence: 0.963 - 0.989 miles

⏱️  Travel Time Analysis:
   Expected time: 2.3 minutes
   Std deviation: 0.8 minutes
   Probability of taking >10min: 0.0%

🛣️  Route Decision Analysis:
   Main route faster: 85.0% confidence
   ✅ Recommendation: Take main route

⛽ Fuel Consumption Analysis:
   Expected fuel: 0.036 gallons
   Confidence we have enough fuel: 100.0%
```

### Medical Diagnosis (`medical_diagnosis.rs`)

**Purpose**: Demonstrates evidence-based medical decision making with uncertain test
results and measurement errors.

**Key Features**:

- Blood pressure, cholesterol, glucose, and BMI measurements with uncertainty
- Risk factor assessment for hypertension, diabetes, and cardiovascular disease
- Combined risk scoring with confidence intervals
- Treatment recommendations based on uncertainty analysis
- Diagnostic confidence assessment and follow-up recommendations

**Sample Output**:

```bash
🏥 Medical Diagnosis with Uncertainty Analysis
=============================================

📊 Patient Measurements:
   Blood Pressure: 145.0 ± 8.0 mmHg
   Cholesterol: 220.0 ± 15.0 mg/dL
   Glucose: 126.0 ± 12.0 mg/dL
   BMI: 25.6 (calculated with uncertainty)

🔬 Risk Factor Analysis:
   Hypertension risk: 73.3% confidence
   High cholesterol risk: 89.9% confidence
   Diabetes risk: 54.4% confidence
   Obesity risk: 0.0% confidence

❤️  Cardiovascular Risk Assessment:
   Average risk score: 4.1
   Low risk probability: 12.1%
   Moderate risk probability: 36.1%
   High risk probability: 53.9%

💊 Treatment Recommendations:
   🚨 HIGH RISK: Immediate intervention recommended
      - Start medication for hypertension/cholesterol
      - Strict dietary modifications
      - Regular monitoring required
```

### Climate Modeling (`climate_modeling.rs`)

**Purpose**: Demonstrates uncertainty propagation in climate change impact assessment,
including temperature projections, sea level rise, and economic modeling.

**Key Features**:

- CO2 concentration and temperature sensitivity modeling
- Future emissions scenarios with uncertainty
- Sea level rise calculations from multiple factors
- Economic impact assessment with GDP projections
- Policy recommendations with carbon pricing
- Tipping point risk assessment

**Sample Output**:

```bash
🌍 Climate Change Impact Assessment
==================================

🌡️  Current Climate State:
   Baseline warming: 1.1°C ± 0.2°C
   CO2 concentration: 420 ± 5 ppm
   Climate sensitivity: 3.0°C ± 1.0°C per CO2 doubling

📈 Future CO2 Projections (2050):
   Expected CO2: 507 ± 50 ppm
   Range (95% confidence): 410 - 604 ppm

🌡️  Temperature Projections:
   Expected warming: 3.7°C ± 1.0°C
   Probability > 1.5°C: 99.2%
   Probability > 2.0°C: 96.9%
   Probability > 3.0°C: 74.0%

⚠️  Risk Assessment Summary:
   🚨 HIGH RISK: >50% chance of exceeding 2°C target
      - Immediate aggressive action required
      - International cooperation essential

⚡ Tipping Point Assessment:
   Amazon rainforest collapse risk: 53.7%
   Arctic ice loss risk: 96.3%
   Permafrost melting risk: 98.3%
   🚨 CRITICAL: High risk of irreversible tipping points!
```

### Sensor Processing (`sensor_processing.rs`)

**Purpose**: Comprehensive example of robust sensor data processing with error handling,
sensor fusion, and system reliability assessment.

**Key Features**:

- Multiple sensor types with various failure modes
- Error handling for sensor failures, calibration drift, and communication errors
- Sensor fusion with uncertainty-weighted averaging
- Anomaly detection and corrective action recommendations
- Fallback strategies and graceful degradation
- Long-term reliability assessment and maintenance planning

**Sample Output**:

```bash
🔧 Robust Sensor Data Processing with Error Handling
===================================================

📊 Raw Sensor Data Status:
   ⚠️  humidity_1: 45.2 ±3.5 (Degraded) [t=999]
   ❌ humidity_2: N/A unknown (Failed) [t=980]
   ✅ temp_1: 23.2 ±0.5 (Healthy) [t=1000]
   📡 temp_3: N/A unknown (CommunicationError) [t=995]
   ✅ pressure_1: 1013.2 ±2.0 (Healthy) [t=1001]
   ⚡ temp_2: 85.7 ±5.0 (OutOfRange) [t=1002]
   🔧 pressure_2: 1015.8 ±8.0 (CalibrationDrift) [t=1003]

⚙️  Sensor Fusion with Uncertainty Propagation:
   Temperature fusion with error handling:
      ✅ Fusing 2 temperature sensors
         temp_1: 23.2°C ±0.5 (weight: 1.66)
         temp_2: 85.8°C ±9.8 (weight: 0.10)
      🎯 Fused temperature: 26.8°C ±0.8
      ⚠️  Large sensor disagreement (62.6°C) - possible sensor failure

📈 Long-term Reliability Assessment:
      📊 System availability: 42.9%
      📊 System health: 28.6%
      🎯 Overall system risk: CRITICAL
      💡 Recommendations:
         - Implement redundant sensor deployment
         - Increase monitoring frequency
         - Review maintenance procedures
```
