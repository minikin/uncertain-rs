#![allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]

use std::collections::HashMap;
use uncertain_rs::Uncertain;

/// Sensor data processing with comprehensive error handling
///
/// This example demonstrates how to handle common real-world scenarios:
/// - Sensor failures and missing data
/// - Out-of-range readings and calibration drift
/// - Network timeouts and communication errors
/// - Data validation and outlier detection
/// - Graceful degradation and fallback strategies
fn main() {
    println!("🔧 Robust Sensor Data Processing with Error Handling");
    println!("===================================================\n");

    // Simulate a multi-sensor system with various failure modes
    let sensor_readings = simulate_sensor_data();

    println!("📊 Raw Sensor Data Status:");
    print_sensor_status(&sensor_readings);

    // Process sensor data with error handling and validation
    let processed_data = process_sensor_data_robust(&sensor_readings);

    println!("\n🛡️  Error Handling and Data Validation:");
    validate_and_process(&processed_data);

    println!("\n⚙️  Sensor Fusion with Uncertainty Propagation:");
    sensor_fusion_with_errors(&processed_data);

    println!("\n🚨 Anomaly Detection and Outlier Handling:");
    detect_and_handle_anomalies(&sensor_readings);

    println!("\n🔄 Fallback Strategies and Graceful Degradation:");
    demonstrate_fallback_strategies(&sensor_readings);

    println!("\n📈 Long-term Reliability Assessment:");
    assess_system_reliability(&sensor_readings);
}

#[derive(Debug, Clone)]
struct SensorReading {
    id: String,
    value: Option<f64>,
    timestamp: u64,
    status: SensorStatus,
    uncertainty: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
enum SensorStatus {
    Healthy,
    Degraded,
    Failed,
    OutOfRange,
    CalibrationDrift,
    CommunicationError,
}

fn simulate_sensor_data() -> HashMap<String, SensorReading> {
    let mut sensors = HashMap::new();

    // Temperature sensors with various issues
    sensors.insert(
        "temp_1".to_string(),
        SensorReading {
            id: "temp_1".to_string(),
            value: Some(23.2),
            timestamp: 1000,
            status: SensorStatus::Healthy,
            uncertainty: Some(0.5),
        },
    );

    sensors.insert(
        "temp_2".to_string(),
        SensorReading {
            id: "temp_2".to_string(),
            value: Some(85.7), // Unrealistic reading - likely sensor failure
            timestamp: 1002,
            status: SensorStatus::OutOfRange,
            uncertainty: Some(5.0), // High uncertainty due to suspected failure
        },
    );

    sensors.insert(
        "temp_3".to_string(),
        SensorReading {
            id: "temp_3".to_string(),
            value: None, // Communication timeout
            timestamp: 995,
            status: SensorStatus::CommunicationError,
            uncertainty: None,
        },
    );

    // Pressure sensors
    sensors.insert(
        "pressure_1".to_string(),
        SensorReading {
            id: "pressure_1".to_string(),
            value: Some(1013.25),
            timestamp: 1001,
            status: SensorStatus::Healthy,
            uncertainty: Some(2.0),
        },
    );

    sensors.insert(
        "pressure_2".to_string(),
        SensorReading {
            id: "pressure_2".to_string(),
            value: Some(1015.8),
            timestamp: 1003,
            status: SensorStatus::CalibrationDrift, // Systematic bias detected
            uncertainty: Some(8.0), // Increased uncertainty due to calibration issues
        },
    );

    // Humidity sensors
    sensors.insert(
        "humidity_1".to_string(),
        SensorReading {
            id: "humidity_1".to_string(),
            value: Some(45.2),
            timestamp: 999,
            status: SensorStatus::Degraded, // Aging sensor with reduced accuracy
            uncertainty: Some(3.5),
        },
    );

    sensors.insert(
        "humidity_2".to_string(),
        SensorReading {
            id: "humidity_2".to_string(),
            value: None, // Complete sensor failure
            timestamp: 980,
            status: SensorStatus::Failed,
            uncertainty: None,
        },
    );

    sensors
}

fn print_sensor_status(sensors: &HashMap<String, SensorReading>) {
    for reading in sensors.values() {
        let value_str = match reading.value {
            Some(v) => format!("{v:.1}"),
            None => "N/A".to_string(),
        };

        let uncertainty_str = match reading.uncertainty {
            Some(u) => format!("±{u:.1}"),
            None => "unknown".to_string(),
        };

        let status_icon = match reading.status {
            SensorStatus::Healthy => "✅",
            SensorStatus::Degraded => "⚠️ ",
            SensorStatus::Failed => "❌",
            SensorStatus::OutOfRange => "⚡",
            SensorStatus::CalibrationDrift => "🔧",
            SensorStatus::CommunicationError => "📡",
        };

        println!(
            "   {status_icon} {}: {value_str} {uncertainty_str} ({:?}) [t={}]",
            reading.id, reading.status, reading.timestamp
        );
    }
}

fn process_sensor_data_robust(
    sensors: &HashMap<String, SensorReading>,
) -> HashMap<String, Result<Uncertain<f64>, String>> {
    let mut processed = HashMap::new();

    for (id, reading) in sensors {
        let result = match (&reading.status, reading.value, reading.uncertainty) {
            // Healthy sensors: use as-is
            (SensorStatus::Healthy, Some(value), Some(uncertainty)) => {
                Ok(Uncertain::normal(value, uncertainty))
            }

            // Degraded sensors: increase uncertainty
            (SensorStatus::Degraded, Some(value), Some(uncertainty)) => {
                let degraded_uncertainty = uncertainty * 2.0; // Double uncertainty for degraded sensors
                Ok(Uncertain::normal(value, degraded_uncertainty))
            }

            // Out-of-range sensors: try to salvage with high uncertainty
            (SensorStatus::OutOfRange, Some(value), _) => {
                if is_physically_plausible(id, value) {
                    // Use reading but with very high uncertainty
                    Ok(Uncertain::normal(value, 10.0))
                } else {
                    Err(format!(
                        "Sensor {id} reading {value} is physically implausible"
                    ))
                }
            }

            // Calibration drift: apply correction with increased uncertainty
            (SensorStatus::CalibrationDrift, Some(value), Some(uncertainty)) => {
                let corrected_value = apply_calibration_correction(id, value);
                let drift_uncertainty = uncertainty * 1.5 + 2.0; // Account for correction uncertainty
                Ok(Uncertain::normal(corrected_value, drift_uncertainty))
            }

            // Failed or communication error sensors
            (SensorStatus::Failed | SensorStatus::CommunicationError, _, _) => {
                Err(format!("Sensor {id} is unavailable"))
            }

            _ => Err(format!("Sensor {id} has invalid data configuration")),
        };

        processed.insert(id.clone(), result);
    }

    processed
}

fn is_physically_plausible(sensor_id: &str, value: f64) -> bool {
    match sensor_id {
        id if id.starts_with("temp") => (-50.0..=100.0).contains(&value), // Celsius
        id if id.starts_with("pressure") => (800.0..=1200.0).contains(&value), // hPa
        id if id.starts_with("humidity") => (0.0..=100.0).contains(&value), // %
        _ => true, // Unknown sensor type, assume plausible
    }
}

fn apply_calibration_correction(sensor_id: &str, value: f64) -> f64 {
    // Simulate known calibration corrections
    match sensor_id {
        "pressure_2" => value - 2.3, // Known systematic bias
        id if id.starts_with("temp") => value * 0.98 + 0.5, // Linear correction
        _ => value,                  // No correction needed
    }
}

fn validate_and_process(processed_data: &HashMap<String, Result<Uncertain<f64>, String>>) {
    let mut healthy_sensors = 0;
    let mut _failed_sensors = 0;
    let mut total_uncertainty = 0.0;

    for (id, result) in processed_data {
        match result {
            Ok(uncertain_value) => {
                healthy_sensors += 1;

                // Sample uncertainty to estimate data quality
                let samples = uncertain_value.take_samples(100);
                let mean = samples.iter().sum::<f64>() / samples.len() as f64;
                let std_dev = (samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                    / samples.len() as f64)
                    .sqrt();
                total_uncertainty += std_dev;

                println!("   ✅ {id}: μ={mean:.1}, σ={std_dev:.1}");

                // Validate uncertainty bounds
                if std_dev > 5.0 {
                    println!("      ⚠️  High uncertainty detected - consider sensor maintenance");
                }
            }
            Err(error) => {
                _failed_sensors += 1;
                println!("   ❌ {id}: {error}");
            }
        }
    }

    let total_sensors = processed_data.len();
    let reliability = f64::from(healthy_sensors) / total_sensors as f64 * 100.0;
    let avg_uncertainty = if healthy_sensors > 0 {
        total_uncertainty / f64::from(healthy_sensors)
    } else {
        0.0
    };

    println!(
        "\n   📊 System Health: {healthy_sensors}/{total_sensors} sensors operational ({reliability:.1}%)"
    );
    println!("   📊 Average uncertainty: {avg_uncertainty:.1}");

    if reliability < 60.0 {
        println!("   🚨 CRITICAL: System reliability below acceptable threshold!");
    } else if reliability < 80.0 {
        println!("   ⚠️  WARNING: System reliability degraded");
    }
}

fn sensor_fusion_with_errors(processed_data: &HashMap<String, Result<Uncertain<f64>, String>>) {
    println!("   Temperature fusion with error handling:");

    // Collect all healthy temperature sensors
    let temp_sensors: Vec<_> = processed_data
        .iter()
        .filter(|(id, _)| id.starts_with("temp"))
        .filter_map(|(id, result)| match result {
            Ok(uncertain) => Some((id, uncertain)),
            Err(_) => None,
        })
        .collect();

    if temp_sensors.is_empty() {
        println!("      ❌ No healthy temperature sensors available!");
        return;
    }

    if temp_sensors.len() == 1 {
        println!("      ⚠️  Only one temperature sensor available - no redundancy");
        let (_id, temp) = temp_sensors[0];
        let samples = temp.take_samples(1000);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        println!("      📊 Single sensor reading: {mean:.1}°C");
        return;
    }

    // Multi-sensor fusion with uncertainty weighting
    println!("      ✅ Fusing {} temperature sensors", temp_sensors.len());

    // Weight sensors by inverse of their uncertainty (lower uncertainty = higher weight)
    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;

    for (id, temp_uncertain) in &temp_sensors {
        let samples = temp_uncertain.take_samples(1000);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let std_dev =
            (samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64).sqrt();

        // Weight by inverse uncertainty (with minimum threshold to avoid division by zero)
        let weight = 1.0 / (std_dev + 0.1);
        weighted_sum += mean * weight;
        total_weight += weight;

        println!("         {id}: {mean:.1}°C ±{std_dev:.1} (weight: {weight:.2})");
    }

    let fused_temperature = weighted_sum / total_weight;
    let fused_uncertainty = 1.0 / total_weight.sqrt(); // Combined uncertainty

    println!("      🎯 Fused temperature: {fused_temperature:.1}°C ±{fused_uncertainty:.1}");

    // Detect sensor disagreement
    let temp_values: Vec<f64> = temp_sensors
        .iter()
        .map(|(_, temp)| {
            let samples = temp.take_samples(100);
            samples.iter().sum::<f64>() / samples.len() as f64
        })
        .collect();

    if temp_values.len() > 1 {
        let max_diff = temp_values
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            - temp_values
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();

        if max_diff > 5.0 {
            println!(
                "      ⚠️  Large sensor disagreement ({max_diff:.1}°C) - possible sensor failure"
            );
        }
    }
}

fn detect_and_handle_anomalies(sensors: &HashMap<String, SensorReading>) {
    for (id, reading) in sensors {
        if let Some(value) = reading.value {
            let is_anomaly = detect_anomaly(id, value, &reading.status);

            if is_anomaly {
                println!("   🚨 Anomaly detected in {id}: {value}");

                // Suggest corrective actions
                match reading.status {
                    SensorStatus::OutOfRange => {
                        println!(
                            "      💡 Action: Check sensor calibration and physical installation"
                        );
                    }
                    SensorStatus::CalibrationDrift => {
                        println!("      💡 Action: Schedule sensor recalibration");
                    }
                    SensorStatus::Degraded => {
                        println!("      💡 Action: Replace sensor within maintenance window");
                    }
                    _ => {
                        println!(
                            "      💡 Action: Investigate sensor and environmental conditions"
                        );
                    }
                }
            }
        }
    }
}

fn detect_anomaly(sensor_id: &str, value: f64, status: &SensorStatus) -> bool {
    // Multiple anomaly detection criteria
    let out_of_normal_range = match sensor_id {
        id if id.starts_with("temp") => !(15.0..=35.0).contains(&value), // Typical indoor range
        id if id.starts_with("pressure") => !(980.0..=1050.0).contains(&value), // Normal atmospheric range
        id if id.starts_with("humidity") => !(20.0..=80.0).contains(&value),    // Comfortable range
        _ => false,
    };

    let sensor_status_indicates_problem = matches!(
        status,
        SensorStatus::Failed | SensorStatus::OutOfRange | SensorStatus::CalibrationDrift
    );

    out_of_normal_range || sensor_status_indicates_problem
}

fn demonstrate_fallback_strategies(sensors: &HashMap<String, SensorReading>) {
    println!("   Implementing fallback strategies:");

    // Strategy 1: Use historical data when sensor fails
    let temp_sensors: Vec<_> = sensors
        .iter()
        .filter(|(id, _)| id.starts_with("temp"))
        .collect();

    let healthy_temp_count = temp_sensors
        .iter()
        .filter(|(_, reading)| reading.status == SensorStatus::Healthy)
        .count();

    if healthy_temp_count == 0 {
        println!("      🔄 No healthy temperature sensors - using historical model");
        let historical_temp = Uncertain::normal(22.0, 3.0); // Based on historical data
        let samples = historical_temp.take_samples(100);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        println!("         Historical estimate: {mean:.1}°C ±3.0 (high uncertainty)");
    }

    // Strategy 2: Cross-validation between sensor types
    println!("      🔄 Cross-validation between sensor types:");

    if let (Some(pressure_reading), Some(temp_reading)) =
        (sensors.get("pressure_1"), sensors.get("temp_1"))
    {
        if let (Some(pressure), Some(temp)) = (pressure_reading.value, temp_reading.value) {
            // Sanity check: pressure and temperature correlation
            let expected_temp_from_pressure = estimate_temperature_from_pressure(pressure);
            let temp_diff = (temp - expected_temp_from_pressure).abs();

            if temp_diff > 10.0 {
                println!("         ⚠️  Temperature-pressure correlation check failed");
                println!(
                    "         Expected temp: {expected_temp_from_pressure:.1}°C, Measured: {temp:.1}°C"
                );
            } else {
                println!("         ✅ Temperature-pressure correlation validated");
            }
        }
    }

    // Strategy 3: Graceful degradation
    println!("      🔄 Graceful degradation modes:");
    let operational_sensors = sensors
        .values()
        .filter(|r| matches!(r.status, SensorStatus::Healthy | SensorStatus::Degraded))
        .count();

    match operational_sensors {
        0 => println!("         🚨 EMERGENCY MODE: All sensors failed - system shutdown required"),
        1..=2 => println!("         ⚠️  REDUCED MODE: Limited sensors - increased uncertainty"),
        3..=4 => println!("         📊 NORMAL MODE: Adequate sensor coverage"),
        _ => println!("         ✅ FULL MODE: All sensors operational"),
    }
}

fn estimate_temperature_from_pressure(pressure: f64) -> f64 {
    // Simplified barometric formula for temperature estimation
    // This is a rough approximation for demonstration
    20.0 + (pressure - 1013.25) * 0.02
}

fn assess_system_reliability(sensors: &HashMap<String, SensorReading>) {
    println!("   Long-term reliability metrics:");

    let total_sensors = sensors.len() as f64;
    let healthy_sensors = sensors
        .values()
        .filter(|r| r.status == SensorStatus::Healthy)
        .count() as f64;
    let degraded_sensors = sensors
        .values()
        .filter(|r| r.status == SensorStatus::Degraded)
        .count() as f64;
    let failed_sensors = sensors
        .values()
        .filter(|r| {
            matches!(
                r.status,
                SensorStatus::Failed | SensorStatus::CommunicationError
            )
        })
        .count() as f64;

    let system_availability = (healthy_sensors + degraded_sensors) / total_sensors * 100.0;
    let system_health = healthy_sensors / total_sensors * 100.0;

    println!("      📊 System availability: {system_availability:.1}%");
    println!("      📊 System health: {system_health:.1}%");
    println!("      📊 Failed sensors: {failed_sensors:.0}/{total_sensors:.0}");

    // Predict maintenance needs
    if degraded_sensors > 0.0 {
        let maintenance_urgency = degraded_sensors / total_sensors * 100.0;
        println!(
            "      🔧 Maintenance needed for {degraded_sensors:.0} sensors ({maintenance_urgency:.1}% of system)"
        );

        if maintenance_urgency > 30.0 {
            println!("         🚨 HIGH PRIORITY: Schedule immediate maintenance");
        } else if maintenance_urgency > 15.0 {
            println!("         ⚠️  MEDIUM PRIORITY: Schedule maintenance within week");
        } else {
            println!("         📅 LOW PRIORITY: Include in next routine maintenance");
        }
    }

    // Risk assessment
    let risk_level = if system_health < 50.0 {
        "CRITICAL"
    } else if system_health < 70.0 {
        "HIGH"
    } else if system_health < 85.0 {
        "MEDIUM"
    } else {
        "LOW"
    };

    println!("      🎯 Overall system risk: {risk_level}");

    // Recommendations
    if system_health < 70.0 {
        println!("      💡 Recommendations:");
        println!("         - Implement redundant sensor deployment");
        println!("         - Increase monitoring frequency");
        println!("         - Review maintenance procedures");
    }
}
