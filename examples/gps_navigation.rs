use uncertain_rs::Uncertain;

/// GPS Navigation with Uncertainty-Aware Route Planning
///
/// This example demonstrates how uncertain GPS readings affect
/// arrival time predictions and route decisions.

fn main() {
    println!("🚗 GPS Navigation with Uncertainty Analysis");
    println!("===========================================\n");

    // GPS position has uncertainty due to satellite positioning errors
    let current_lat = Uncertain::normal(37.7749, 0.0001); // San Francisco
    let current_lon = Uncertain::normal(-122.4194, 0.0001);

    let destination_lat = Uncertain::point(37.7849); // 1 mile north
    let destination_lon = Uncertain::point(-122.4094); // 1 mile east

    // Calculate distance with uncertainty propagation
    let lat_diff = destination_lat - current_lat.clone();
    let lon_diff = destination_lon - current_lon.clone();

    // Simplified distance calculation (in miles, approximate)
    let distance_sq = lat_diff.clone() * lat_diff + lon_diff.clone() * lon_diff;
    let distance = distance_sq.map(|x| x.sqrt() * 69.0); // Convert to miles

    println!("📍 Distance Analysis:");
    let samples: Vec<f64> = distance.take_samples(1000);
    let mean_distance = samples.iter().sum::<f64>() / samples.len() as f64;
    let std_distance = (samples.iter().map(|x| (x - mean_distance).powi(2)).sum::<f64>()
                       / samples.len() as f64).sqrt();

    println!("   Mean distance: {:.3} miles", mean_distance);
    println!("   Std deviation: {:.4} miles", std_distance);
    println!("   95% confidence: {:.3} - {:.3} miles",
             mean_distance - 1.96 * std_distance,
             mean_distance + 1.96 * std_distance);

    // Speed varies due to traffic, weather, driver behavior
    let base_speed = Uncertain::normal(35.0, 8.0); // mph, with uncertainty
    let traffic_factor = Uncertain::uniform(0.6, 1.0); // Traffic slows us down
    let actual_speed = base_speed * traffic_factor;

    // Calculate arrival time
    let travel_time_hours = distance.clone() / actual_speed;
    let travel_time_minutes = travel_time_hours.map(|h| h * 60.0);

    println!("\n⏱️  Travel Time Analysis:");
    let time_samples: Vec<f64> = travel_time_minutes.take_samples(1000);
    let mean_time = time_samples.iter().sum::<f64>() / time_samples.len() as f64;
    let std_time = (time_samples.iter().map(|x| (x - mean_time).powi(2)).sum::<f64>()
                   / time_samples.len() as f64).sqrt();

    println!("   Expected time: {:.1} minutes", mean_time);
    println!("   Std deviation: {:.1} minutes", std_time);

    // Probability analysis for arrival predictions
    let late_threshold = 10.0; // minutes
    let will_be_late = travel_time_minutes.map(move |t| t > late_threshold);

    let late_samples: Vec<bool> = will_be_late.take_samples(1000);
    let late_probability = late_samples.iter().filter(|&&x| x).count() as f64 / 1000.0;

    println!("   Probability of taking >{}min: {:.1}%", late_threshold, late_probability * 100.0);

    // Route decision with uncertainty
    println!("\n🛣️  Route Decision Analysis:");

    // Alternative route: longer but more predictable
    let alt_distance = Uncertain::point(2.2); // Slightly longer
    let alt_speed = Uncertain::normal(45.0, 3.0); // Highway, more predictable
    let alt_time = alt_distance / alt_speed * Uncertain::point(60.0);

    // Compare routes using evidence-based reasoning
    let main_faster = travel_time_minutes.less_than(&alt_time);
    let confidence_main_faster = main_faster.take_samples(1000)
        .iter().filter(|&&x| x).count() as f64 / 1000.0;

    println!("   Main route faster: {:.1}% confidence", confidence_main_faster * 100.0);

    if confidence_main_faster > 0.7 {
        println!("   ✅ Recommendation: Take main route");
    } else if confidence_main_faster < 0.3 {
        println!("   ✅ Recommendation: Take alternative route");
    } else {
        println!("   ⚠️  Recommendation: Routes are similar, consider other factors");
    }

    // Fuel consumption analysis
    println!("\n⛽ Fuel Consumption Analysis:");
    let fuel_efficiency = Uncertain::normal(28.0, 4.0); // mpg
    let fuel_needed = distance / fuel_efficiency;

    let fuel_samples: Vec<f64> = fuel_needed.take_samples(1000);
    let mean_fuel = fuel_samples.iter().sum::<f64>() / fuel_samples.len() as f64;

    println!("   Expected fuel: {:.3} gallons", mean_fuel);

    // Check if we have enough fuel
    let current_fuel = Uncertain::uniform(0.8, 1.2); // Uncertain fuel gauge reading
    let enough_fuel = current_fuel.greater_than(&fuel_needed);
    let fuel_confidence = enough_fuel.take_samples(1000)
        .iter().filter(|&&x| x).count() as f64 / 1000.0;

    println!("   Confidence we have enough fuel: {:.1}%", fuel_confidence * 100.0);

    if fuel_confidence < 0.8 {
        println!("   ⚠️  Consider refueling before the trip!");
    }
}
