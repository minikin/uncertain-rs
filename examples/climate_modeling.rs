use uncertain_rs::Uncertain;

/// Climate Change Impact Assessment with Uncertainty
///
/// This example demonstrates how climate models with inherent uncertainty
/// can be used to assess environmental risks and policy decisions.
#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
fn main() {
    println!("🌍 Climate Change Impact Assessment");
    println!("==================================\n");

    // Climate model parameters (all uncertain due to model limitations)
    let baseline_temp = Uncertain::normal(1.1, 0.2).unwrap(); // °C warming since pre-industrial
    let co2_sensitivity = Uncertain::normal(3.0, 1.0).unwrap(); // °C per CO2 doubling
    let current_co2 = Uncertain::normal(420.0, 5.0).unwrap(); // ppm ± measurement uncertainty

    // Future emissions scenarios (highly uncertain)
    let emission_reduction = Uncertain::uniform(0.2, 0.8).unwrap(); // 20-80% reduction by 2050
    let _economic_growth = Uncertain::normal(0.025, 0.015).unwrap(); // 2.5% ± 1.5% annual GDP growth

    println!("🌡️  Current Climate State:");
    println!("   Baseline warming: {:.1}°C ± {:.1}°C", 1.1, 0.2);
    println!("   CO2 concentration: {:.0} ± {:.0} ppm", 420.0, 5.0);
    println!(
        "   Climate sensitivity: {:.1}°C ± {:.1}°C per CO2 doubling",
        3.0, 1.0
    );

    // Project future CO2 concentrations
    let years_ahead = 25.0; // 2050 projection
    let business_as_usual_growth = Uncertain::normal(0.015, 0.005).unwrap(); // 1.5% annual CO2 growth
    let reduction_factor = Uncertain::point(1.0) - emission_reduction.clone();

    let future_co2_growth = business_as_usual_growth * reduction_factor;
    let future_co2 =
        current_co2 * (Uncertain::point(1.0) + future_co2_growth).map(move |x| x.powf(years_ahead));

    println!("\n📈 Future CO2 Projections (2050):");
    let co2_samples: Vec<f64> = future_co2.take_samples(1000);
    let mean_co2 = co2_samples.iter().sum::<f64>() / co2_samples.len() as f64;
    let std_co2 = (co2_samples
        .iter()
        .map(|x| (x - mean_co2).powi(2))
        .sum::<f64>()
        / co2_samples.len() as f64)
        .sqrt();

    println!("   Expected CO2: {mean_co2:.0} ± {std_co2:.0} ppm");
    println!(
        "   Range (95% confidence): {:.0} - {:.0} ppm",
        mean_co2 - 1.96 * std_co2,
        mean_co2 + 1.96 * std_co2
    );

    // Calculate temperature increase
    let co2_doubling_ratio = future_co2.map(|co2| (co2 / 280.0).log2()); // Pre-industrial = 280 ppm
    let temperature_increase = baseline_temp + co2_sensitivity * co2_doubling_ratio;

    println!("\n🌡️  Temperature Projections:");
    let temp_samples: Vec<f64> = temperature_increase.take_samples(1000);
    let mean_temp = temp_samples.iter().sum::<f64>() / temp_samples.len() as f64;
    let std_temp = (temp_samples
        .iter()
        .map(|x| (x - mean_temp).powi(2))
        .sum::<f64>()
        / temp_samples.len() as f64)
        .sqrt();

    println!("   Expected warming: {mean_temp:.1}°C ± {std_temp:.1}°C");

    // Probability of exceeding critical thresholds
    let exceeds_1_5c = temperature_increase.gt(1.5);
    let exceeds_2c = temperature_increase.gt(2.0);
    let exceeds_3c = temperature_increase.gt(3.0);

    let prob_1_5c = exceeds_1_5c
        .take_samples(1000)
        .iter()
        .filter(|&&x| x)
        .count() as f64
        / 1000.0;
    let prob_2c = exceeds_2c.take_samples(1000).iter().filter(|&&x| x).count() as f64 / 1000.0;
    let prob_3c = exceeds_3c.take_samples(1000).iter().filter(|&&x| x).count() as f64 / 1000.0;

    println!("   Probability > 1.5°C: {:.1}%", prob_1_5c * 100.0);
    println!("   Probability > 2.0°C: {:.1}%", prob_2c * 100.0);
    println!("   Probability > 3.0°C: {:.1}%", prob_3c * 100.0);

    println!("\n🌊 Sea Level Rise Assessment:");

    let thermal_expansion = temperature_increase.map(|t| t * 0.2); // 20cm per °C (simplified)
    let ice_sheet_melting = temperature_increase.map(|t| {
        if t > 2.0 { (t - 2.0) * 0.5 } else { 0.0 } // Accelerated melting above 2°C
    });
    let glacier_melting = temperature_increase.map(|t| t * 0.15); // 15cm per °C

    let total_sea_level_rise = thermal_expansion + ice_sheet_melting + glacier_melting;

    let slr_samples: Vec<f64> = total_sea_level_rise.take_samples(1000);
    let mean_slr = slr_samples.iter().sum::<f64>() / slr_samples.len() as f64;
    let std_slr = (slr_samples
        .iter()
        .map(|x| (x - mean_slr).powi(2))
        .sum::<f64>()
        / slr_samples.len() as f64)
        .sqrt();

    println!("   Expected sea level rise: {mean_slr:.1} ± {std_slr:.1} meters");

    // Coastal flooding risk
    let flood_risk_1m = total_sea_level_rise.gt(1.0);
    let flood_risk_0_5m = total_sea_level_rise.gt(0.5);

    let prob_flood_1m = flood_risk_1m
        .take_samples(1000)
        .iter()
        .filter(|&&x| x)
        .count() as f64
        / 1000.0;
    let prob_flood_0_5m = flood_risk_0_5m
        .take_samples(1000)
        .iter()
        .filter(|&&x| x)
        .count() as f64
        / 1000.0;

    println!("   Risk of >0.5m rise: {:.1}%", prob_flood_0_5m * 100.0);
    println!("   Risk of >1.0m rise: {:.1}%", prob_flood_1m * 100.0);

    println!("\n💰 Economic Impact Assessment:");

    // GDP impact from climate change (uncertain relationship)
    let gdp_impact_per_degree = Uncertain::normal(-0.08, 0.04).unwrap(); // -8% ± 4% per degree
    let total_gdp_impact = temperature_increase.clone() * gdp_impact_per_degree;

    // Adaptation costs
    let adaptation_cost_per_slr = Uncertain::normal(2000.0, 500.0).unwrap(); // $2000B ± $500B per meter
    let adaptation_costs = total_sea_level_rise * adaptation_cost_per_slr;

    let gdp_impact_samples: Vec<f64> = total_gdp_impact.take_samples(1000);
    let adaptation_samples: Vec<f64> = adaptation_costs.take_samples(1000);

    let mean_gdp_impact = gdp_impact_samples.iter().sum::<f64>() / gdp_impact_samples.len() as f64;
    let mean_adaptation = adaptation_samples.iter().sum::<f64>() / adaptation_samples.len() as f64;

    println!("   Expected GDP impact: {:.1}%", mean_gdp_impact * 100.0);
    println!("   Adaptation costs: ${mean_adaptation:.0}B");

    println!("\n🏛️  Policy Recommendations:");

    // Carbon pricing needed
    let carbon_price_per_ton = Uncertain::normal(100.0, 50.0).unwrap(); // $100 ± $50 per ton CO2
    let global_emissions = Uncertain::normal(40.0, 5.0).unwrap(); // 40 ± 5 GtCO2/year
    let required_reduction = emission_reduction.clone() * global_emissions;
    let carbon_tax_revenue = required_reduction * carbon_price_per_ton;

    let revenue_samples: Vec<f64> = carbon_tax_revenue.take_samples(1000);
    let mean_revenue = revenue_samples.iter().sum::<f64>() / revenue_samples.len() as f64;

    println!(
        "   Recommended carbon price: ${:.0} ± ${:.0} per ton",
        100.0, 50.0
    );
    println!("   Potential revenue: ${mean_revenue:.0}B annually");

    // Renewable energy investment
    let renewable_capacity_needed = emission_reduction.map(|r| r * 10000.0); // GW capacity
    let cost_per_gw = Uncertain::normal(1.5, 0.3).unwrap(); // $1.5B ± $0.3B per GW
    let renewable_investment = renewable_capacity_needed * cost_per_gw;

    let investment_samples: Vec<f64> = renewable_investment.take_samples(1000);
    let mean_investment = investment_samples.iter().sum::<f64>() / investment_samples.len() as f64;

    println!(
        "   Required renewable investment: ${:.0}T",
        mean_investment / 1000.0
    );

    println!("\n⚠️  Risk Assessment Summary:");

    if prob_2c > 0.5 {
        println!("   🚨 HIGH RISK: >50% chance of exceeding 2°C target");
        println!("      - Immediate aggressive action required");
        println!("      - International cooperation essential");
    } else if prob_1_5c > 0.8 {
        println!("   🟡 MODERATE RISK: Likely to exceed 1.5°C target");
        println!("      - Accelerated mitigation needed");
        println!("      - Adaptation planning critical");
    } else {
        println!("   🟢 MANAGEABLE RISK: Targets achievable with current policies");
        println!("      - Continue current trajectory");
        println!("      - Monitor and adjust as needed");
    }

    println!("\n🎯 Uncertainty Analysis:");

    let temp_uncertainty = std_temp / mean_temp;
    let slr_uncertainty = std_slr / mean_slr;
    let economic_uncertainty = adaptation_samples
        .iter()
        .map(|x| (x - mean_adaptation).abs())
        .sum::<f64>()
        / adaptation_samples.len() as f64
        / mean_adaptation;

    println!(
        "   Temperature uncertainty: ±{:.1}%",
        temp_uncertainty * 100.0
    );
    println!("   Sea level uncertainty: ±{:.1}%", slr_uncertainty * 100.0);
    println!(
        "   Economic uncertainty: ±{:.1}%",
        economic_uncertainty * 100.0
    );

    if temp_uncertainty > 0.3 {
        println!("   📊 High uncertainty - invest in better climate models");
    }

    if economic_uncertainty > 0.5 {
        println!("   💼 Economic projections highly uncertain - scenario planning needed");
    }

    println!("\n⚡ Tipping Point Assessment:");

    let amazon_tipping = temperature_increase.gt(3.5);
    let arctic_ice_tipping = temperature_increase.gt(2.0);
    let permafrost_tipping = temperature_increase.gt(1.8);

    let prob_amazon = amazon_tipping
        .take_samples(1000)
        .iter()
        .filter(|&&x| x)
        .count() as f64
        / 1000.0;
    let prob_arctic = arctic_ice_tipping
        .take_samples(1000)
        .iter()
        .filter(|&&x| x)
        .count() as f64
        / 1000.0;
    let prob_permafrost = permafrost_tipping
        .take_samples(1000)
        .iter()
        .filter(|&&x| x)
        .count() as f64
        / 1000.0;

    println!(
        "   Amazon rainforest collapse risk: {:.1}%",
        prob_amazon * 100.0
    );
    println!("   Arctic ice loss risk: {:.1}%", prob_arctic * 100.0);
    println!(
        "   Permafrost melting risk: {:.1}%",
        prob_permafrost * 100.0
    );

    if prob_arctic > 0.3 || prob_permafrost > 0.5 {
        println!("   🚨 CRITICAL: High risk of irreversible tipping points!");
    }
}
