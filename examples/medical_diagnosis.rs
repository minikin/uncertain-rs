use uncertain_rs::Uncertain;

// Medical thresholds
const HYPERTENSION_THRESHOLD: f64 = 140.0;
const HIGH_CHOLESTEROL_THRESHOLD: f64 = 200.0;
const DIABETES_THRESHOLD: f64 = 125.0;
const OBESITY_BMI_THRESHOLD: f64 = 30.0;
const SAMPLE_COUNT: usize = 1000;

// Risk scoring
const HYPERTENSION_RISK_POINTS: f64 = 2.0;
const CHOLESTEROL_RISK_POINTS: f64 = 1.5;
const DIABETES_RISK_POINTS: f64 = 2.5;
const OBESITY_RISK_POINTS: f64 = 1.0;

#[allow(clippy::cast_precision_loss)]
fn calculate_probability(uncertain_bool: &Uncertain<bool>) -> f64 {
    uncertain_bool
        .take_samples(SAMPLE_COUNT)
        .iter()
        .filter(|&&x| x)
        .count() as f64
        / SAMPLE_COUNT as f64
}

/// Medical Diagnosis with Uncertainty-Aware Decision Making
///
/// This example shows how medical test results with uncertainty
/// can be combined to make evidence-based diagnostic decisions.
#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
fn main() {
    println!("🏥 Medical Diagnosis with Uncertainty Analysis");
    println!("=============================================\n");

    // Patient symptoms and test results (all have measurement uncertainty)

    // Blood pressure reading (systolic)
    let blood_pressure = Uncertain::normal(145.0, 8.0); // mmHg ± measurement error

    // Cholesterol test result
    let cholesterol = Uncertain::normal(220.0, 15.0); // mg/dL ± lab uncertainty

    // Blood sugar (glucose) level
    let glucose = Uncertain::normal(126.0, 12.0); // mg/dL ± measurement variance

    // BMI calculation with uncertain measurements
    let weight_kg = Uncertain::normal(78.5, 1.2); // kg ± scale uncertainty
    let height_m = Uncertain::normal(1.75, 0.02); // m ± measurement error
    let bmi = weight_kg / (height_m.clone() * height_m);

    println!("📊 Patient Measurements:");
    println!("   Blood Pressure: {:.1} ± {:.1} mmHg", 145.0, 8.0);
    println!("   Cholesterol: {:.1} ± {:.1} mg/dL", 220.0, 15.0);
    println!("   Glucose: {:.1} ± {:.1} mg/dL", 126.0, 12.0);

    let bmi_samples: Vec<f64> = bmi.take_samples(SAMPLE_COUNT);
    let mean_bmi = bmi_samples.iter().sum::<f64>() / bmi_samples.len() as f64;
    println!("   BMI: {mean_bmi:.1} (calculated with uncertainty)");

    // Evidence-based risk assessment
    println!("\n🔬 Risk Factor Analysis:");

    // Hypertension risk (blood pressure > 140 mmHg)
    let hypertension_evidence = blood_pressure.gt(HYPERTENSION_THRESHOLD);
    let hypertension_prob = calculate_probability(&hypertension_evidence);
    println!(
        "   Hypertension risk: {:.1}% confidence",
        hypertension_prob * 100.0
    );

    // High cholesterol risk (> 200 mg/dL)
    let high_cholesterol_evidence = cholesterol.gt(HIGH_CHOLESTEROL_THRESHOLD);
    let cholesterol_prob = calculate_probability(&high_cholesterol_evidence);
    println!(
        "   High cholesterol risk: {:.1}% confidence",
        cholesterol_prob * 100.0
    );

    // Diabetes risk (glucose > 125 mg/dL)
    let diabetes_evidence = glucose.gt(DIABETES_THRESHOLD);
    let diabetes_prob = calculate_probability(&diabetes_evidence);
    println!("   Diabetes risk: {:.1}% confidence", diabetes_prob * 100.0);

    // Obesity risk (BMI > 30)
    let obesity_evidence = bmi.gt(OBESITY_BMI_THRESHOLD);
    let obesity_prob = calculate_probability(&obesity_evidence);
    println!("   Obesity risk: {:.1}% confidence", obesity_prob * 100.0);

    // Combined cardiovascular risk assessment
    println!("\n❤️  Cardiovascular Risk Assessment:");

    // Multiple risk factors compound the risk
    let blood_pressure_clone = blood_pressure.clone();
    let cholesterol_clone = cholesterol.clone();
    let glucose_clone = glucose.clone();
    let bmi_clone = bmi.clone();

    let cv_risk_score = Uncertain::new(move || {
        let bp_risk = if blood_pressure_clone.sample() > HYPERTENSION_THRESHOLD {
            HYPERTENSION_RISK_POINTS
        } else {
            0.0
        };
        let cholesterol_risk = if cholesterol_clone.sample() > HIGH_CHOLESTEROL_THRESHOLD {
            CHOLESTEROL_RISK_POINTS
        } else {
            0.0
        };
        let diabetes_risk = if glucose_clone.sample() > DIABETES_THRESHOLD {
            DIABETES_RISK_POINTS
        } else {
            0.0
        };
        let obesity_risk = if bmi_clone.sample() > OBESITY_BMI_THRESHOLD {
            OBESITY_RISK_POINTS
        } else {
            0.0
        };

        bp_risk + cholesterol_risk + diabetes_risk + obesity_risk
    });

    let risk_samples: Vec<f64> = cv_risk_score.take_samples(SAMPLE_COUNT);
    let mean_risk = risk_samples.iter().sum::<f64>() / risk_samples.len() as f64;

    // Risk categories based on combined score
    let low_risk = cv_risk_score.lt(2.0);
    let moderate_risk = cv_risk_score.map(|score| (2.0..4.0).contains(&score));
    let high_risk = cv_risk_score.ge(4.0);

    let low_prob = calculate_probability(&low_risk);
    let moderate_prob = calculate_probability(&moderate_risk);
    let high_prob = calculate_probability(&high_risk);

    println!("   Average risk score: {mean_risk:.1}");
    println!("   Low risk probability: {:.1}%", low_prob * 100.0);
    println!(
        "   Moderate risk probability: {:.1}%",
        moderate_prob * 100.0
    );
    println!("   High risk probability: {:.1}%", high_prob * 100.0);
    println!("\n💊 Treatment Recommendations:");

    if high_prob > 0.5 {
        println!("   🚨 HIGH RISK: Immediate intervention recommended");
        println!("      - Start medication for hypertension/cholesterol");
        println!("      - Strict dietary modifications");
        println!("      - Regular monitoring required");
    } else if moderate_prob > 0.3 {
        println!("   ⚠️  MODERATE RISK: Lifestyle changes recommended");
        println!("      - Dietary modifications");
        println!("      - Increase physical activity");
        println!("      - Monitor progress in 3 months");
    } else {
        println!("   ✅ LOW RISK: Continue current lifestyle");
        println!("      - Maintain healthy habits");
        println!("      - Annual check-ups sufficient");
    }

    println!("\n🔍 Diagnostic Confidence & Follow-up:");

    // Calculate overall diagnostic confidence
    let max_uncertainty = [
        hypertension_prob,
        cholesterol_prob,
        diabetes_prob,
        obesity_prob,
    ]
    .iter()
    .map(|&p| (p - 0.5).abs())
    .fold(0.0, f64::max);

    let diagnostic_confidence = 1.0 - max_uncertainty;

    println!(
        "   Overall diagnostic confidence: {:.1}%",
        diagnostic_confidence * 100.0
    );

    if diagnostic_confidence < 0.7 {
        println!("   📋 Recommendation: Repeat tests for confirmation");
        println!("      - High measurement uncertainty detected");
        println!("      - Consider additional diagnostic tests");
    } else {
        println!("   ✅ Diagnosis confidence is adequate for treatment decisions");
    }

    // Medication dosage with uncertainty
    if hypertension_prob > 0.8 {
        println!("\n💊 Blood Pressure Medication Dosage:");

        // Base dosage depends on severity (with uncertainty)
        let severity_factor = blood_pressure.map(|bp| ((bp - 120.0) / 20.0).clamp(0.5, 2.0));
        let base_dosage = Uncertain::point(10.0); // mg baseline
        let recommended_dosage = base_dosage * severity_factor;

        let dosage_samples: Vec<f64> = recommended_dosage.take_samples(SAMPLE_COUNT);
        let mean_dosage = dosage_samples.iter().sum::<f64>() / dosage_samples.len() as f64;
        let std_dosage = (dosage_samples
            .iter()
            .map(|x| (x - mean_dosage).powi(2))
            .sum::<f64>()
            / dosage_samples.len() as f64)
            .sqrt();

        println!("   Recommended dosage: {mean_dosage:.1} ± {std_dosage:.1} mg");
        println!(
            "   Start with: {:.0} mg (conservative approach)",
            mean_dosage - std_dosage
        );
    }
}
