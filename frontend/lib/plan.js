// General, educational steps matched to the values that are above the usual ranges.
// Based on common diabetes-prevention guidance (e.g. ADA Standards of Care, Diabetes Prevention Program).
// These are not personal medical advice and never mention starting or stopping medicines.
export function buildPlan(p, tier) {
  const out = [];
  const add = (id, tag, title, why) => out.push({ id, tag, title, why });
  const confirm = p.hba1c >= 6.5 || p.glucose >= 126 || tier === "high";
  if (confirm) add("see-doctor", "First step", "Book a doctor visit for confirmatory lab tests", "Only a laboratory test and a clinician can confirm or rule out diabetes. Do not start or stop any medicine on your own.");
  else if (p.hba1c >= 5.7 || p.glucose >= 100) add("recheck", "Check", "Repeat HbA1c and fasting glucose in the next few months", "Values in the prediabetes range should be re-measured and tracked over time.");
  if (p.hba1c >= 5.7) add("carbs", "HbA1c " + p.hba1c, "Swap sugary drinks and refined carbs for whole grains, pulses and vegetables", "HbA1c reflects average sugar over about three months, and food choices move it most steadily.");
  if (p.glucose >= 100) add("walk", "Glucose " + p.glucose, "Take a 10 to 15 minute walk after your main meals", "Light activity after eating helps muscles take up sugar from the blood.");
  if (p.bmi >= 25) add("weight", "BMI " + p.bmi, "Aim for a gradual 5 to 7 percent weight loss over several months", "In prevention trials this much weight loss cut the chance of developing diabetes substantially.");
  if (p.bmi >= 25 || p.hba1c >= 5.7 || p.glucose >= 100) add("activity", "Activity", "Work up to 150 minutes a week of brisk walking or similar exercise", "Spread over most days; add two short strength sessions if you can.");
  if (p.hypertension) add("bp", "Blood pressure", "Keep checking blood pressure and follow your doctor's plan; cut back on salt", "High blood pressure and diabetes together raise heart and kidney risk.");
  if (p.heart_disease) add("heart", "Heart", "Keep your regular heart check-ups and mention your sugar results at the next one", "Heart disease and high blood sugar affect each other.");
  if (p.smoking_history === "current") add("smoke", "Smoking", "Ask a doctor or pharmacist about quit-smoking support", "Smoking raises diabetes and heart risk.");
  if (p.age >= 45) add("annual", "Age " + p.age, "Get blood sugar screening at least once a year", "Risk rises with age, so regular screening catches changes early.");
  add("sleep", "Everyday", "Aim for 7 to 8 hours of sleep and manage stress", "Poor sleep and long-term stress can push blood sugar up.");
  return out;
}
