import pandas as pd

df = pd.read_csv("output.csv", keep_default_na=False)

print("STATUS DISTRIBUTION:")
print(df["affordability_status"].value_counts())
print("\nMETHOD DISTRIBUTION:")
print(df["recommended_payment_method"].value_counts())
print("\nSPENDING CHANGES DISTRIBUTION:")
has_changes = df[df["spending_changes_needed"] != "none"]
print(f"Requests requiring spending changes: {len(has_changes)}")

print("\n--- SAMPLE EXPLANATIONS BY DECISION CATEGORY ---")

print("\n1. AFFORDABLE_NOW (Full Payment):")
for _, r in df[df["affordability_status"] == "affordable_now"].head(3).iterrows():
    print(f"[{r['request_id']}] {r['decision_explanation']}")

print("\n2. AFFORDABLE_LATER (Wait):")
for _, r in df[df["affordability_status"] == "affordable_later"].head(3).iterrows():
    print(f"[{r['request_id']}] Plan: {r['payment_plan']} | Expl: {r['decision_explanation']}")

print("\n3. AFFORDABLE_WITH_PLAN (Installments):")
for _, r in df[df["recommended_payment_method"] == "installments"].head(3).iterrows():
    print(f"[{r['request_id']}] Changes: {r['spending_changes_needed']} | Expl: {r['decision_explanation']}")

print("\n4. AFFORDABLE_WITH_PLAN (Partial Payment):")
for _, r in df[df["recommended_payment_method"] == "partial_payment"].head(3).iterrows():
    print(f"[{r['request_id']}] Plan: {r['payment_plan']} | Expl: {r['decision_explanation']}")

print("\n5. NOT_AFFORDABLE (Not Recommended):")
for _, r in df[df["affordability_status"] == "not_affordable"].head(3).iterrows():
    print(f"[{r['request_id']}] SafeAmt: {r['amount_safe_to_pay']} | Expl: {r['decision_explanation']}")
