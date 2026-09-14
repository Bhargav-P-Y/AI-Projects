import pandas as pd
from datetime import date

df = pd.read_csv("output.csv", keep_default_na=False)
requests_df = pd.read_csv("dataset/requests.csv", keep_default_na=False)
profiles_df = pd.read_csv("dataset/financial_profiles.csv", keep_default_na=False)

req_map = {r["request_id"]: r for r in requests_df.to_dict("records")}
prof_map = {p["user_id"]: p for p in profiles_df.to_dict("records")}

print(f"Total rows in output.csv: {len(df)}")
issues = []

for idx, row in df.iterrows():
    rid = row["request_id"]
    req = req_map.get(rid)
    if not req:
        issues.append((rid, "Unknown request_id"))
        continue
    prof = prof_map.get(req["user_id"])
    
    # 1. amount_safe_to_pay check
    try:
        amt_safe = float(row["amount_safe_to_pay"])
        req_amt = float(req["requested_amount"])
        if amt_safe < 0 or amt_safe > req_amt:
            issues.append((rid, f"amount_safe_to_pay out of bounds: {amt_safe} (req: {req_amt})"))
    except Exception as e:
        issues.append((rid, f"amount_safe_to_pay not float: {e}"))

    # 2. affordability_status check
    status = row["affordability_status"]
    if status not in ["affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"]:
        issues.append((rid, f"Invalid affordability_status: {status}"))

    # 3. recommended_payment_method check
    method = row["recommended_payment_method"]
    if method not in ["full_payment", "partial_payment", "installments", "wait", "not_recommended"]:
        issues.append((rid, f"Invalid recommended_payment_method: {method}"))

    # 4. payment_plan syntax
    plan = row["payment_plan"]
    if method == "not_recommended":
        if plan != "none":
            issues.append((rid, f"not_recommended must have payment_plan == none, got {plan}"))
    elif plan == "none":
        issues.append((rid, f"Method {method} has payment_plan == none"))
    else:
        parts = plan.split("|")
        for p in parts:
            if ":" not in p:
                issues.append((rid, f"Malformed plan entry: {p}"))
            else:
                d_str, a_str = p.split(":", 1)
                try:
                    date.fromisoformat(d_str)
                    val = float(a_str)
                    if val <= 0:
                        issues.append((rid, f"Plan amount non-positive: {val} in {p}"))
                except Exception as ex:
                    issues.append((rid, f"Invalid plan date/amt: {p} ({ex})"))

    # 5. earliest_date_for_full_payment
    earliest = row["earliest_date_for_full_payment"]
    if status == "affordable_now":
        if earliest != req["request_date"]:
            issues.append((rid, f"affordable_now earliest date must equal request_date {req['request_date']}, got {earliest}"))
    elif earliest:
        try:
            date.fromisoformat(earliest)
        except Exception:
            issues.append((rid, f"Invalid earliest date format: {earliest}"))

    # 6. spending_changes_needed
    changes = row["spending_changes_needed"]
    if changes != "none":
        ch_parts = changes.split("|")
        if len(ch_parts) > 3:
            issues.append((rid, f"More than 3 spending changes: {len(ch_parts)}"))
        for cp in ch_parts:
            if not (cp.startswith("stop:") or cp.startswith("reduce_to:")):
                issues.append((rid, f"Invalid spending change action syntax: {cp}"))

    # 7. decision_explanation quality check
    expl = str(row["decision_explanation"]).strip()
    if len(expl) < 15:
        issues.append((rid, f"Decision explanation too short: {expl}"))
    if expl.endswith(("..", "...", "and", "then", "with", "to", "or", ",")):
        issues.append((rid, f"Decision explanation appears truncated: {expl}"))

print(f"Total invariant issues found across 250 rows: {len(issues)}")
if issues:
    for iss in issues[:30]:
        print(" -", iss)
