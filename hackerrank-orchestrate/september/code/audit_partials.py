import pandas as pd

df = pd.read_csv("output.csv", keep_default_na=False)
requests_df = pd.read_csv("dataset/requests.csv", keep_default_na=False)
req_map = {r["request_id"]: r for r in requests_df.to_dict("records")}

partials = df[df["recommended_payment_method"] == "partial_payment"]
print(f"Total partial payments: {len(partials)}")
discrepancies = []

for _, r in partials.iterrows():
    rid = r["request_id"]
    req = req_map[rid]
    req_amt = float(req["requested_amount"])
    parts = r["payment_plan"].split("|")
    if len(parts) != 2:
        discrepancies.append((rid, "Partial payment plan does not have exactly 2 entries", r["payment_plan"]))
        continue
    d1, a1 = parts[0].split(":")
    d2, a2 = parts[1].split(":")
    p1 = float(a1)
    p2 = float(a2)
    total = round(p1 + p2, 2)
    if abs(total - req_amt) > 0.05:
        discrepancies.append((rid, f"Payments sum {total} != requested_amount {req_amt}"))
    if d1 != req["request_date"]:
        discrepancies.append((rid, f"First payment date {d1} != request_date {req['request_date']}"))
    if d2 != r["earliest_date_for_full_payment"]:
        discrepancies.append((rid, f"Second payment date {d2} != earliest_date_for_full_payment {r['earliest_date_for_full_payment']}"))

print(f"Partial payment discrepancies: {len(discrepancies)}")
if discrepancies:
    for d in discrepancies:
        print(" -", d)
