from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


RNG_SEED = 42


def safe_divide(numer, denom):
    return numer / denom if denom not in (0, None, np.nan) else 0.0


def generate_all(output_dir: str | Path = "data") -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(RNG_SEED)
    rng = np.random.default_rng(RNG_SEED)

    # Borrower setup
    occupations = [
        "Vendor",
        "Tailor",
        "Driver",
        "Farmer",
        "Shopkeeper",
        "Fisherman",
        "Carpenter",
        "Mechanic",
        "SalonOwner",
        "MarketTrader",
        "Teacher",
        "Cleaner",
    ]
    genders = ["Female", "Male", "Non-binary"]

    borrowers = []
    group_ids = [f"G{i:02d}" for i in range(1, 16)]
    location_ids = [f"L{i:02d}" for i in range(1, 9)]

    demo_ids = ["A", "B", "C", "D"]
    for borrower_id in demo_ids:
        borrowers.append(
            {
                "borrower_id": borrower_id,
                "group_id": "G01",
                "location_id": "L01",
                "occupation": "Vendor" if borrower_id in {"A", "B"} else "Shopkeeper",
                "gender": "Female" if borrower_id in {"A", "C"} else "Male",
                "join_month": "2024-01",
                "initial_loan_amount": 50000 + 15000 * demo_ids.index(borrower_id),
            }
        )

    for i in range(1, 147):
        borrower_id = f"B{i:03d}"
        group_id = group_ids[(i - 1) % len(group_ids)]
        location_id = location_ids[(i - 1) % len(location_ids)]
        occupation = occupations[(i - 1) % len(occupations)]
        gender = genders[(i - 1) % len(genders)]
        join_month = f"2024-{(i % 12) + 1:02d}"
        initial_loan_amount = int(35000 + rng.normal(18000, 9000))
        borrowers.append(
            {
                "borrower_id": borrower_id,
                "group_id": group_id,
                "location_id": location_id,
                "occupation": occupation,
                "gender": gender,
                "join_month": join_month,
                "initial_loan_amount": max(20000, initial_loan_amount),
            }
        )

    borrowers_df = pd.DataFrame(borrowers)
    borrowers_df = borrowers_df[[
        "borrower_id",
        "group_id",
        "location_id",
        "occupation",
        "gender",
        "join_month",
        "initial_loan_amount",
    ]]
    borrowers_df.to_csv(output_dir / "borrowers.csv", index=False)

    # Economic conditions by location and month
    months = pd.date_range("2024-01-01", periods=12, freq="MS").strftime("%Y-%m")
    econ_records = []
    for location_id in location_ids:
        loc_index = int(location_id[1:]) - 1
        base = 0.15 + 0.05 * (loc_index % 4)
        for month_idx, month in enumerate(months):
            market_shock = float(np.clip(0.05 + 0.12 * math.sin((month_idx + loc_index) / 2) + (0.2 if (month_idx + loc_index) % 6 == 0 else 0.0), 0, 1))
            weather_shock = float(np.clip(0.08 + 0.09 * ((month_idx + loc_index) % 4 == 0), 0, 1))
            unemployment_rate = 0.04 + 0.006 * (loc_index + 1) + 0.005 * ((month_idx + 1) % 5 == 0)
            price_inflation = 0.08 + 0.012 * (loc_index % 3) + 0.015 * ((month_idx + 2) % 4 == 0)
            economic_stress_score = float(np.clip(0.25 * market_shock + 0.2 * weather_shock + 0.35 * unemployment_rate + 0.20 * price_inflation, 0, 1))
            econ_records.append(
                {
                    "location_id": location_id,
                    "month": month,
                    "market_shock": round(market_shock, 4),
                    "weather_shock": round(weather_shock, 4),
                    "unemployment_rate": round(unemployment_rate, 4),
                    "price_inflation": round(price_inflation, 4),
                    "economic_stress_score": round(economic_stress_score, 4),
                }
            )
    economic_df = pd.DataFrame(econ_records)
    economic_df.to_csv(output_dir / "economic_conditions.csv", index=False)

    # Relationships
    relationship_types = [
        ("shared_guarantee", 0.90),
        ("common_income_source", 0.75),
        ("same_lending_group", 0.50),
        ("shared_collection_center", 0.35),
        ("same_location", 0.25),
        ("same_occupation", 0.20),
    ]

    relationship_rows = []
    borrower_ids = borrowers_df["borrower_id"].tolist()
    demo_cluster = ["A", "B", "C", "D"]

    for i, source in enumerate(borrower_ids):
        for j in range(i + 1, len(borrower_ids)):
            target = borrower_ids[j]
            if source == target:
                continue

            source_row = borrowers_df[borrowers_df["borrower_id"] == source].iloc[0]
            target_row = borrowers_df[borrowers_df["borrower_id"] == target].iloc[0]

            if source in demo_cluster and target in demo_cluster:
                if set((source, target)) in {("A", "B"), ("A", "C"), ("C", "D")}: 
                    relationship_rows.append((source, target, "shared_guarantee", 0.9, "2024-01", 1))
                elif set((source, target)) in {("A", "D"), ("B", "C")}: 
                    relationship_rows.append((source, target, "common_income_source", 0.75, "2024-01", 1))

            if source_row["group_id"] == target_row["group_id"]:
                relationship_rows.append((source, target, "same_lending_group", 0.50, "2024-01", 1))
            if source_row["location_id"] == target_row["location_id"]:
                relationship_rows.append((source, target, "same_location", 0.25, "2024-01", 1))
            if source_row["occupation"] == target_row["occupation"]:
                relationship_rows.append((source, target, "same_occupation", 0.20, "2024-01", 1))

            if abs(i - j) % 13 == 0 and source_row["location_id"] == target_row["location_id"]:
                relationship_rows.append((source, target, "shared_collection_center", 0.35, "2024-01", 1))

            if abs(i - j) % 17 == 0 and source_row["group_id"] == target_row["group_id"]:
                relationship_rows.append((source, target, "common_income_source", 0.75, "2024-01", 1))

    # Deduplicate and cap to a compact graph
    seen = set()
    unique_rows = []
    for row in relationship_rows:
        key = (row[0], row[1], row[2])
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)

    relationships_df = pd.DataFrame(unique_rows, columns=[
        "source_borrower_id",
        "target_borrower_id",
        "relationship_type",
        "relationship_strength",
        "start_month",
        "active",
    ])
    relationships_df = relationships_df[relationships_df["source_borrower_id"] != relationships_df["target_borrower_id"]]
    relationships_df.to_csv(output_dir / "relationships.csv", index=False)

    # Financial behaviour generation
    financial_records = []
    borrower_ids_list = borrowers_df["borrower_id"].tolist()
    month_values = pd.date_range("2024-01-01", periods=12, freq="MS").strftime("%Y-%m")

    for borrower in borrowers_df.itertuples(index=False):
        base_income = 3200 + (hash(borrower.borrower_id) % 2000)
        local_econ = economic_df[economic_df["location_id"] == borrower.location_id].copy()
        local_econ = local_econ.set_index("month")

        previous_income = base_income
        previous_balance = borrower.initial_loan_amount
        for month in month_values:
            local_row = local_econ.loc[month]
            trend = 1 + 0.08 * np.sin((len(financial_records) % 6) / 2.0)
            group_factor = 1.0 if borrower.group_id != "G01" else 0.94
            local_factor = 1.0 - 0.18 * local_row["economic_stress_score"]
            borrower_specific = 1.0 + 0.08 * ((hash(borrower.borrower_id) + len(month_values)) % 5) / 10
            income = max(1500, base_income * trend * group_factor * local_factor * borrower_specific)
            if borrower.borrower_id in {"A", "B"} and month in {"2024-04", "2024-06"}:
                income *= 0.72
            if borrower.borrower_id == "C" and month in {"2024-05", "2024-07"}:
                income *= 0.82
            if borrower.borrower_id == "D" and month in {"2024-03", "2024-08"}:
                income *= 0.9

            income_change_pct = ((income - previous_income) / previous_income) * 100 if previous_income else 0.0
            repayment_due = max(500, income * 0.18 + previous_balance * 0.04)
            payment_pressure = max(0.0, (0.65 - local_row["economic_stress_score"]) * 0.6 + (0.18 if income_change_pct < -10 else 0.0))
            repayment_amount = repayment_due * max(0.55, 1.0 - payment_pressure)
            days_past_due = 0
            missed_payment = 0
            if income_change_pct < -15 or local_row["economic_stress_score"] > 0.68 or repayment_amount < 0.75 * repayment_due:
                missed_payment = 1
                days_past_due = int(np.clip(14 + 22 * (1 - repayment_amount / repayment_due) + 18 * local_row["economic_stress_score"], 0, 90))
            elif income_change_pct < -8:
                days_past_due = int(np.clip(8 + 10 * abs(income_change_pct) / 30, 0, 40))

            debt_to_income = previous_balance / max(income, 1)
            financial_stress_score = float(np.clip(
                0.35 * max(0, -income_change_pct) / 30
                + 0.30 * missed_payment
                + 0.20 * min(days_past_due / 60, 1)
                + 0.15 * min(debt_to_income, 1.5) / 1.5,
                0,
                1,
            ))

            next_3m_risk = 0
            # this is a future-aware signal, but the dataset keeps it in the same row as the target label for the prototype.
            # It is used for the classifier target and not for feature generation.
            if month in {"2024-03", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08", "2024-09"} and (missed_payment or financial_stress_score > 0.68):
                next_3m_risk = 1

            if borrower.borrower_id in {"A", "B", "C", "D"} and month in {"2024-05", "2024-06"}:
                next_3m_risk = 1

            loan_balance = max(0, previous_balance + repayment_due - repayment_amount)
            financial_records.append(
                {
                    "borrower_id": borrower.borrower_id,
                    "month": month,
                    "income": round(income, 2),
                    "income_change_pct": round(income_change_pct, 2),
                    "loan_balance": round(loan_balance, 2),
                    "repayment_due": round(repayment_due, 2),
                    "repayment_amount": round(repayment_amount, 2),
                    "days_past_due": int(days_past_due),
                    "missed_payment": int(missed_payment),
                    "debt_to_income": round(debt_to_income, 4),
                    "financial_stress_score": round(financial_stress_score, 4),
                    "distress_next_3m": int(next_3m_risk),
                }
            )
            previous_income = income
            previous_balance = loan_balance

    financial_df = pd.DataFrame(financial_records)
    financial_df.to_csv(output_dir / "monthly_financials.csv", index=False)


def main() -> None:
    generate_all()


if __name__ == "__main__":
    main()

