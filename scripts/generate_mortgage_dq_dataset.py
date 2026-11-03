
import os
import uuid
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
rng = np.random.default_rng(SEED)

N_ACCOUNTS = 20_000
N_MONTHS = 36

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_MONTH = pd.Timestamp("2023-01-01")
MONTHS = [
    START_MONTH + relativedelta(months=i)
    for i in range(N_MONTHS)
]

REGIONS = [
    "Capital", "Zealand", "Southern",
    "Central", "Northern", "Western"
]

PRODUCTS = ["Fixed", "Tracker", "SVR"]
CHANNELS = ["Branch", "Online", "Broker"]
ARREARS = ["Up-to-date", "1-2 months", "3+ months"]

LTV_BANDS = [
    "0-60%", "60-75%", "75-90%", "90-100%"
]

FIXED_THRESHOLD = 0.15

# ============================================================
# 1. GENERATE SYNTHETIC ACCOUNT BASE
# ============================================================

seasonality = np.array([
    0.78, 0.82, 1.00, 1.30,
    1.20, 1.05, 0.98, 0.95,
    0.92, 0.90, 0.82, 0.72
])

trend = np.linspace(1.0, 1.12, N_MONTHS)

weights = np.array([
    seasonality[m.month - 1] * trend[i]
    for i, m in enumerate(MONTHS)
])

monthly_prob = weights / weights.sum()

monthly_counts = rng.multinomial(
    N_ACCOUNTS, monthly_prob
)

opening_dates = np.repeat(MONTHS, monthly_counts)
rng.shuffle(opening_dates)

accounts = pd.DataFrame({
    "account_id": [
        f"SYN-MTG-{i:06d}"
        for i in range(1, N_ACCOUNTS + 1)
    ],
    "open_date": opening_dates,
    "region": rng.choice(
        REGIONS, N_ACCOUNTS,
        p=[0.25, 0.20, 0.13, 0.17, 0.13, 0.12]
    ),
    "product_subtype": rng.choice(
        PRODUCTS, N_ACCOUNTS,
        p=[0.55, 0.25, 0.20]
    ),
    "balance": np.round(
        rng.lognormal(
            mean=np.log(210000),
            sigma=0.38,
            size=N_ACCOUNTS
        ), 2
    ),
    "ltv_band": rng.choice(
        LTV_BANDS, N_ACCOUNTS,
        p=[0.20, 0.35, 0.35, 0.10]
    ),
    "arrears_status": rng.choice(
        ARREARS, N_ACCOUNTS,
        p=[0.92, 0.055, 0.025]
    ),
    "channel": rng.choice(
        CHANNELS, N_ACCOUNTS,
        p=[0.30, 0.35, 0.35]
    )
})

ltv_mapping = {
    "0-60%": 0.50,
    "60-75%": 0.675,
    "75-90%": 0.825,
    "90-100%": 0.95
}

accounts["_ltv_numeric"] = (
    accounts["ltv_band"].map(ltv_mapping)
    + rng.normal(0, 0.025, N_ACCOUNTS)
).clip(0.01, 1.0)

accounts.drop(columns="_ltv_numeric").to_csv(
    f"{OUTPUT_DIR}/mortgage_accounts.csv",
    index=False
)

# ============================================================
# 2. GENERATE MONTHLY METRICS
# ============================================================

monthly_records = []

for i, month in enumerate(MONTHS):

    active = accounts[
        accounts["open_date"] <= month
    ]

    new = accounts[
        accounts["open_date"] == month
    ]

    seasonal_dip = (
        0.003 * np.cos(2 * np.pi * month.month / 12)
    )

    active_rate = np.clip(
        0.985 + seasonal_dip
        + rng.normal(0, 0.0015),
        0.970, 0.995
    )

    arrears_rate = (
        (active["arrears_status"] != "Up-to-date").mean()
        + rng.normal(0, 0.0004)
    )

    record = {
        "month": month.strftime("%Y-%m-01"),
        "new_mortgages_count": len(new),
        "balance_sum": round(active["balance"].sum(), 2),
        "avg_ltv": round(active["_ltv_numeric"].mean(), 5),
        "active_rate": round(active_rate, 5),
        "arrears_rate": round(
            np.clip(arrears_rate, 0.005, 0.20), 5
        )
    }

    product_shares = active[
        "product_subtype"
    ].value_counts(normalize=True)

    for product in PRODUCTS:
        record[f"product_{product.lower()}_prop"] = (
            product_shares.get(product, 0)
        )

    region_shares = active[
        "region"
    ].value_counts(normalize=True)

    for region in REGIONS:
        record[f"region_{region.lower()}_prop"] = (
            region_shares.get(region, 0)
        )

    monthly_records.append(record)

monthly = pd.DataFrame(monthly_records)

# ============================================================
# 3. ANOMALY MANIFEST
# ============================================================

manifest = []

def add_anomaly(
    month, metric, anomaly_type,
    description, fixed_catch, learned_catch
):
    manifest.append({
        "month": month,
        "metric_name": metric,
        "anomaly_type": anomaly_type,
        "description": description,
        "should_fixed_rule_catch": fixed_catch,
        "should_learned_method_catch": learned_catch
    })


# LEVEL SHIFT: active rate definition change.

level_month = "2024-10-01"
idx = monthly.index[monthly.month == level_month][0]

monthly.loc[idx:, "active_rate"] = (
    monthly.loc[idx:, "active_rate"] - 0.035
)

add_anomaly(
    level_month,
    "active_rate",
    "LEVEL_SHIFT",
    "Sudden downward step change of 3.5 percentage points.",
    True, True
)


# SLOW DRIFT: six gradual increases in arrears.

drift_months = pd.date_range(
    "2024-04-01", periods=6, freq="MS"
)

for month in drift_months:

    month_string = month.strftime("%Y-%m-01")
    idx = monthly.index[
        monthly.month == month_string
    ][0]

    monthly.loc[idx:, "arrears_rate"] += 0.0015

    add_anomaly(
        month_string,
        "arrears_rate",
        "SLOW_DRIFT",
        "Gradual cumulative arrears increase; "
        "individual monthly movements remain below 15%.",
        False, True
    )


# SEASONAL FALSE ALARM: legitimate spring peak.

spring_month = "2025-04-01"
idx = monthly.index[monthly.month == spring_month][0]

monthly.loc[idx, "new_mortgages_count"] = int(
    monthly.loc[idx, "new_mortgages_count"] * 1.35
)

add_anomaly(
    spring_month,
    "new_mortgages_count",
    "SEASONAL_FALSE_ALARM",
    "Legitimate spring origination peak exceeding "
    "the fixed 15% threshold.",
    True, False
)


# CATEGORY DRIFT: product composition migration.

category_months = [
    "2025-03-01",
    "2025-04-01",
    "2025-05-01",
    "2025-06-01"
]

targets = [
    [0.53, 0.26, 0.21],
    [0.50, 0.27, 0.23],
    [0.47, 0.29, 0.24],
    [0.44, 0.31, 0.25]
]

for month_string, target in zip(
    category_months, targets
):

    idx = monthly.index[
        monthly.month == month_string
    ][0]

    for product, value in zip(
        ["fixed", "tracker", "svr"], target
    ):
        monthly.loc[
            idx, f"product_{product}_prop"
        ] = value

    add_anomaly(
        month_string,
        "product_subtype_proportions",
        "CATEGORY_DRIFT",
        "Gradual migration from Fixed towards Tracker and SVR.",
        False, True
    )


# MISSING COLUMN ANALOGUE.
# Retain consistent CSV schema and represent the missing
# monthly field using a null value.

missing_month = "2025-09-01"
idx = monthly.index[monthly.month == missing_month][0]

monthly.loc[idx, "avg_ltv"] = np.nan

add_anomaly(
    missing_month,
    "avg_ltv",
    "MISSING_COLUMN",
    "avg_ltv unavailable for this reporting month.",
    True, True
)


# ============================================================
# 4. NORMALISE CATEGORY PROPORTIONS
# ============================================================

for idx in monthly.index:

    product_cols = [
        "product_fixed_prop",
        "product_tracker_prop",
        "product_svr_prop"
    ]

    region_cols = [
        f"region_{r.lower()}_prop"
        for r in REGIONS
    ]

    monthly.loc[idx, product_cols] = (
        monthly.loc[idx, product_cols]
        / monthly.loc[idx, product_cols].sum()
    )

    monthly.loc[idx, region_cols] = (
        monthly.loc[idx, region_cols]
        / monthly.loc[idx, region_cols].sum()
    )

monthly = monthly.round(5)

# ============================================================
# 5. GENERATE DQ RUN LOG
# ============================================================

RULES = [
    "has_data",
    "valid_values",
    "is_unique",
    "matches_pattern",
    "is_timely",
    "is_conforming",
    "monthly_volume_check",
    "datatype_conformity_check",
    "monthly_amount_sum_check",
    "numeric_field_profiling",
    "query_comparison"
]

METRICS = [
    "new_mortgages_count",
    "balance_sum",
    "avg_ltv",
    "active_rate",
    "arrears_rate",
    "product_subtype_proportions",
    "region_proportions"
]

dq_records = []

for _, row in monthly.iterrows():

    month = row["month"]

    for metric in METRICS:

        for rule in RULES:

            status = "PASS"
            rows_checked = N_ACCOUNTS
            rows_failed = 0
            threshold = "N/A"

            # Schema/null anomaly.
            missing = (
                metric == "avg_ltv"
                and pd.isna(row["avg_ltv"])
            )

            if missing and rule in [
                "has_data",
                "valid_values",
                "is_conforming",
                "datatype_conformity_check",
                "numeric_field_profiling",
                "query_comparison"
            ]:
                status = "FAIL"
                rows_failed = 1

            # Fixed month-on-month volume threshold.
            if (
                rule == "monthly_volume_check"
                and metric == "new_mortgages_count"
            ):
                threshold = "+/-15%"

                idx = monthly.index[
                    monthly.month == month
                ][0]

                if idx > 0:
                    previous = monthly.loc[
                        idx - 1, metric
                    ]
                    current = row[metric]

                    if previous > 0:
                        change = abs(current / previous - 1)

                        if change > FIXED_THRESHOLD:
                            status = "FAIL"
                            rows_failed = 1

            # Fixed monthly amount threshold.
            if (
                rule == "monthly_amount_sum_check"
                and metric == "balance_sum"
            ):
                threshold = "+/-15%"

                idx = monthly.index[
                    monthly.month == month
                ][0]

                if idx > 0:
                    previous = monthly.loc[
                        idx - 1, metric
                    ]
                    current = row[metric]

                    if previous > 0:
                        change = abs(current / previous - 1)

                        if change > FIXED_THRESHOLD:
                            status = "FAIL"
                            rows_failed = 1

            if rule == "is_unique":
                threshold = "Unique synthetic identifiers"

            if rule == "matches_pattern":
                threshold = "YYYY-MM-01"

            if rule == "is_timely":
                threshold = "Within 5 days"

            if rule == "numeric_field_profiling":
                threshold = "Numeric range and null profiling"

            if rule == "query_comparison":
                threshold = "Cross-query consistency"

            run_id = str(uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"{month}_{metric}_{rule}"
            ))

            run_timestamp = (
                pd.Timestamp(month)
                + pd.Timedelta(days=2, hours=6)
            )

            dq_records.append({
                "run_id": run_id,
                "month": month,
                "metric_name": metric,
                "rule_type": rule,
                "status": status,
                "rows_checked": rows_checked,
                "rows_failed": rows_failed,
                "threshold_used": threshold,
                "run_timestamp": run_timestamp
            })

dq_log = pd.DataFrame(dq_records)

# ============================================================
# 6. SAVE FOUR CSV FILES
# ============================================================

accounts.to_csv(
    OUTPUT_DIR / "mortgage_accounts.csv",
    index=False
)

monthly.to_csv(
    OUTPUT_DIR / "monthly_metrics.csv",
    index=False
)

dq_log.to_csv(
    OUTPUT_DIR / "dq_run_log.csv",
    index=False
)

manifest_df = pd.DataFrame(manifest)

manifest_df.to_csv(
    OUTPUT_DIR / "anomalies_manifest.csv",
    index=False
)

# ============================================================
# 7. VALIDATION AND SUMMARY
# ============================================================

assert len(accounts) == 20_000
assert len(monthly) == 36
assert len(dq_log) == 36 * 7 * 11
assert manifest_df["anomaly_type"].nunique() == 5

print("\n========================================")
print("SYNTHETIC MORTGAGE DATASET GENERATED")
print("========================================")

print(f"Random seed: {SEED}")
print(f"Account records: {len(accounts):,}")
print(f"Monthly records: {len(monthly):,}")
print(f"DQ executions: {len(dq_log):,}")
print(f"Manifest entries: {len(manifest_df):,}")

print("\nInjected anomalies:")

for _, anomaly in manifest_df.iterrows():
    print(
        f"- {anomaly['anomaly_type']}: "
        f"{anomaly['month']} "
        f"({anomaly['metric_name']})"
    )

print("\nOutput directory:", OUTPUT_DIR)

print("\nGenerated files:")
for filename in [
    "mortgage_accounts.csv",
    "monthly_metrics.csv",
    "dq_run_log.csv",
    "anomalies_manifest.csv"
]:
    print(" -", filename)

print("\nGeneration completed successfully.")
