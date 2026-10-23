import numpy as np
import pandas as pd

def safe_ratio(a, b):
    return (a / b).replace([np.inf, -np.inf], np.nan)


def merge_features(df, agg, count_cols):
    n_before = len(df)
    out = df.merge(agg, on="SK_ID_CURR", how="left")
    assert len(out) == n_before, "строки задвоились или потерялись!"
    out[count_cols] = out[count_cols].fillna(0)
    return out


def load_application(data_dir="data"):
    df = pd.read_csv(f"{data_dir}/application_train.csv")

    # 365243 — служебный код «не работает» (почти все — пенсионеры):
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    return df


def add_bureau(df, data_dir="data"):
    bureau = pd.read_csv(f"{data_dir}/bureau.csv")

    bureau["IS_ACTIVE"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(int)
    bureau["HAS_OVERDUE"] = (bureau["CREDIT_DAY_OVERDUE"] > 0).astype(int)

    agg = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT=("IS_ACTIVE", "sum"),
        BUREAU_DAYS_CREDIT_MAX=("DAYS_CREDIT", "max"),
        BUREAU_DAYS_CREDIT_MIN=("DAYS_CREDIT", "min"),
        BUREAU_CREDIT_SUM=("AMT_CREDIT_SUM", "sum"),
        BUREAU_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_OVERDUE_DAYS_MAX=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_OVERDUE_AMT_SUM=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUREAU_OVERDUE_COUNT=("HAS_OVERDUE", "sum"),
    ).reset_index()

    agg["BUREAU_DEBT_RATIO"] = safe_ratio(agg["BUREAU_DEBT_SUM"], agg["BUREAU_CREDIT_SUM"])

    count_cols = ["BUREAU_COUNT", "BUREAU_ACTIVE_COUNT", "BUREAU_OVERDUE_COUNT"]
    return merge_features(df, agg, count_cols)


def add_prev(df, data_dir="data"):
    df_prev_app = pd.read_csv(f"{data_dir}/previous_application.csv")

    df_prev_app["IS_REFUSED"] = (df_prev_app["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    df_prev_app["CREDIT_TO_APP"] = safe_ratio(df_prev_app["AMT_CREDIT"], df_prev_app["AMT_APPLICATION"])

    agg = df_prev_app.groupby("SK_ID_CURR").agg(
        PREV_COUNT=("SK_ID_PREV", "count"),
        PREV_REFUSED_COUNT=("IS_REFUSED", "sum"),
        PREV_REFUSED_SHARE=("IS_REFUSED", "mean"),
        PREV_DAYS_DECISION_MAX=("DAYS_DECISION", "max"),
        PREV_CREDIT_TO_APP_MEAN=("CREDIT_TO_APP", "mean"),
    ).reset_index()

    count_cols = ["PREV_COUNT", "PREV_REFUSED_COUNT"]
    return merge_features(df, agg, count_cols)


def add_installments(df, data_dir="data"):
    df_inst_pay = pd.read_csv(f"{data_dir}/installments_payments.csv")

    df_inst_pay["DPD"] = (df_inst_pay["DAYS_ENTRY_PAYMENT"] - df_inst_pay["DAYS_INSTALMENT"]).clip(lower=0)
    df_inst_pay["UNDERPAID"] = (df_inst_pay["AMT_INSTALMENT"] - df_inst_pay["AMT_PAYMENT"]).clip(lower=0)
    df_inst_pay["IS_LATE"] = (df_inst_pay["DPD"] > 0).astype(int)


    agg = df_inst_pay.groupby("SK_ID_CURR").agg(
        INST_COUNT=("SK_ID_PREV", "count"),
        INST_PAID_COUNT=("AMT_PAYMENT", "count"),
        INST_DPD_MEAN=("DPD", "mean"),
        INST_DPD_MAX=("DPD", "max"),
        INST_LATE_SHARE=("IS_LATE", "mean"),
        INST_UNDERPAID_SUM=("UNDERPAID", "sum"),
        INST_PAYMENT_SUM=("AMT_PAYMENT", "sum"),
        INST_INSTALMENT_SUM=("AMT_INSTALMENT", "sum"),
    ).reset_index()

    agg["INST_PAID_RATIO"] = safe_ratio(agg["INST_PAYMENT_SUM"], agg["INST_INSTALMENT_SUM"])
    agg["INST_UNPAID_COUNT"] = agg["INST_COUNT"] - agg["INST_PAID_COUNT"]

    count_cols = ["INST_COUNT", "INST_PAID_COUNT", "INST_UNPAID_COUNT"]
    return merge_features(df, agg, count_cols)


def make_xy(df):
    """Признаки X и таргет y. Текстовые колонки -> category (для LightGBM)."""
    y = df["TARGET"]
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    X[cat_cols] = X[cat_cols].astype("category")
    return X, y
