import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def discretize_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    bins = {
        'Age': [18, 30, 40, 50, 60],
        'MonthlyIncome': [0, 3000, 6000, 10000, 20000],
        'TotalWorkingYears': [0, 5, 10, 20, 40],
        'YearsAtCompany': [0, 2, 5, 10, 40]
    }

    for col, edges in bins.items():
        df[f"{col}_bin"] = pd.cut(df[col], bins=edges)

    df['AttritionLabel'] = df['Attrition'].map({0: 'Stay', 1: 'Leave'})
    return df


def build_transactions(df: pd.DataFrame) -> pd.DataFrame:
    selected_cols = [
        'Age_bin',
        'MonthlyIncome_bin',
        'TotalWorkingYears_bin',
        'YearsAtCompany_bin',
        'OverTime_Yes',
        'JobLevel',
        'AttritionLabel'
    ]

    df_trans = df[selected_cols].astype(str)
    return pd.get_dummies(df_trans)


def mine_association_rules(transactions: pd.DataFrame,
                           min_support=0.01) -> pd.DataFrame:
    freq_items = apriori(
        transactions,
        min_support=min_support,
        use_colnames=True
    )

    rules = association_rules(
        freq_items,
        metric="confidence",
        min_threshold=0.0
    )

    return rules


def split_rules_by_attrition(rules: pd.DataFrame):
    leave_rules = rules[
        rules["antecedents"].astype(str).str.contains("AttritionLabel_Leave") |
        rules["consequents"].astype(str).str.contains("AttritionLabel_Leave")
    ]

    stay_rules = rules[
        rules["antecedents"].astype(str).str.contains("AttritionLabel_Stay") |
        rules["consequents"].astype(str).str.contains("AttritionLabel_Stay")
    ]

    return leave_rules, stay_rules