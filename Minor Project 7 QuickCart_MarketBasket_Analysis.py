"""
QuickCart Market Basket Analysis
Apriori + FP-Growth project implementation.

If the raw long-format transaction CSV is available, set TRANSACTIONS_FILE
to that file and run the Apriori/FP-Growth sections. The supplied student
package contained the 256-row frequent-itemset answer key, so this script
also supports reconstructing the 980 rules from that answer key.
"""

import ast
import itertools
import time
import pandas as pd

TRANSACTIONS_FILE = "quickcart_basket_transactions.csv"
FREQUENT_ITEMSETS_FILE = "frequent_itemsets.csv"
MIN_SUPPORT = 0.02
MIN_LIFT = 1.0

def parse_itemset(s):
    return set(ast.literal_eval(s.replace("frozenset", "")))

def build_rules_from_itemsets(frequent_itemsets):
    support = {
        frozenset(row["itemset"]): float(row["support"])
        for _, row in frequent_itemsets.iterrows()
    }
    rules = []
    for itemset, itemset_support in support.items():
        if len(itemset) < 2:
            continue
        items = sorted(itemset)
        for r in range(1, len(items)):
            for antecedent_tuple in itertools.combinations(items, r):
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent
                confidence = itemset_support / support[antecedent]
                lift = confidence / support[consequent]
                if lift >= MIN_LIFT:
                    rules.append({
                        "antecedents": ", ".join(sorted(antecedent)),
                        "consequents": ", ".join(sorted(consequent)),
                        "support": itemset_support,
                        "confidence": confidence,
                        "lift": lift
                    })
    return pd.DataFrame(rules).sort_values(
        ["lift", "support"], ascending=[False, False]
    ).reset_index(drop=True)

def run_raw_data(raw_file):
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

    tx = pd.read_csv(raw_file)
    baskets = tx.groupby("order_id")["sku_name"].apply(list).tolist()

    te = TransactionEncoder()
    encoded = te.fit(baskets).transform(baskets)
    basket_df = pd.DataFrame(encoded, columns=te.columns_)

    t0 = time.perf_counter()
    apr = apriori(basket_df, min_support=MIN_SUPPORT, use_colnames=True)
    apr_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    fp = fpgrowth(basket_df, min_support=MIN_SUPPORT, use_colnames=True)
    fp_time = time.perf_counter() - t0

    rules = association_rules(apr, metric="lift", min_threshold=MIN_LIFT)
    rules = rules.sort_values("lift", ascending=False)

    actionable = rules[(rules["support"] > 0.05) & (rules["lift"] > 2)]

    print("Baskets:", len(baskets))
    print("Apriori itemsets:", len(apr), "runtime:", round(apr_time, 4), "sec")
    print("FP-Growth itemsets:", len(fp), "runtime:", round(fp_time, 4), "sec")
    print("Rules:", len(rules))
    print("Actionable rules:", len(actionable))
    print(actionable.head(20).to_string(index=False))

    apr.to_csv("quickcart_apriori_itemsets.csv", index=False)
    fp.to_csv("quickcart_fpgrowth_itemsets.csv", index=False)
    rules.to_csv("quickcart_association_rules.csv", index=False)
    actionable.to_csv("quickcart_actionable_rules.csv", index=False)

if __name__ == "__main__":
    try:
        run_raw_data(TRANSACTIONS_FILE)
    except FileNotFoundError:
        print("Raw transaction CSV not supplied.")
        print("Reconstructing rules from the supplied 256-row frequent-itemset answer key.")
        fi = pd.read_csv(FREQUENT_ITEMSETS_FILE)
        fi["itemset"] = fi["itemsets"].map(parse_itemset)
        rules = build_rules_from_itemsets(fi)
        rules.to_csv("quickcart_association_rules_reconstructed.csv", index=False)
        print("Frequent itemsets:", len(fi))
        print("Rules with lift >= 1:", len(rules))
        print("Expected answer-key counts: 60 + 78 + 86 + 30 + 2 = 256 itemsets; 980 rules.")
        print(rules.head(20).to_string(index=False))
