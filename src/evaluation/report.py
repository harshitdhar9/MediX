import pandas as pd

EXPERT_CSV = "src/test_results_t5_small.csv"
GENERATED_CSV = "src/test_results_t5_small_flan_fk.csv"

EXPERT_FK_COL = "fk_grade"
EXPERT_ARI_COL = "ari"

GEN_FK_COL = "fk_flan"
GEN_ARI_COL = "ari_flan"

expert_df = pd.read_csv(EXPERT_CSV)
gen_df = pd.read_csv(GENERATED_CSV)

expert_fk_mean = expert_df[EXPERT_FK_COL].mean()
expert_ari_mean = expert_df[EXPERT_ARI_COL].mean()

gen_fk_mean = gen_df[GEN_FK_COL].mean()
gen_ari_mean = gen_df[GEN_ARI_COL].mean()

fk_drop = expert_fk_mean - gen_fk_mean
ari_drop = expert_ari_mean - gen_ari_mean

print(" READABILITY STATISTICS ")
print(f"Expert FKGL Mean     : {expert_fk_mean:.2f}")
print(f"Generated FKGL Mean  : {gen_fk_mean:.2f}")
print(f"FKGL Reduction       : {fk_drop:.2f}")
print()
print(f"Expert ARI Mean      : {expert_ari_mean:.2f}")
print(f"Generated ARI Mean   : {gen_ari_mean:.2f}")
print(f"ARI Reduction        : {ari_drop:.2f}")
