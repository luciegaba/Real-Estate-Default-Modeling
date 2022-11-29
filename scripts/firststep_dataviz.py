import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
def min_max_for_datetime_col(df,col):
    print(f"Date minimale pour {col}:", df["date_debloc_avec_crd"].min())
    print(f"Date maximale pour {col}:", df["date_debloc_avec_crd"].max())


def missing_rate_report(X_train):
    missing_rate=pd.DataFrame({'count': X_train.isna().sum(),
                             'rate': (X_train.isna().sum()*100/X_train.shape[0])}).sort_values(by = 'rate', ascending = False)
    display(missing_rate)
    return missing_rate

def stabilite_global_temps(data):
    sns.histplot(data["date_debloc_avec_crd"].astype(str).str[:7])
    plt.title("Stabilité de l'ensemble de l'échantillon sur la période d'étude (par mois)")
    plt.xticks(rotation=80, size = 8)
    plt.figure(figsize=(14, 14))
    plt.show()

    sns.histplot(data['date_debloc_avec_crd'].astype(str).str[:4])
    plt.title("Stabilité de l'ensemble de l'échantillon sur la période d'étude (par année)")
    plt.xticks(rotation=80, size = 8)
    plt.figure(figsize=(14, 14))