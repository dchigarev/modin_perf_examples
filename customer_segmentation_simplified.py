"""
This code is adopted from the below notebook
https://www.kaggle.com/code/fabiendaniel/customer-segmentation/notebook

This workload writes 'data_aug.csv' to the disk as a result.

The script reads two environment variables:
1. 'PANDAS_MODE': which library to use for this workload ("Pandas", "Modin", "Mixed")
2. 'CS_DATA_PATH': path to a directory with 'data.csv' https://www.kaggle.com/code/fabiendaniel/customer-segmentation/input

Example:
'''
PANDAS_MODE=Mixed CS_DATA_PATH=~/downloads/ python customer_segmentation_simplified.py
'''
"""

import pandas
import datetime
import nltk
import os
from pathlib import Path
from timeit import default_timer as timer

# Ensure that the required NLTK libraries are downloaded
# (they cannot be installed via pip)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from sklearnex import patch_sklearn

patch_sklearn()
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA

from contextlib import contextmanager

measurements = {}

@contextmanager
def measure(name):
    time_start = timer()
    try:
        yield
    finally:
        print(f"{name}: {round(timer() - time_start, 2)}s.")
        measurements[name] = round(timer() - time_start, 2)


mode = os.environ.get("PANDAS_MODE", "Mixed").capitalize()
path = os.environ.get("CS_DATA_PATH", "")

if mode in ("Pandas", "Mixed"):
    import pandas as pd
elif mode == "Modin":
    import modin.pandas as pd
else:
    raise ValueError(f"Unknown mode: {mode=}. The only supported modes are: ('Pandas', 'Modin', 'Mixed')")

print(f"\n\n========= Running in a {mode} mode =========\n\n")

rawtraindata = Path(path) / Path("data.csv")

# read the datafile
with measure("reading & basic preprocessing"):
    try:
        df_initial = pd.read_csv(
            rawtraindata,
            encoding="ISO-8859-1",
            dtype={"CustomerID": str, "InvoiceID": str},
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{e}. Make sure you've specified 'CS_DATA_PATH' variable correctly.")

    df_initial["InvoiceDate"] = pd.to_datetime(df_initial["InvoiceDate"])
    # gives some infos on columns types and number of null values
    tab_info = pd.DataFrame(df_initial.dtypes).T.rename(index={0: "column type"})
    tab_info = pd.concat(
        [
            tab_info,
            pd.DataFrame(df_initial.isnull().sum()).T.rename(
                index={0: "null values (nb)"}
            ),
        ]
    )
    tab_info = pd.concat(
        [
            tab_info,
            pd.DataFrame(
                df_initial.isnull().sum() / df_initial.shape[0] * 100
            ).T.rename(index={0: "null values (%)"}),
        ]
    )
    df_initial.dropna(axis=0, subset=["CustomerID"], inplace=True)

    # gives some infos on columns types and number of null values
    tab_info = pd.DataFrame(df_initial.dtypes).T.rename(index={0: "column type"})
    tab_info = pd.concat(
        [
            tab_info,
            pd.DataFrame(df_initial.isnull().sum()).T.rename(
                index={0: "null values (nb)"}
            ),
        ]
    )
    tab_info = pd.concat(
        [
            tab_info,
            pd.DataFrame(
                df_initial.isnull().sum() / df_initial.shape[0] * 100
            ).T.rename(index={0: "null values (%)"}),
        ]
    )
    df_initial.drop_duplicates(inplace=True)

    temp = (
        df_initial[["CustomerID", "InvoiceNo", "Country"]]
        .groupby(["CustomerID", "InvoiceNo", "Country"])
        .count()
    )
    temp = temp.reset_index(drop=False)
    countries = temp["Country"].value_counts()

    pd.DataFrame(
        [
            {
                "products": len(df_initial["StockCode"].value_counts()),
                "transactions": len(df_initial["InvoiceNo"].value_counts()),
                "customers": len(df_initial["CustomerID"].value_counts()),
            }
        ],
        columns=["products", "transactions", "customers"],
        index=["quantity"],
    )
    temp = df_initial.groupby(by=["CustomerID", "InvoiceNo"], as_index=False)[
        "InvoiceDate"
    ].count()
    nb_products_per_basket = temp.rename(
        columns={"InvoiceDate": "Number of products"}
    )
    nb_products_per_basket[:10].sort_values("CustomerID")

    nb_products_per_basket["order_canceled"] = nb_products_per_basket[
        "InvoiceNo"
    ].apply(lambda x: int("C" in x))

    n1 = nb_products_per_basket["order_canceled"].sum()
    n2 = nb_products_per_basket.shape[0]

    df_check = df_initial[df_initial["Quantity"] < 0][
        ["CustomerID", "Quantity", "StockCode", "Description", "UnitPrice"]
    ]
    # this for-loop seems to have no sense
    for _, col in df_check.iterrows():
        if (
            df_initial[
                (df_initial["CustomerID"] == col.iloc[0])
                & (df_initial["Quantity"] == -col.iloc[1])
                & (df_initial["Description"] == col.iloc[2])
            ].shape[0]
            == 0
        ):
            break

    df_check = df_initial[
        (df_initial["Quantity"] < 0) & (df_initial["Description"] != "Discount")
    ][["CustomerID", "Quantity", "StockCode", "Description", "UnitPrice"]]

    # this for-loop seems to have no sense
    for _, col in df_check.iterrows():
        if (
            df_initial[
                (df_initial["CustomerID"] == col.iloc[0])
                & (df_initial["Quantity"] == -col.iloc[1])
                & (df_initial["Description"] == col.iloc[2])
            ].shape[0]
            == 0
        ):
            break

def groupby_filtering(df):
    entry_to_remove = []
    doubtfull_entry = []
    df_cleaned = {}
    for index, col in df.iterrows():
        if (col["Quantity"] > 0) or col["Description"] == "Discount":
            continue
        df_test = df[
            (df["InvoiceDate"] < col["InvoiceDate"]) & (df["Quantity"] > 0)
        ].copy()

        # Cancelation WITHOUT counterpart
        if df_test.shape[0] == 0:
            doubtfull_entry.append(index)
        # Cancelation WITH a counterpart
        elif df_test.shape[0] == 1:
            index_order = df_test.index[0]
            df_cleaned[index_order] = -col["Quantity"]
            entry_to_remove.append(index)
        # Various counterparts exist in orders: we delete the last one
        elif df_test.shape[0] > 1:
            df_test.sort_index(axis=0, ascending=False, inplace=True)
            for ind, val in df_test.iterrows():
                if val["Quantity"] < -col["Quantity"]:
                    continue
                df_cleaned[ind] = -col["Quantity"]
                entry_to_remove.append(index)
                break
    res = df.copy()
    res["QuantityCanceled"] = 0
    res.loc[list(df_cleaned.keys()), "QuantityCanceled"] = list(df_cleaned.values())
    res = res.drop(entry_to_remove + doubtfull_entry, axis=0)
    return res

with measure("groupby.apply(complex_aggregation)"):
    if mode == "Mixed":
        from modin.pandas.io import from_pandas

        grp = from_pandas(df_initial).groupby(["CustomerID", "StockCode"], as_index=False)
    else:
        grp = df_initial.groupby(["CustomerID", "StockCode"], as_index=False)
    df_cleaned = grp.apply(groupby_filtering)

    if mode == "Mixed":
        df_cleaned = df_cleaned.to_pandas()

with measure("prepare for fit/predict"):
    remaining_entries = df_cleaned[
        (df_cleaned["Quantity"] < 0) & (df_cleaned["StockCode"] != "D")
    ]
    list_special_codes = df_cleaned[
        df_cleaned["StockCode"].str.contains("^[a-zA-Z]+", regex=True)
    ]["StockCode"].unique()
    df_cleaned["TotalPrice"] = df_cleaned["UnitPrice"] * (
        df_cleaned["Quantity"] - df_cleaned["QuantityCanceled"]
    )
    temp = df_cleaned.groupby(by=["CustomerID", "InvoiceNo"], as_index=False)[
        "TotalPrice"
    ].sum()
    basket_price = temp.rename(columns={"TotalPrice": "Basket Price"})
    # date of order
    df_cleaned["InvoiceDate_int"] = df_cleaned["InvoiceDate"].astype("int64")
    temp = df_cleaned.groupby(by=["CustomerID", "InvoiceNo"], as_index=False)[
        "InvoiceDate_int"
    ].mean()
    df_cleaned.drop("InvoiceDate_int", axis=1, inplace=True)
    basket_price.loc[:, "InvoiceDate"] = pd.to_datetime(temp["InvoiceDate_int"])
    # selection of significant entries:
    basket_price = basket_price[basket_price["Basket Price"] > 0]

    # Purchase countdown
    price_range = [0, 50, 100, 200, 500, 1000, 5000, 50000]
    count_price = []
    for i, price in enumerate(price_range):
        if i == 0:
            continue
        val = basket_price[
            (basket_price["Basket Price"] < price)
            & (basket_price["Basket Price"] > price_range[i - 1])
        ]["Basket Price"].count()
        count_price.append(val)

    def keywords_inventory(dataframe, column="Description"):
        """Stemming"""
        stemmer = nltk.stem.SnowballStemmer("english")
        keywords_roots = dict()  # collect the words / root
        keywords_select = dict()  # association: root <-> keyword
        category_keys = []
        count_keywords = dict()
        for s in dataframe[column]:
            if pd.isnull(s):
                continue
            lines = s.lower()
            tokenized = nltk.word_tokenize(lines)
            nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if pos[:2] == "NN"]

            for t in nouns:
                t = t.lower()
                racine = stemmer.stem(t)
                if racine in keywords_roots:
                    keywords_roots[racine].add(t)
                    count_keywords[racine] += 1
                else:
                    keywords_roots[racine] = {t}
                    count_keywords[racine] = 1

        for s in keywords_roots.keys():
            if len(keywords_roots[s]) > 1:
                min_length = 1000
                for k in keywords_roots[s]:
                    if len(k) < min_length:
                        clef = k
                        min_length = len(k)
                category_keys.append(clef)
                keywords_select[s] = clef
            else:
                category_keys.append(list(keywords_roots[s])[0])
                keywords_select[s] = list(keywords_roots[s])[0]

        return category_keys, keywords_roots, keywords_select, count_keywords

    df_produits = pd.DataFrame(df_initial["Description"].unique()).rename(
        columns={0: "Description"}
    )

    _, _, keywords_select, count_keywords = keywords_inventory(df_produits)

    list_products = []
    for k, v in count_keywords.items():
        list_products.append([keywords_select[k], v])
    list_products.sort(key=lambda x: x[1], reverse=True)

    liste = sorted(list_products, key=lambda x: x[1], reverse=True)

    list_products = []
    for k, v in count_keywords.items():
        word = keywords_select[k]
        if word in ["pink", "blue", "tag", "green", "orange"]:
            continue
        if len(word) < 3 or v < 13:
            continue
        if ("+" in word) or ("/" in word):
            continue
        list_products.append([word, v])

    list_products.sort(key=lambda x: x[1], reverse=True)

    liste_produits = df_cleaned["Description"].unique()
    X_data = {}
    for key, occurence in list_products:
        value = list(map(lambda x: int(key.upper() in x), liste_produits))
        X_data[key] = value

    X = pd.DataFrame(X_data)

    threshold = [0, 1, 2, 3, 5, 10]
    label_col = []
    for i in range(len(threshold)):
        if i == len(threshold) - 1:
            col = ".>{}".format(threshold[i])
        else:
            col = "{}<.<{}".format(threshold[i], threshold[i + 1])
        label_col.append(col)
        X.loc[:, col] = 0

    prix_values = (
        df_cleaned[["Description", "UnitPrice"]][
            df_cleaned["Description"].isin(liste_produits)
        ]
        .groupby("Description")["UnitPrice"]
        .mean()
    )
    # random-access works poorly on modin objects, so converting it to dict
    prix_values = prix_values.to_dict()
    for i, prod in enumerate(liste_produits):
        prix = prix_values[prod]
        j = 0
        while prix > threshold[j]:
            j = j + 1
            if j == len(threshold):
                break
        X.loc[i, label_col[j - 1]] = 1

    for i in range(len(threshold)):
        if i == len(threshold) - 1:
            col = ".>{}".format(threshold[i])
        else:
            col = "{}<.<{}".format(threshold[i], threshold[i + 1])

with measure("fit/predict"):
    matrix = X.to_numpy()  # as_matrix()
    for n_clusters in range(3, 10):
        kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=30)
        kmeans.fit(matrix)
        clusters = kmeans.predict(matrix)
        silhouette_avg = silhouette_score(matrix, clusters)

    n_clusters = 5
    silhouette_avg = -1
    while silhouette_avg < 0.145:
        kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=30)
        kmeans.fit(matrix)
        clusters = kmeans.predict(matrix)
        silhouette_avg = silhouette_score(matrix, clusters)

    pd.Series(clusters).value_counts()

    # define individual silouhette scores
    sample_silhouette_values = silhouette_samples(matrix, clusters)

    liste = pd.DataFrame(liste_produits)
    liste_words = [word for (word, _) in list_products]

    def count_words(df):
        occurence = {}
        for word in liste_words:
            if word in ["art", "set", "heart", "pink", "blue", "tag"]:
                continue
            occurence[word] = df[0].str.contains(word.upper()).sum()
        return pandas.Series(occurence)

    liste_cl = pd.concat([liste, pd.Series(clusters, name="cluster")], axis=1)
    occurence = liste_cl.groupby("cluster").apply(count_words)
    occurence = list(occurence.T.to_dict().values())

    pca = PCA()
    pca.fit(matrix)
    pca_samples = pca.transform(matrix)

    pca = PCA(n_components=50)
    matrix_9D = pca.fit_transform(matrix)
    mat = pd.DataFrame(matrix_9D)
    mat["cluster"] = pd.Series(clusters)

    corresp = dict()
    for key, val in zip(liste_produits, clusters):
        corresp[key] = val

    df_cleaned["categ_product"] = df_cleaned.loc[:, "Description"].map(corresp)
    for i in range(5):
        col = "categ_{}".format(i)
        df_temp = df_cleaned[df_cleaned["categ_product"] == i]
        price_temp = df_temp["UnitPrice"] * (
            df_temp["Quantity"] - df_temp["QuantityCanceled"]
        )
        price_temp = price_temp.apply(lambda x: x if x > 0 else 0)
        df_cleaned.loc[:, col] = price_temp
        df_cleaned[col].fillna(0, inplace=True)

    temp = df_cleaned.groupby(by=["CustomerID", "InvoiceNo"], as_index=False)[
        "TotalPrice"
    ].sum()
    basket_price = temp.rename(columns={"TotalPrice": "Basket Price"})
    # percentage of order price / product category
    for i in range(5):
        col = "categ_{}".format(i)
        temp = df_cleaned.groupby(by=["CustomerID", "InvoiceNo"], as_index=False)[
            col
        ].sum()
        basket_price.loc[:, col] = temp[col]

    # date of order
    df_cleaned["InvoiceDate_int"] = df_cleaned["InvoiceDate"].astype("int64")
    temp = df_cleaned.groupby(by=["CustomerID", "InvoiceNo"], as_index=False)[
        "InvoiceDate_int"
    ].mean()
    df_cleaned.drop("InvoiceDate_int", axis=1, inplace=True)
    basket_price.loc[:, "InvoiceDate"] = pd.to_datetime(temp["InvoiceDate_int"])

with measure("interpret results"):
    # selection of significant entries:
    basket_price = basket_price[basket_price["Basket Price"] > 0]

    set_entrainement = basket_price[
        basket_price["InvoiceDate"] < pd.to_datetime(datetime.date(2011, 10, 1))
    ]
    set_test = basket_price[
        basket_price["InvoiceDate"] >= pd.to_datetime(datetime.date(2011, 10, 1))
    ]
    basket_price = set_entrainement.copy(deep=True)

    # number of visits and stats on basket amount / users
    transactions_per_user = basket_price.groupby(by=["CustomerID"])[
        "Basket Price"
    ].agg(["count", "min", "max", "mean", "sum"])
    for i in range(5):
        col = "categ_{}".format(i)
        transactions_per_user.loc[:, col] = (
            basket_price.groupby(by=["CustomerID"])[col].sum()
            / transactions_per_user["sum"]
            * 100
        )

    transactions_per_user.reset_index(drop=False, inplace=True)
    basket_price.groupby(by=["CustomerID"])["categ_0"].sum()

    last_date = basket_price["InvoiceDate"].max().date()

    first_registration = pd.DataFrame(
        basket_price.groupby(by=["CustomerID"])["InvoiceDate"].min()
    )
    last_purchase = pd.DataFrame(
        basket_price.groupby(by=["CustomerID"])["InvoiceDate"].max()
    )

    test = first_registration.map(lambda x: (last_date - x.date()).days)
    test2 = last_purchase.map(lambda x: (last_date - x.date()).days)

    transactions_per_user.loc[:, "LastPurchase"] = test2.reset_index(drop=False)[
        "InvoiceDate"
    ]
    transactions_per_user.loc[:, "FirstPurchase"] = test.reset_index(drop=False)[
        "InvoiceDate"
    ]

    n1 = transactions_per_user[transactions_per_user["count"] == 1].shape[0]
    n2 = transactions_per_user.shape[0]

    list_cols = [
        "count",
        "min",
        "max",
        "mean",
        "categ_0",
        "categ_1",
        "categ_2",
        "categ_3",
        "categ_4",
    ]

    selected_customers = transactions_per_user.copy(deep=True)
    matrix = selected_customers[list_cols].to_numpy()

    scaler = StandardScaler()
    scaler.fit(matrix)
    scaled_matrix = scaler.transform(matrix)

    pca = PCA()
    pca.fit(scaled_matrix)
    pca_samples = pca.transform(scaled_matrix)

    n_clusters = 11
    kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=100)
    kmeans.fit(scaled_matrix)
    clusters_clients = kmeans.predict(scaled_matrix)
    silhouette_avg = silhouette_score(scaled_matrix, clusters_clients)

    pca = PCA(n_components=6)
    matrix_3D = pca.fit_transform(scaled_matrix)
    mat = pd.DataFrame(matrix_3D)
    mat["cluster"] = pd.Series(clusters_clients)

    # define individual silouhette scores
    sample_silhouette_values = silhouette_samples(scaled_matrix, clusters_clients)

    selected_customers["cluster"] = clusters_clients

    selected_customers = selected_customers.astype({"CustomerID": int})
    merged_df = pd.concat(
        [
            selected_customers.groupby("cluster").mean(),
            selected_customers.groupby("cluster").size().to_frame("size"),
        ],
        axis=1,
    )

    merged_df = merged_df.sort_values("sum")
    liste_index = []
    for i in range(5):
        COLUMN = f"categ_{i}"
        liste_index.append(merged_df[merged_df[COLUMN] > 45].index.values[0])

    liste_index_reordered = liste_index
    liste_index_reordered += [s for s in merged_df.index if s not in liste_index]

    merged_df = merged_df.reindex(index=liste_index_reordered)
    merged_df = merged_df.reset_index(drop=False)
    selected_customers = pd.concat([selected_customers], ignore_index=True)

    # Save to a csv file
    file_name = rawtraindata.stem
    SUFFIX = "_aug"
    newdatafile = f"{file_name}{SUFFIX}.csv"
    selected_customers.to_csv(newdatafile, index=False)

print(f"\n\n========= Results for {mode} mode =========")
for key, value in measurements.items():
    print(f"\t{key}: {value}s.")
