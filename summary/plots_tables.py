import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import plotly.io as pio

pio.kaleido.scope.mathjax = None

from summary.mappings import set_group_amount_groups, baseline_method_maps

fsb_count = 70
srb_count = 4
fsb_dataset_groups = {"sine": ["sine-"],
                      "ecg-cbf-rw": ["ecg-", "cbf-", "rw-"],
                      "increasing": ["increasing-"],
                      "saw": ["saw-"],
                      "wave": ["wave-"],
                      "corr": ["corr-"]}
fsb_dataset_count = {'corr': 5, 'ecg-cbf-rw': 20, 'increasing': 7, 'saw': 2, 'sine': 16, 'wave': 14}
srb_dataset_groups = {"lotka_volterra": ["lotka_volterra-"], }
srb_dataset_count = {"lotka_volterra": 4}

with open("anomaly_groups.json", "r") as f:
    anomaly_groups = json.load(f)

fsb_dataset_anomaly_groups = {}
fsb_dataset_anomaly_counts = {}
for g in anomaly_groups["anomaly2sequence"].items():
    fsb_dataset_anomaly_counts[g[0]] = len(g[1])
    fsb_dataset_anomaly_groups[g[0]] = g[1]


def calc_by_type(data, prefix, expected_count):
    table_view = []
    for g, rows in data.groupby(["model_type", "dataset"]):
        table_view.append([g[0], g[1], rows["AUC_ROC"].max(), rows["VUS_ROC"].max(), rows["R_AUC_ROC"].max()])

    table_view = pd.DataFrame(table_view, columns=["model_type", "dataset", "AUC ROC", "VUS ROC", "R_AUC ROC"])
    table_results = []
    for g, rows in table_view.groupby("model_type"):
        missing_percentage = (expected_count - rows.shape[0]) * 100 / expected_count
        table_results.append([g, f"{rows['AUC ROC'].mean():.2f} ± {rows['AUC ROC'].std():.2f}",
                              f"{rows['VUS ROC'].mean():.2f} ± {rows['VUS ROC'].std():.2f}",
                              f"{rows['R_AUC ROC'].mean():.2f} ± {rows['R_AUC ROC'].std():.2f}",
                              f"{missing_percentage:.1f}%"])

    table_results = pd.DataFrame(table_results, columns=["model_type", "AUC ROC", "VUS ROC", "R AUC ROC", "Missing %"])
    return table_results


def create_pox_plot_by_type_plotly(data, title: str = None):
    data["model_type"] = data["model_type"].apply(
        lambda x: baseline_method_maps[x]["name"] if x in baseline_method_maps else x)
    first_order_group = ["tcNF-base", "tcNF-cnn",
                         "tcNF-stateless", "tcNF-mlp",
                         "tcNF-stateful"]
    second_order_group = ["RealNVP",
                          "PCA", "KNN", "HBOS", "iForest", "GDN", "DAMP", "PCC", "IF-LOF", "CBLOF", "Torks",
                          "KMeans", "mSTAMP", "COP"]

    first_order_ranks = (data[data["model_type"].str.contains("|".join(first_order_group))]
                         .groupby("model_type")["AUC_ROC"].mean())
    second_order_ranks = (data[data["model_type"].str.contains("|".join(second_order_group))]
                          .groupby("model_type")["AUC_ROC"].mean())
    # sort by mean auc roc
    first_order_ranks = list(first_order_ranks.sort_values(ascending=False).index.values)
    second_order_ranks = list(second_order_ranks.sort_values(ascending=False).index.values)
    ordering = first_order_ranks + second_order_ranks

    data["model_type"] = data["model_type"].astype("category")
    data["model_type"] = data["model_type"].cat.set_categories(ordering)
    # sort by model type
    data.sort_values("model_type", inplace=True)
    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        shared_yaxes=True,
                        subplot_titles=("AUC", "VUS"),
                        vertical_spacing=0.05)
    fig.add_trace(go.Box(x=data["model_type"], y=data["AUC_ROC"], name="AUC ROC",
                         boxpoints="all", boxmean=True), row=1, col=1)
    fig.add_trace(go.Box(x=data["model_type"], y=data["VUS_ROC"], name="VUS ROC",
                         boxpoints="all", boxmean=True), row=2, col=1),
    # fig.add_trace(go.Box(x=data["model_type"], y=data["R_AUC_ROC"], name="R AUC ROC",
    #                      boxpoints="all", boxmean=False), row=3, col=1)
    # add a different background to first 4 box plots
    # for each box plot
    fig.update_layout(shapes=[
        dict(type="rect", xref="x", yref="y1", x0=3.4, x1=13.5, y0=-0.05, y1=1.05, fillcolor="lightgray", opacity=0.7,
             layer="below", line_width=0),
        dict(type="rect", xref="x", yref="y2", x0=3.4, x1=13.5, y0=-0.05, y1=1.05, fillcolor="lightgray", opacity=0.7,
             layer="below", line_width=0),
    ])

    # make points in box plot smaller
    fig.update_traces(marker=dict(size=2))
    fig.update_layout(
        # xaxis_title="Model Type",
        yaxis=dict(range=[-0.05, 1.05], title="Score"),
        yaxis2=dict(range=[-0.05, 1.05], title="Score"),
        # xaxis=dict(
        #     categoryorder='array',  # Specifies sorting order
        #     categoryarray=ordering  # Custom sorting order
        # ),
        # xaxis2=dict(
        #     categoryorder='array',  # Specifies sorting order
        #     categoryarray=ordering  # Custom sorting order
        # ),
        # xaxis3=dict(
        #     categoryorder='array',  # Specifies sorting order
        #     categoryarray=ordering  # Custom sorting order
        # )
    )
    # fig.update_traces(orientation='h')
    # hide legend
    fig.update_layout(showlegend=False)
    # reduce padding on the right side
    fig.update_layout(margin=dict(r=0, l=0))
    if title is not None:
        fig.update_layout(title_text=title)
    # fig.show()
    fig.write_image(f"fsb_overview_results.pdf", width=1000, height=600)


def create_pox_plot_py_type_matplotlib(data, title: str = None):
    ordering = ["tcNF-base", "tcNF-mlp",
                "tcNF-cnn", "tcNF-stateless",
                "RealNVP",
                "pcc", "hbos", "knn", "kmeans", "mstamp", "damp", "pca", "torsk",
                "gdn", "copod", "cblof", "cof", "iforest", "if_lof"
                ]
    x_lables = ["tcNF-base", "tcNF-mlp",
                "tcNF-cnn", "tcNF-stateless",
                "RealNVP",
                "PCC", "HBOS", "KNN", "KMeans", "mSTAMP", "DAMP", "PCA", "Torsk",
                "GDN", "COP", "CBLOF", "COF", "iForest", "IF-LOF"
                ]

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    fig.suptitle(title if title else "Plot")

    sns.boxplot(x="model_type", y="AUC_ROC", data=data, order=ordering, ax=axes[0])
    axes[0].set_title("AUC ROC")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("Score")

    sns.boxplot(x="model_type", y="VUS_ROC", data=data, order=ordering, ax=axes[1])
    axes[1].set_title("VUS ROC")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel("Model Type")
    axes[1].set_ylabel("Score")

    # set the x labels
    axes[1].set_xticklabels(x_lables, rotation=45, ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def calc_by_dataset_group(data, prefix, group, group_counts, group_order):
    table_view = []
    for g, rows in data.groupby(["model_type"]):
        for n, dataset in group.items():
            reg = "|".join(dataset)
            matches = rows[rows["dataset"].str.contains(reg, case=False, na=False)]
            count = group_counts[n] // 2
            missing_percentage = (count - matches.shape[0]) * 100 / count
            table_view.append([g[0], n,
                               f"{matches['AUC_ROC'].mean():.2f} ± {matches['AUC_ROC'].std():.2f}" if matches.shape[
                                                                                                          0] > 1 else f"{matches['AUC_ROC'].mean():.2f} ± 0.00",
                               f"{matches['VUS_ROC'].mean():.2f} ± {matches['VUS_ROC'].std():.2f}" if matches.shape[
                                                                                                          0] > 1 else f"{matches['VUS_ROC'].mean():.2f} ± 0.00",
                               # f"{matches['R_AUC_ROC'].mean():.2f} ± {matches['R_AUC_ROC'].std():.2f}",
                               f"{missing_percentage:.1f}%"])

    table_view = pd.DataFrame(table_view,
                              columns=["model_type", "dataset", "AUC ROC", "VUS ROC", "Missing %"])
    table_view["model_type"] = table_view["model_type"].apply(
        lambda x: baseline_method_maps[x]["name"] if x in baseline_method_maps else x)
    table_view["AUC ROC"] = table_view["AUC ROC"].apply(lambda x: "-" if "nan" in x else x)
    table_view["VUS ROC"] = table_view["VUS ROC"].apply(lambda x: "-" if "nan" in x else x)
    ordering = ["tcNF-base", "tcNF-mlp",
                "tcNF-cnn", "tcNF-stateless",
                "RealNVP",
                "PCC", "HBOS", "KNN", "KMeans", "mSTAMP", "DAMP", "PCA", "Torsk",
                "GDN", "COP", "CBLOF", "COF", "iForest", "IF-LOF"
                ]
    table_view["model_type"] = table_view["model_type"].astype("category")
    table_view["model_type"] = table_view["model_type"].cat.set_categories(ordering)
    table_view["dataset"] = table_view["dataset"].astype("category")
    table_view["dataset"] = table_view["dataset"].cat.set_categories(group_order)
    table_view.sort_values(["dataset", "model_type"], inplace=True)
    # drop if dataset is nature
    table_view = table_view[table_view["dataset"] != "nature"]

    # write table to latex and highlight the best values per column
    if prefix == "srb_dataset":
        pivot_view = table_view.pivot_table(index="model_type", columns="dataset",
                                                values=["AUC ROC", "VUS ROC"],
                                                aggfunc='first')
        column_format = "lcc"
        pivot_view.style.highlight_max(axis=0, props='bfseries: ;').to_latex(f"{prefix}_table.tex", hrules=True,
                                                                             column_format=column_format)
    else:
        auc_pivot_view = table_view.pivot_table(index="model_type", columns="dataset",
                                                values=["AUC ROC"],
                                                aggfunc='first')
        vus_pivot_view = table_view.pivot_table(index="model_type", columns="dataset",
                                                values=["VUS ROC"],
                                                aggfunc='first')
        column_format = "l" + "c" * len(group.keys())
        auc_pivot_view.style.highlight_max(axis=0, props='bfseries: ;').to_latex(f"{prefix}_auc_table.tex", hrules=True,
                                                                                 column_format=column_format)
        vus_pivot_view.style.highlight_max(axis=0, props='bfseries: ;').to_latex(f"{prefix}_vus_table.tex", hrules=True,
                                                                                 column_format=column_format)
    return table_view


if __name__ == "__main__":

    source = "fsb"
    if source == "fsb":
        srb_results = pd.read_csv("fsb/results-TFselfopt-fsb.csv")
        base_results = pd.read_csv("../results/mTADS-baseline/updated_wandb_export_2023-11-21.csv")

        # baseline methods
        list_of_unsupervised_methods = ["pcc", "hbos", "knn", "kmeans", "mstamp", "damp", "pca", "torsk",
                                        "gdn", "copod", "cblof", "cof", "iforest", "if_lof"]
        base_unsupervised_results = base_results[
            base_results["model_type"].str.contains("|".join(list_of_unsupervised_methods))]
        base_unsupervised_results = base_unsupervised_results[
            ~base_unsupervised_results["model_type"].str.contains("|".join([
                "robust_pca", "subsequence_knn", "copod", "hybrid_knn"
            ]))
        ]

        fsb_full_results = pd.concat([
            srb_results[["model_type", "AUC_ROC", "VUS_ROC"]],
            base_results[["model_type", "AUC_ROC", "VUS_ROC"]]
        ])

        srb_full_unsupervised_results = pd.concat([
            srb_results[["model_type", "dataset", "AUC_ROC", "VUS_ROC"]],
            base_unsupervised_results[["model_type", "dataset", "AUC_ROC", "VUS_ROC"]]
        ])
        srb_full_unsupervised_results.reset_index(inplace=True)
        result_table = srb_full_unsupervised_results
        title = "Overview FSB Benchmark Suite and mTADS Baseline"
        create_pox_plot_by_type_plotly(result_table, title)
        calc_by_dataset_group(srb_full_unsupervised_results, "fsb", fsb_dataset_groups,
                              fsb_dataset_count, set_group_amount_groups.keys())
        calc_by_dataset_group(srb_full_unsupervised_results, "fsb_anomaly", fsb_dataset_anomaly_groups,
                              fsb_dataset_anomaly_counts, fsb_dataset_anomaly_groups.keys())

    elif source == "srb":
        srb_results = pd.read_csv("srb/results-TFselfopt-srb.csv")
        base_results = pd.read_csv("../results/mTADS-baseline/srb_updated_wandb_export_2023-11-21.csv")

        # baseline methods
        list_of_unsupervised_methods = ["pcc", "hbos", "knn", "kmeans", "mstamp", "damp", "pca", "torsk",
                                        "gdn", "copod", "cblof", "cof", "iforest", "if_lof"]
        base_unsupervised_results = base_results[
            base_results["model_type"].str.contains("|".join(list_of_unsupervised_methods))]
        base_unsupervised_results = base_unsupervised_results[
            ~base_unsupervised_results["model_type"].str.contains("|".join([
                "robust_pca", "subsequence_knn", "copod", "hybrid_knn", "torsk"
            ]))
        ]

        srb_full_unsupervised_results = pd.concat([
            srb_results[["model_type", "dataset", "AUC_ROC", "VUS_ROC"]],
            base_unsupervised_results[["model_type", "dataset", "AUC_ROC", "VUS_ROC"]]
        ])
        srb_full_unsupervised_results.reset_index(inplace=True)
        result_table = srb_full_unsupervised_results
        title = "Overview SRB Benchmark Suite and mTADS Baseline"
        # create_pox_plot_by_type_plotly(result_table, title)
        calc_by_dataset_group(srb_full_unsupervised_results, "srb_dataset", srb_dataset_groups,
                              srb_dataset_count, set_group_amount_groups.keys())

    elif source == "real":
        real_results = pd.read_csv("real/results-TFselfopt-real.csv")

        result_table = real_results[["model_type", "dataset", "AUC_ROC", "VUS_ROC"]]
        result_table = result_table[~result_table["dataset"].str.contains("smd|ghl")]

        # map AUC_ROC and VUS_ROC to string with 2 decimal places
        result_table["AUC_ROC"] = result_table["AUC_ROC"].apply(lambda x: f'{x:.2f}')
        result_table["VUS_ROC"] = result_table["VUS_ROC"].apply(lambda x: f'{x:.2f}')

        for i, g in real_results[real_results["dataset"].str.contains("ghl")].groupby("model_type"):
            config = g["img"].values[0].replace("overview.png", "config.json")
            with open(config, "r") as f:
                args = json.load(f)
            test = args["test"]
            auc = []
            vus = []
            for t, v in test.items():
                if "t" == "mean":
                    continue
                auc.append(v["AUC_ROC"])
                vus.append(v["VUS_ROC"])
            s = pd.DataFrame.from_records([[i, "ghl", f'{np.mean(auc):.2f} ± {np.std(auc):.2f}',
                                            f'{np.mean(vus):.2f} ± {np.std(vus):.2f}']],
                                          columns=["model_type", "dataset", "AUC_ROC", "VUS_ROC"])
            # add s into result_table
            result_table = pd.concat([result_table, s])

        for i, g in real_results[real_results["dataset"].str.contains("smd")].groupby("model_type"):
            s = pd.DataFrame.from_records([[i, "smd", f'{g["AUC_ROC"].mean():.2f} ± {g["AUC_ROC"].std():.2f}',
                                            f'{g["VUS_ROC"].mean():.2f} ± {g["VUS_ROC"].std():.2f}']],
                                          columns=["model_type", "dataset", "AUC_ROC", "VUS_ROC"])
            # add s into result_table
            result_table = pd.concat([result_table, s])

        title = "Overview Real-World Datasets"
        for v, n in zip(["auc", "vus"], ["AUC_ROC", "VUS_ROC"]):
            c_result_table = pd.pivot_table(result_table, index="model_type", columns="dataset",
                                            values=[n],
                                            aggfunc='first')
            # replace nan with -
            c_result_table = c_result_table.replace(np.nan, "-")

            # write only :.2f values to latex
            (c_result_table.style
             .highlight_max(axis=0, props='bfseries: ;')
             # .format("{:.2f}")
             .to_latex(f"real_{v}_table.tex",
                       hrules=True,
                       column_format="l|ccccc|ccccc"))
    else:
        raise ValueError("Unknown source")
