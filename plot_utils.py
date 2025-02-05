import io
from datetime import datetime
from typing import Union

import numpy
import numpy as np
import pandas as pd
import plotly
from PIL import Image
from altair import selection
from loguru import logger
from matplotlib import pyplot as plt, gridspec
from matplotlib.pyplot import subplots, title
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.colors as colors
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve

from global_utils import re_range
from summary.mappings import optimization_mapping


def matplot_to_pil(assets):
    buf = io.BytesIO()
    assets.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def plot_samples(s=None, input_dim=None, validation_loader=None) -> plt.Figure:
    cols = input_dim // 2 + 1
    d = validation_loader.dataset.data1
    fig, ax = plt.subplots(nrows=1, ncols=cols, figsize=(10 * cols, 12))
    # ax[0].axis('off')
    # ax[1].axis('off')
    ax[0].set_title('Data', fontsize=24)
    ax[1].set_title('Samples', fontsize=24)
    ax[0].hist2d(d[..., 0], d[..., 1], bins=256, range=[[-2, 2], [-2, 2]])
    if s is not None:
        s = s.detach().cpu().numpy()
        for i, d_ in enumerate(range(0, input_dim, 2)):
            ax[i + 1].hist2d(s[..., d_], s[..., d_ + 1], bins=256, range=[[-2, 2], [-2, 2]])
        # ax[2].hist2d(s[..., 2], s[...,  3], bins=256, range=[[-2, 2], [-2, 2]])
    plt.tight_layout()
    return fig


def make_simple_anomaly_detection_plot(df, column, method):
    matches = df[df['anomaly'] == 1][column]
    plt.figure(figsize=(18, 6))
    plt.plot(df[column], color='blue', label=column)
    plt.plot(matches.index, matches, linestyle='none', marker='x', color='red', label=method)
    plt.xlabel('Date and Time')
    plt.ylabel('Sensor Reading')
    plt.title(f'{column} Anomalies')
    plt.legend()
    plt.show()


def make_auc_plot(y_values: list, m_type: str, title: str) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot([0] + list(range(len(y_values))), [0] + y_values)
    ax.axhline(0, color="black")
    ax.set_xticks(range(0, len(y_values)))
    ax.set_ylim([-1.1, 1.1])
    ax.set_xlabel(f"Buffer size")
    ax.set_ylabel(f"{m_type} score")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def consecutive_w_list_comprehension(arr, step_size=7200):
    idx = np.r_[0, np.where(np.diff(arr) > step_size)[0] + 1, len(arr)]
    return [arr[i:j] for i, j in zip(idx, idx[1:])]


def date_transform(date):
    if isinstance(date, numpy.datetime64):
        return date
    if not isinstance(date, np.dtypes.DateTime64DType) and np.int64(date) > 1_356_998_400:
        return datetime.fromtimestamp(date)
    else:
        return date


def plot_all_multiple_detection(train_seq, test_seq, prob, is_anomaly, test_dates, train_dates, title) -> go.Figure:
    test_dates_formated = [date_transform(d) for d in test_dates]
    train_dates_formated = [date_transform(d) for d in train_dates]
    fig = make_subplots(rows=4, cols=1,
                        shared_xaxes=True,
                        subplot_titles=(["Train Features", "Test Features", "Loss", "Anomaly"]),
                        vertical_spacing=0.05, horizontal_spacing=0.05)

    for d in range(min(train_seq.shape[1], 20)):
        fig.add_trace(go.Scatter(x=train_dates_formated, y=train_seq[:, d], name=f"Train {d}"),
                      row=1, col=1)

    for d in range(min(test_seq.shape[1], 20)):
        fig.add_trace(go.Scatter(x=test_dates_formated, y=test_seq[:, d], name=f"Test {d}"),
                      row=2, col=1)

    if prob is not None and len(prob.shape) == 1:
        # one class probability prediction
        fig.add_trace(go.Scatter(x=test_dates_formated, y=prob, name="Prediction"),
                      row=3, col=1)
    elif prob is not None and len(prob.shape) == 2 and prob.shape[1] == 2:
        # two class probability prediction
        for i, n in enumerate(["Normal", "Anomaly"]):
            fig.add_trace(go.Scatter(x=test_dates_formated, y=prob[:, i],
                                     name=n),
                          row=3, col=1)
    elif prob is not None and len(prob.shape) == 2 and prob.shape[1] == 3:
        # one class probability prediction from Normalizing Flow with all components
        for i, n in enumerate(["NLL", "LogProb", "LogDet"]):
            fig.add_trace(go.Scatter(x=test_dates_formated, y=prob[:, i],
                                     name=n),
                          row=3, col=1)
    elif prob is not None and len(prob.shape) == 2 and prob.shape[1] > 3:
        for i in range(min(prob.shape[1], 6)):
            fig.add_trace(go.Scatter(x=test_dates_formated, y=prob[:, i],
                                     name=f"dim {i}"),
                          row=3, col=1)
    else:
        logger.info("Plotting probability/prediction is not implemented for this type")

    if is_anomaly is not None:
        for i, n in enumerate(["Test Anomalies", "Detection"]):
            if is_anomaly[i] is not None:
                fig.add_trace(go.Scatter(x=test_dates_formated, y=is_anomaly[i],
                                         name=n),
                              row=4, col=1)

    # if np.abs(len(test_dates) - len(train_dates)) > 500:
    #     train_sections = consecutive_w_list_comprehension(train_dates)
    #     for i, s in enumerate(train_sections):
    #         fig.add_vrect(date_transform(s[0]),
    #                       date_transform(s[-1]),
    #                       fillcolor="gray", opacity=0.2, row=3, col=1)

    fig.update_layout(title_text=title)
    return fig


def plot_2d_latent_dist_space(train_seq, test_seq, latent, probs, sample_anomalies, is_anomaly, title) -> go.Figure:
    if train_seq.shape[1] > 2:
        # create input space transformer
        pca = PCA(n_components=2)
        if len(test_seq) == len(train_seq):
            complete_data_space = np.vstack([test_seq, train_seq])
            # fit the pca to all samples
            pca.fit(complete_data_space)
            # use input space transformer for all samples
            test_seq = pca.transform(test_seq)
            train_seq = pca.transform(train_seq)
        else:
            test_seq = pca.fit_transform(test_seq)
            # use input space transformer for train samples
            train_seq = pca.transform(train_seq)
    # if len(probs) > 0 and probs.shape[1] > 2:
    #     # make new fit for probs values
    #     pca = PCA(n_components=2)
    #     probs = pca.fit_transform(probs)

    # jitter = True
    # if latent.shape[1] > 2:
    #     # make new fit for latent values
    #     pca = PCA(n_components=2)
    #     latent = pca.fit_transform(latent)
    #     jitter = False

    train_indexes = list(range(train_seq.shape[0]))
    test_indexes = list(range(test_seq.shape[0]))

    train_red_flag = np.where(sample_anomalies == 1)[0]
    test_red_flag = np.where(is_anomaly == 1)[0]
    fig = make_subplots(rows=2, cols=3,
                        specs=[[{}, {"rowspan": 2}, {"rowspan": 2}],
                               [{}, None, None]],
                        shared_xaxes="columns",
                        shared_yaxes="columns",
                        subplot_titles=(["Input Train", "Latent space", "Probs space", "Input All"]),
                        vertical_spacing=0.05, horizontal_spacing=0.05)
    # Input space
    fig.add_trace(go.Scatter(x=train_seq[:, 0], y=train_seq[:, 1], mode='markers',
                             marker=dict(color=train_indexes, colorscale='Viridis'), name="Train Input space"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=train_seq[train_red_flag, 0], y=train_seq[train_red_flag, 1], mode='markers',
                             opacity=0.7, marker=dict(color="red"), name="Train anomalies"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=test_seq[:, 0], y=test_seq[:, 1], mode='markers',
                             marker=dict(color=test_indexes, colorscale='Viridis'),
                             name="Full Input space"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=test_seq[test_red_flag, 0], y=test_seq[test_red_flag, 1], mode='markers',
                             opacity=0.7, marker=dict(color="red"), name="Input anomalies"),
                  row=2, col=1)

    # Latent space
    for i in range(0, latent.shape[1], 2):
        fig.add_trace(go.Scatter(x=latent[:, i], y=latent[:, i + 1], mode='markers',
                                 marker=dict(color=test_indexes, colorscale='Viridis'), name="Latent normal"),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=latent[test_red_flag, i], y=latent[test_red_flag, i + 1], mode='markers',
                                 opacity=0.7, marker=dict(color="red"), name="Latent anomalies"),
                      row=1, col=2)

    for i in range(0, probs.shape[1], 2):
        # probs space
        fig.add_trace(go.Scatter(x=probs[:, i], y=probs[:, i + 1], mode='markers',
                                 marker=dict(color=test_indexes, colorscale='Viridis'), name="Probs"),
                      row=1, col=3)
        fig.add_trace(go.Scatter(x=probs[test_red_flag, i], y=probs[test_red_flag, i + 1], mode='markers',
                                 opacity=0.7, marker=dict(color="red"), name="Probs anomalies"),
                      row=1, col=3)

    fig.update_layout(title=title)
    return fig


def plot_roc_curve(prob, is_anomaly, title) -> go.Figure:
    fpr, tpr, thresholds = roc_curve(is_anomaly, prob)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines'
    ))
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_layout(title=title,
                      xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate")
    fig.update_yaxes(range=[-0.1, 1.1])
    fig.update_xaxes(range=[-0.1, 1.1])

    return fig


def __plot_sequence(ax, data, number_of_sequences):
    nos = min(data.shape[0], number_of_sequences)
    l = data.shape[1]
    for row in range(nos):
        ax.plot(range(l), data[row, :])


def __plot_y_latent_scatter(ax, data, number_of_sequences):
    nos = min(data.shape[0], number_of_sequences)
    l = data.shape[1]
    for row in range(nos):
        ax.scatter(range(l), data[row, :])


def __plot_z_latent_scatter(ax, data, number_of_sequences):
    nos = min(data.shape[0], number_of_sequences)
    for row in range(nos):
        ax.scatter(data[row, 0], data[row, 1])


def plot_all(in_, y_, z_, out_, sy_, sz_, sx_, title=None) -> plt.Figure:
    fig = plt.figure(figsize=(15, 15))
    axs = fig.subplots(4, 4)

    for i in range(in_.shape[2]):
        __plot_sequence(axs[0, i], in_[:, :, i])
        axs[0, i].set(title=f"Input F{i + 1}")
        # axs[0, i].set_ylim(-0.1, 1.1)

    for i in range(out_.shape[2]):
        __plot_sequence(axs[1, i], out_[:, :, i])
        axs[1, i].set(title=f"Output F{i + 1}")
        # axs[1, i].set_ylim(-0.1, 1.1)

    if y_ is not None:
        __plot_y_latent_scatter(axs[2, 0], y_)
        axs[2, 0].set(title="AE latent space | neg_log_likelihood")
        # axs[2, 0].set_ylim(-1, 1)

    if z_ is not None:
        __plot_y_latent_scatter(axs[2, 1], z_)
        axs[2, 1].set(title="Flow distribution space")
        # axs[2, 1].set_ylim(-1, 1)

    if sz_ is not None:
        __plot_y_latent_scatter(axs[2, 2], sz_)
        axs[2, 2].set(title="Flow sample distribution space")
        # axs[2, 2].set_ylim(-1, 1)

    if sy_ is not None:
        __plot_y_latent_scatter(axs[2, 3], sy_)
        axs[2, 3].set(title="AE sample latent space | neg_log_likelihood")
        # axs[2, 3].set_ylim(-1, 1)

    for i in range(sx_.shape[2]):
        __plot_sequence(axs[3, i], sx_[:, :, i])
        axs[3, i].set(title=f"Output sample F{i + 1}")
        # axs[3, i].set_ylim(-0.1, 1.1)

    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_history(history) -> plt.Figure:
    losses = list(filter(lambda x: "val" not in x, history.history.keys()))
    fig = plt.figure(figsize=(10, 6))
    axs = fig.subplots(2, 3).flatten()

    for i, l in enumerate(losses):
        if i >= 6:
            break
        axs[i].plot(history.history[l], label=l)
        axs[i].plot(history.history["val_" + l], label="val_" + l)
        axs[i].set(title=l)
        axs[i].legend()

    fig.suptitle("Training history")
    plt.tight_layout()
    return fig


def plot_results_based_on_dataset(base_prob, base_data, base_dist, base_nll,
                                  diff_prob, diff_data, diff_dist, diff_nll, title) -> plt.Figure:
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle(title)

    gs0 = gridspec.GridSpec(2, 1, figure=fig)

    for i, data in zip(range(2), [(base_prob, base_data, base_dist, base_nll),
                                  (diff_prob, diff_data, diff_dist, diff_nll)
                                  ]):
        gs00 = gridspec.GridSpecFromSubplotSpec(1, 7, subplot_spec=gs0[i])

        prob, data, dist, nll = data

        # hist
        ax = fig.add_subplot(gs00[:, :2])
        for p, n in prob:
            if n == "th":
                ax.axvline(p, color="black", ls="--", alpha=0.3)
            else:
                ax.hist(p, bins=50, alpha=0.5, label=n)
        ax.legend()
        ax.set(title="NLL histogram (normalized)")

        # dataspace
        ax = fig.add_subplot(gs00[:, 2])
        __plot_sequence(ax, data[:, :, 0])
        # ax.set_ylim(-0.1, 1.1)
        ax.set(title="Feature 1")

        if data.shape[2] > 1:
            ax = fig.add_subplot(gs00[:, 3])
            __plot_sequence(ax, data[:, :, 1])
            # ax.set_ylim(-0.1, 1.1)
            ax.set(title="Feature 2")

        # dist
        ax = fig.add_subplot(gs00[:, 4:6])
        __plot_y_latent_scatter(ax, dist)
        ax.set(title="Distributions x value")

        # nll
        ax = fig.add_subplot(gs00[:, 6])
        __plot_y_latent_scatter(ax, np.expand_dims(nll, axis=1))
        ax.set(title="NLL")

    plt.tight_layout()

    return fig


def plot_detection(seq, dates, prob, prob_dates, z, title) -> go.Figure:
    latent_space = z.shape[1] // 2
    fig = make_subplots(rows=latent_space, cols=3,
                        specs=[[{"rowspan": 5}, {}, {}],
                               [None, {}, {}],
                               [None, {}, {}],
                               [None, {}, {}],
                               [None, {}, {}],
                               [{"rowspan": 5}, {}, {}],
                               [None, {}, {}],
                               [None, {}, {}],
                               [None, {}, {}],
                               [None, {}, {}]],
                        shared_xaxes=True,
                        subplot_titles=("Feature 1",
                                        "", "", "", "", "", "",
                                        "", "", "", "", "", "",
                                        "Feature 2",
                                        "", "", "", "", "", "",
                                        "", "", "", "", "", ""))

    for i in range(seq.shape[1]):
        fig.add_trace(go.Scatter(x=dates, y=seq[:, i], name=f"Feature {i + 1}"), row=i * 5 + 1, col=1)
        fig.add_trace(go.Scatter(x=prob_dates, y=prob), row=i * 5 + 1, col=1)

    samples = list(range(z.shape[0]))
    for c in range(2):
        for i in range(latent_space):
            fig.add_trace(go.Scatter(x=samples, y=z[:, c * latent_space + i]), row=i + 1, col=c + 2)
            fig.add_hline(y=1, opacity=0.2, line_dash="dash", row=i + 1, col=c + 2)
            fig.add_hline(y=-1, opacity=0.2, line_dash="dash", row=i + 1, col=c + 2)
            fig.add_vline(x=z.shape[0] // 2, opacity=0.2, line_dash="dash", row=i + 1, col=c + 2)

    fig.update_layout(title_text=title)
    return fig


def plot_all_detection(seq, prob, dates, threshold, window_size, selection, title) -> go.Figure:
    fig = make_subplots(rows=3, cols=1,
                        shared_xaxes=True,
                        subplot_titles=("Feature 1", "Feature 2", "NLL (normalized)"))

    data_colors = ["blue", "red"]
    selection_colors = ["gray", "goldenrod"]

    cut_point = seq.shape[1]
    if seq.shape[0] > 1:
        selection_pairs = [[], []]
    else:
        selection_pairs = [[]]
    if selection is not None:
        start = selection[0]
        prev = selection[0]
        for s in selection[1:]:
            if abs(s - prev) > 1:
                if start > cut_point:
                    selection_pairs[1].append((start - cut_point, prev - cut_point))
                else:
                    selection_pairs[0].append((start, prev))
                start = s
            prev = s

    for seq_, prob_, dates_, selection_pairs_, data_color_, selection_color_ in \
            zip(seq, prob, dates, selection_pairs, data_colors, selection_colors):
        for i in range(seq_.shape[1]):
            fig.add_trace(go.Scatter(x=dates_, y=seq_[:, i], line=dict(color=data_color_)), row=i + 1, col=1)
            if selection is not None:
                for (s, e) in selection_pairs_:
                    fig.add_vrect(dates_[s - 5], dates_[e + 5], fillcolor=selection_color_, opacity=0.2,
                                  row=i + 1, col=1)

        fig.add_trace(go.Scatter(x=dates_[5:-5], y=prob_, line=dict(color=data_color_)), row=3, col=1)
        for th in threshold:
            fig.add_hline(y=th, line_color="black", opacity=0.2, row=3, col=1)
        if selection is not None:
            for (s, e) in selection_pairs_:
                fig.add_vrect(dates_[s - 5], dates_[e + 5], fillcolor=selection_color_, opacity=0.2,
                              row=3, col=1)

    fig.update_layout(title_text=title)
    return fig, None


def plot_optimization_results(results, group) -> go.Figure:
    iterations = results["iterations"]
    features = list(results[0][0].keys())
    features.remove("opt_goal")
    features.remove("details")

    cols = iterations // 2
    if iterations % 2 != 0:
        cols += 1

    fig = make_subplots(rows=2, cols=cols,
                        subplot_titles=[f"Gen {i}" for i in range(iterations)],
                        shared_yaxes=True,
                        vertical_spacing=0.05)

    best_candidate = None
    best_loss = 1000

    for gen, (k, v) in enumerate(results.items()):
        if not isinstance(k, int):
            continue
        p_col = gen % cols
        p_row = gen // cols
        df = pd.DataFrame.from_dict(v).T

        if isinstance(df.opt_goal[0], list):
            df["details"] = df["details"].apply(lambda x: x[1:])
            df["loss"] = df["opt_goal"].apply(lambda x: x[0])

        min_loss = df["opt_goal"].min()
        max_loss = df["opt_goal"].max()
        fig.add_annotation(xref=f"x{gen + 1}", x=1, yref=f"y{gen + 1}", y=0,
                           text=f"Min: {min_loss:.3f}, Max: {max_loss:.3f}",
                           showarrow=False, row=p_row + 1, col=p_col + 1)

        loss_normal = re_range(df["opt_goal"])
        colors_mapped = plotly.colors.sample_colorscale("Viridis", loss_normal)  # Viridis, Cividis

        for kr, vr in df.iterrows():
            if vr["opt_goal"] < best_loss:
                best_loss = vr["opt_goal"]
                best_candidate = (gen, kr, vr)

            fig.add_trace(go.Scatter(x=features, y=vr[features] + np.random.normal(0, 0.15, len(features)),
                                     mode='lines',
                                     marker=dict(
                                         color=colors_mapped[kr],  # set color equal to a variable
                                     ),
                                     text=f"Can: {kr} | \n"
                                          f"Loss: {vr['opt_goal']:.3f} \n"
                                          f"Hidden Multi.: {vr['hidden_multiplier']} \n"
                                          f"ST Layers: {vr['st_net_layers']} \n"
                                          f"Coupling Layers: {vr['coupling_layers']} \n"
                                          f"Seed: {vr['seed']} \n"
                                          f"Past: {vr['past']} \n",
                                     textposition="bottom right",
                                     hoverinfo="text",
                                     ),
                          row=p_row + 1, col=p_col + 1)

    fig.update(layout_showlegend=False)
    # backup = fig.layout.annotations
    # fig.update_layout(annotations=annotations)
    # fig.update_layout(annotations=list(backup) + list(fig.layout.annotations))
    # fig['layout'].update(annotations=annotations)
    fig.update_layout(title_text=f"{group} | \n"
                                 f"Best: Gen: {best_candidate[0]} Can: {best_candidate[1]} "
                                 f"Loss: {best_candidate[2]['opt_goal']:.3f} | "
                                 f"Hidden Multi.: {best_candidate[2]['hidden_multiplier']} "
                                 f"ST Layers: {best_candidate[2]['st_net_layers']} "
                                 f"Coupling Layers: {best_candidate[2]['coupling_layers']} "
                                 f"Past: {best_candidate[2]['past']}")
    fig.show()

    return fig


def plot_parallel_coordinates(results, group, model_type) -> go.Figure:
    # convert results to DataFrame
    result_list = []
    for gen_id, can in results.items():
        if not isinstance(gen_id, int):
            try:
                gen_id = int(gen_id)
            except ValueError:
                continue
        for can_id, can_data in can.items():
            if np.isinf(can_data["opt_goal"]) or can_data["opt_goal"] >= 1000:
                continue
            result_list.append({**can_data, "gen": gen_id, "can_id": can_id,
                                "test_auc": can_data["details"][1]["values"]["AUC_ROC"],
                                "test_vus": can_data["details"][1]["values"]["VUS_ROC"]})

    df = pd.DataFrame(result_list)
    # df["test_auc"] = df["details"].apply(lambda x: x[1]['values']['AUC_ROC'])
    # df["test_vus"] = df["details"].apply(lambda x: x[1]['values']['VUS_ROC'])

    features = list(results[0][0].keys())
    features.remove("opt_goal")
    features.remove("details")
    try:
        features.remove("seed")
    except ValueError:
        pass
    features = ["gen"] + features + ["opt_goal", "test_auc", "test_vus"]
    # features = list(map(lambda x: optimization_mapping[x], features))

    fig = go.Figure()
    fig.add_trace(go.Parcoords(line=dict(color=df["opt_goal"],
                                         colorscale='Viridis',
                                         showscale=True,
                                         cmin=df["opt_goal"].min(),
                                         cmax=df["opt_goal"].max(),
                                         colorbar=dict(title='Optim. Goal')),

                               dimensions=[dict(range=[df[f].min(), df[f].max()],
                                                label=optimization_mapping[f],
                                                values=df[f]) for f in features])
                  )

    # make lines thicker

    best = df['opt_goal'].argmin()
    best_candidate = df.iloc[best]
    fig.update_layout(title_text=f"{group} | {model_type} | "
                                 f"Best: Gen.: {best_candidate['gen']} Can.: {best_candidate['can_id']} "
                                 f"Optim. Goal: {best_candidate['opt_goal']:.3f}")
    fig.show()

    return fig


def plot_kl_matrix(sections, kl, r_kl):
    grid_size = len(sections)
    fig = make_subplots(rows=grid_size + 1, cols=grid_size + 1,
                        # shared_xaxes=True,
                        # shared_yaxes=True,
                        vertical_spacing=0.05, horizontal_spacing=0.05)

    for d, i in sections:
        for j in range(d.shape[1]):
            x_val = list(range(d.shape[0]))
            fig.add_trace(go.Scatter(y=d[:, j], x=x_val, mode='lines', showlegend=False),
                          row=1, col=i + 2)
            fig.add_trace(go.Scatter(y=d[:, j], x=x_val, mode='lines', showlegend=False),
                          row=i + 2, col=1)

    # gray colors
    hist_colors = ["#808080", "#A9A9A9", "#C0C0C0", "#D3D3D3", "#DCDCDC", "#F5F5F5"]
    fig.add_annotation(text="KL and R_KL", x=24, y=0.5, row=1, col=1)
    for i in range(grid_size):
        for j in range(grid_size):
            if i != j:
                fig.add_annotation(text=f"KL: {kl[i, j]:.3f}", x=0, y=0.8, row=i + 2, col=j + 2, showarrow=False)
                fig.add_annotation(text=f"R_KL: {r_kl[i, j]:.3f}", x=0, y=-0.1, row=i + 2, col=j + 2, showarrow=False)

            fig.add_trace(go.Scatter(x=sections[i][0][:, 0], y=sections[i][0][:, 1], mode='markers',
                                     showlegend=False, marker=dict(color=hist_colors[0])),
                          row=i + 2, col=j + 2)
            if i != j:
                fig.add_trace(go.Scatter(x=sections[j][0][:, 0], y=sections[j][0][:, 1], mode='markers',
                                         showlegend=False, marker=dict(color=hist_colors[1])),
                              row=i + 2, col=j + 2)

    fig.show()


def plot_spectrogram_matrix(sections: list, mel_sections: list, mel_l2_diff: np.ndarray):
    grid_size = len(sections)

    # mel_titles = np.reshape([f"L2 {i:.3f}" for i in mel_l2_diff.flatten()], (grid_size, grid_size))
    sub_titles = np.full((grid_size + 1, grid_size + 1), "", dtype=object)
    sub_titles[1:, 1:] = mel_l2_diff

    fig = make_subplots(rows=grid_size + 1, cols=grid_size + 1,
                        # shared_xaxes=True,
                        # shared_yaxes=True,
                        vertical_spacing=0.05, horizontal_spacing=0.05,
                        subplot_titles=np.reshape(sub_titles, -1))

    fig.add_annotation(text="Mel L2 diff", x=0, y=0, row=1, col=1, showarrow=False)
    for section in sections:
        for j in range(section[0].shape[1]):
            x_val = list(range(section[0].shape[0]))
            fig.add_trace(go.Scatter(y=section[0][:, j], x=x_val, mode='lines', showlegend=False),
                          row=1, col=section[1] + 2)
            fig.add_trace(go.Scatter(y=section[0][:, j], x=x_val, mode='lines', showlegend=False),
                          row=section[1] + 2, col=1)

    for i in range(grid_size):
        for j in range(grid_size):
            if i == j:
                fig.add_trace(go.Heatmap(z=np.vstack(mel_sections[i]), coloraxis="coloraxis",
                                         zmid=0), row=i + 2, col=j + 2)
            else:
                mell_diff = mel_sections[i] - mel_sections[j]
                fig.add_trace(go.Heatmap(z=np.vstack(mell_diff), coloraxis="coloraxis",
                                         zmid=0), row=i + 2, col=j + 2)

    fig.update_layout(title_text="Mel Spectrogram L2 Diff (L2 | MSE | DTW)",
                      coloraxis={'colorscale': 'rdbu', 'cmid': 0})

    fig.show()


def plot_sampled_imputation(data: np.ndarray, full_new_data: np.ndarray, all_probs: np.ndarray, update_history: bool):
    fig = make_subplots(rows=3, cols=1,
                        shared_xaxes=True,
                        shared_yaxes=False,
                        vertical_spacing=0.05, horizontal_spacing=0.05,
                        subplot_titles=("Data", "Imputed Data", "Imputed Probabilities"))

    for c in range(data.shape[1]):
        fig.add_trace(go.Scatter(x=list(range(data.shape[0])), y=data[:, c],
                                 mode='lines', name=f"Original {c}"),
                      row=1, col=1)

    for c in range(full_new_data.shape[1]):
        fig.add_trace(go.Scatter(x=list(range(full_new_data.shape[0])), y=full_new_data[:, c],
                                 mode='lines', name=f"Imputed {c}"),
                      row=2, col=1)

    for c, n in enumerate(["NLL", "LogProb", "LogDet"]):
        fig.add_trace(go.Scatter(x=list(range(all_probs.shape[0])), y=all_probs[:, c],
                                 mode='lines', name=f"{n}"),
                      row=3, col=1)

    # add figure title
    fig.update_layout(title_text=f"Sampled Imputation with guide and history update {update_history}")

    fig.show()


def simple_pair_sequence_plot(train_seq: np.ndarray, test_seq: np.ndarray, nll: np.ndarray, is_anomaly: np.ndarray,
                              stored_nll: np.ndarray, scores: dict):
    rows = 4 if stored_nll is not None else 3
    fig = make_subplots(rows=rows, cols=1,
                        shared_xaxes=True,
                        shared_yaxes=True,
                        subplot_titles=("Data", "Anomaly"),
                        vertical_spacing=0.05)
    for i in range(train_seq.shape[1]):
        fig.add_trace(
            go.Scatter(x=list(range(train_seq.shape[0])), y=train_seq[:, i], mode='lines', name=f"Feature {i}"),
            row=1, col=1)
    for i in range(test_seq.shape[1]):
        fig.add_trace(go.Scatter(x=list(range(test_seq.shape[0])), y=test_seq[:, i], mode='lines', name=f"Feature {i}"),
                      row=2, col=1)
    fig.add_trace(go.Scatter(x=list(range(nll.shape[0])), y=nll, mode='lines', name='Score'),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=list(range(is_anomaly.shape[0])), y=is_anomaly, mode='lines', name='Anomaly'),
                  row=3, col=1)

    if stored_nll is not None:
        diff_start = test_seq.shape[0] - stored_nll.shape[0]
        for i in range(stored_nll.shape[1]):
            fig.add_trace(go.Scatter(x=list(range(diff_start, stored_nll.shape[0] + diff_start)), y=stored_nll[:, i],
                                     mode='lines', name=f"Stored NLL {i}"),
                          row=4, col=1)

    fig.update_layout(title=f"AUC ROC: {scores['AUC_ROC']:.3f} | VUS ROC: {scores['VUS_ROC']:.3f} | "
                            f"R_AUC ROC: {scores['R_AUC_ROC']:.3f} | F: {scores['F']:.3f} | "
                            f"RF: {scores['RF']:.3f}")
    fig.show()


def plot_simple_plotly(data: pd.DataFrame, title: str):
    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        row_heights=[0.8, 0.2], )

    for c in data.columns[1:-1]:
        fig.add_trace(go.Scatter(x=data["timestamp"], y=data[c], mode='lines', name=f"Feature {c}"),
                      row=1, col=1)
    fig.add_trace(go.Scatter(x=data["timestamp"], y=data["is_anomaly"], mode='lines', name="Anomaly"),
                  row=2, col=1)

    fig.update_layout(title=title)
    fig.show()


def plot_sequence_and_latent_space(train_seq: np.ndarray, train_dates: np.ndarray,
                                   test_seq: np.ndarray, test_dates: np.ndarray,
                                   latent_train: np.ndarray, latent_test: np.ndarray,
                                   test_prob: np.ndarray, test_anomaly: np.ndarray,
                                   channel_ids: Union[list, None], title: str):
    fig = make_subplots(rows=3, cols=2,
                        column_widths=[0.7, 0.3],
                        subplot_titles=("Train Sequence", "Train Latent Space",
                                        "Test Sequence", "Test Latent Space",
                                        "Test probability", ""),
                        vertical_spacing=0.11,
                        # shared_xaxes="columns",
                        )
    # 6 colorblind save colors
    colors = ["#E69F00", "#56B4E9", "#009E73",
              "#0072B2", "#D55E00", "#CC79A7",
              "#F0E442"]
    # colors = ["blue", "red", "green", "orange", "purple", "brown"]
    # plot train and test sequences
    for i in range(train_seq.shape[1]):
        name = f"Train ch.: {channel_ids[i] if channel_ids is not None else i}"
        if train_seq.shape[1] > len(colors):
            fig.add_trace(go.Scatter(x=train_dates, y=train_seq[:, i], mode='lines', name=name,
                                     opacity=0.7, xaxis="x1"), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=train_dates, y=train_seq[:, i], mode='lines', name=name,
                                     line=dict(color=colors[i]), opacity=0.7, xaxis="x1"),
                          row=1, col=1)
    for i in range(test_seq.shape[1]):
        name = f"Test ch.: {channel_ids[i] if channel_ids is not None else i}"
        if test_seq.shape[1] > len(colors):
            fig.add_trace(go.Scatter(x=test_dates, y=test_seq[:, i], mode='lines', name=name,
                                     opacity=0.7, xaxis="x1"), row=2, col=1)
        else:
            fig.add_trace(go.Scatter(x=test_dates, y=test_seq[:, i], mode='lines', name=name,
                                     line=dict(color=colors[i]), opacity=0.7, xaxis="x1"),
                          row=2, col=1)

    # plot test probability and anomaly
    fig.add_trace(
        go.Scatter(x=test_dates, y=test_prob, mode='lines', name="Test Prob.",
                   line=dict(color="blue"), xaxis="x1"),
        row=3, col=1)
    fig.add_trace(
        go.Scatter(x=test_dates, y=test_anomaly, mode='lines', name="Anomaly Label",
                   line=dict(color="red"), opacity=0.6, xaxis="x1"),
        row=3, col=1)

    # plot latent spaces
    latent_train_indexes = list(range(latent_train.shape[0]))
    fig.add_trace(
        go.Scatter(x=latent_train[:, 0], y=latent_train[:, 1], mode='markers', name="Train U",
                   marker=dict(color=latent_train_indexes, colorscale='Viridis',
                               showscale=True, colorbar=dict(len=0.3, y=0.3)),
                   showlegend=False),
        row=1, col=2)
    latent_test_indexes = list(range(latent_test.shape[0]))
    non_anomaly_flagged = np.where(test_anomaly == 0)[0]
    fig.add_trace(
        go.Scatter(x=latent_test[non_anomaly_flagged, 0], y=latent_test[non_anomaly_flagged, 1],
                   mode='markers', name="Test U",
                   marker=dict(color=latent_test_indexes, colorscale='Viridis'),
                   showlegend=False),
        row=2, col=2)
    if latent_test.shape[1] >= 3:
        fig.add_trace(
            go.Scatter(x=latent_test[non_anomaly_flagged, 2], y=latent_test[non_anomaly_flagged, 3],
                       mode='markers', name="Test U",
                       marker=dict(color=latent_test_indexes, colorscale='Viridis'),
                       showlegend=False, xaxis="x2", yaxis="y2"),
            row=3, col=2)
    ## add data points in red if they are flagged as anomalies
    for i in np.where(test_anomaly == 1)[0]:
        fig.add_trace(
            go.Scatter(x=[latent_test[i, 0]], y=[latent_test[i, 1]], mode='markers', name="Anomaly",
                       marker=dict(color="red", symbol="triangle-up", size=10),
                       opacity=0.8,
                       showlegend=False),
            row=2, col=2)
        if latent_test.shape[1] >= 3:
            fig.add_trace(
                go.Scatter(x=[latent_test[i, 2]], y=[latent_test[i, 3]], mode='markers', name="Anomaly",
                           marker=dict(color="red", symbol="triangle-up", size=10),
                           opacity=0.8,
                           showlegend=False),
                row=3, col=2)

    fig.update_layout(title=dict(text=title, y=0.998))
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    if latent_test.shape[1] < 2:
        fig.update_xaxes(showticklabels=True, row=2, col=2)

    fig.update_xaxes(title_text="Timestamp", row=3, col=1)

    fig.update_xaxes(title_text="Latent Dim. 1", row=1, col=2)
    fig.update_yaxes(title_text="Latent Dim. 2", row=1, col=2)
    fig.update_xaxes(title_text="Latent Dim. 1", row=2, col=2)
    fig.update_yaxes(title_text="Latent Dim. 2", row=2, col=2)
    if latent_test.shape[1] >= 3:
        fig.update_xaxes(title_text="Latent Dim. 3", row=3, col=2)
        fig.update_yaxes(title_text="Latent Dim. 4", row=3, col=2)

    # change ticks on colorbar and add title
    fig.update_traces(marker=dict(colorbar=dict(tickvals=[20, latent_train.shape[0] - 21],
                                                ticktext=["Start", "End"],
                                                title="Sequence")),
                      row=1, col=2)
    fig.update_xaxes(
        title_standoff=6
    )
    # fig.show()
    return fig
