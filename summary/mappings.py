set_groups = {
    "sine (16)": ["-sine-"],
    "ecg-cbf-rw (20)": ["-ecg-", "-cbf-", "-rw-"],
    "increasing (13)": ["-increasing-", "_increasing-", "_increasing_"],
    "saw (2)": ["-saw-"],
    "wave (14)": ["-wave", "_wave"],
    "corr (5)": ["-corr-"],
    "lotka_volterra (4)": ["-lotka_volterra-"]
}

set_group_amount_groups = {
    "sine": "16",
    "ecg-cbf-rw": "20",
    "increasing": "13",
    "saw": "2",
    "wave": "14",
    "corr": "5",
    "lotka_volterra": "4"
}

anomaly_groups = ['platform', 'mean', 'frequency', 'pattern', 'pattern-shift', 'amplitude', 'extremum', 'variance',
                  'trend', 'signal-cancellation', 'signal-reset', 'signal-cut', 'signal-cut-match', 'disconnect',
                  'nature']

method_maps = {0: 'RealNVP',
               1: 'tcNF-b',
               4: 'tcNF-c',
               2: 'tcNF-l',
               3: 'tcNF-f'}

method_order = {0: 0, 1: 1, 4: 2, 2: 3, 3: 4}

baseline_method_maps = {
    "autoencoder": {"supervised": True, "unsupervised": True, "name": "AE"},
    "cblof": {"supervised": False, "unsupervised": True, "name": "CBLOF"},
    "cof": {"supervised": False, "unsupervised": True, "name": "COF"},
    "copod": {"supervised": False, "unsupervised": True, "name": "COPOD"},
    "dae": {"supervised": True, "unsupervised": True, "name": "DAE"},
    "damp": {"supervised": True, "unsupervised": True, "name": "DAMP"},
    "deepant": {"supervised": True, "unsupervised": True, "name": "DeepAnT"},
    "deepnap": {"supervised": True, "unsupervised": True, "name": "DeepNAP"},
    "encdec_ad": {"supervised": True, "unsupervised": True, "name": "EncDec-AD"},
    "fast_mcd": {"supervised": True, "unsupervised": True, "name": "Fast-MCD"},
    "gdn": {"supervised": False, "unsupervised": True, "name": "GDN"},
    "hbos": {"supervised": False, "unsupervised": True, "name": "HBOS"},
    "health_esn": {"supervised": True, "unsupervised": True, "name": "Health-ESN"},
    "hif": {"supervised": True, "unsupervised": False, "name": "HIF"},
    "hybrid_knn": {"supervised": True, "unsupervised": True, "name": "HybridKNN"},
    "if_lof": {"supervised": False, "unsupervised": True, "name": "IF-LOF"},
    "iforest": {"supervised": False, "unsupervised": True, "name": "iForest"},
    "kmeans": {"supervised": False, "unsupervised": True, "name": "k-Means"},
    "knn": {"supervised": False, "unsupervised": True, "name": "KNN"},
    "laser_dbn": {"supervised": True, "unsupervised": True, "name": "LaserDBN"},
    "lstm-vae": {"supervised": True, "unsupervised": True, "name": "LSTM-VAE"},
    "lstm_ad": {"supervised": True, "unsupervised": True, "name": "LSTM-AD"},
    "mscred": {"supervised": True, "unsupervised": True, "name": "MSCRED"},
    "mstamp": {"supervised": False, "unsupervised": True, "name": "MSTAMP"},
    "mtad_gat": {"supervised": True, "unsupervised": True, "name": "MTAD-GAT"},
    "multi_hmm": {"supervised": True, "unsupervised": False, "name": "MultiHMM"},
    "multi_subsequence_lof": {"supervised": False, "unsupervised": True, "name": "LOF"},
    "normalizing_flows": {"supervised": True, "unsupervised": False, "name": "NF"},
    "omni": {"supervised": True, "unsupervised": True, "name": "OmniAnomaly"},
    "pca": {"supervised": False, "unsupervised": True, "name": "PCA"},
    "pcc": {"supervised": False, "unsupervised": True, "name": "PCC"},
    "random_black_forest": {"supervised": True, "unsupervised": True, "name": "RBForest"},
    "robust_pca": {"supervised": True, "unsupervised": True, "name": "RobustPCA"},
    "subsequence_knn": {"supervised": False, "unsupervised": True, "name": "Subsequence KNN"},
    "tanogan": {"supervised": True, "unsupervised": True, "name": "TAnoGan"},
    "telemanom": {"supervised": True, "unsupervised": True, "name": "Telemanom"},
    "torsk": {"supervised": False, "unsupervised": True, "name": "Torsk"},
    "usad": {"supervised": True, "unsupervised": True, "name": "USAD"}
}

metric_maps = {'VUS_ROC': 'VUS',
               'AUC_ROC': 'AUC'}
metric_order = {'VUS_ROC': 0, 'AUC_ROC': 0.5}

real_set_groups = [['SMD', 'smd_'],
                   ['SWaT', 'SWat'],
                   ['CalIt2', 'calit'],
                   ['GHL', 'ghl'],
                   ['Metro', 'metro'],
                   ['Occupancy', 'occupancy-']]


optimization_mapping = {
    "gen": "Generation",
    "can_id": "Candidate ID",
    "opt_goal": "Optim. Goal",
    "test_auc": "Test AUC",
    "test_vus": "Test VUS",
    "hidden_multiplier": "ST Multiplier",
    "st_net_layers": "ST Layers",
    "st_dropout": "ST Dropout",
    "st_funnel_factor": "ST Funnel",
    "coupling_layers": "Coupling Layers",
    "past": "Past",
    "encoder_layers": "Enc. Layers",
    "encoder_size": "Enc. Size",
    "encoder_dropout": "Enc. Dropout",
    "encoder_compression_factor": "Enc. Funnel",
    "encoder_channel_depth": "Enc. Depth",
}