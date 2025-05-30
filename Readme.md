# HistCondFlow 

## Model Training

### SLURM (full experiment)

```bash
sh slurm_manage.sh
```

### Local (quick experiment)

```bash
python execute_run_opt_only.py
```

## Tables and Plots

* Full result tables are available in the `tables` [directory](./tables). 
* Plots of the latent space and the training curves are available in the `plots` [directory](./plots).
* Experiment tables are available in the `summary` directory.
  * FSB: [summary/fsb/results-TFselfopt-fsb.csv](./summary/fsb/results-TFselfopt-fsb.csv)
  * SRB: [summary/srb/results-TFselfopt-srb.csv](./summary/srb/results-TFselfopt-srb.csv)
  * Real: [summary/real/results-TFselfopt-real.csv](./summary/real/results-TFselfopt-real.csv)
* To generate the tables and plots, run the following commands in the `summary` [directory](./summary):

```bash
cd summary
python plots_tables.py
python latent_space_plots.py
```

## Models

Some trained models are available in the `results` [folder](./models).