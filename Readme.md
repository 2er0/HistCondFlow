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

Full result tables are available in the `summary` folder. 
To generate the tables and plots, run the following commands:

```bash
cd summary
python plots_tables.py
python latent_space_plots.py
```

## Models

Some trained models are available in the `results` folder.