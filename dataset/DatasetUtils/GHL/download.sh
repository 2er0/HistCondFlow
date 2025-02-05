wget https://github.com/n1tk/researchLSTM/raw/master/Data/GHL/train_1500000_seed_11_vars_23.csv.zip
rm -rf output && mkdir output
mv train_1500000_seed_11_vars_23.csv.zip output
cd output
unzip train_1500000_seed_11_vars_23.csv.zip && rm train_1500000_seed_11_vars_23.csv.zip