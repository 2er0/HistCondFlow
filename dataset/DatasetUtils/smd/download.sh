wget https://github.com/NetManAIOps/OmniAnomaly/archive/refs/heads/master.zip && unzip master.zip && rm master.zip
rm -rf input/*
cd OmniAnomaly-master && mv ServerMachineDataset ../input/
cd ..
rm -rf OmniAnomaly-master
cp -R input/ServerMachineDataset/* output