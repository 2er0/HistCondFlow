wget https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip
rm -rf output && mkdir output
mv metro+interstate+traffic+volume.zip output
cd output
unzip metro+interstate+traffic+volume.zip && rm metro+interstate+traffic+volume.zip
gunzip Metro_Interstate_Traffic_Volume.csv.gz