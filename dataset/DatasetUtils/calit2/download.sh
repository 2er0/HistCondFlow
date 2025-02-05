wget https://archive.ics.uci.edu/static/public/156/calit2+building+people+counts.zip
rm -rf output && mkdir output
mv calit2+building+people+counts.zip output
cd output
unzip calit2+building+people+counts.zip && rm calit2+building+people+counts.zip