# normal period
python ../gdrivedl.py "https://drive.google.com/open?id=1rVJ5ry5GG-ZZi5yI4x9lICB8VhErXwCw" input/
# anomalies
python ../gdrivedl.py "https://drive.google.com/open?id=1iDYc0OEmidN712fquOBRFjln90SbpaE7" input/
# original excel files
python ../gdrivedl.py "https://drive.google.com/drive/folders/1zkPNPgZ0fIneML21KlLnnkOBuL8X7qK7" input/

cp input/*.csv output/