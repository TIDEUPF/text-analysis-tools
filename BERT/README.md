
## Getting started

python train.py --train ../code/dataTraining.csv

python test.py --test dataTest.csv --model ../code/path to checkpoint (.ckpt)


# Predicting by sentences
python test.py --test predictSet.csv --model ../code/lightning_logs/path to checkpoint (.ckpt)
