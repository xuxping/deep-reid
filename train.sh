# train with Softmax cross-entropy loss and triplet loss
python train_imgreid_xent_htri.py -s market1501  -t market1501  --height 128 --width 64 --optim adam  --label-smooth  --lr 0.003  --max-epoch 60  --stepsize 20 40 --train-batch-size 128 --test-batch-size 128 -a resnet50 --eval-freq 20 --save-dir log/market1501-xent-htri

# train with Softmax cross-entropy loss
# python train_imgreid_xent.py -s market1501  -t market1501  --height 128 --width 64 --optim adam  --label-smooth  --lr 0.003  --max-epoch 60  --stepsize 20 40 --train-batch-size 128 --test-batch-size 128 -a resnet50 --eval-freq 20 --save-dir log/market1501-xent
