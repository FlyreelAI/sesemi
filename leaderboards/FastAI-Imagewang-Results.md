# FastAI Imagewang Results

Our Approach:

* We jointly train the supervised task on labeled data with a self-supervised pretext task of [predicting image rotations](https://arxiv.org/abs/1803.07728) on unlabeled data for semi-supervised learning.
* We use a standard training protocol: vanilla ResNet-50 backbone, conventional data augmentation (random cropping, gamma correction, horizontal flipping, mean-variance normalization), Nesterov SGD optimization, and simple polynomial learning rate decay with warm-up.
* No tricks, no advanced techniques, no pretrained weights; we simply add unlabeled data for improved accuracy performance with *self-supervised regularization*.
* We don't excessively tune hyper-parameters on a per-dataset or per-experiment basis beyond the traditional set for training CNNs, namely just learning rate.

Our Results:

In practical semi-supervised applications, one would use all available examples (training, validation, and even test) as unlabeled data. When trained with the validation set included as unlabeled examples using ResNet-50, we report the following results on the [FastAI leaderboard](https://github.com/fastai/imagenette#image%E7%BD%91-leaderboard) for the Imagewang dataset as produced by our model at the final epoch of training.

* 78.41% &pm; 0.39 for 80 epochs over five runs
* 79.27% &pm; 0.20 for 200 epochs over three runs

We also report slightly lower accuracy performance without including the validation set as part of unlabeled data examples:

* 77.61% &pm; 0.34 for 80 epochs over five runs
* 77.79% &pm; 0.38 for 200 epochs over three runs

## Usage

* Download the full size [Imagewang dataset](https://s3.amazonaws.com/fast-ai-imageclas/imagewang.tgz) and extract it to a directory of your choice
* In the `train` directory of Imagewang, **remove all sub-directories except** for those also found in the `val` directory. In other words, we **keep** 10 sub-directories found in both `train` and `val`, namely:
  * `n02086240`
  * `n02087394`
  * `n02088364`
  * `n02089973`
  * `n02093754`
  * `n02096294`
  * `n02099601`
  * `n02105641`
  * `n02111889`
  * `n02115641`
* Modify and run the following command to reproduce our results (as of commit `de60e31`) on the [FastAI leaderboard](https://github.com/fastai/imagenette#image%E7%BD%91-leaderboard) for the Imagewang dataset in the 256 size category:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -u open_sesemi.py \
  --data-dir </path/to/imagewang/> \
  --unlabeled-dir </path/to/imagewang/unsup/> \
  --lr 0.1 --optimizer sgd --batch-size 32 \
  --epochs <80,200> --warmup-lr 0.001 --warmup-epochs 10 \
  --backbone resnet50 --run-id imagewang_run01
```

* The above run configuration requires two GPUs with 12GB of video memory each and takes between 6 and 18 hours to complete. Ideas to reduce training time while maintaining comparable accuracy performance are welcome!

## Ideas for Further Exploration

* Train with additional unlabeled data from Imagenette
* Does a better backbone help improve performance? What about advanced optimizers, robust data augmentation techniques, and other [tricks of the trade](https://arxiv.org/abs/1812.01187)?
* What about another self-supervised learning algorithm?
