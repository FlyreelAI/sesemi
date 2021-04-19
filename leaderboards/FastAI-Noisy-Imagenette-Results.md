# FastAI Noisy Imagenette Results

Our Approach:

* We use the same training protocol as with semi-supervised learning on the Imagewang dataset.
* We find SESEMI is robust to label noise and corruption by way of using unlabeled data as a guiding signal. This [NeurIPS 2019 paper](https://arxiv.org/abs/1906.12340) provides some insight and analysis.

We report the following results at the final epoch of training.

* Imagenette with 5% Label Noise - 256 size
  * 88.87% &pm; 0.67 for 5 epochs over five runs
  * 92.95% &pm; 0.12 for 80 epochs over three runs
  * 93.96% &pm; 0.23 for 200 epochs over three runs
* Imagenette with 50% Label Noise - 256 size
  * 76.72% &pm; 0.83 for 5 epochs over five runs
  * 57.76% &pm; 0.39 for 80 epochs over three runs
  * 61.48% &pm; 0.33 for 200 epochs over three runs

Interestingly, we find overfitted performance when training for 20 epochs using standard SGD and Adam optimizers. Looking at the baseline results for 20 epochs, the Ranger optimizer seems to work well in this setting. An implementation of the Ranger optimizer for SESEMI is most welcome.

## Usage

* Download the full size [Imagenette dataset](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz) and extract it to a directory of your choice
* Run this code snippet to create `noisy5` and `noisy50` training datasets:

```python
import os
import shutil
import pandas as pd

noise_level = '5' # '5' or '50'
outdir = 'noisy' + noise_level
df = pd.read_csv('noisy_imagenette.csv', sep=',')
for inpath, label, is_val in zip(df['path'], df['noisy_labels_' + noise_level], df['is_valid']):
    if is_val:
        subdir = 'val'
    else:
        subdir = 'train'
    outpath = os.path.join(outdir, subdir, label)
    os.makedirs(outpath, exist_ok=True)
    shutil.copy2(inpath, outpath)
```

* Modify and run the following command to reproduce our results (as of commit `de60e31`) on the [FastAI leaderboard](https://github.com/fastai/imagenette#imagenette-wlabel-noise--5) for the noisy Imagenette dataset in the 256 size category:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -u open_sesemi.py \
  --data-dir </path/to/noisy/imagenette/> \
  --lr 0.001 --optimizer adam --batch-size 24 \
  --epochs 5 --warmup-lr 0.001 --warmup-epochs 0 \
  --backbone resnext50_32x4d --run-id imagenette_run01
```

```bash
CUDA_VISIBLE_DEVICES=0,1 python -u open_sesemi.py \
  --data-dir </path/to/noisy/imagenette/> \
  --lr 0.1 --optimizer sgd --batch-size 32 \
  --epochs <80,200> --warmup-lr 0.001 --warmup-epochs 10 \
  --backbone resnet50 --run-id imagenette_run01
```

* The above run configuration requires two GPUs with 12GB of video memory each and takes between 1 hour for 5 epochs and 20 hours for 200 epochs.
