# Using StratGAN

options are passed to the script by command line, for example, you can paint with:
```
python main.py --paint --run_dir=line6 --image_dir=multi_line_bw_128 --gf_dim=32 --df_dim=16 --paint_label=2 --paint_ncores=2
```

## options to be passed to `main.py`
 * `run_dir`, Directory run name to save/load samp, log, chkp under. If none, auto select [None])
 * `gf_dim`, Number of filters in generator [64])
 * `df_dim`, Number of filters in discriminator [64])


### training related flags
 * `train`, FTrue for training [False])
 * `epoch`, Epoch to train [5])
 * `learning_rate`,  Learning rate of for adam [0.0005])
 * `beta1`, Momentum term of adam [0.6])
 * `batch_size`, Size of batch images [64])
 * `gener_iter`, Number of times to iterate generator per batch [2])
 * `image_dir`, Root directory of dataset [multi_line_bw_128])


### painting related flags
 * `paint`, True for painting [False])
 * `paint_label`, The label to paint with)
 * `checkpoint_dir`,  Directory name to save the checkpoints [checkpoint])
 * `paint_width`, The size of the paint images to produce. If None, same value as paint_height [1000])
 * `paint_height`, The size of the paint images to produce. If None, value of paint_width/4 [None])
 * `paint_overlap`, The size of the overlap during painting [24])
 * `paint_overlap_thresh`, The threshold L2 norm error for overlapped patch areas [10.0])
 * `(paint_core_source`, Method for generating cores, if not recognized assume file name ['block'])
 * `paint_ncores`, The number of cores to generate in the painting process, [0])
 * `paint_core_thresh`, The threshold L2 norm error for overlapped core areas [2.0])


### post sampling related flags
 * `post`, True for post sampling [False])