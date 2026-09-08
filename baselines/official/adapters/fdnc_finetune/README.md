# Official fDNC fine-tuning

This directory fine-tunes the authors' untouched `NIT_Registration` source from:

`baselines/official/third_party/fdnc_official/src/model.py`

Initialization is the authors' downloaded:

`baselines/official/third_party/fdnc_official/model/model.bin`

SHA256 locked for this benchmark:

`ab529eb6a886cb6ab3f199b7aaa4e49b82562dc280b3ac736ed61e25d5138ec9`

## What is and is not changed

Unchanged:
- official `NIT_Registration` architecture;
- official `forward()` implementation;
- official `p_m` matching distribution;
- official outlier head;
- official pretrained initialization.

Project adapter responsibilities:
- convert unified NPZs to XYZ point sets;
- construct the `match_dict` expected by official `forward()`;
- fine-tune selected Transformer layers;
- validation-only early stopping;
- common direct Top-1 / MRR / Hungarian evaluation.

The official source itself is never edited.

## Training protocol

For each dataset × outer fold × seed:

1. initialize from official `model.bin`;
2. use only that fold's `seed_<seed>/train.txt`;
3. train 4 predeclared candidates:

   - last 1 layer, LR 5e-6
   - last 1 layer, LR 1e-5
   - last 2 layers, LR 5e-6
   - last 2 layers, LR 1e-5

4. choose the candidate by validation direct Top-1 (MRR tie-break only);
5. only after selection, evaluate the chosen checkpoint on `fold_<k>/test.txt`.

The training script has no test-list argument.

## Recommended smoke test

```bash
cd /home/ubuntu/klb/nuclr/nuclr
conda activate nuclr310

CUDA_VISIBLE_DEVICES=0 python -u \
  baselines/official/adapters/fdnc_finetune/train_fdnc_official_finetune.py \
  --dataset atanas \
  --fold 1 \
  --seed 42 \
  --save-dir baselines/official/runs/fdnc_smoke \
  --unfreeze-last-n 2 \
  --backbone-lr 5e-6 \
  --epochs 2 \
  --pairs-per-epoch 10 \
  --patience 2 \
  --device cuda:0
```

If that succeeds, delete the smoke directory and run the full grid:

```bash
rm -rf baselines/official/runs/fdnc_smoke

GPUS="0 1" bash \
  baselines/official/adapters/fdnc_finetune/run_fdnc_official_finetune_cv5x3.sh
```
