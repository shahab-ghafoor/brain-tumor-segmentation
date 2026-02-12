# Brain Tumor Auto-Segmentation in Magnetic Resonance Imaging (MRI)

## Project summary

This repository implements a volumetric deep-learning pipeline for multi-class brain-tumor segmentation from multi-sequence MRI (FLAIR, T1, T1-GD, T2). The approach treats segmentation as a per-voxel multi-class problem (edema, non-enhancing tumor, enhancing tumor) and solves it using a 3D U-Net trained on randomly sampled sub-volumes (patches). The pipeline includes data loading (NIfTI), patch sampling with tumor-presence biasing, per-slice standardization, a 3D U-Net model, multi-class soft-Dice loss for optimization, and voxel-level evaluation (Dice, sensitivity, specificity).


## Key contributions / highlights

* Reproducible preprocessing that extracts spatially consistent sub-volumes of size **[160, 160, 16]** from source images of shape **(240, 240, 155, 4)** while enforcing a maximum background fraction of **95%** per patch.
* Channel-first input format for 3D CNNs: `(num_channels, X, Y, Z)` → model input `(4, 160, 160, 16)`.
* Multi-class soft-Dice loss (differentiable) and mean Dice metric for robust training under class imbalance.
* Patch generator implemented as a Keras `Sequence` (on-disk h5py patches) to enable training on datasets that exceed GPU memory.
* Tools to stitch patch predictions into whole-scan probability maps and to compute scan-level sensitivity/specificity per class.


## Data format and description

* **Input images**: NIfTI-1 files containing 4 MRI sequences concatenated as a 4D array of shape `(240, 240, 155, 4)` where sequence index ≔ {0: FLAIR, 1: T1, 2: T1-GD, 3: T2}.
* **Labels**: NIfTI files with integer per-voxel annotations `(240, 240, 155)` with values `{0: background, 1: edema, 2: non-enhancing tumor, 3: enhancing tumor}`.
* Typical dataset sources used in this implementation: BraTS / Medical Decathlon preprocessing (data provided in `data/imagesTr` and `data/labelsTr`).


## Preprocessing pipeline

1. **Load case**: `load_case(image_nifty_file, label_nifty_file)` uses `nibabel` to obtain NumPy arrays.
2. **Patch sampling**: `get_sub_volume(image, label, output_x=160, output_y=160, output_z=16, background_threshold=0.95)`

   * Randomly samples corner `(start_x, start_y, start_z)` for sub-volume.
   * One-hot encodes labels and moves channels to front.
   * Excludes the background channel from model targets (resulting `y` shape: `(3, 160,160,16)`).
3. **Standardization**: `standardize(image)`—per channel and per Z-slice standardization to mean 0 and std 1 (guarding against zero std).
4. **Patch storage**: Collected patches saved in h5py files and referenced by a `config.json` for generators.


## Model architecture

* **Backbone**: 3D U-Net (encoder–decoder with skip connections). Example input shape used in `util.unet_model_3d`: `(None, 4, 160, 160, 16)` (batch dimension first).
* **Final layer**: 3 feature maps corresponding to the three abnormality classes (no background channel).
* **Parameters**: ~16 million trainable parameters (architecture summary provided by `model.summary()` in the notebook).
* **Activation**: final per-voxel sigmoid/probability outputs (for thresholding at inference).

Key references: U-Net and its 3D variants, for details see `https://arxiv.org/abs/1606.06650`.


## Loss function and metrics

* **Soft Dice Loss** (multi-class): implemented as the differentiable variant of Dice using squared sums in denominator:

  ```
  dice_numerator = 2 * Σ(y_true * y_pred) + ε
  dice_denominator = Σ(y_true^2) + Σ(y_pred^2) + ε
  loss = 1 − mean(dice_numerator / dice_denominator)
  ```
* **Evaluation metrics**:

  * Mean Dice coefficient across abnormality classes (implemented by `dice_coefficient`).
  * Per-class sensitivity and specificity implemented by `compute_class_sens_spec(pred, label, class_num)`.


## Training

* **Data loader**: `VolumeDataGenerator` implements the Keras `Sequence` interface and streams patches from `processed/train/` and `processed/valid/`.
* **Example training settings used in the assignment** (illustrative):

  ```
  batch_size = 3
  steps_per_epoch = 20
  validation_steps = 20
  epochs = 10
  model.compile(optimizer, loss=soft_dice_loss, metrics=[dice_coefficient])
  model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch,
                      epochs=n_epochs, validation_data=valid_generator,
                      validation_steps=validation_steps, use_multiprocessing=True)
  ```
* **Pretrained weights**: `model_pretrained.hdf5` may be provided for experiments that skip long training runs.


## Inference and whole-scan reconstruction

* **Patch prediction**: Add batch dimension and run `model.predict` on standardized patches. Apply threshold (default 0.5) to convert probabilities into binary maps.
* **Stitching patches**: `util.predict_and_viz(image, label, model, threshold, loc)` generates overlapping patches across the full volume, aggregates probabilities, thresholds, and produces whole-scan predictions and visualizations.
* **Postprocessing**: Probabilistic aggregation, optional connected-component filtering, and optional class-specific morphological operations.


## Example usage (high level)

```python
from util import load_case, get_sub_volume, standardize, unet_model_3d, VolumeDataGenerator, predict_and_viz
# Load
image, label = load_case('data/imagesTr/BRATS_003.nii.gz', 'data/labelsTr/BRATS_003.nii.gz')
# Sample patch
X, y = get_sub_volume(image, label)
X_norm = standardize(X)
# Load model and weights
model = unet_model_3d(loss_function=soft_dice_loss, metrics=[dice_coefficient])
model.load_weights('weights/model_pretrained.hdf5')
# Predict and visualize
pred = predict_and_viz(image, label, model, threshold=0.5, loc=(130,130,77))
```

---

## Results (representative / expected from assignment)

* Example validation run (expected approximate values reported in the assignment):

  * **Validation soft Dice loss** ≈ **0.4742**
  * **Validation mean Dice coefficient** ≈ **0.5152**
* Example patch-level sensitivity/specificity for a representative scan:

  ```
               Edema  Non-Enhancing  Enhancing
  Sensitivity  0.9085     0.9505      0.7891
  Specificity  0.9848     0.9961      0.9960
  ```
* Whole-scan sensitivity/specificity (example scan):

  ```
               Edema  Non-Enhancing  Enhancing
  Sensitivity  0.902   0.2617         0.8496
  Specificity  0.9894  0.9998         0.9982
  ```

> Note: These figures were obtained with the predefined patch-sampling, model architecture, and the supplied pretrained weights. Variations are expected due to stochastic patch selection and training randomness.


## Limitations and possible extensions

* The patch-based training strategy reduces memory but can miss long-range 3D context; evaluating 3D cell-based or full-volume approaches is advised when resources permit.
* Class imbalance remains challenging for small, sparse tumor classes; consider hybrid losses (Dice + cross-entropy), focal Dice, or class-adaptive sampling.
* Ensemble strategies (multi-scale or model ensembles) and 3D pretraining (MedicalNet, self-supervised) can improve robustness.
* Postprocessing (conditional random fields, connected component filtering) may reduce false positives.

---

## References

* Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI 2015.*
* Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016). 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. *MICCAI 2016.*
* Menze, B. H., et al. (2015). The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS). *IEEE Trans. Med. Imaging.*
* Simpson, A. L., et al. (2019). A large annotated medical image dataset for the development and evaluation of segmentation algorithms. *(Medical Segmentation Decathlon / associated publications).*
