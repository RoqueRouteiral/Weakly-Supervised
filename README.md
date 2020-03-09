# Weakly-Supervised

In this project we aim to explore a weakly supervised approach for tumor lozalization.


### Main Idea
* 2D Lesions on brain from BRATS dataset.
* Creating classification labels from segmentation masks.
* Training SoA network for classification.
* Applying Gradcam like methods for localizing (+BBox).
* Evaluating with IoU like metrics.

### Dataset

Modified BRATS. To Do

### To Do
[x] Create simple dataset
[x] Train simple classification network
[] Get to good performance
[] Add data augmentation
[] Add grad cam or similar
[] Add bboxes generator and evaluation for detection task

### Future
* Upgrade to 3D