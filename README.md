# four_shapes_classifier
Step by step classification CNN of shapes images (square, circle, star, triangle). Written in TensorFlow/Keras and Python 3.7.

## Dataset

- Training dataset: https://www.kaggle.com/smeschke/four-shapes (consists of 200x200 images of shapes labelled as either circle, triangle, star or square.
- Bonus test images (very different from training data) were added by me: see subfolder /data.

## Code

Using Python 3 and Tensorflow 2.

- Baseline CNN: very basic CNN with 2 conv. layers, no regularization.
- Regularized CNN: added data augmentation strategy and dropout to baseline.
- Use the [notebook](https://github.com/alzaia/four_shapes_classifier/blob/main/four_shapes_classifier_notebook.ipynb) to run all the main code and visualize results.

