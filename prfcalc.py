from PIL import Image
import numpy as np
import sklearn.metrics
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
# import tensorflow as tf
import pandas as pd

# Open the uploaded images
mask_img_gt = Image.open('C:/Users/swava/Desktop/CRACK-SAM/Dataset2/masksam_bb/CRACK500_20160222_164141_1281_1_bb.png')
mask_img_pred = Image.open('C:/Users/swava/Desktop/CRACK-SAM/Dataset2/masksam_gt/CRACK500_20160222_164141_1281_1.png')

if mask_img_gt.mode == 'RGBA':
    mask_img_gt = mask_img_gt.convert('RGB')

if mask_img_pred.mode == 'RGBA':
    mask_img_pred = mask_img_pred.convert('RGB')
# Resize the predicted mask to the same shape as the ground truth mask
mask_img_pred = mask_img_pred.resize(mask_img_gt.size)

# Convert the images to NumPy arrays
mask_gt = np.array(mask_img_gt)
mask_pred = np.array(mask_img_pred)

# Print the shape of the arrays
print(mask_gt.shape)
print(mask_pred.shape)

# Compute precision and recall
# precision = sklearn.metrics.precision_score(mask_gt.ravel(), mask_pred.ravel())
precision = precision_score(mask_gt.ravel(), mask_pred.ravel(), average='weighted')
recall = sklearn.metrics.recall_score(mask_gt.ravel(), mask_pred.ravel(), average='weighted')



f1_score = f1_score(mask_gt.ravel(), mask_pred.ravel(), average='weighted')


true_labels = pd.Series(mask_gt.flatten() )
predicted_labels = pd.Series(mask_pred.flatten() )

threshold = 0.6  # Experiment with different threshold values

predicted_labels_above_threshold = predicted_labels >= threshold

true_positives = ((true_labels == 1) & (predicted_labels_above_threshold == 1)).sum()
false_positives = ((true_labels == 0) & (predicted_labels_above_threshold == 1)).sum()
false_negatives = ((true_labels == 1) & (predicted_labels_above_threshold == 0)).sum()

print("true_positives:", true_positives)
print("false_positives:", false_positives)
print("false_negatives:", false_negatives)


precisionPD = 2*true_positives / (true_positives + false_positives)
recallPD = true_positives / (true_positives + false_negatives)
dice_coeffPD = 2*true_positives/ (2*(true_positives+false_positives+false_negatives))

print("PrecisionPD:", precisionPD)
print("RecallPD:", recallPD)
print("Dice coefficientPD:", dice_coeffPD)


# Print the results
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1_score)


def compute_dice_coefficient(mask_gt, mask_pred):

  volume_sum = mask_gt.sum() + mask_pred.sum()
  if volume_sum == 0:
    return np.NaN
  volume_intersect = (mask_gt & mask_pred).sum()
  # print("volume intersect:", volume_intersect)
  return 2*volume_intersect / volume_sum 

# Compute the Dice coefficient
dice_coeff = compute_dice_coefficient(mask_gt.ravel(), mask_pred.ravel())

# Print the result
print("Dice coefficient:", dice_coeff)
