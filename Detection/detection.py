import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load speckle images
speckle_imgs = np.load("Detection/speckle_imgs.npy")

num_iterations = speckle_imgs.shape[0]
reference_count = 15  # Increased reference count
class_size = 25       # Narrower class definition

false_alarms = 0
misses = 0
total_H0 = 0
total_H1 = 0
all_true_labels = []
all_predicted_labels = []

# Flatten PCA data for correlation
def flatten(imgs):
    return imgs.reshape(imgs.shape[0], -1)

# Normalize vectors
def normalize(v):
    return (v - np.mean(v)) / np.std(v)

# Correlate a sample against reference set, return average correlation
def average_correlation(sample, references):
    sample = normalize(sample)
    corrs = [np.mean(sample * normalize(ref)) for ref in references]
    return np.mean(corrs)

for iteration in range(num_iterations):
    data = speckle_imgs[iteration, :100]  # Use only first 100 wavelengths (400–499nm)
    flattened = flatten(data)

    # Define new indices for narrower class bands
    indices_H0 = np.arange(0, 25)         # 400–424 nm
    indices_H1 = np.arange(75, 100)       # 475–499 nm

    # Sample references
    ref_H0_idx = np.random.choice(indices_H0, reference_count, replace=False)
    ref_H1_idx = np.random.choice(indices_H1, reference_count, replace=False)
    ref_H0 = flattened[ref_H0_idx]
    ref_H1 = flattened[ref_H1_idx]

    # Test samples from the same narrower bands
    for i in np.concatenate((indices_H0, indices_H1)):
        sample = flattened[i]
        label = 0 if i in indices_H0 else 1

        corr_H0 = average_correlation(sample, ref_H0)
        corr_H1 = average_correlation(sample, ref_H1)
        predicted = 0 if corr_H0 > corr_H1 else 1

        all_true_labels.append(label)
        all_predicted_labels.append(predicted)

        if label == 0:
            total_H0 += 1
            if predicted == 1:
                false_alarms += 1
        elif label == 1:
            total_H1 += 1
            if predicted == 0:
                misses += 1

# Compute probabilities
P_FA = false_alarms / total_H0
P_M = misses / total_H1

# Display results
print(f"False Alarm Probability (P_FA): {P_FA:.4f}")
print(f"Miss Probability (P_M): {P_M:.4f}")

# Plot confusion matrix
cm = confusion_matrix(all_true_labels, all_predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["H0", "H1"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
