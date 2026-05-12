
import numpy as np
import matplotlib.pyplot as plt


def denormalize(image, device, mean_param=[0.485, 0.456, 0.406], std_param=[0.229, 0.224, 0.225]):
    image = image.to(device)
    image = image.numpy().transpose((1, 2, 0))
    
    mean = np.array(mean_param)
    std = np.array(std_param)
    
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image

def visualize_samples(dataloader_train, device):
  
  # Visualize one example for each class for 10 classes
  fig, axes = plt.subplots(2, 5, figsize=(15, 6))
  classes_sampled = []
  found_classes = 0

  for i, (inputs, classes) in enumerate(dataloader_train):

    for j in range(len(classes)):
      label = classes[j].item()

      if label not in classes_sampled:
        row = found_classes // 5
        col = found_classes % 5

        img = denormalize(inputs[j], device) # Use inputs[j] to get the image corresponding to the label
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"Class {label}")
        axes[row, col].axis("off")

        classes_sampled.append(label)
        found_classes += 1

        if found_classes == 10:
          break
    if found_classes == 10:
      break

  plt.tight_layout()
  plt.show()