import matplotlib.pyplot as plt
import numpy as np

# Data for test set
test_classes = ['All', 'Macropus fuliginosus', 'Macropus giganteus', 'Notamacropus rufogriseus', 'Onychogalea fraenata', 'Ospranter rufus', 'Petrogale penicillata', 'Wallabia bicolor']
test_average_precision = [90, 96, 67, 99, 100, 90, 83, 97]

# Data for validation set (including a placeholder value for 'Onychogalea fraenata')
validation_average_precision = [94, 95, 85, 0, 95, 100, 96, 0]

# Create the bar plot
plt.figure(figsize=(10, 6))

bar_width = 0.35
# Calculate the position for the bars
index = np.arange(len(test_classes))
# Plot data for test set
test_bars = plt.bar(index - bar_width/2, test_average_precision, color='#2e3d49', label='Test Set', width=bar_width)

# Add text annotations for each bar in the test set
for i, bar in enumerate(test_bars):
    plt.text(bar.get_x() + bar.get_width()/2, 2, str(test_classes[i]), ha='center', va='bottom', rotation=90, color='#ffffff')
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height()-4, str(test_average_precision[i]), ha='center', va='bottom',
             color='#ffffff')

#
# Plot data for validation set
validation_bars = plt.bar(index + bar_width/2, validation_average_precision, color='#246473', label='Validation Set', width=bar_width)
for i, bar in enumerate(validation_bars):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height()-4, str(validation_average_precision[i]), ha='center', va='bottom',
             color='#ffffff')

# Add labels and title
plt.ylabel('Average Precision (%)')
plt.title('Average Precision by Class (Test vs Validation)')
plt.legend()

# Rotate x-axis labels for better readability
plt.xticks([])

# Show plot
plt.tight_layout()
plt.show()
