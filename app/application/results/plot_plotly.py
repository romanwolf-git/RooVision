import plotly.graph_objects as go
import numpy as np

# Data for test set
test_classes = ['All', 'Macropus fuliginosus', 'Macropus giganteus', 'Notamacropus rufogriseus', 'Onychogalea fraenata', 'Ospranter rufus', 'Petrogale penicillata', 'Wallabia bicolor']
test_average_precision = [90, 96, 67, 99, 100, 90, 83, 97]

# Data for validation set (including a placeholder value for 'Onychogalea fraenata')
validation_average_precision = [94, 95, 85, 0, 95, 100, 96, 0]

# Create the bar plot
fig = go.Figure()

# Plot data for test set
fig.add_trace(go.Bar(
    x=test_classes,
    y=test_average_precision,
    name='Test Set',
    marker_color='#2e3d49'
))

# Plot data for validation set
fig.add_trace(go.Bar(
    x=test_classes,
    y=validation_average_precision,
    name='Validation Set',
    marker_color='#246473'
))

# Add text annotations for each bar
for i in range(len(test_classes)):
    fig.add_annotation(
        x=test_classes[i],
        y=test_average_precision[i],
        text=str(test_average_precision[i]),
        font=dict(color='#ffffff'),
        showarrow=False
    )
    fig.add_annotation(
        x=test_classes[i],
        y=validation_average_precision[i],
        text=str(validation_average_precision[i]),
        font=dict(color='#ffffff'),
        showarrow=False
    )

# Update layout
fig.update_layout(
    title='Average Precision by Class (Test vs Validation)',
    xaxis_title='Classes',
    yaxis_title='Average Precision (%)',
    barmode='group',
    xaxis_tickangle=-90
)

# Show plot
fig.show()
