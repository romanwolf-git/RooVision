// Data, i.e. species mAP50 and mAP50-95
var roo_species = ['All', 'Bridled-nail-tail wallaby', 'Brush-tailed rock-wallaby', 'Eastern grey kanga-roo', 'Red kangaroo', 'Red-necked wallaby', 'Swamp wallaby', 'Western grey kan-garoo'];
var map50 = [0.926, 0.917, 0.990, 0.904, 0.938, 0.899, 0.989, 0.846];
var map50_95 = [0.784, 0.853, 0.800, 0.785, 0.793, 0.714, 0.826, 0.720];

// Create traces
var trace_map50 = {
    x: roo_species,
    y: map50,
    name: 'mAP50',
    type: 'bar',
    marker: {
        color: '#2e3d49'
    }
};

var trace_map50_95 = {
    x: roo_species,
    y: map50_95,
    name: 'mAP50-95',
    type: 'bar',
    marker: {
        color: '#246473'
    }
};

var data = [trace_map50, trace_map50_95];

var barWidth = 1 / roo_species.length;

// Set layout options
var layout = {
    xaxis: {
        showticklabels: false // Hide tick labels under x-axis
    },
    yaxis: {
        title: 'mean Average Precision (mAP)'
    },
    barmode: 'group',
    autosize: true, // Automatically adjust size to content
    responsive: true, // Make plot responsive to viewport

    legend: {
        orientation: "h", // Horizontal orientation for legend
        yanchor: "bottom", // Anchor legend to bottom of plot
        y: 1, // Adjust position of legend under the plot
        xanchor: "left",
        x: 0
    },
    annotations: roo_species.map((className, index) => ({
        x: index - 1.5*barWidth,
        y: 0,
        xanchor: 'center',
        yanchor: 'bottom',
        xref: 'x',
        yref: 'y',
        text: className,
        showarrow: false,
        textangle: -90,
        font: {
            size: 10,
            color: 'white'
        }
    })),
    margin: {
        l: 50, // Left margin
        r: 50, // Right margin
        b: 50, // Bottom margin
        t: 0, // Top margin
        pad: 0 // Padding between plot and container
    }
};

// Plot the graph
Plotly.newPlot('plot', data, layout);
