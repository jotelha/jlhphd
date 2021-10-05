# Scripts for setting up workflows

## Mode

Several fields are used to distinguish between prduction and trial.
At least adapt the fields `description`,  `dtool_target`, and `mode`.


## Useful snippets

Autoreload in ipython

```
%load_ext autoreload
%autoreload 2
```

Plot a workflow

```python
from fireworks.utilities.dagflow import plot_wf

visual_style = {}

# generic plotting defaults
visual_style["layout"] = 'kamada_kawai'
visual_style["bbox"] = (1600, 1200)
visual_style["margin"] = [400, 100, 400, 200]

visual_style["vertex_label_angle"] = -3.14/4.0
visual_style["vertex_size"] = 8
visual_style["vertex_shape"] = 'rectangle'
visual_style["vertex_label_size"] = 10
visual_style["vertex_label_dist"] = 4

# edge defaults
visual_style["edge_color"] = 'black'
visual_style["edge_width"] = 1
visual_style["edge_arrow_size"] = 1
visual_style["edge_arrow_width"] = 1
visual_style["edge_label_size"] = 8

p = plot_wf(wf, labels=True, **visual_style)
# BUG: direct inline display drops labels, thus view externally
p.save('wf.png')
!display wf.png
```


