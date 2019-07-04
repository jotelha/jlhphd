# Fireworks Workflows

## Intro

This subdirectory contains Workflow descriptions in yaml format,
executable by FireWorks. They are based on the
[jotelha/fireworks](https://github.com/jotelha/fireworks) fork and might not
work properly with standard
[materialsproject/fireworks](https://github.com/materialsproject/fireworks).

## Tree

* AFM:        workflows for AFM tip model insertion, minimization,
              equilibration, and normal AFM tip approach.
* interface:  workflows for merging substrate and solution systems,
              minimization, and equilibration.
* substrate:  workflows for crystal substrate creation, minimization,
              and equilibration.
* legacy:     outdated, for reference purposes
              Two kinds of `.yaml` files are to be distinguished:
              Single FireWorks and full workflow descriptions.
              They are to be distinguished by prefix `fw_`and `wf_`
              respectively.
  * benchmarks
  * manual
  * postprocess
  * prep
  * production
  * recover
  * specific
  * trials
  * utility

## Concept

In static yaml files, the same parameters
have to be specified multiple times at different places.

Thus, yaml files here are formatted as [Jinja2](http://jinja.pocoo.org)
templates and go through a two preprocessing layers.
In the following, we refer to (AFM/AU/25Ang/AU/150Ang/cibe/SDS/production) as
an example.

On the topmost layer, there is a `system_generic.yaml` file. This file describes
the specific workflow and does not prescribe any machine and environment for
execution. It must have the top level keys `name`, `metadata`, `std`,
`transient`, `persistent` and `dependencies`.

```yaml
name: "Workflow's descriptive title"
metadata: "Workflow's metadata, just put {} if empty"
std:
  description: >-
    The >- operator allows a multi-line description here in our yaml file. The
    "std" section contains parameters provided to all templates at the following
    lower layer of preprocessing, such as...
  temperature:      298
  temperature_unit: K
  pressure:         1
  pressure_unit:    atm

  # Settings dependent on the environment of execution should be reduced to
  # a minimum here and always be due to simple case discriminations in the
  # Jinja2 template language:
  {%- if machine is in ['JUWELS'] %}
  worker:           juwels_noqueue
  queue_worker:     juwels_queue
  {%- elif machine is in ['NEMO'] %}
  worker:           nemo_noqueue
  queue_worker:     nemo_queue
  {%- endif %}

# the following 'transient' section can contain Fireworks-wise entries, i.e...
transient:
  fw_050_production.yaml:
  - description: >-
      Parameters defined here will only be available to the very Firework and
      possibly override a parameter of the same name defined within 'std'.
      These transient settings will not affect any children Fireworks.
      The double braces above mark a Jinja2 placeholder to be filled in.
  - worker: {{ queue_worker }}
    description: >-
      If two or more list entries are defined for the same Firework, then the
      workflow will fork at this point, with one Firework instance for each
      parameter set in this list.

# the following 'persistent' section works equivalently to the 'transient'
# section, with one crucial difference: children Fireworks inherit all
# newly defined and overridden parameters
persistent:
  fw_050_production.yaml:
  - production_steps: 100000
    comment:  long run
  - production_steps: 1000
    comment: short run
# all transient will be combined with all persistent parameter sets, resulting
# in four instances of 'fw_050_production.yaml' here.

# finally, the 'dependencies' section corresponds to its static workflow
# description's namesake, with the difference of using full file names instead
# of integer indices here for readability:
dependencies:
  fw_030_datafile_retrieval.yaml:
  - fw_050_production.yaml

  fw_050_production.yaml:
  - fw_100_postprocessing.yaml
  - fw_200_results_to_filepad.yaml

  fw_100_extract_property.yaml:
  - fw_200_results_to_filepad.yaml
```

The decision for an execution environment can be made via

```bash
render --context '{"machine":"NEMO"}' system_generic.yaml system_nemo.yaml
```

for rendering the template with NEMO-specific settings.

The actual Fireworks are to be found below the `templates` subfolder with
content

* fw_030_datafile_retrieval.yaml
* fw_050_production.yaml
* fw_100_postprocessing.yaml
* fw_200_results_to_filepad.yaml
* fw_base.yaml

The prefix `fw` and integer indices are no imperative and only serve better
readability within the file system. Each template just is a normal Fireworks
description in yaml format, enhanced by Jinja2 bits, i.e.
`fw_030_datafile_retrieval.yaml`:

```yaml
{% extends "fw_base.yaml" %}
{% block body %}
name: {{ title }}, file retrieval
spec:
  _category: {{ worker|default("nemo_noqueue",true) }}
  _files_out:
    input_file: "lammps.input"
    data_file:  "*.lammps"
  _tasks:

  - _fw_name: GetFilesTask
    identifiers:
    - lammps.input

  - _fw_name: GetFilesByQueryTask
    query:
      metadata:
        temperature:      {{ temperature }}
        pressure:         {{ pressure }}
        type:             initial_config

  metadata:
    step: "datafile retrieval"
    {{ render_metadata()|indent(4)}}
{% endblock %}
```
The template `fw_base.yaml` has a special status. Just as in object oriented
programming languages, all Fireworks templates are *derived* from
`fw_base.yaml`:

```yaml
{%- set title = "%s %s, %s %s sample workflow"|format(
  temperature,
  temperature_unit.
  pressure,
  pressure_unit -%}

{# depending on machine, different queue settings are necessary #}
{# JUWELS uses SLURM, NEMO MOAB #}
{%- macro render_queueadapter() -%}
{%- if worker is in ['nemo_queue'] -%}
nodes:            {{ nodes|default(1,true)|int }}
ppn:              {{ ppn|default(20,true)|int }}
queue:            {{ queue|default('express',true) }}
walltime:         {{ walltime|default("00:01:00") }}
{%- elif worker is in ['juwels_queue'] -%}
account:          {{ account|default("hfr13") }}
cpus_per_task:    {{ cpus_per_task|default(1,true)|int }}
ntasks_per_node:  {{ ntasks_per_node|default(96,true)|int }}
ntasks:           {{ ntasks|default(96,true)|int }}
queue:            {{ queue|default("express") }}
walltime:         {{ walltime|default("00:01:00") }}
{%- endif -%}
{%- endmacro -%}

{%- macro render_metadata() -%}
production_steps: {{ production_steps }}
pressure:         {{ pressure }}
pressure_unit:    {{ pressure_unit }}
temperature:      {{ temperature }}
temperature_unit: {{ temperature_unit }}
{%- endmacro -%}
{%- block body -%}{%- endblock -%}
```

There are several advantages in doing so:
* The placeholder `title` is generated automatically for each Firework.
* The macro `render_queueadapter` can insert the machine-dependent queue
  settings wherever necessary.
* The macro `render_metadata` can insert the same set of metadata at any desired
  position.


## Notes

The workflow descriptions shall become independent on the actual execution of
tasks on different machines (NEMO, JUWELS, ...) - except the queue parameters
(walltime, nodes, ppn, ...). This is achieved by

* CmdTask implemented in [`fireworks/user_objects/firetasks/jlh_tasks.py`](https://github.com/jotelha/fireworks/blob/master/fireworks/user_objects/firetasks/jlh_tasks.py) of the fork [jotelha/fireworks](https://github.com/jotelha/fireworks) together with environment-specific command aliases in worker files such as [`etc/nemo_noqueue_worker.yaml`](https://github.com/jotelha/fw-hpc-worker-jlh/blob/master/etc/nemo_noqueue_worker.yaml) in [jotelha/fw-hpc-worker-jlh](https://github.com/jotelha/fw-hpc-worker-jlh)
