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
templates and go through two preprocessing layers before handing them to
FireWorks. In the following, we refer to the `fw/sample` for illustration.

On the topmost layer, there is the `system_generic.yaml` file. This file
describes the specific workflow and does not prescribe any machine and
environment for execution. It must have the top level keys `name`, `metadata`,
`std`, `transient`, `persistent` and `dependencies`:

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
  {%- set queue_worker='juwels_queue' %}
  worker:           juwels_noqueue
  {%- elif machine is in ['NEMO'] %}
  {%- set queue_worker='nemo_queue' %}
  worker:           nemo_noqueue
  {%- endif %}

# the following 'transient' section can contain Fireworks-wise entries, i.e...
transient:
  fw_050_production.yaml:
  - pressure: 0
    description: >-
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

  fw_100_postprocessing.yaml:
  - fw_200_results_to_filepad.yaml
```

The decision for an execution environment can be made via the tiny tool

```console
$ render -h
usage: render [-h] [--context CONTEXT] [--verbose] [--debug] infile outfile

Quickly renders a single jinja2 template file from command line.

positional arguments:
  infile             Template .yaml input file
  outfile            Rendered .yaml output file

optional arguments:
  -h, --help         show this help message and exit
  --context CONTEXT  Context (default: {'machine': 'NEMO', 'mode':
                     'PRODUCTION'})
  --verbose, -v      Make this tool more verbose (default: False)
  --debug            Make this tool print debug info (default: False)
```

evoked like

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
`fw_050_production.yaml`:

```yaml
{% extends "fw_base.yaml" %}
{% block body %}
name: {{ title }}, production, {{ comment }}
spec:
  _category: {{ worker|default("juwels_queue",true) }}
  _queueadapter:
    {{ render_queueadapter()|safe|indent(4) }}
  _files_in:
    data_file:  datafile.lammps
    input_file: production.input
  _files_out:
    data_file: final.lammps
    log_file:  log.lammps
  _tasks:
  - _fw_name: CmdTask
    cmd: lmp
    opt:
    - -in production.input
    - -v productionSteps  {{ production_steps|default(10000,true)|int }}
    - -v pressureP        {{ pressure|default(0.0,true)|float }}
    - -v temperatureT     {{ temperature|default(298.0)|float }}
    stderr_file:    std.err
    stdout_file:    std.out
    store_stdout:   true
    store_stderr:   true
    use_shell:      true
    fizzle_bad_rc:  true
  _trackers:
  - filename: log.lammps
    nlines: 25
  metadata:
    step:  production, {{ comment }}
    {{ render_metadata()|indent(4)}}
{% endblock %}
```

The template `fw_base.yaml` has a special status. Just as in object oriented
programming languages, all Fireworks templates are *derived* from
`fw_base.yaml`:

```yaml
{%- set title = "%s %s, %s %s sample workflow"|format(
  temperature,
  temperature_unit,
  pressure,
  pressure_unit) -%}

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

Now, create an empty directory `build` and use the *Workflow Builder Tool*

```console
$ wfb -h
usage: wfb [-h] [--template-dir template-directory] [--verbose] [--debug]
           system.yaml build-directory

Workflow Builder, facilitates workflow construction

positional arguments:
  system.yaml           .yaml input file.
  build-directory       output directory.

optional arguments:
  -h, --help            show this help message and exit
  --template-dir template-directory
                        Directory containing templates. (default: templates)
  --verbose, -v         Make this tool more verbose (default: False)
  --debug               Make this tool print debug info (default: False)
```

to compile `system_nemo.yaml` and Fireworks templates into a static workflow
description `build/wf.yaml` with

```bash
wfb --debug system_nemo.yaml build
```

The `--debug` switch is recommended to see exactly where failures occur.
Within `build`, you will find the `wf.yaml` file ready to hand to Fireworks for
execution with

```bash
lpad add build/wf.yaml
```

Beyond that, you will find one single yaml file for each rendered instance
of a Firework, i.e. `build/fw_050_production.yaml_000030` with content

```yaml
name: 298 K, 1 atm sample workflow, production, long run
spec:
  _category: nemo_queue
  _queueadapter:
    nodes:            1
    ppn:              20
    queue:            express
    walltime:         00:01:00
  _files_in:
    data_file:  datafile.lammps
    input_file: production.input
  _files_out:
    data_file:  final.lammps
    log_file:   log.lammps
  _tasks:
  - _fw_name: CmdTask
    cmd: lmp
    opt:
    - -in production.input
    - -v productionSteps  100000
    - -v pressureP        1.0
    - -v temperatureT     298.0
    stderr_file:    std.err
    stdout_file:    std.out
    store_stdout:   true
    store_stderr:   true
    use_shell:      true
    fizzle_bad_rc:  true
  _trackers:
  - filename: log.lammps
    nlines: 25
  metadata:
    step:  production, long run
    production_steps: 100000
    pressure:         1
    pressure_unit:    atm
    temperature:      298
    temperature_unit: K
```

among `fw_050_production.yaml_000020`, `fw_050_production.yaml_000040`, and  
`fw_050_production.yaml_000050` for the other three parameter sets.
Such a file can be modified, renamed and then appended manually with

```bash
lpad append_wflow -i 123,125 -f fw_050_modified_production.yaml
```
by explicitly specifying the Firework IDs of its designated parents
(here 123 and 125). Fireworks will only accept `.yaml`-suffixed files.

## Notes

The workflow descriptions as introduced above are to be as independent as
possible on the actual execution environment (except for the queue parameters
walltime, nodes, ppn, ...). This is achieved by

* `CmdTask` implemented in [`fireworks/user_objects/firetasks/jlh_tasks.py`](https://github.com/jotelha/fireworks/blob/master/fireworks/user_objects/firetasks/jlh_tasks.py) of the fork [jotelha/fireworks](https://github.com/jotelha/fireworks) together with environment-specific command aliases in worker files such as [`etc/nemo_noqueue_worker.yaml`](https://github.com/jotelha/fw-hpc-worker-jlh/blob/master/etc/nemo_noqueue_worker.yaml) in [jotelha/fw-hpc-worker-jlh](https://github.com/jotelha/fireworks/blob/master/fireworks/user_objects/firetasks/jlh_tasks.py)
* `render` implemented in [`fireworks/utilities/render_template.py`](https://github.com/jotelha/fireworks/blob/master/fireworks/user_objects/firetasks/jlh_tasks.py) of the fork [jotelha/fireworks](https://github.com/jotelha/fireworks) together with environment-specific command aliases in worker files such as [`etc/nemo_noqueue_worker.yaml`](https://github.com/jotelha/fw-hpc-worker-jlh/blob/master/etc/nemo_noqueue_worker.yaml) in [jotelha/fw-hpc-worker-jlh](https://github.com/jotelha/fireworks/blob/master/fireworks/utilities/render_template.py)
* `wfb` implemented in [`fireworks/utilities/wfb.py`](https://github.com/jotelha/fireworks/blob/master/fireworks/user_objects/firetasks/jlh_tasks.py) of the fork [jotelha/fireworks](https://github.com/jotelha/fireworks) together with environment-specific command aliases in worker files such as [`etc/nemo_noqueue_worker.yaml`](https://github.com/jotelha/fw-hpc-worker-jlh/blob/master/etc/nemo_noqueue_worker.yaml) in [jotelha/fw-hpc-worker-jlh](https://github.com/jotelha/fireworks/blob/master/fireworks/utilities/wfb.py)
