# Fireworks

This subdirectory contains Workflow descriptions in the yaml format
executable by Fireworks. In these static files, the same parameters
have to be specified mutliple times at different places.

The workflow descriptions shall become independent on the actual execution of tasks on different machines (NEMO, JUWELS, ...) - except the queue parameters (walltime, nodes, ppn, ...). This is achieved by

* CmdTask implemented in [`fireworks/user_objects/firetasks/jlh_tasks.py`](https://github.com/jotelha/fireworks/blob/master/fireworks/user_objects/firetasks/jlh_tasks.py) of the fork [jotelha/fireworks](https://github.com/jotelha/fireworks) together with environment-specific command aliases in worker files such as [`etc/nemo_noqueue_worker.yaml`](https://github.com/jotelha/fw-hpc-worker-jlh/blob/master/etc/nemo_noqueue_worker.yaml) in [jotelha/fw-hpc-worker-jlh](https://github.com/jotelha/fw-hpc-worker-jlh)
