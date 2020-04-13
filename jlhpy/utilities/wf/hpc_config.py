# -*- coding: utf-8 -*-

HPC_SPECS = {
    'forhlr2': {
        'fw_queue_category':   'forhlr2_queue',
        'fw_noqueue_category': 'forhlr2_noqueue',
        'queue': 'develop',
        'physical_cores_per_node': 20,
        'logical_cores_per_node':  40,
        'nodes': 4,
        'walltime':  '00:60:00',
    },
    'juwels_devel': {
        'fw_queue_category':   'juwels_queue',
        'fw_noqueue_category': 'juwels_noqueue',
        'queue': 'devel',
        'physical_cores_per_node': 48,
        'logical_cores_per_node':  96,
        'nodes': 8,
        'walltime':  '00:30:00',
    },
    'juwels': {
        'fw_queue_category':   'juwels_queue',
        'fw_noqueue_category': 'juwels_noqueue',
        'queue': 'batch',
        'physical_cores_per_node': 48,
        'logical_cores_per_node':  96,
        'nodes': 1024,
        'walltime':  '00:30:00',
    },
    'ubuntu': {
        'fw_noqueue_category': 'ubuntu_noqueue',
        'fw_queue_category':   'ubuntu_noqueue',
        'queue': 'NONE',
        'physical_cores_per_node': 1,
        'logical_cores_per_node':  1,
        'nodes': 1,
        'walltime':  '99:99:99',
    },
}

HPC_EXPORTS = {  # standard settings for environment variables
    'forhlr2': {
        'OMP_NUM_THREADS': 1,
        'KMP_AFFINITY':    "'verbose,compact,1,0'",
        'I_MPI_PIN_DOMAIN': 'core',
    },
    'juwels': {
        'OMP_NUM_THREADS':  1,
        'KMP_AFFINITY':     "'verbose,compact,1,0'",
        'I_MPI_PIN_DOMAIN': 'core',
    }
}
