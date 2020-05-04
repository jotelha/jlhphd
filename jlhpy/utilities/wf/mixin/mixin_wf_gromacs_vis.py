# -*- coding: utf-8 -*-
"""GROMACS trajectory visualization sub workflow."""

import datetime

from abc import ABC, abstractmethod

from fireworks import Firework
from fireworks.user_objects.firetasks.fileio_tasks import ArchiveDirTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask
from imteksimfw.fireworks.user_objects.firetasks.dtool_tasks import CreateDatasetTask, CopyDatasetTask

from jlhpy.utilities.wf.workflow_generator import SubWorkflowGenerator
