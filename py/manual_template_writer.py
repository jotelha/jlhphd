# a sample snippet on how to quickly generate a filled template via FireWorks 
from fireworks.user_objects.firetasks.templatewriter_task import TemplateWriterTask
import yaml

with open("indenter_insertion_context.yaml") as stream: fw_spec = yaml.safe_load(stream)

t = TemplateWriterTask.from_dict(fw_spec)
t.run_task(fw_spec)