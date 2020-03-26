#!/usr/bin/env python
"""Find all variables within jinja2 template and dump to stdout or file."""
import json
import jinja2
import jinja2.meta
import yaml
import sys

def find_undeclared_variables(infile):
    env = jinja2.Environment()
    with open(infile) as template_file:
        parsed = env.parse(template_file.read())

    undefined = jinja2.meta.find_undeclared_variables(parsed)
    return undefined


def main():
    import logging, argparse

    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('template',
        help='jija2 template file.', metavar='template.j2')
    parser.add_argument('outfile', nargs='?',
        help='output file', default=None, metavar='OUT')
    parser.add_argument('--format', choices=('json','yaml'),
        help='file format', default='yaml', metavar='FORMAT')
    parser.add_argument('--verbose', '-v', action='store_true',
        help='Make this tool more verbose')
    parser.add_argument('--debug', action='store_true',
        help='Make this tool print debug info')
    args = parser.parse_args()

    if args.debug:
        loglevel = logging.DEBUG
    elif args.verbose:
        loglevel = logging.INFO
    else:
        loglevel = logging.WARNING

    logging.basicConfig(level=loglevel)

    undefined = list(find_undeclared_variables(args.template))

    if not args.outfile:
        outfile = sys.stdout
    else:
        outfile = open(args.outfile,'w')

    if args.format == 'yaml':
        yaml.dump(undefined, outfile, default_flow_style=False)
    elif args.format == 'json':
        json.dump(undefined, outfile)
    else:
        raise ValueError("Not implemented.")

    try:
        outfile.close()
    except:
        pass

if __name__ == '__main__':
    main()