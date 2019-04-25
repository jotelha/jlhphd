#!/usr/bin/env python
"""
Pushes or pulls a file to or from filepad
"""

actions = ['pull','push','delete']

import logging
logger = logging.getLogger(__name__)
logfmt = "[%(levelname)s - %(filename)s:%(lineno)s - %(funcName)s() ] %(message)s (%(asctime)s)"
logging.basicConfig( format = logfmt )

def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--host',
        help="MongoDB server",
        default='localhost')
    parser.add_argument('--port',
        help="MongoDB port",
        type=int,
        default=27018)
    parser.add_argument('--database',
        help="MongoDB name",
        default='fireworks-jhoermann')
    parser.add_argument('--user',
        help="MongoDB user",
        default='fireworks')
    parser.add_argument('--password',
        help="MongoDB password",
        default='fireworks')
    parser.add_argument('--file',
        help='Local file name',
        default=None)
    parser.add_argument('--metadata-file',
        help="Metadata text file to attach to pushed or to store for pulled document",
        default=None)
    parser.add_argument('--format',
        help='Format of metadata file',
        choices=('yaml','json'),
        default='yaml')
    parser.add_argument('--verbose', '-v', action='store_true',
        help='Make this tool more verbose')
    parser.add_argument('--debug', action='store_true',
        help='Make this tool print debug info')
    parser.add_argument('action',
        help='Action to perform',
        choices=actions,
        nargs='+')
    parser.add_argument('identifier',
        help='Filepad identifier')
    args = parser.parse_args()

    if args.debug:
      loglevel = logging.DEBUG
    elif args.verbose:
      loglevel = logging.INFO
    else:
      loglevel = logging.WARNING

    logger.setLevel(loglevel)

    import json, yaml
    import os.path
    from fireworks.utilities.filepad import FilePad

    logger.info("Connecting to MongoDB {:s}@{:s}:{:d}/{:s}...".format(
        args.host, args.user, args.port, args.database ) )

    fp = FilePad(
        host=args.host,
        port=args.port,
        database=args.database,
        username=args.user,
        password=args.password )

    file = args.file
    if file is None:
        file = os.path.basename(args.identifier)
        logger.info("No file name specified, using '{:s}'...".format( file ) )


    for i, action in enumerate(args.action):
        if action == 'pull':
            logger.info("Pulling '{:s}' from filepad...".format( args.identifier ) )
            content, doc = fp.get_file(identifier=args.identifier)
            # 646_SDS_on_AU_111_51x30x2_hemicylinders_with_counterion_10ns.lammps

            if args.metadata_file:
              logger.info("Writing metadata to '{:s}'...".format(args.metadata_file))
              with open(args.metadata_file,'w') as metafile:
                if args.format == 'json':
                  json.dump( doc, metafile, skipkeys=True, indent=2, default=lambda d: str(d) )
                else: # yaml
                  yaml.dump( doc, metafile, default_flow_style=False )
                # filepad entry carries 'ObjectID of type 'bson.objectid.ObjectId'
                # serializable by simple str conversion
            logger.info("Writing content to '{:s}'...".format(file))
            with open(file,'wb') as datafile:
              datafile.write(content)
        elif action == 'push':
            metadata = None
            if args.metadata_file:
              logger.info("Reading metadata from '{:s}'...".format(args.metadata_file))
              with open(args.metadata_file,'r') as metafile:
                if args.format == 'json':
                  logger.exception("JSON metadta not implemented!")
                  raise ValueError()
                else: # yaml
                  metadata = yaml.safe_load(stream)

            logger.info("Pushing '{:s}' to filepad with identifier ...".format( file, args.identifier ) )
            gridfsid, identifier = fp.get_file(
                path=file, identifier=args.identifier, metadata=metadata )
            logger.info("Stored as '{:s}': '{:s}.".format( gridfsid, identifier ) )

if __name__ == '__main__':
    main()
