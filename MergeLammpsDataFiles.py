#!/usr/bin/env python
# needs pizza.py
import re
from pprint import pprint

type_section_regex = {
    'Pair Coeffs':     re.compile(r'''
        ^\#\ Pair\ Coeffs\ *[\n\r]
        ^\#\ *[\n\r]
        (?P<type_mapping>(?:^\#\ *\d+\ *\d+\ *[\n\r])+)
        ''', re.MULTILINE | re.VERBOSE),
    'Bond Coeffs':     re.compile(r'''
        ^\#\ Bond\ Coeffs\ *[\n\r]
        ^\#\ *[\n\r]
        (?P<type_mapping>(?:^\#\ *\d+\ *\d+\ *[\n\r])+)
        ''', re.MULTILINE | re.VERBOSE),
    'Angle Coeffs':     re.compile(r'''
        ^\#\ Angle\ Coeffs\ *[\n\r]
        ^\#\ *[\n\r]
        (?P<type_mapping>(?:^\#\ *\d+\ *\d+\ *[\n\r])+)
        ''', re.MULTILINE | re.VERBOSE),
    'Dihedral Coeffs':     re.compile(r'''
        ^\#\ Dihedral\ Coeffs\ *[\n\r]
        ^\#\ *[\n\r]
        (?P<type_mapping>(?:^\#\ *\d+\ *\d+\ *[\n\r])+)
        ''', re.MULTILINE | re.VERBOSE) }

# same for all object entities:
type_mapping_regex = re.compile(r'''
        ^\#\ *(?P<index>\d+)\ *(?P<name>\d+)
        ''', re.MULTILINE | re.VERBOSE )

section_header_dict = {
    'Masses':'atom types',
    'Atoms':'atoms',
    'Angles':'angles',
    'Angle Coeffs':'angle types',
    'Bonds':'bonds',
    'Bond Coeffs':'bond types',
    'Dihedrals':'dihedrals',
    'Dihedral Coeffs':'dihedral types'
    #'Pair Coeffs', no header
    #'Velocities', no header
}

header_section_dict = {
    header: section for section, header in section_header_dict.items() }

header_ordering_list = [
    'atoms',
    'atom types',
    'angles',
    'angle types',
    'bonds',
    'bond types',
    'dihedrals',
    'dihedral types' ]

header_ordering_dict = dict(
    zip( header_ordering_list, range(len(header_ordering_list)) ) )

def header_key(key):
    if key in header_ordering_dict:
        return header_ordering_dict[key]
    else:
        return len(header_ordering_dict)

# re.VERBOSE
#
# This flag allows you to write regular expressions that look nicer and are more
# readable by allowing you to visually separate logical sections of the pattern
# and add comments. Whitespace within the pattern is ignored, except when in a
# character class, or when preceded by an unescaped backslash, or within tokens
# like *?, (?: or (?P<...>. When a line contains a # that is not in a character
# class and is not preceded by an unescaped backslash, all characters from the
# leftmost such # through the end of the line are ignored.

def strip_comments(infile, outfile):
    """Removes all trailing comments from a LAMMPS data file.
       Necessary to make them pizza.py-processible"""
    import re
    regex = re.compile(r"\s*#.*$")
    with open(infile) as i:
        with open(outfile, 'w') as o:
            for line in i:
                line = regex.sub('',line)
                o.write(line)

def map_types(datafile):
    """Expects a datafile written by TopoTools writelammpsdata
       and determines type mappings from auto-generated comments"""

    with open(datafile, 'r') as f:
        content = f.read()

    mapping_dict = {}
    for key, regex in type_section_regex.items():
        print("Parsing section '{}'...".format(key))
        mapping_table = []
        for mapping_section in regex.finditer(content): # should not loop, only 1 section for each key expected
            print("Found: \n{}".format(mapping_section.group('type_mapping').rstrip()))
            for mapping in type_mapping_regex.finditer(mapping_section.group('type_mapping')):
                print("Add mapping index: {} <--> name: {}".format(
                    mapping.group('index'), mapping.group('name') ) )
                mapping_table.append( ( int(mapping.group('name')), int(mapping.group('index')) ) )

        # only if mapping table not empty:
        if len(mapping_table) > 0:
            mapping_dict[key] = dict(mapping_table)

    print("Created mapping dict:")
    pprint(mapping_dict)
    return mapping_dict

def merge_lammps_datafiles(datafile,reffile,outfile):
    """Compares sections in datafile and reffile (reference),
       appends missing sections to datafile andd writes result to outfile."""
    # global data
    from data import data

    # LAMMPS datafile produced by TopoTools 1.7 contains type mappings
    mapping_dict = map_types(datafile)

    reffile_stripped  = reffile + '_stripped'
    datafile_stripped = datafile + '_stripped'

    strip_comments(reffile, reffile_stripped)
    strip_comments(datafile, datafile_stripped)

    #pizzapy_data = data()
    ref = data(reffile_stripped)
    dat = data(datafile_stripped)

    print("Atom types in reference data file:")
    for line in ref.sections["Masses"]: print(line.rstrip())

    if "Masses" in dat.sections:
        print("Atom types in data file:")
        for line in dat.sections["Masses"]: print(line.rstrip())
    else:
        print("No atom types in data file!")

    print("Sections in reference data file:")
    pprint(ref.sections.keys())

    print("Sections in data file:")
    pprint(dat.sections.keys())

    # very weird: pizza.py apparenlty creates an object called "list"
    # containing its command line arguments
    # try:
    #    del list

    missing_sections = list( set( ref.sections.keys() ) - set( dat.sections.keys() ) )
    print("Sections missing in data file:")
    pprint(missing_sections)

    for section in missing_sections:
        if section in mapping_dict:
            print("Missing section {} requires specific mapping.".format(section))
            dat_object_lines = []
            dat_object_list = []
            ref_object_list = ref.get(section)
            print("Reference section contains {} objects.".format( len(ref_object_list) ) )
            for object in ref_object_list:
                # (consecutive) object index in 1st column
                ref_id = int(object[0])
                print("Checking for reference object {} in section {}...".format(object, section))
                if ref_id in mapping_dict[section]:
                    object[0] = mapping_dict[section][ref_id]
                    dat_object_list.append( object )
                    print("Mapping reference object {} onto new {}...".format(ref_id, object))

                    object_type_test = [ int(el) if (type(el) is float and el.is_integer()) else el for el in object ]
                else:
                    print("Reference object {} not in new data file, dropped.".format(ref_id))
            print("Mapped section contains {} objects.".format( len(dat_object_list) ) )
            for object in dat_object_list:
                # make sure all integer values are written as integers
                object = [ int(el) if (type(el) is float and el.is_integer()) else el for el in object ]
                # it seems LAMMPS can read integers as floats, but not floats as integers
                object_str = ' '.join( map(str, object) )
                print("Object will be written as {}...".format(object_str))
                object_str += '\n'
                dat_object_lines.append(object_str)
            dat.sections[section] = dat_object_lines
        else:
            print("Missing section {} does not require specific mapping, copy as is.".format(section))
            dat.sections[section] = ref.sections[section]

    # if new sections have been added (or the lenght of sections has been
    # altered), the header must be updated accordingly
    print("Check header for completeness...")
    for section in dat.sections.keys():
        if section in section_header_dict:
            header = section_header_dict[section]
            if header in dat.headers:
                print( ' '.join((
                    "Section '{:s}' corresponds".format(section),
                    "to existent header entry '{:s}'".format(header) )) )
                if dat.headers[header] == len(dat.sections[section]):
                    print(
                        "Header value {} agrees with section length.".format(
                            dat.headers[header] ))
                else:
                    print( ' '.join((
                        "Header value {} does not".format(dat.headers[header]),
                        "agree with section length {}! Updated.".format(
                            len(dat.sections[section] ) ) )) )
                    dat.headers[header] = len(dat.sections[section])
            else:
                print( ' '.join((
                    "No corresponding header entry '{}'".format(header),
                    "for section '{}' of length {}! Created.".format(section,
                        len(dat.sections[section] ) ) )) )
                dat.headers[header] = len(dat.sections[section])
        else:
            print("No header entry required for section '{}'.".format(section) )


    print("Write merged data to {}...".format(outfile))
    dat.write(outfile)
