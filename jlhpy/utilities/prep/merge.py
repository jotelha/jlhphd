#!/usr/bin/env python
#
# MergeLammpsDataFiles.py
#
# Copyright (C) 2018, 2019 IMTEK Simulation
# Author: Johannes Hoermann, johannes.hoermann@imtek.uni-freiburg.de
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# needs pizza.py, see merge.py in same directory for more information
#
# TODO: 2019-08-16 treat impropers as well
#
import re
import numpy as np
from pprint import pprint

# Below follows excerpt from
#
# Pizza.py toolkit, www.cs.sandia.gov/~sjplimp/pizza.html
# Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories
#
# Copyright (2005) Sandia Corporation.  Under the terms of Contract
# DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
# certain rights in this software.  This software is distributed under
# the GNU General Public License.
#
# adapted for standalone usage in Python 3

# data tool

# History
#   8/05, Steve Plimpton (SNL): original version
#   11/07, added triclinic box support

# ToDo list

# Variables
#   title = 1st line of data file
#   names = dictionary with atom attributes as keys, col #s as values
#   headers = dictionary with header name as key, value or tuple as values
#   sections = dictionary with section name as key, array of lines as values
#   nselect = 1 = # of snapshots

# Imports and external programs

from os import popen

# Class definition

class data:

  # --------------------------------------------------------------------

  def __init__(self,*list):
    self.nselect = 1

    if len(list) == 0:
      self.title = "LAMMPS data file"
      self.names = {}
      self.headers = {}
      self.sections = {}
      return

    file = list[0]
    f = open(file)

    self.title = f.readline()
    self.names = {}

    headers = {}
    while 1:
      line = f.readline()
      line = line.strip()
      if len(line) == 0:
        continue
      found = 0
      for keyword in hkeywords:
        if line.find(keyword) >= 0:
          found = 1
          words = line.split()
          if keyword == "xlo xhi" or keyword == "ylo yhi" or \
            keyword == "zlo zhi":
            headers[keyword] = (float(words[0]),float(words[1]))
          elif keyword == "xy xz yz":
            headers[keyword] = \
              (float(words[0]),float(words[1]),float(words[2]))
          else:
            headers[keyword] = int(words[0])
      if not found:
        break

    sections = {}
    while 1:
      found = 0
      for pair in skeywords:
        keyword,length = pair[0],pair[1]
        if keyword == line:
          found = 1
          if length not in headers:
            raise Exception("data section %s has no matching header value" % line)
          f.readline()
          list = []
          for i in range(headers[length]): list.append(f.readline())
          sections[keyword] = list
      if not found:
        raise Exception("invalid section %s in data file" % line)
      f.readline()
      line = f.readline()
      if not line:
        break
      line = line.strip()

    f.close()
    self.headers = headers
    self.sections = sections

  # --------------------------------------------------------------------
  # assign names to atom columns

  def map(self,*pairs):
    if len(pairs) % 2 != 0:
      raise Exception("data map() requires pairs of mappings")
    for i in range(0,len(pairs),2):
      j = i + 1
      self.names[pairs[j]] = pairs[i]-1

  # --------------------------------------------------------------------
  # extract info from data file fields

  def get(self,*list):
    if len(list) == 1:
      field = list[0]
      array = []
      lines = self.sections[field]
      for line in lines:
        words = line.split()
        # values = list(map(float,words))
        values = [m for m in map(float,words)]
        array.append(values)
      return array
    elif len(list) == 2:
      field = list[0]
      n = list[1] - 1
      vec = []
      lines = self.sections[field]
      for line in lines:
        words = line.split()
        vec.append(float(words[n]))
      return vec
    else:
      raise Exception("invalid arguments for data.get()")

  # --------------------------------------------------------------------
  # reorder columns in a data file field

  def reorder(self,name,*order):
    n = len(order)
    natoms = len(self.sections[name])
    oldlines = self.sections[name]
    newlines = natoms*[""]
    for index in order:
      for i in range(len(newlines)):
        words = oldlines[i].split()
        newlines[i] += words[index-1] + " "
    for i in range(len(newlines)):
      newlines[i] += "\n"
    self.sections[name] = newlines

  # --------------------------------------------------------------------
  # replace a column of named section with vector of values

  def replace(self,name,icol,vector):
    lines = self.sections[name]
    newlines = []
    j = icol - 1
    for i in range(len(lines)):
      line = lines[i]
      words = line.split()
      words[j] = str(vector[i])
      newline = ' '.join(words) + '\n'
      newlines.append(newline)
    self.sections[name] = newlines

  # --------------------------------------------------------------------
  # replace x,y,z in Atoms with x,y,z values from snapshot ntime of dump object
  # assumes id,x,y,z are defined in both data and dump files
  # also replaces ix,iy,iz if they are defined

  def newxyz(self,dm,ntime):
    nsnap = dm.findtime(ntime)

    dm.sort(ntime)
    x,y,z = dm.vecs(ntime,"x","y","z")

    self.replace("Atoms",self.names['x']+1,x)
    self.replace("Atoms",self.names['y']+1,y)
    self.replace("Atoms",self.names['z']+1,z)

    if "ix" in dm.names and "ix" in self.names:
      ix,iy,iz = dm.vecs(ntime,"ix","iy","iz")
      self.replace("Atoms",self.names['ix']+1,ix)
      self.replace("Atoms",self.names['iy']+1,iy)
      self.replace("Atoms",self.names['iz']+1,iz)

  # --------------------------------------------------------------------
  # delete header value or section from data file

  def delete(self,keyword):

    if keyword in self.headers: del self.headers[keyword]
    elif keyword in self.sections: del self.sections[keyword]
    else: raise Exception("keyword not found in data object")

  # --------------------------------------------------------------------
  # write out a LAMMPS data file

  def write(self,file):
    f = open(file,"w")
    print(self.title, file=f)
    for keyword in hkeywords:
      if keyword in self.headers:
        if keyword == "xlo xhi" or keyword == "ylo yhi" or \
               keyword == "zlo zhi":
          pair = self.headers[keyword]
          print(pair[0],pair[1],keyword, file=f)
        elif keyword == "xy xz yz":
          triple = self.headers[keyword]
          print(triple[0],triple[1],triple[2],keyword, file=f)
        else:
          print(self.headers[keyword],keyword, file=f)
    for pair in skeywords:
      keyword = pair[0]
      if keyword in self.sections:
        print("\n%s\n" % keyword, file=f)
        for line in self.sections[keyword]:
          print(line, end=' ', file=f)
    f.close()

  # --------------------------------------------------------------------
  # iterator called from other tools

  def iterator(self,flag):
    if flag == 0: return 0,0,1
    return 0,0,-1

  # --------------------------------------------------------------------
  # time query from other tools

  def findtime(self,n):
    if n == 0: return 0
    raise Exception("no step %d exists" % (n))

  # --------------------------------------------------------------------
  # return list of atoms and bonds to viz for data object

  def viz(self,isnap):
    if isnap: raise Exception("cannot call data.viz() with isnap != 0")

    id = self.names["id"]
    type = self.names["type"]
    x = self.names["x"]
    y = self.names["y"]
    z = self.names["z"]

    xlohi = self.headers["xlo xhi"]
    ylohi = self.headers["ylo yhi"]
    zlohi = self.headers["zlo zhi"]
    box = [xlohi[0],ylohi[0],zlohi[0],xlohi[1],ylohi[1],zlohi[1]]

    # create atom list needed by viz from id,type,x,y,z

    atoms = []
    atomlines = self.sections["Atoms"]
    for line in atomlines:
      words = line.split()
      atoms.append([int(words[id]),int(words[type]),
                    float(words[x]),float(words[y]),float(words[z])])

    # create list of current bond coords from list of bonds
    # assumes atoms are sorted so can lookup up the 2 atoms in each bond

    bonds = []
    if "Bonds" in self.sections:
      bondlines = self.sections["Bonds"]
      for line in bondlines:
        words = line.split()
        bid,btype   = int(words[0]),int(words[1])
        atom1,atom2 = int(words[2]),int(words[3])
        atom1words  = atomlines[atom1-1].split()
        atom2words  = atomlines[atom2-1].split()
        bonds.append([bid,btype,
                      float(atom1words[x]),float(atom1words[y]),
                      float(atom1words[z]),
                      float(atom2words[x]),float(atom2words[y]),
                      float(atom2words[z]),
                      float(atom1words[type]),float(atom2words[type])])

    tris = []
    lines = []
    return 0,box,atoms,bonds,tris,lines

  # --------------------------------------------------------------------
  # return box size

  def maxbox(self):
    xlohi = self.headers["xlo xhi"]
    ylohi = self.headers["ylo yhi"]
    zlohi = self.headers["zlo zhi"]
    return [xlohi[0],ylohi[0],zlohi[0],xlohi[1],ylohi[1],zlohi[1]]

  # --------------------------------------------------------------------
  # return number of atom types

  def maxtype(self):
    return self.headers["atom types"]

# --------------------------------------------------------------------
# data file keywords, both header and main sections

hkeywords = ["atoms","ellipsoids","lines","triangles","bodies",
             "bonds","angles","dihedrals","impropers",
             "atom types","bond types","angle types","dihedral types",
             "improper types","xlo xhi","ylo yhi","zlo zhi","xy xz yz"]

skeywords = [["Masses","atom types"],
             ["Atoms","atoms"],["Ellipsoids","ellipsoids"],
             ["Lines","lines"],["Triangles","triangles"],["Bodies","bodies"],
             ["Bonds","bonds"],
             ["Angles","angles"],["Dihedrals","dihedrals"],
             ["Impropers","impropers"],["Velocities","atoms"],
             ["Pair Coeffs","atom types"],
             ["Bond Coeffs","bond types"],["Angle Coeffs","angle types"],
             ["Dihedral Coeffs","dihedral types"],
             ["Improper Coeffs","improper types"],
             ["BondBond Coeffs","angle types"],
             ["BondAngle Coeffs","angle types"],
             ["MiddleBondTorsion Coeffs","dihedral types"],
             ["EndBondTorsion Coeffs","dihedral types"],
             ["AngleTorsion Coeffs","dihedral types"],
             ["AngleAngleTorsion Coeffs","dihedral types"],
             ["BondBond13 Coeffs","dihedral types"],
             ["AngleAngle Coeffs","improper types"],
             ["Molecules","atoms"],
             ["Tinker Types","atoms"]]

# End of pizza.py excerpt

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

type_instance_dict = {
    'Angle Coeffs':'Angles',
    'Bond Coeffs':'Bonds',
    'Dihedral Coeffs':'Dihedrals'
    #'Pair Coeffs', no header
    #'Velocities', no header
}

instance_type_dict = { val: key for (key,val) in type_instance_dict.items() }

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
        # should not loop, only 1 section for each key expected:
        for mapping_section in regex.finditer(content):
            print(
                "Found: \n{}".format(
                    mapping_section.group('type_mapping').rstrip()))
            for mapping in type_mapping_regex.finditer(
                mapping_section.group('type_mapping')):
                print("Add mapping index: {} <--> name: {}".format(
                    mapping.group('index'), mapping.group('name') ) )
                mapping_table.append( ( int(mapping.group('index')),
                    int(mapping.group('name')) ) )

        # only if mapping table not empty:
        if len(mapping_table) > 0:
            mapping_dict[key] = dict(mapping_table)

    print("Created mapping dict:")
    pprint(mapping_dict)
    return mapping_dict

def merge_lammps_datafiles(datafile,reffile,outfile,exceptions=[]):
    """Compares sections in datafile and reffile (reference),
       appends missing sections to datafile andd writes result to outfile."""

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

    missing_sections = list(
        set( ref.sections.keys() ) - set( dat.sections.keys() ) )
    print("Sections missing in data file:")
    pprint(missing_sections)

    for section in ref.sections.keys():
        if section in exceptions:
            print("Section {} is marked to be skipped.".format(section))
            continue
        if (section in instance_type_dict) and (
            instance_type_dict[section] in mapping_dict):

            type_section = instance_type_dict[section]
            print("Section {} requires specific mapping.".format(section))
            print("Mapping applied to according to type section {}.".format(
                type_section))

            dat_object_lines = []
            new_dat_object_list = []
            dat_object_list = dat.get(section)
            print("Datafile section contains {} objects.".format(
                len(dat_object_list) ) )
            for object in dat_object_list:
                dat_id   = int(object[0]) # (consecutive) object idx in 1st col
                dat_type = int(object[1]) # type index in 2nd column
                print("Checking for mapping of object {} in type section {}...".format(
                    object, type_section))
                if dat_type in mapping_dict[type_section]:
                    object[1] = mapping_dict[type_section][dat_type]
                    new_dat_object_list.append( object )
                    print("Mapping object {} onto new entry {}...".format(
                        dat_id, object))
                    object_type_test = [
                        int(el) if (type(el) is float and el.is_integer()) \
                        else el for el in object ]
                else:
                    print("No mapping for object {}, type {} in new data file, dropped.".format(
                        dat_id, dat_type))
            print("Mapped section contains {} objects.".format(
                len(dat_object_list) ) )

            for object in new_dat_object_list:
                # make sure all integer values are written as integers
                object = [
                    int(el) if (type(el) is float and el.is_integer()) \
                    else el for el in object ]
                # it seems LAMMPS can read integers as floats,
                # but not floats as integers
                object_str = ' '.join( map(str, object) )
                print("Object will be written as {}...".format(object_str))
                object_str += '\n'
                dat_object_lines.append(object_str)
            dat.sections[section] = dat_object_lines

        if section in missing_sections:
            print(
                "Missing section {} does not require specific mapping, copy as is.".format(
                    section))
            dat.sections[section] = ref.sections[section]

    # if new sections have been added (or the lenght of sections has been
    # altered), the header must be updated accordingly
    print("Check header for completeness...")
    for section in ref.sections.keys():
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

        # if the section has a type header, but no type section, then check:
        if section in instance_type_dict \
            and instance_type_dict[section] not in dat.sections.keys():

            print( ' '.join((
               "Section '{:s}' exists, but not the".format(section),
               "corresponding type section '{:s}'".format(
                  instance_type_dict[section]) )) )
            header = section_header_dict[ instance_type_dict[ section ] ]
            maximum_type_index = int( np.max( np.array( dat.get(section) )[:,1] ) )
            if header in dat.headers:
                print( ' '.join((
                    "Types of section '{:s}' correspond".format(section),
                    "to existent header entry '{:s}'".format(header) )) )
                if dat.headers[header] == maximum_type_index:
                    print(
                        "Type header value {} agrees with number of types in {}.".format(
                            dat.headers[header], section ))
                else:
                    print( ' '.join((
                        "Header value {} does not".format(dat.headers[header]),
                        "agree with number of types {} in {}! Updated.".format(
                             maximum_type_index, section ) )) )
                    dat.headers[header] = maximum_type_index
            else:
                print( ' '.join((
                    "No corresponding type header entry '{}'".format(header),
                    "for section '{}' with {} types! Created.".format(section,
                        section, maximum_type_index ) )) )
                dat.headers[header] = maximum_type_index

    print("Write merged data to {}...".format(outfile))
    dat.write(outfile)
