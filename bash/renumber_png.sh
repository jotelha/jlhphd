#!/bin/bash
# Pymol outputs png files without leading zeroes beyond 4 digits.
# Renumber for correct processing order with ffmpeg.
# This assures 6 leading zeroes.

prefix=$1

# for 4 digit frames:
for f in ${prefix}[0-9][0-9][0-9][0-9].png; do
    ftail=${f#$prefix}
    g="${prefix}0${ftail}"
    #echo "${f} --> ${g}"
    mv "${f}" "${g}"
done

# for 5 digit frames:
for f in ${prefix}[0-9][0-9][0-9][0-9][0-9].png; do
    ftail=${f#$prefix}
    g="${prefix}0${ftail}"
    #echo "${f} --> ${g}"
    mv "${f}" "${g}"
done
