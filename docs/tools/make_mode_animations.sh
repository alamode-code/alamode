#!/bin/sh
# Regenerate the phonon-mode animations shown in the tutorials
# (docs/tools/mode_animations/*.xyz) with anphon's ANIME option.
#
#   sh docs/tools/make_mode_animations.sh [path/to/anphon]
#
# Inputs: docs/tools/mode_animations/*.in (harmonic IFCs from example/).
#   sto_R: SrTiO3, R point, NONANALYTIC = 3 as in the SCPH tutorial; mode 1 is one
#          of the three degenerate unstable octahedron rotations (-76 cm^-1).
#   bto_G: BaTiO3, Gamma, cBTO222_harmonic.xml; mode 3 is the polar soft mode
#          (-199 cm^-1) with Ti moving along z against the O cage.
# Degenerate modes can come out as different combinations with another LAPACK;
# pick the mode whose displacement pattern is as described above.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
anphon=${1:-$root/_build/anphon/anphon}
case $anphon in /*) ;; */*) anphon=$(cd "$(dirname "$anphon")" && pwd)/$(basename "$anphon") ;; esac
out=$root/docs/tools/mode_animations
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir -p "$out"

cd "$work"
bunzip2 -c "$root/example/SrTiO3/reference/STO_anharm.xml.bz2" > STO_anharm.xml
cp "$root/example/SrTiO3/reference/BORN" .
cp "$root/example/BaTiO3/anharm_IFCs/cBTO222_harmonic.xml" .

"$anphon" "$here/mode_animations/sto_R.in" > sto_R.log
cp sto_R.anime01.xyz "$out/sto_R_soft.xyz"

"$anphon" "$here/mode_animations/bto_G.in" > bto_G.log
cp bto_G.anime03.xyz "$out/bto_G_soft.xyz"

head -2 "$out"/*.xyz
