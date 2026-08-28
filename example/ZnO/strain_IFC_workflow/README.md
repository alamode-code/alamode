# Strain-IFC coupling and elastic constants of wurtzite ZnO

Template inputs for the tools that prepare the cell-relaxation inputs of anphon
(`STRAIN_IFC_DIR`), see the tutorial "QHA structural optimization of ZnO".
Pseudopotentials are not included: edit `pseudo_dir` in the `pw.in` files.

* `template_primitive_QE/pw.in` : 4-atom primitive cell (PBEsol structure of the tutorial)
* `template_supercell_QE/pw.in` : 4x4x2 supercell (the cell of `ZnO442_harmonic.xml`)
* `strain_modes.json`           : the six strain modes (xx, yy, zz, yz, zx, xy), weight 1, smag 0.005
* `job.sh`, `DFT_command.sh`    : optional job-script template (`--job-template`, `--dft-command`)

## 1. Elastic constants (elastic_constants.in, C1_array.in)

    elastic.py generate --code QE --template template_primitive_QE --outdir elastic --smag 0.01 --nmag 2
    (run pw.x in elastic/strain_*/ ; single-point, fixed cell and ions)
    elastic.py fit --outdir elastic --fit stress --fcs ../qha_relax/ZnO442_harmonic.xml \
               --anphon-cell ../qha_relax/ZnO_qha_thermo.in

## 2. Strain-force coupling (strain_force.in)

    strainifc.py generate --coupling force --code QE --template template_primitive_QE --outdir force
    (run pw.x in force/strain_*/primitive/)
    strainifc.py collect --outdir force --fcs ../qha_relax/ZnO442_harmonic.xml \
               --anphon-cell ../qha_relax/ZnO_qha_thermo.in

## 3. Strain-harmonic coupling (strain_harmonic.in + force-constant files)

    strainifc.py generate --coupling harmonic --code QE --template template_supercell_QE --outdir harmonic
    (run pw.x in harmonic/strain_*/nodisp/ and harmonic/strain_*/disp_*/)
    strainifc.py collect --outdir harmonic --fcs ../qha_relax/ZnO442_harmonic.xml

Copy `elastic/results/elastic_constants.in`, `force/results/strain_force.in`,
`harmonic/results/strain_harmonic.in` and the `strain_00N.xml` files into
`STRAIN_IFC_DIR`, and `elastic/results/C1_array.in` into the anphon working directory.
