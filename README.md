# pekeris-waveguide-solver

Python script for calculating the acoustic pressure (and gradients) in a Pekeris waveguide, which is a cannonical simplistic model of a shallow-water waveguide.

The Python script `pekeris.py` was generated from original Fortran/C code (written around 2010 as part of my academic research) with assistance from Claude Code.  The original Fortran/C source code is also included for reference.

When the Fortran implementation was originally written, its output was verified against a finite-element model .  The Claude-generated Python code has not been rigorously verified.  I have only inspected the Python output visually.

## Python dependencies
- numpy
- scipy (Brent solver, Bessel functions, adaptive integration)
- matplotlib

## Usage
Running the script with no arguments will calculate the pressure field for an example waveguide and display the results:
```
python pekeris.py
```

A few command-line options are available:
```
python pekeris.py --help
```
