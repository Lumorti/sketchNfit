# sketchNfit

A program to try to fit many-to-one data to a sketched 2D line

It also contains the ability to randomly test observables given monomial data.

This is built specifically for the output of another code for a paper that we are testing. But if you want to use it, you'll need a file called monomials.txt as well as a bunch of numpy files in the same directory, each of which contains moment data for a system parameter given in their filename. For example: a folder containing monomials.txt the files "y_J0.100.npm" which contains the monomial values (in order, labelled by monomials.txt) for J = 0.1.

It requires just numpy, that can be installed with pip:
```bash
pip3 install numpy
```

