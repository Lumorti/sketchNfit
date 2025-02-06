# sketchNfit

A program to try to fit many-to-one data to a sketched 2D line

It also contains the ability to randomly test observables given monomial data.

This is built specifically for the output of another code that we are testing. But if you want to use it, you'll need a file called monomials.txt as well as a bunch of numpy files in the same directory, each of which contains moment data for a system parameter given in their filename. For example: a folder containing monomials.txt as well as the files "y_J0.100.npm" and "y_J0.200.npm" which contain the monomial values (in order, labelled by monomials.txt) for J = 0.1 and J = 0.2 respectively.

It requires just numpy, that can be installed with pip:
```bash
pip3 install numpy
```

