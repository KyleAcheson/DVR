# DVR

A series of python scripts that perform solve the 1D
TISE using DVR based methods.

## Lobatto
Uses a basis of lobatto shaped functions evaluated at the
quadrature nodes. PE and KE matrix elements evalauted
according to the rules of Gauss-Lobatto quadrature.

## PIB
Uses a basis of particle in a box basis functions,
and numerically calculates the PE and KE matrix elements
through standard trapeziodal intergration.

## HEG
Diagonalizes the position operator matrix in the basis
of orthnormal polynomails. Allows one to go from a
FBR to a DVR representation. The transformation works
to get the DVR basis, and the PE matrix elements are evaluated,
but there is currently a problem with KE matrix elements.

