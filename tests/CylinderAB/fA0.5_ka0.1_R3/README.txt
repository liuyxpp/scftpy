AB (0.5, 0.5) chiN = 18
ka 0.1 for A
kb -0.1 for A

grid (32, 32, 64) lam (0.02, 0.02)
Converge too slow

grid (32, 32, 32) lam (0.02, 0.02)
Converge a bit faster, but local density of A exceeds 1.0

grid (32, 32, 48) lam (0.03, 0.03)
Do not converge.

grid (32, 32, 48) lam (0.02, 0.02)
Converge too slow

grid (32, 32, 48) lam (0.1, 0.1, 20)
Converge too slow

AB (0.5, 0.5) chiN=15, DL=1.4 (L=3.71, R=D/2=1.4*3.71/2=2.597)
ka=-kb=0.1 for A

we should really try small lamA=lamB

grid (32, 32, 48) lam (0.01, 0.01, 100)


