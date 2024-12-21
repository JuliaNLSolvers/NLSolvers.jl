
a = ∇f[1]
c = ∇f[2]
b = Λ[1]
d = Λ[2]
e = Δ^2

k1 = a+c-b^2*e-d^2*e
k2 = d*e
K = k1 -4*b*k2
J = k1 +2*b*k2
L = -b*c-a*d +b^2*k2+b*d^2*e
H = b^2*c+a*d^2-b^2*d^2*e
B = b*e+k2
C = 108*B^2*H+72*e*H*K+2*K^3-36*B*K*L-108*e*L^2+sqrt(-4*(-12*e*H+K^2-12*B*L)^3+(108*B^2*H+72*e*H*K+2*K^3-36*B*K*L-108*e*L^2)^2)
g = b+d

U = -((2^(2/3)*C^(1/3) - 6*e*g^2 + (2*2^(1/3)*J^2)/C^(1/3) - 4*K)/e)
W =  ((2^(2/3)*C^(1/3) - 6*e*g^2 + (2*2^(1/3)*J^2)/C^(1/3) - 4*K)/e)
V = C^(1/3)/(3*2^(1/3)*e) + 2*g^2 + (2^(1/3)*J^2)/(3*C^(1/3)*e) + (4*K)/(3*e)
Y = 2 *sqrt(6)*(e*g^3 + g*K + 2*L)
Q = V - Y/(e Sqrt[U])
R = V + Y/(e Sqrt[U])


sol1 = 1/12 * (-6g-6sqrt(R)-sqrt(6)*sqrt(U))
sol2 = 1/12 * (-6g+6sqrt(R)-sqrt(6)*sqrt(U))
sol3 = 1/12 * (-6g-6sqrt(R)+sqrt(6)*sqrt(U))
sol4 = 1/12 * (-6g+6sqrt(R)+sqrt(6)*sqrt(U))


 