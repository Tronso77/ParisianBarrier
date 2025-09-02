from pricing.Parisian.paris_pde import ParisianParams, GridParams, price_parisian, bs_price, parity_error

# simple test case
S0 = 100.0; K = 100.0; T = 0.25
r = 0.02; q = 0.00; sigma = 0.20
B = 95.0; D = 5/252

g = GridParams(Nx=300, Ntau=150)

p_out = ParisianParams(S0,K,T,r,q,sigma,B,D, option_type='call', direction='down', inout='out')
v_out = price_parisian(p_out, g)
v_van = bs_price(S0,K,T,r,q,sigma,'call')
v_in  = price_parisian(ParisianParams(S0,K,T,r,q,sigma,B,D,'call','down','in'), g)
pe    = parity_error(p_out, g)

print("Vanilla     :", v_van)
print("Parisian OUT:", v_out)
print("Parisian IN :", v_in)
print("Parity error:", pe)