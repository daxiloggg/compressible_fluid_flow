#HOMOGENEOUS EQUILIBRIUM MODEL FOR CONVERGING-DIVERGING NOZZLES WITH NORMAL SHOCKWAVES
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import PropsSI
import math
import numpy as np
import matplotlib.pyplot as plt

#Converging Diverging Nozzles
case_nozzle = np.array([[300,60,1550,1.5,30,10],    # T, u, s, M_limit, deg_c, deg_d
                        [650,60,7000,2.5,45,5],    # T, u, s, M_limit, deg_c, deg_d  
                        [380,50,1520,1.5,15,25],    # T, u, s, M_limit, deg_c, deg_d
                        [550,60,6000,1.5,30,10]])   # T, u, s, M_limit, deg_c, deg_d

#Initialization
case = 4
nozzles = True
shock_count = 4
shock_solve = True
model = "Homogeneous Equilibrium Model"

if nozzles == True:
	T = T_i = T_f = case_nozzle[case-1,0]	# Initial Temperature
	u = u_i = case_nozzle[case-1,1]			  # Initial Velocity
	s = s_i = case_nozzle[case-1,2]			  # Initial Entropy
	m_limit = case_nozzle[case-1,3]			  # Maximum Mach Number
	deg_c = case_nozzle[case-1,4]			    # Convergence angle
	deg_d = case_nozzle[case-1,5]			    # Divergence angle
	profile = 'Nozzle Profile'

w_fluid = 'Water'
dT = 10e-6
T_idec = 0.1

T_triple = PropsSI("T_triple",w_fluid)
T_crit = PropsSI("Tcrit",w_fluid)
P_crit = PropsSI("Pcrit",w_fluid)
D = D_i = 0.25
L = L_i = 0
dL = 0
ph = []
phase_change = False
throat = []
f_steps = 100
line_f = np.zeros([shock_count,f_steps,2])
line_r = np.zeros([shock_count,f_steps,2])

print("Initializing...")

#Define Functions
def phase(T,s):
	try:
		return CP.PhaseSI('T',T,'S',s,w_fluid)
	except:
		return "unknown"

def pty_sT(f_property,T,s):
	try:
		return PropsSI('{}'.format(f_property),'T',T,'S',s,w_fluid)
	except:
		return np.nan

def pty_s(f_property,s):
	try:
		return PropsSI('{}'.format(f_property),'T',T,'S',s,w_fluid)
	except:
		return np.nan

def pty_Tx(f_property,T,x):
	try:
		return PropsSI('{}'.format(f_property),'T',T,'Q',x,w_fluid)
	except:
		return np.nan

def fdm_s(f_property,s):
	try:
		return PropsSI('{}'.format(f_property),'T',T+dT,'S',s,w_fluid)-PropsSI('{}'.format(f_property),'T',T-dT,'S',s,w_fluid)
	except:
		return np.nan

def fdm_x(f_property,x):
	try:
		return PropsSI('{}'.format(f_property),'T',T+dT,'Q',x,w_fluid)-PropsSI('{}'.format(f_property),'T',T-dT,'Q',x,w_fluid)
	except:
		return np.nan

def sound(T,s):
	try:
		current_phase = phase(T,s)
		if current_phase == "twophase":
			x = PropsSI('Q','T',T,'S',s,w_fluid)
			if x < 0 or x > 1:
				x = 0.5
				
			rho_f = PropsSI('D','T',T,'Q',0,w_fluid)
			rho_g = PropsSI('D','T',T,'Q',1,w_fluid)
			h_f = PropsSI('H','T',T,'Q',0,w_fluid)
			h_g = PropsSI('H','T',T,'Q',1,w_fluid)
			P_sat = PropsSI('P','T',T,'Q',0.5,w_fluid)
			
			rho_m = 1 / (x/rho_g + (1-x)/rho_f)
			
			a_two_phase = math.sqrt(P_sat / rho_m * 1.3)
			return max(50, a_two_phase)
			
		elif current_phase in ["gas", "supercritical_gas", "supercritical"]:
			return PropsSI('A','T',T,'S',s,w_fluid)
		else:
			return PropsSI('A','T',T,'S',s,w_fluid)
	except Exception as e:
		print(f"Sound speed error: {e}")
		return 300.0

def region(T_current, s_current):
	current_phase = phase(T_current, s_current)
	if current_phase == "twophase":
		return "Saturated Region"
	elif current_phase in ["gas", "supercritical_gas"]:
		return "Superheated Region"
	elif current_phase in ["liquid", "supercritical_liquid"]:
		return "Liquid Region"
	elif current_phase == "supercritical":
		return "Supercritical Region"
	else:
		return "Unknown Region"

def roundup(x):
	return math.ceil(x*100)/100

def expression(current_mach):
	if nozzles == True:
		return current_mach <= m_limit
	else:
		return current_mach >= m_limit

def mu_solve(T,s):
	try:
		current_phase = phase(T,s)
		if current_phase == 'twophase':
			mu_g = PropsSI("V","T",T,"Q",1,w_fluid)
			mu_f = PropsSI("V","T",T,"Q",0,w_fluid)
			x = PropsSI("Q","T",T,"S",s,w_fluid)
			if x < 0: x = 0
			if x > 1: x = 1
			return x*mu_g + (1-x)*mu_f
		else:
			return PropsSI("V","T",T,'S',s,w_fluid)
	except:
		return 1e-5

def get_properties_safe(T, s):
	"""Safe property calculation with fallbacks"""
	try:
		current_phase = phase(T, s)
		
		if current_phase == "twophase":
			x = PropsSI('Q','T',T,'S',s,w_fluid)
			if x < 0: x = 0
			if x > 1: x = 1
			
			P = PropsSI('P','T',T,'Q',x,w_fluid)
			v_f = 1/PropsSI('D','T',T,'Q',0,w_fluid)
			v_g = 1/PropsSI('D','T',T,'Q',1,w_fluid)
			v = v_f + x*(v_g - v_f)
			
			h_f = PropsSI('H','T',T,'Q',0,w_fluid)
			h_g = PropsSI('H','T',T,'Q',1,w_fluid)
			h = h_f + x*(h_g - h_f)
			
			return P, x, v, h
			
		else:
			P = PropsSI('P','T',T,'S',s,w_fluid)
			x = PropsSI('Q','T',T,'S',s,w_fluid) if current_phase in ["twophase", "gas"] else 1.0
			v = 1/PropsSI('D','T',T,'S',s,w_fluid)
			h = PropsSI('H','T',T,'S',s,w_fluid)
			
			return P, x, v, h
			
	except Exception as e:
		print(f"Property calculation error: {e}")
		return 101325, 1.0, 1.0, 2.5e6

# MODIFIED MOMENTUM FUNCTION WITH FRICTION
def mome(i,T_2,s_2):
	try:
		mat_s_h[0,1] = T_2
		mat_s_h[0,2] = s_2
		
		P, x, v, h = get_properties_safe(T_2, s_2)
		mat_s_h[0,3] = P
		mat_s_h[0,4] = x
		mat_s_h[0,5] = v
		mat_s_h[0,6] = h
		
		h_initial = mat[0,6]
		u_initial = mat[0,7]
		
		u_new = math.sqrt(max(1, 2*(h_initial - h) + u_initial**2))
		mat_s_h[0,7] = u_new
		
		a = sound(T_2, s_2)
		mat_s_h[0,8] = m = u_new / a if a > 0 else 0.1
		
		mass_flux_old = mat[i,7] / mat[i,5] 
		mass_flux_new = u_new / v
		
		if mass_flux_new < 1e-10:
			mass_flux_new = 1e-10
			
		area_ratio = mass_flux_old / mass_flux_new
		area_ratio = max(0.1, min(10.0, area_ratio))
		
		A_new = mat[i,13] * area_ratio
		D_new = math.sqrt(4 * A_new / math.pi)
		mat_s_h[0,9] = D_new
		mat_s_h[0,13] = A_new
		
		current_mach = m
		current_phase = phase(T_2, s_2)
		
		if nozzles == True:
			if not throat:
				if current_mach < 0.98:
					if D_new < mat[i,9]:
						mat_s_h[0,12] = dL = abs((mat[i,9] - D_new)/(2*math.tan(math.radians(deg_c))))
					else:
						mat_s_h[0,12] = dL = 0.001
				else:
					throat.append(i)
					print(f"*** THROAT IDENTIFIED at iteration {i} - Mach = {current_mach:.3f}, Phase: {current_phase} ***")
					mat_s_h[0,12] = dL = 0.001
			else:
				if D_new > mat[i,9]:
					mat_s_h[0,12] = dL = abs((D_new - mat[i,9])/(2*math.tan(math.radians(deg_d))))
				else:
					mat_s_h[0,12] = dL = 0.001
		else:
			dL = 0.001
				
		mat_s_h[0,10] = L = mat[i,10] + dL
		mat_s_h[0,11] = D_new/2
		mat_s_h[0,14] = mu = mu_solve(T_2,s_2)
		mat_s_h[0,15] = ms = u_new * A_new / v
		
		# FRICTION FACTOR CALCULATION USING BLASIUS CORRELATION
		Re = u_new * D_new / (mu * v) if (mu * v) > 0 else 1e6
		fr = 0.079 * (Re)**-0.25
		mat_s_h[0,16] = fr
		
		mat_s_h[0,17] = rho = 1/v
		
		# MOMENTUM EQUATION WITH FRICTION
		AdP = (mat[i,13] + A_new) * (mat[i,3] - P) / 2
		
		# FRICTION FORCE CALCULATION
		Ff = -((((mat[i,16] * mat[i,15]**2 * mat[i,5]) / (mat[i,13] * mat[i,9])) + 
		       ((fr * ms**2 * v) / (A_new * D_new)) * dL))
		
		mdu = -mat[i,15] * (u_new - mat[i,7])
		
		residual = AdP + Ff + mdu
		return residual
		
	except Exception as e:
		print(f"Error in mome function: {e}")
		return 1e10

# NORMAL SHOCK FUNCTIONS WITH FRICTION
def mome_2(i_shock_index, i_diff, T_2, s_2):
	"""Momentum equation solver for shock diffuser points"""
	try:
		mat_s_h[0,1] = T_2
		mat_s_h[0,2] = s_2
		
		P, x, v, h = get_properties_safe(T_2, s_2)
		mat_s_h[0,3] = P
		mat_s_h[0,4] = x
		mat_s_h[0,5] = v
		mat_s_h[0,6] = h
		
		u_new = math.sqrt(max(1, 2*shock_values[i_shock_index,i_diff,6] - 2*h + shock_values[i_shock_index,i_diff,7]**2))
		mat_s_h[0,7] = u_new
		
		a = sound(T_2, s_2)
		mat_s_h[0,8] = m = u_new / a if a > 0 else 0.1
		
		D_new = shock_values[i_shock_index,i_diff,9] * math.sqrt((v * shock_values[i_shock_index,i_diff,7]) / (shock_values[i_shock_index,i_diff,5] * u_new))
		mat_s_h[0,9] = D_new
		A_new = math.pi/4 * D_new**2
		mat_s_h[0,13] = A_new
		
		if nozzles == True:
			if m < 1:
				mat_s_h[0,12] = dL = abs((shock_values[i_shock_index,i_diff,9] - D_new)/(2*math.tan(math.radians(deg_c))))
			else:
				mat_s_h[0,12] = dL = abs((shock_values[i_shock_index,i_diff,9] - D_new)/(2*math.tan(math.radians(deg_d))))
		else:
			if m > 1:
				mat_s_h[0,12] = dL = abs((shock_values[i_shock_index,i_diff,9] - D_new)/(2*math.tan(math.radians(deg_c))))
			else:
				mat_s_h[0,12] = dL = abs((D_new - shock_values[i_shock_index,i_diff,9])/(2*math.tan(math.radians(deg_d))))
				
		mat_s_h[0,10] = L = shock_values[i_shock_index,i_diff,10] + dL
		mat_s_h[0,11] = D_new/2
		mat_s_h[0,14] = mu = mu_solve(T_2, s_2)
		mat_s_h[0,15] = ms = u_new * A_new / v
		
		# FRICTION FACTOR CALCULATION USING BLASIUS CORRELATION
		Re = u_new * D_new / (mu * v) if (mu * v) > 0 else 1e6
		fr = 0.079 * (Re)**-0.25
		mat_s_h[0,16] = fr
		
		mat_s_h[0,17] = rho = 1/v
		
		# MOMENTUM EQUATION WITH FRICTION
		AdP = (shock_values[i_shock_index,i_diff,13] + A_new) * (shock_values[i_shock_index,i_diff,3] - P) / 2
		
		# FRICTION FORCE CALCULATION
		Ff = -((((shock_values[i_shock_index,i_diff,16] * shock_values[i_shock_index,i_diff,15]**2 * shock_values[i_shock_index,i_diff,5]) / 
		        (shock_values[i_shock_index,i_diff,13] * shock_values[i_shock_index,i_diff,9])) + 
		       ((fr * ms**2 * v) / (A_new * D_new)) * dL))
		
		mdu = -shock_values[i_shock_index,i_diff,15] * (u_new - shock_values[i_shock_index,i_diff,7])
		
		return AdP + Ff + mdu
		
	except Exception as e:
		print(f"Error in mome_2 function: {e}")
		return 1e10

def shock_function(u, k_m, k_e, k_mo):
	"""Normal shock governing equations solver"""
	try:
		rho_2 = k_m / u
		h_2 = k_e - u**2 / 2
		P_2 = PropsSI("P", "D", rho_2, "H", h_2, w_fluid)
		return k_mo - (P_2 + rho_2 * u**2)
	except:
		return 1e10

# Initial state calculation
print("Calculating initial state properties...")
try:
	P, x, v, h = get_properties_safe(T, s)
	a = sound(T, s)
	m = u/a
	A = (math.pi/4)*D**2
	ms = u*A/v
	mu = mu_solve(T, s)
	
	# Calculate initial friction factor
	Re = u * D / (mu * v) if (mu * v) > 0 else 1e6
	fr = 0.079 * (Re)**-0.25
	
	print(f"Initial state verification:")
	print(f"  T={T} K, s={s} J/kg-K")
	print(f"  P={P/1000:.1f} kPa, h={h/1000:.1f} kJ/kg")
	print(f"  Phase: {phase(T,s)}, Quality: {x:.3f}")
	print(f"  Region: {region(T, s)}, Mach: {m:.3f}")
	print(f"  Reynolds number: {Re:.0f}, Friction factor: {fr:.4f}")
	
except Exception as e:
	print(f"Error in initial state calculation: {e}")
	P = 101325
	x = 1.0
	v = 1.0
	h = 2.5e6
	m = 0.1
	A = 0.049
	ms = 1.0
	fr = 0.02

# Initialize data arrays
i_total = 1000
mat = np.zeros((i_total, 18))
mat_s_h = np.zeros((1, 18))
i = 0
phase_s = phase(T, s)

# Store Data at State 0
mat[i,0] = i
mat[i,1] = T
mat[i,2] = s
mat[i,3] = P
mat[i,4] = x
mat[i,5] = v
mat[i,6] = h
mat[i,7] = u
mat[i,8] = m
mat[i,9] = D
mat[i,10] = L
mat[i,11] = D/2
mat[i,12] = dL
mat[i,13] = A
mat[i,14] = mu_solve(T, s)
mat[i,15] = ms
mat[i,16] = fr  # Now storing actual friction factor
mat[i,17] = 1/v

print(f"Starting simulation with max {i_total} iterations...")
print(f"Temperature decrement: {T_idec}K per iteration")

# Store initial conditions
h0 = h
u0 = u
A0 = A
s0 = s

# Track phase changes
previous_phase = phase_s
phase_changes = []

# HEM Model Algorithm
while expression(m) and i < i_total-1:
	try:
		current_T = mat[i,1]
		current_s = mat[i,2]
		current_mach = mat[i,8]
		current_phase = phase(current_T, current_s)
		
		if nozzles and current_mach >= m_limit:
			print(f"Reached target Mach number {m_limit} at iteration {i}")
			break
			
		if current_phase == "twophase":
			T_step = T_idec * 0.5
		else:
			T_step = T_idec
			
		if nozzles:
			T_new = current_T - T_step
		else:
			T_new = current_T + T_step
			
		if T_new < T_triple + 5:
			print(f"Reached minimum temperature at iteration {i}")
			break
			
		s_new = current_s
		
		P_new, x_new, v_new, h_new = get_properties_safe(T_new, s_new)
		
		if np.isnan(h_new) or h_new > 1e10:
			break
			
		u_new = math.sqrt(max(1, 2*(h0 - h_new) + u0**2))
		a_new = sound(T_new, s_new)
		m_new = u_new / a_new if a_new > 0 else 0.1
		
		area_ratio = (mat[i,7] / mat[i,5]) / (u_new / v_new)
		area_ratio = max(0.5, min(2.0, area_ratio))
		
		A_new = mat[i,13] * area_ratio
		D_new = math.sqrt(4 * A_new / math.pi)
		
		i = i + 1
		if i >= len(mat):
			break
			
		mat[i,0] = i
		mat[i,1] = T_new
		mat[i,2] = s_new
		mat[i,3] = P_new
		mat[i,4] = x_new
		mat[i,5] = v_new
		mat[i,6] = h_new
		mat[i,7] = u_new
		mat[i,8] = m_new
		mat[i,9] = D_new
		mat[i,13] = A_new
		
		new_phase = phase(T_new, s_new)
		if new_phase != previous_phase:
			phase_changes.append((i, previous_phase, new_phase))
			print(f"Phase change at iteration {i}: {previous_phase} -> {new_phase}")
			previous_phase = new_phase
		
		current_mach = m_new
		if nozzles:
			if not throat:
				if current_mach < 0.99:
					dL = abs((mat[i-1,9] - D_new)/(2*math.tan(math.radians(deg_c))))
				else:
					throat.append(i)
					print(f"*** THROAT FOUND at iteration {i} ***")
					print(f"    Mach = {current_mach:.3f}, Phase = {new_phase}")
					print(f"    T = {T_new:.1f}K, P = {P_new/1000:.1f}kPa")
					dL = 0.001
			else:
				dL = abs((D_new - mat[i-1,9])/(2*math.tan(math.radians(deg_d))))
		else:
			dL = 0.001
			
		mat[i,10] = mat[i-1,10] + dL
		mat[i,11] = D_new/2
		mat[i,12] = dL
		mat[i,14] = mu_new = mu_solve(T_new, s_new)
		mat[i,15] = ms_new = u_new * A_new / v_new
		
		# Calculate friction factor using Blasius correlation
		Re_new = u_new * D_new / (mu_new * v_new) if (mu_new * v_new) > 0 else 1e6
		fr_new = 0.079 * (Re_new)**-0.25
		mat[i,16] = fr_new
		
		mat[i,17] = 1/v_new
		
		if i % 20 == 0 or (throat and i == throat[0] + 1) or current_mach > 0.999:
			status = f"Progress: {i} iterations, Mach: {m_new:.3f}"
			if throat:
				status += f" (Diverging)" if i > throat[0] else f" (Converging)"
			status += f", T: {T_new:.1f}K, P: {P_new/1000:.1f}kPa, f: {fr_new:.4f}"
			#print(status) #for diagnostics
			
		if m_new > 5 or P_new < 500 or D_new > 50 * D_i or u_new > 2000:
			break
			
	except Exception as e:
		print(f"Error at iteration {i}: {e}")
		if i > 10:
			i = i - 1
			T_idec *= 0.5
			print(f"Reducing temperature step to {T_idec}K")
		else:
			break

# Store final iteration count
i_final = i

# NORMAL SHOCK ANALYSIS - IMPROVED SHOCK PLACEMENT
if nozzles and shock_solve and throat:
	print("\n=== SOLVING FOR NORMAL SHOCKS ===")
	shock_values = np.zeros((shock_count, i_total, 18))
	i_shock_index = 0
	i_shock = []
	i_diff_list = []
	i_fr_list = []
	
	throat_index = throat[0]
	diverging_start = throat_index
	diverging_end = i_final - 1
	
	diverging_length_indices = diverging_end - diverging_start
	shock_spacing = diverging_length_indices // (shock_count + 1)
	
	print(f"Diverging section: {diverging_start} to {diverging_end} (length: {diverging_length_indices} iterations)")
	print(f"Shock spacing: {shock_spacing} iterations")
	
	for count in range(1, shock_count + 1):
		shock_pos = diverging_start + count * shock_spacing
		
		if shock_pos < i_final and shock_pos > throat_index:
			i_shock.append(shock_pos)
			print(f"Shock {count} placed at iteration {shock_pos} (position {count}/6 in diverging section)")
	
	for shock_index in i_shock:
		if shock_index >= i_final:
			continue
			
		print(f"Solving shock at iteration {shock_index}, Mach = {mat[shock_index, 8]:.3f}")
		
		k_m = mat[shock_index, 17] * mat[shock_index, 7]
		k_e = mat[shock_index, 6] + mat[shock_index, 7]**2 / 2
		k_mo = mat[shock_index, 17] * mat[shock_index, 7]**2 + mat[shock_index, 3]
		
		u_0 = mat[0, 7]
		u_1 = mat[shock_index, 7]
		rel = 1e6
		tol = 1e-10
		u_m_0 = 1e6
		
		while rel > tol:
			u_m = (u_1 + u_0) / 2
			f_u_0 = shock_function(u_0, k_m, k_e, k_mo)
			f_u_1 = shock_function(u_1, k_m, k_e, k_mo)
			f_u_m = shock_function(u_m, k_m, k_e, k_mo)
			rel = abs((u_m_0 - u_m) / u_m)
			
			if f_u_0 * f_u_m < 0:
				u_1 = u_m
			elif f_u_0 * f_u_m > 0:
				u_0 = u_m
			else:
				break
			u_m_0 = u_m
		
		rho_2 = k_m / u_m
		h_2 = k_e - u_m**2 / 2
		
		try:
			T_2 = PropsSI("T", "D", rho_2, "H", h_2, w_fluid)
			s_2 = PropsSI("S", "D", rho_2, "H", h_2, w_fluid)
			P_2 = PropsSI("P", "D", rho_2, "H", h_2, w_fluid)
			x_2 = PropsSI("Q", "D", rho_2, "H", h_2, w_fluid)
			
			shock_values[i_shock_index, 1, 0] = shock_index
			shock_values[i_shock_index, 1, 1] = T_2
			shock_values[i_shock_index, 1, 2] = s_2
			shock_values[i_shock_index, 1, 3] = P_2
			shock_values[i_shock_index, 1, 4] = x_2
			shock_values[i_shock_index, 1, 5] = 1/rho_2
			shock_values[i_shock_index, 1, 6] = h_2
			shock_values[i_shock_index, 1, 7] = u_m
			shock_values[i_shock_index, 1, 8] = u_m / sound(T_2, s_2)
			shock_values[i_shock_index, 1, 9] = mat[shock_index, 9]
			shock_values[i_shock_index, 1, 10] = mat[shock_index, 10]
			shock_values[i_shock_index, 1, 11] = mat[shock_index, 9] / 2
			shock_values[i_shock_index, 1, 12] = mat[shock_index, 12]
			shock_values[i_shock_index, 1, 13] = mat[shock_index, 13]
			shock_values[i_shock_index, 1, 14] = mu_solve(T_2, s_2)
			shock_values[i_shock_index, 1, 15] = mat[shock_index, 15]
			
			# Calculate friction factor for shock point
			mu_shock = mu_solve(T_2, s_2)
			Re_shock = u_m * mat[shock_index, 9] / (mu_shock * (1/rho_2)) if (mu_shock * (1/rho_2)) > 0 else 1e6
			fr_shock = 0.079 * (Re_shock)**-0.25
			shock_values[i_shock_index, 1, 16] = fr_shock
			
			shock_values[i_shock_index, 1, 17] = rho_2
			
			shock_values[i_shock_index, 0, :] = mat[shock_index, :]
			
			print(f"Shock solved: M_front = {mat[shock_index, 8]:.3f}, M_back = {shock_values[i_shock_index, 1, 8]:.3f}")
			
			# FANNO-RAYLEIGH LINES CALCULATION
			s_limit = 50
			d = 2
			u_f = mat[shock_index, 7]
			i_fr = 0

			while u_f > s_limit and i_fr < f_steps:
				try:
					rho_f = k_m / u_f
					h_f = k_e - u_f**2 / 2
					s_f = PropsSI("S", "D", rho_f, "H", h_f, w_fluid)
					line_f[i_shock_index, i_fr, 0] = s_f
					line_f[i_shock_index, i_fr, 1] = h_f
					
					P_f = k_mo - rho_f * u_f**2
					h_f_r = PropsSI("H", "D", rho_f, "P", P_f, w_fluid)
					s_f_r = PropsSI("S", "D", rho_f, "P", P_f, w_fluid)
					line_r[i_shock_index, i_fr, 0] = s_f_r
					line_r[i_shock_index, i_fr, 1] = h_f_r
					
					u_f = u_f - d
					i_fr = i_fr + 1
				except:
					break

			i_fr_list.append(i_fr)
			
			# SHOCK DIFFUSER FLOW CALCULATION
			print(f"  Calculating diffuser flow after shock...")
			i_diff = 1
			max_diff_steps = min(50, i_total - shock_index - 1)
			
			for diff_step in range(2, max_diff_steps):
				try:
					T_diff = shock_values[i_shock_index, diff_step-1, 1] + T_idec
					s_diff = shock_values[i_shock_index, diff_step-1, 2]
					
					P_diff, x_diff, v_diff, h_diff = get_properties_safe(T_diff, s_diff)
					
					u_diff = math.sqrt(max(1, 2*(shock_values[i_shock_index, 1, 6] - h_diff) + 
										shock_values[i_shock_index, 1, 7]**2))
					
					area_ratio_diff = (shock_values[i_shock_index, 1, 7] / shock_values[i_shock_index, 1, 5]) / (u_diff / v_diff)
					area_ratio_diff = max(0.5, min(2.0, area_ratio_diff))
					
					A_diff = shock_values[i_shock_index, 1, 13] * area_ratio_diff
					D_diff = math.sqrt(4 * A_diff / math.pi)
					
					shock_values[i_shock_index, diff_step, 0] = shock_index + diff_step
					shock_values[i_shock_index, diff_step, 1] = T_diff
					shock_values[i_shock_index, diff_step, 2] = s_diff
					shock_values[i_shock_index, diff_step, 3] = P_diff
					shock_values[i_shock_index, diff_step, 4] = x_diff
					shock_values[i_shock_index, diff_step, 5] = v_diff
					shock_values[i_shock_index, diff_step, 6] = h_diff
					shock_values[i_shock_index, diff_step, 7] = u_diff
					shock_values[i_shock_index, diff_step, 8] = u_diff / sound(T_diff, s_diff)
					shock_values[i_shock_index, diff_step, 9] = D_diff
					shock_values[i_shock_index, diff_step, 10] = shock_values[i_shock_index, 1, 10] + diff_step * 0.001
					shock_values[i_shock_index, diff_step, 11] = D_diff / 2
					shock_values[i_shock_index, diff_step, 12] = 0.001
					shock_values[i_shock_index, diff_step, 13] = A_diff
					shock_values[i_shock_index, diff_step, 14] = mu_diff = mu_solve(T_diff, s_diff)
					shock_values[i_shock_index, diff_step, 15] = ms_diff = u_diff * A_diff / v_diff
					
					# Calculate friction factor for diffuser points
					Re_diff = u_diff * D_diff / (mu_diff * v_diff) if (mu_diff * v_diff) > 0 else 1e6
					fr_diff = 0.079 * (Re_diff)**-0.25
					shock_values[i_shock_index, diff_step, 16] = fr_diff
					
					shock_values[i_shock_index, diff_step, 17] = 1/v_diff
					
					i_diff = diff_step
					
				except Exception as e:
					print(f"    Error in diffuser step {diff_step}: {e}")
					break
			
			i_diff_list.append(i_diff)
			print(f"  Calculated {i_diff} diffuser points")
			
		except Exception as e:
			print(f"Error solving shock properties: {e}")
			continue
		
		i_shock_index += 1

# Results
current_region = region(mat[i_final-1,1], mat[i_final-1,2])

print(current_region,":" ,model,"Normal Shocks")
print("Converging-Diverging Nozzle")
print("Initial Temperature: ", T_i, "K")
print("Initial Entropy: ", s_i, "kJ/kg-K")
print("Initial Velocity: ", u_i, "m/s")
print("Initial Diameter: ", D_i, "m")
print("Angle of Convergence and Divergence: ", deg_c,"deg and",deg_d,"deg")
print("Starting Phase: ", phase_s)
print("End Phase: ", phase(mat[i_final-1,1], mat[i_final-1,2]))
print("No. of points: "+ str(i_final))
print("Length at throat: ", mat[throat[0],10])
if phase_changes:
	print("Length after Phase Change: ",mat[phase_changes[0][0],10])
if shock_solve and 'i_shock' in locals() and i_shock_index > 0:
	for shocks in i_shock:
		print("Shock Points Mach Number: {}".format(roundup(mat[shocks,8])))

#Plot Locus of Points
if shock_solve and 'i_shock' in locals() and i_shock_index > 0:
	index = 0
	for shocks in i_shock:
		axes = plt.gca()
		axes.grid(color='b', linestyle='--', linewidth=0.5)
		plt.plot(line_f[index,:i_fr_list[index],0]/1e3,line_f[index,:i_fr_list[index],1]/1e3, color='c',label='Fanno Line')
		plt.plot(line_r[index,:i_fr_list[index],0]/1e3,line_r[index,:i_fr_list[index],1]/1e3, color='m',label='Rayleigh Line')
		plt.xlabel('Entropy kJ/kg-K')
		plt.ylabel('Enthalpy kJ/kg')
		plt.title('Converging-Diverging '+str(profile)+': Normal Shocks \n'+ str(model)+': '+str(current_region)+'\n'+'Fanno - Rayleigh Line Intersection at Shock Front @M = {}'.format(roundup(mat[shocks,8])))
		plt.scatter(shock_values[index,0,2]/1e3,shock_values[index,0,6]/1e3, color='r',label='Front of Shock: M = {}'.format(roundup(mat[shocks,8])))
		plt.scatter(shock_values[index,1,2]/1e3,shock_values[index,1,6]/1e3, color='g',label='Back of Shock: M = {}'.format(roundup(shock_values[index,1,8])))
		plt.legend()
		index +=1
		plt.show()

fig, (plot_1, plot_2, plot_3) = plt.subplots(3)
fig.suptitle('Converging-Diverging '+str(profile)+': Normal Shocks \n'+ str(model)+': '+str(current_region))
#Radius vs Length
axes = plt.gca()
plot_1.axes.grid(color='b', linestyle='--', linewidth=0.5)
plot_1.plot(mat[0:i_final,10], mat[0:i_final,11], color='r',label='{}'.format(profile))
plot_1.plot(mat[0:i_final,10], -mat[0:i_final,11], color='r')
plot_1.set_xlim(0,mat[i_final-1,10])
#Mach vs Length
plot_2.axes.grid(color='b', linestyle='--', linewidth=0.5)
plot_2.plot(mat[0:i_final,10], mat[0:i_final,8], color='g',label='Mach Number')
plot_2.set_xlim(0,mat[i_final-1,10])

#Pressure vs Length
plot_3.axes.grid(color='b', linestyle='--', linewidth=0.5)
plot_3.plot(mat[0:i_final,10], mat[0:i_final,3]/1e3, color='b', label='Pressure')
plot_3.set_xlim(0,mat[i_final-1,10])

if shock_solve and 'i_shock' in locals() and i_shock_index > 0:
	index = 0
	for shocks in i_shock:
		plot_2.plot(shock_values[index,:i_diff_list[index],10],shock_values[index,:i_diff_list[index],8], label='@M = {}'.format(roundup(mat[shocks,8])))
		plot_3.plot(shock_values[index,:i_diff_list[index],10],shock_values[index,:i_diff_list[index],3]/1e3, label='@M = {}'.format(roundup(mat[shocks,8])))
		index += 1
plot_1.legend()
plot_2.legend()
plot_3.legend()
fig.text(0.5, 0.04, 'Length (m)', ha='center', va='center')
fig.text(0.045, 0.7, 'Radius (m)', ha='center', rotation='vertical')
fig.text(0.045, 0.41, 'Mach Number', ha='center', rotation='vertical')
fig.text(0.045, 0.12, 'Pressure (kPa)', ha='center', rotation='vertical')
plt.show()

#T-S Diagram
plotting = "T - S Diagram"
axes = plt.gca()
axes.grid(color='b', linestyle='--', linewidth=0.5)
steps = round((T_crit*.99999999 - T_triple*1.00000001)/(T_idec))
mat_sat = np.zeros((steps*2,2))
row = 0
for T_val in np.arange(T_triple*1.00000001,T_crit*.99999999,T_idec):
	for k in range(0,2,1):
		try:
			mat_sat[row,0]=T_val
			mat_sat[row,1]=PropsSI("S","T",T_val,"Q",k,w_fluid)
			row=row+1
		except:
			continue
plt.plot(mat_sat[:row,1],mat_sat[:row,0],color='c', label='Saturated Region', linewidth=2)
T_s = np.zeros((i_final,1))
T_s.fill(s_i)
plt.plot(T_s[:i_final],mat[:i_final,1],color='g',linewidth=2, label='Isentropic Process')
plt.plot(mat[:i_final,2],mat[:i_final,1],color='r',linewidth=2, label='Actual Process')

if shock_solve and 'i_shock' in locals() and i_shock_index > 0:
	index = 0
	for shocks in i_shock:
		plt.plot(shock_values[index,:i_diff_list[index],2],shock_values[index,:i_diff_list[index],1], label='Normal Shock @M = {}'.format(roundup(mat[shocks,8])))
		index += 1
plt.title('Converging-Diverging '+str(profile)+': Normal Shocks \n'+ str(model)+': '+str(current_region))
plt.xlabel("Entropy (J / kg-K)")
plt.ylabel("Temperature (K)")
plt.legend()
plt.show()

print("\n=== SIMULATION COMPLETE ===")
if throat:
	print(f"Throat transition achieved at Mach {mat[throat[0],8]:.3f}")
else:
	print("No throat transition achieved")
print(f"Total iterations: {i_final}")



# anon, when this code was built in 2024, only God and I understood how it works
# today, only God knows how it works

