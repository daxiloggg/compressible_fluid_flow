#HOMOGENEOUS EQUILIBRIUM MODEL FOR CONVERGING-DIVERGING NOZZLES AND DIFFUSERS
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import PropsSI
import math
import numpy as np
import matplotlib.pyplot as plt

#Converging Diverging Nozzles and Diffusers
case_nozzle = np.array([
    # 1st Case - Saturated Region (inside saturation dome) - NOZZLE
    [373.15, 20, 4500, 1.5, 20, 45],  # T, u, s, M_limit, deg_c, deg_d
  
    # 2nd Case - Superheated Region (outside saturation dome) - NOZZLE
    [600, 100, 8500, 1.5, 20, 20],  # T, u, s, M_limit, deg_c, deg_d
  
    # 3rd Case - Supercritical to Saturated (starts supercritical, crosses into saturation) - NOZZLE
    [670, 50, 4400, 1.5, 15, 25],  # T, u, s, M_limit, deg_c, deg_d
  
    # 4th Case - Superheated to Saturated (starts superheated, crosses into saturation) - NOZZLE
    [550, 20, 6000, 1.5, 20, 20],  # T, u, s, M_limit, deg_c, deg_d
    
    # 5th Case - Saturated Region (inside saturation dome) - DIFFUSER
    [373.15, 500, 4500, 0.2, 45, 30],  # T, u, s, M_limit, deg_c, deg_d
    
    # 6th Case - Superheated Region (outside saturation dome) - DIFFUSER
    [600, 900, 8500, 0.1, 30, 20],  # T, u, s, M_limit, deg_c, deg_d
    
    # 7th Case - Supercritical to Saturated (starts supercritical, crosses into saturation) - DIFFUSER
    [630, 900, 4400, 0.1, 15, 25],  # T, u, s, M_limit, deg_c, deg_d
    
    # 8th Case - Superheated to Saturated (starts superheated, crosses into saturation) - DIFFUSER
    [500, 600, 6000, 0.1, 10, 15]  # T, u, s, M_limit, deg_c, deg_d
])

#Initialization
case = 8  # Case selector
nozzles = True if case <= 4 else False  # Nozzle/Diffuser flag
model = "Homogeneous Equilibrium Model"

if nozzles == True:
    T = T_i = case_nozzle[case-1,0]    # Initial Temperature
    u = u_i = case_nozzle[case-1,1]    # Initial Velocity
    s = s_i = case_nozzle[case-1,2]    # Initial Entropy
    m_limit = case_nozzle[case-1,3]    # Maximum Mach Number
    deg_c = case_nozzle[case-1,4]      # Convergence angle
    deg_d = case_nozzle[case-1,5]      # Divergence angle
    profile = 'Nozzle Profile'
else:
    T = T_i = case_nozzle[case-1,0]    # Initial Temperature
    u = u_i = case_nozzle[case-1,1]    # Initial Velocity
    s = s_i = case_nozzle[case-1,2]    # Initial Entropy
    m_limit = case_nozzle[case-1,3]    # Minimum Mach Number
    deg_c = case_nozzle[case-1,4]      # Convergence angle
    deg_d = case_nozzle[case-1,5]      # Divergence angle
    profile = 'Diffuser Profile'

w_fluid = 'Water'
dT = 10e-6

# Adaptive temperature decrement
if nozzles:
    if case == 1:  # Saturated region
        T_idec = 0.1
    elif case == 2:  # Superheated region
        T_idec = 0.5  
    elif case == 3:  # Near saturation
        T_idec = 0.2
    else:  # Default
        T_idec = 0.3
else:  # Diffuser cases
    if case == 5:  # Saturated region
        T_idec = 0.1
    elif case == 6:  # Superheated region
        T_idec = 0.3
    elif case == 7:  # Supercritical
        T_idec = 0.2
    else:  # Default
        T_idec = 0.25

T_triple = PropsSI("T_triple",w_fluid)
T_crit = PropsSI("Tcrit",w_fluid)
P_crit = PropsSI("Pcrit",w_fluid)
D = D_i = 0.25
L = L_i = 0
dL = 0
ph = []
phase_change = False
throat = []

print("Initializing...")
print(f"Case {case}: {profile}")
print(f"T={T_i}K, s={s_i}J/kg-K, u={u_i}m/s, M_limit={m_limit}")

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

def pty_Tx(f_property,T,x):
    try:
        return PropsSI('{}'.format(f_property),'T',T,'Q',x,w_fluid)
    except:
        return np.nan

def sound(T,s):
    try:
        current_phase = phase(T,s)
        if current_phase == "twophase":
            # Homogeneous equilibrium model for two-phase speed of sound
            x = PropsSI('Q','T',T,'S',s,w_fluid)
            if x < 0 or x > 1:
                x = 0.5
                
            # Get properties at saturation
            rho_f = PropsSI('D','T',T,'Q',0,w_fluid)
            rho_g = PropsSI('D','T',T,'Q',1,w_fluid)
            h_f = PropsSI('H','T',T,'Q',0,w_fluid)
            h_g = PropsSI('H','T',T,'Q',1,w_fluid)
            P_sat = PropsSI('P','T',T,'Q',0.5,w_fluid)
            
            # Homogeneous mixture density
            rho_m = 1 / (x/rho_g + (1-x)/rho_f)
            
            # Approximate speed of sound in two-phase flow
            a_two_phase = math.sqrt(P_sat / rho_m * 1.3)
            return max(50, a_two_phase)
            
        elif current_phase in ["gas", "supercritical_gas", "supercritical"]:
            return PropsSI('A','T',T,'S',s,w_fluid)
        else:  # Liquid or supercritical liquid
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

def expression(current_mach, nozzles, m_limit):
    """Simulation continuation condition"""
    if nozzles:
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
            # Linear mixing rule for viscosity
            return x*mu_g + (1-x)*mu_f
        else:
            return PropsSI("V","T",T,"S",s,w_fluid)
    except:
        return 1e-5

def get_properties_safe(T, s):
    """Safe property calculation with fallbacks"""
    try:
        current_phase = phase(T, s)
        
        if current_phase == "twophase":
            # Two-phase properties
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
            # Single-phase properties
            P = PropsSI('P','T',T,'S',s,w_fluid)
            x = PropsSI('Q','T',T,'S',s,w_fluid) if current_phase in ["twophase", "gas"] else 1.0
            v = 1/PropsSI('D','T',T,'S',s,w_fluid)
            h = PropsSI('H','T',T,'S',s,w_fluid)
            
            return P, x, v, h
            
    except Exception as e:
        print(f"Property calculation error: {e}")
        return 101325, 1.0, 1.0, 2.5e6

# MODIFIED MOMENTUM FUNCTION FOR BOTH NOZZLES AND DIFFUSERS
def mome(i,T_2,s_2):
    try:
        mat_s_h[0,1] = T_2
        mat_s_h[0,2] = s_2
        
        # Get properties using safe function
        P, x, v, h = get_properties_safe(T_2, s_2)
        mat_s_h[0,3] = P
        mat_s_h[0,4] = x
        mat_s_h[0,5] = v
        mat_s_h[0,6] = h
        
        # Energy equation
        h_initial = mat[0,6]
        u_initial = mat[0,7]
        
        # Calculate velocity from energy conservation
        u_new = math.sqrt(max(1, 2*(h_initial - h) + u_initial**2))
        mat_s_h[0,7] = u_new
        
        # Calculate Mach number
        a = sound(T_2, s_2)
        mat_s_h[0,8] = m = u_new / a if a > 0 else 0.1
        
        # Mass conservation
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
        
        # MODIFIED GEOMETRY CALCULATION FOR BOTH NOZZLES AND DIFFUSERS
        current_mach = m
        
        if nozzles == True:
            # NOZZLE GEOMETRY
            if not throat:
                if current_mach < 1.01:
                    if D_new < mat[i,9]:
                        mat_s_h[0,12] = dL = abs((mat[i,9] - D_new)/(2*math.tan(math.radians(deg_c))))
                    else:
                        mat_s_h[0,12] = dL = 0.001
                else:
                    throat.append(i)
                    print(f"*** THROAT IDENTIFIED at iteration {i} - Mach = {current_mach:.3f} ***")
                    mat_s_h[0,12] = dL = 0.001
            else:
                if D_new > mat[i,9]:
                    mat_s_h[0,12] = dL = abs((D_new - mat[i,9])/(2*math.tan(math.radians(deg_d))))
                else:
                    mat_s_h[0,12] = dL = 0.001
        else:
            # DIFFUSER GEOMETRY
            if not throat:
                if current_mach > 0.99:
                    if D_new > mat[i,9]:
                        mat_s_h[0,12] = dL = abs((D_new - mat[i,9])/(2*math.tan(math.radians(deg_d))))
                    else:
                        mat_s_h[0,12] = dL = 0.001
                else:
                    throat.append(i)
                    print(f"*** THROAT IDENTIFIED at iteration {i} - Mach = {current_mach:.3f} ***")
                    mat_s_h[0,12] = dL = 0.001
            else:
                if D_new < mat[i,9]:
                    mat_s_h[0,12] = dL = abs((mat[i,9] - D_new)/(2*math.tan(math.radians(deg_c))))
                else:
                    mat_s_h[0,12] = dL = 0.001
                
        mat_s_h[0,10] = L = mat[i,10] + dL
        mat_s_h[0,11] = D_new/2
        mat_s_h[0,14] = mu = mu_solve(T_2,s_2)
        mat_s_h[0,15] = ms = u_new * A_new / v
        mat_s_h[0,16] = 0.079
        mat_s_h[0,17] = 1/v
        
        # Momentum equation residual
        AdP = (mat[i,13] + A_new) * (mat[i,3] - P) / 2
        mdu = -mat[i,15]*(u_new-mat[i,7])
        
        residual = AdP + mdu
        return residual
        
    except Exception as e:
        print(f"Error in mome function: {e}")
        return 1e10

# Initial state calculation
print("Calculating initial state properties...")
try:
    P, x, v, h = get_properties_safe(T, s)
    a = sound(T, s)
    m = u/a
    A = (math.pi/4)*D**2
    ms = u*A/v
    
    print(f"Initial state verification:")
    print(f"  T={T} K, s={s} J/kg-K")
    print(f"  P={P/1000:.1f} kPa, h={h/1000:.1f} kJ/kg")
    print(f"  Phase: {phase(T,s)}, Quality: {x:.3f}")
    print(f"  Region: {region(T, s)}, Mach: {m:.3f}")
    
except Exception as e:
    print(f"Error in initial state calculation: {e}")
    P = 101325
    x = 1.0
    v = 1.0
    h = 2.5e6
    m = 0.1
    A = 0.049
    ms = 1.0

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
mat[i,16] = 0.079
mat[i,17] = 1/v

print(f"Starting simulation with initial conditions:")
print(f"  Device: {profile}")
print(f"  Temperature: {T} K")
print(f"  Entropy: {s} J/kg-K") 
print(f"  Velocity: {u} m/s")
print(f"  Mach: {m:.3f}")
print(f"  Phase: {phase_s}")
print(f"  Region: {region(T, s)}")

# ADAPTIVE ITERATION LOGIC FOR BOTH NOZZLES AND DIFFUSERS
throat_found = False
max_iterations = min(600, i_total - 1)

print(f"Starting simulation with max {max_iterations} iterations...")
print(f"Temperature {'decrement' if nozzles else 'increment'}: {T_idec}K per iteration")
print(f"Geometry angles: Converging={deg_c}°, Diverging={deg_d}°")

# Store initial conditions for conservation laws
h0 = h
u0 = u
A0 = A
s0 = s

# Track phase changes
previous_phase = phase_s
phase_changes = []

while i < max_iterations and expression(m, nozzles, m_limit):
    try:
        current_T = mat[i,1]
        current_s = mat[i,2]
        current_mach = mat[i,8]
        current_phase = phase(current_T, current_s)
        
        # Stop if we've reached the Mach limit
        if nozzles and current_mach >= m_limit:
            print(f"Reached target Mach number {m_limit} at iteration {i}")
            break
        elif not nozzles and current_mach <= m_limit:
            print(f"Reached target Mach number {m_limit} at iteration {i}")
            break
            
        # Adaptive temperature step based on phase
        if current_phase == "twophase":
            T_step = T_idec * 0.5
        else:
            T_step = T_idec
            
        if nozzles == True:
            T_new = current_T - T_step
        else:
            T_new = current_T + T_step
            
        # Temperature bounds
        if nozzles and T_new < T_triple + 5:
            print(f"Reached minimum temperature at iteration {i}")
            break
        elif not nozzles and T_new > T_crit * 1.5:
            print(f"Reached maximum temperature at iteration {i}")
            break
            
        # Use constant entropy process
        s_new = current_s
        
        # Get properties for new state
        P_new, x_new, v_new, h_new = get_properties_safe(T_new, s_new)
        
        if np.isnan(h_new) or h_new > 1e10:
            print(f"Invalid enthalpy at iteration {i}, stopping")
            break
            
        # Calculate velocity from energy equation
        u_new = math.sqrt(max(1, 2*(h0 - h_new) + u0**2))
        
        # Calculate Mach number
        a_new = sound(T_new, s_new)
        m_new = u_new / a_new if a_new > 0 else 0.1
        
        # Calculate area from mass conservation
        area_ratio = (mat[i,7] / mat[i,5]) / (u_new / v_new)
        area_ratio = max(0.5, min(2.0, area_ratio))
        
        A_new = mat[i,13] * area_ratio
        D_new = math.sqrt(4 * A_new / math.pi)
        
        # Store results
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
        
        # Track phase changes
        new_phase = phase(T_new, s_new)
        if new_phase != previous_phase:
            phase_changes.append((i, previous_phase, new_phase))
            print(f"Phase change at iteration {i}: {previous_phase} -> {new_phase}")
            previous_phase = new_phase
        
        # GEOMETRY CALCULATION FOR BOTH NOZZLES AND DIFFUSERS
        current_mach = m_new
        if nozzles == True:
            # NOZZLE GEOMETRY
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
            # DIFFUSER GEOMETRY
            if not throat:
                if current_mach > 1.01:
                    dL = abs((D_new - mat[i-1,9])/(2*math.tan(math.radians(deg_d))))
                else:
                    throat.append(i)
                    print(f"*** THROAT FOUND at iteration {i} ***")
                    print(f"    Mach = {current_mach:.3f}, Phase = {new_phase}")
                    print(f"    T = {T_new:.1f}K, P = {P_new/1000:.1f}kPa")
                    dL = 0.001
            else:
                dL = abs((mat[i-1,9] - D_new)/(2*math.tan(math.radians(deg_c))))
            
        mat[i,10] = mat[i-1,10] + dL
        mat[i,11] = D_new/2
        mat[i,12] = dL
        mat[i,14] = mu_solve(T_new, s_new)
        mat[i,15] = u_new * A_new / v_new
        mat[i,16] = 0.079
        mat[i,17] = 1/v_new
        
        # Adaptive progress reporting
        report_interval = 10 if current_phase != "twophase" else 5
        if i % report_interval == 0 or (throat and i == throat[0] + 1) or (nozzles and current_mach > 0.8) or (not nozzles and current_mach < 0.5):
            status = f"Progress: {i} iterations, Mach: {m_new:.3f}"
            if throat:
                if nozzles:
                    status += f" (Diverging)" if i > throat[0] else f" (Converging)"
                else:
                    status += f" (Converging)" if i > throat[0] else f" (Diverging)"
            status += f", T: {T_new:.1f}K, P: {P_new/1000:.1f}kPa"
            status += f", Phase: {new_phase}"
            if new_phase == "twophase":
                status += f", x: {x_new:.3f}"
            
        # Additional stopping conditions
        if nozzles and (m_new > 5 or P_new < 500 or D_new > 50 * D_i or u_new > 2000):
            break
        elif not nozzles and (m_new < 0.01 or P_new > 1e7 or D_new < 0.01 * D_i or u_new < 1):
            break
            
    except Exception as e:
        print(f"Error at iteration {i}: {e}")
        if i > 10:
            i = i - 1
            T_idec *= 0.5
            print(f"Reducing temperature step to {T_idec}K")
        else:
            break

# Results
i_final = i
if i_final > 0:
    current_region = region(mat[i_final-1,1], mat[i_final-1,2])
else:
    current_region = "Unknown"

print(f"\n=== SIMULATION RESULTS ===")
print(f"{current_region} : {model}")
print(f"Case: {case} - {profile}")
print(f"Initial Temperature: {T_i} K")
print(f"Initial Entropy: {s_i} J/kg-K")
print(f"Initial Velocity: {u_i} m/s")
print(f"Initial Diameter: {D_i} m")
print(f"Final Mach number: {mat[i_final-1,8]:.3f}" if i_final > 0 else "N/A")
print(f"Final Phase: {phase(mat[i_final-1,1], mat[i_final-1,2])}" if i_final > 0 else "N/A")

if phase_changes:
    print(f"\nPhase changes occurred at:")
    for change in phase_changes:
        print(f"  Iteration {change[0]}: {change[1]} -> {change[2]}")

if throat:
    throat_length = mat[throat[0],10]
    if nozzles:
        converging_length = throat_length
        diverging_length = mat[i_final-1,10] - throat_length if i_final > throat[0] else 0
    else:
        diverging_length = throat_length
        converging_length = mat[i_final-1,10] - throat_length if i_final > throat[0] else 0
        
    total_length = mat[i_final-1,10]

    print(f"\n=== {profile.upper()} SECTION LENGTHS ===")
    if nozzles:
        print(f"Converging section length: {converging_length:.4f} m")
        print(f"Diverging section length: {diverging_length:.4f} m")
    else:
        print(f"Diverging section length: {diverging_length:.4f} m")
        print(f"Converging section length: {converging_length:.4f} m")
    print(f"Total length: {total_length:.4f} m")
    print(f"Throat location: {throat_length:.4f} m")
    print(f"Throat diameter: {mat[throat[0],9]:.4f} m")
    print(f"Throat Mach number: {mat[throat[0],8]:.3f}")
    print(f"Throat Phase: {phase(mat[throat[0],1], mat[throat[0],2])}")

# ENHANCED PLOTTING FOR BOTH NOZZLES AND DIFFUSERS
if i_final > 10:
    fig, (plot_1, plot_2, plot_3) = plt.subplots(3)
    fig.suptitle('Converging-Diverging '+str(profile)+'\n'+ str(model)+': '+str(current_region))
    
    # Radius vs Length
    axes = plt.gca()
    plot_1.axes.grid(color='b', linestyle='--', linewidth=0.5)
    plot_1.plot(mat[0:i_final,10], mat[0:i_final,11], color='r', label=profile, linewidth=2)
    plot_1.plot(mat[0:i_final,10], -mat[0:i_final,11], color='r', linewidth=2)
    plot_1.set_xlim(0, mat[i_final-1,10])
    plot_1.set_ylabel('Radius (m)')
    
    # Mark throat if exists
    if throat:
        throat_idx = throat[0]
        plot_1.axvline(x=mat[throat_idx,10], color='k', linestyle='--', alpha=0.7, label='Throat')
    
    # Mach vs Length
    plot_2.axes.grid(color='b', linestyle='--', linewidth=0.5)
    plot_2.plot(mat[0:i_final,10], mat[0:i_final,8], color='g', label='Mach Number', linewidth=2)
    plot_2.set_xlim(0, mat[i_final-1,10])
    plot_2.set_ylabel('Mach Number')
    plot_2.axhline(y=1, color='k', linestyle='--', alpha=0.7, label='Sonic')
    
    # Mark throat if exists
    if throat:
        plot_2.axvline(x=mat[throat_idx,10], color='k', linestyle='--', alpha=0.7)
    
    # Pressure vs Length
    plot_3.axes.grid(color='b', linestyle='--', linewidth=0.5)
    plot_3.plot(mat[0:i_final,10], mat[0:i_final,3]/1e3, color='b', label='Pressure', linewidth=2)
    plot_3.set_xlim(0, mat[i_final-1,10])
    plot_3.set_ylabel('Pressure (kPa)')
    plot_3.set_xlabel('Length (m)')
    
    # Mark throat if exists
    if throat:
        plot_3.axvline(x=mat[throat_idx,10], color='k', linestyle='--', alpha=0.7)
    
    plot_1.legend()
    plot_2.legend()
    plot_3.legend()
    
    # Add common labels
    fig.text(0.5, 0.04, 'Length (m)', ha='center', va='center')
    fig.text(0.045, 0.7, 'Radius (m)', ha='center', rotation='vertical')
    fig.text(0.045, 0.41, 'Mach Number', ha='center', rotation='vertical')
    fig.text(0.045, 0.12, 'Pressure (kPa)', ha='center', rotation='vertical')
    
    plt.tight_layout()
    plt.show()
    
    # T-S Diagram
    plotting = "T - S Diagram"
    fig_ts, ax_ts = plt.subplots(figsize=(10, 8))
    ax_ts.grid(color='b', linestyle='--', linewidth=0.5)
    
    # Generate saturation curve
    steps = round((T_crit*.99999999 - T_triple*1.00000001)/(T_idec))
    mat_sat = np.zeros((steps*2,2))
    row = 0
    for T_val in np.arange(T_triple*1.00000001, T_crit*.99999999, T_idec):
        for k in range(0,2,1):
            try:
                mat_sat[row,0] = T_val
                mat_sat[row,1] = PropsSI("S","T",T_val,"Q",k,w_fluid)
                row = row + 1
            except:
                continue
    
    # Plot saturation region
    ax_ts.plot(mat_sat[:row,1], mat_sat[:row,0], color='c', label='Saturated Region', linewidth=2)
    
    # Plot isentropic process line
    T_s = np.zeros((i_final,1))
    T_s.fill(s_i)
    ax_ts.plot(T_s[:i_final], mat[:i_final,1], color='g', linewidth=2, label='Isentropic Process')
    
    # Plot actual process
    ax_ts.plot(mat[:i_final,2], mat[:i_final,1], color='r', linewidth=2, label='Actual Process')
    
    ax_ts.set_title('Converging-Diverging '+str(profile)+'\n'+ str(model)+': '+str(current_region))
    ax_ts.set_xlabel("Entropy (J / kg-K)")
    ax_ts.set_ylabel("Temperature (K)")
    ax_ts.legend()
    plt.tight_layout()
    plt.show()

print("\n=== SIMULATION COMPLETE ===")
if throat:
    print(f"Throat transition achieved at Mach {mat[throat[0],8]:.3f}")
else:
    print("No throat transition achieved")

print(f"Total iterations: {i_final}")

# anon, when this code was built in 2024, only God and I understood how it works
# today, only God knows how it works
