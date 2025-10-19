#ISENTROPIC HOMOGENEOUS EQUILIBRIUM MODEL FOR CONVERGING-DIVERGING NOZZLES AND DIFFUSER
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import PropsSI
from CoolProp.Plots import PropertyPlot
import math
import numpy as np
import matplotlib.pyplot as plt

#Converging Diverging Nozzles and Diffuser Cases
case_nozzle = np.array([
    # 1st Case - Saturated Region
    [373.15, 20, 4500, 1.5, 20, 45],  # T, u, s, M_limit, deg_c, deg_d
    
    # 2nd Case - Superheated Region
    [600, 100, 8500, 1.5, 20, 20],  # T, u, s, M_limit, deg_c, deg_d
    
    # 3rd Case - Supercritical to Saturated
    [670, 50, 4400, 1.5, 15, 25],  # T, u, s, M_limit, deg_c, deg_d
    
    # 4th Case - Superheated to Saturated
    [550, 20, 6000, 1.5, 20, 20],  # T, u, s, M_limit, deg_c, deg_d
    
    # 5th Case - Saturated Region
    [373.15, 400, 4500, 0.2, 20, 45],  # T, u, s, M_limit, deg_c, deg_d
    
    # 6th Case - Superheated Region
    [600, 900, 8500, 0.1, 20, 20],  # T, u, s, M_limit, deg_c, deg_d
    
    # 7th Case - Supercritical to Saturated
    [630, 350, 4400, 0.1, 15, 25],  # T, u, s, M_limit, deg_c, deg_d
    
    # 8th Case - Superheated to Saturated
    [500, 700, 5700, 0.1, 20, 20]  # T, u, s, M_limit, deg_c, deg_d
])

##########################################
########### Case Conditions ############
case = 8 
##########################################

# Set nozzle or diffuser mode based on case number
nozzles = True if case <= 4 else False

model = "Isentropic Homogeneous Equilibrium Model"

# For diffuser operation, we need to modify the algorithm
if nozzles == False:
    case_idx = case - 1
    T_i = case_nozzle[case_idx,0]				            # Initial Temperature
    u_i = case_nozzle[case_idx,1]				            # Initial Velocity
    s_i = case_nozzle[case_idx,2]				            # Initial Entropy
    m_limit = case_nozzle[case_idx,3]				        # Maximum Mach Number
    deg_c = case_nozzle[case_idx,4]				        # Convergence angle
    deg_d = case_nozzle[case_idx,5]				        # Divergence angle
else:
    T_i = case_nozzle[case-1,0]				            # Initial Temperature
    u_i = case_nozzle[case-1,1]				            # Initial Velocity
    s_i = case_nozzle[case-1,2]				            # Initial Entropy
    m_limit = case_nozzle[case-1,3]				        # Maximum Mach Number
    deg_c = case_nozzle[case-1,4]				        # Convergence angle
    deg_d = case_nozzle[case-1,5]				        # Divergence angle

T = T_i
u = u_i
s = s_i
w_fluid = 'Water'
dT = 10e-6
T_idec = 1

T_triple = PropsSI("T_triple",w_fluid)
T_crit = PropsSI("Tcrit",w_fluid)
D = D_i = 10
L = L_i = 0
dL = 0
ph = []
phase_change = False
throat = []

print("Initializing...")
#Define Functions
def phase():
    return CP.PhaseSI('T',T,'S',s,w_fluid)

def pty_s(f_property,s):
    return PropsSI('{}'.format(f_property),'T',T,'S',s,w_fluid)

def pty_x(f_property,x):
    return PropsSI('{}'.format(f_property),'T',T,'Q',x,w_fluid)

def fdm_s(f_property,s):
    return PropsSI('{}'.format(f_property),'T',T+dT,'S',s,w_fluid)-PropsSI('{}'.format(f_property),'T',T-dT,'S',s,w_fluid)

def fdm_x(f_property,x):
    return PropsSI('{}'.format(f_property),'T',T+dT,'Q',x,w_fluid)-PropsSI('{}'.format(f_property),'T',T-dT,'Q',x,w_fluid)

def sound():
    if phase() == "twophase":
        s_g = pty_x("S",1)
        s_f = pty_x("S",0)
        v_g = 1 / pty_x("D",1)
        v_f = 1 / pty_x("D",0)
        dP_dT = (s_g-s_f)/(v_g-v_f)
        x = pty_s("Q",s)
        v = v_f + (v_g-v_f)*x
        dsg_dT = fdm_x("S",1)/(2*dT)
        dsf_dT = fdm_x("S",0)/(2*dT)
        dvg_dT = -v_g**2*fdm_x("D",1)/(2*dT)
        dvf_dT = -v_f**2*fdm_x("D",0)/(2*dT)
        dv_dT = -1*(x*dsg_dT+(1-x)*dsf_dT)/dP_dT+x*dvg_dT+(1-x)*dvf_dT
        return math.sqrt((-1*v**2)*dP_dT/dv_dT)
    else:
        dP = fdm_s("P",s)
        drho = fdm_s("D",s)
        return math.sqrt(dP/drho)

def region():
    if phase_s==phase():
        if phase()=="twophase":
            return "Saturated Region"
        else:
            return "Superheated Region"
    else:
        return "Superheated - Saturated Region"

def roundup(x):
    return math.ceil(x*100)/100

#Assigning of variables
P = pty_s("P",s)
x = pty_s("Q",s)
v = 1/pty_s("D",s)
h = pty_s("H",s)
m = u/sound()
i_total = round((T_crit - T_triple)/(abs(T_idec)))
mat = np.empty((i_total,13))
i = 0
phase_s=phase()

#IHEM Model Algorithm (Nozzles/Diffusers)
if nozzles:
    while m <= m_limit and i < i_total:
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
        ph.append(str(phase()))
        T = T - T_idec
        P = pty_s("P",s)
        x = pty_s("Q",s)
        v = 1/pty_s("D",s)
        h = pty_s("H",s)
        u = math.sqrt(2*mat[i,6]-2*h+mat[i,7]**2)
        m = u/sound()
        D = mat[i,9]*math.sqrt((v*mat[i,7])/(mat[i,5]*u))
        if m<1:
            dL = (mat[i,9]-D)/(2*math.tan(math.radians(deg_c)))
        else:
            dL = (D-mat[i,9])/(2*math.tan(math.radians(deg_d)))
            throat.append(i)
        L = dL + mat[i,10]
        if i > 0 and ph[i-1] != ph[i]:
            phase_change_index = i
            phase_change = True
        i=i+1
else:
    T_idec = -1
    target_mach = m_limit
    
    while m >= target_mach and i < i_total:
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
        ph.append(str(phase()))
        
        T = T - T_idec
        
        P = pty_s("P",s)
        x = pty_s("Q",s)
        v = 1/pty_s("D",s)
        h = pty_s("H",s)
        
        u = math.sqrt(2*mat[i,6]-2*h+mat[i,7]**2)
        m = u/sound()
        
        D = mat[i,9]*math.sqrt((v*mat[i,7])/(mat[i,5]*u))
        
        if m > 1:
            dL = (mat[i,9]-D)/(2*math.tan(math.radians(deg_c)))
        else:
            dL = (D-mat[i,9])/(2*math.tan(math.radians(deg_d)))
            throat.append(i)
        
        L = dL + mat[i,10]
        
        if i > 0 and ph[i-1] != ph[i]:
            phase_change_index = i
            phase_change = True
            
        i = i + 1

i_final = i
print(region(),":" ,model)

case_descriptions = [
    "Saturated Region (inside saturation dome) - NOZZLE - T=373.15K (100°C), u=20 m/s, s=4500 J/kg-K (inside dome), M=1.5, conv=20°, div=45°",
    
    "Superheated Region (outside saturation dome) - NOZZLE - T=600K (326.85°C), u=100 m/s, s=8500 J/kg-K (superheated), M=1.5, conv=20°, div=20°",
    
    "Supercritical to Saturated (starts supercritical, crosses into saturation) - NOZZLE - T=700K (426.85°C > Tcrit), u=50 m/s, s=5800 J/kg-K (will cross into saturation), M=1.5, conv=15°, div=25°",
    
    "Superheated to Saturated (starts superheated, crosses into saturation) - NOZZLE - T=550K (276.85°C), u=20 m/s, s=7500 J/kg-K (superheated→saturated), M=1.5, conv=20°, div=20°",
    
    "Saturated Region (inside saturation dome) - DIFFUSER - T=373.15K (100°C), u=300 m/s, s=4500 J/kg-K (inside dome), M=0.3, conv=20°, div=45°",
    
    "Superheated Region (outside saturation dome) - DIFFUSER - T=600K (326.85°C), u=400 m/s, s=8500 J/kg-K (superheated), M=0.3, conv=20°, div=20°",
    
    "Supercritical to Saturated (starts supercritical, crosses into saturation) - DIFFUSER - T=630K, u=350 m/s, s=4400 J/kg-K, M=0.3, conv=15°, div=25°",
    
    "Superheated to Saturated (starts superheated, crosses into saturation) - DIFFUSER - T=570K, u=250 m/s, s=5700 J/kg-K, M=0.3, conv=20°, div=20°"
]

print(f"Case {case}: {case_descriptions[case-1]}")
print("="*80)

print("Initial Conditions:")
print(f"  Temperature: {T_i} K ({T_i - 273.15:.2f} °C)")
print(f"  Velocity: {u_i} m/s")
print(f"  Entropy: {s_i} J/kg-K")
print(f"  Target Mach Number: {m_limit}")
print(f"  Convergence Angle: {deg_c}°")
print(f"  Divergence Angle: {deg_d}°")

if nozzles == False:
    profile = 'Diffuser Profile'
    print("Converging-Diverging Diffuser")
    print("Initial Temperature: ", T_i, "K")
    print("Initial Entropy: ", s_i, "J/kg-K")
    print("Initial Velocity: ", u_i, "m/s")
    print("Initial Diameter: ", D_i, "m")
    print("Initial Mach Number: ", u_i/sound())
else:
    profile = 'Nozzle Profile'
    print("Converging-Diverging Nozzle")
    print("Initial Temperature: ", T_i, "K")
    print("Initial Entropy: ", s_i, "J/kg-K")
    print("Initial Velocity: ", u_i, "m/s")
    print("Initial Diameter: ", D_i, "m")
    
print("Angle of Convergence and Divergence: ", deg_c,"deg and",deg_d,"deg")
print("Starting Phase: ", phase_s)
print("End Phase: ", phase())
print("No. of iterations: "+ str(i_final))
if throat:
    print("Length at throat: ", mat[throat[0],10])
if phase_change == True:
    print("Length after Phase Change: ",mat[phase_change_index,10])

#Plot Locus of Points
fig, (plot_1, plot_2, plot_3) = plt.subplots(3)
fig.suptitle(f'Case {case}: {case_descriptions[case-1].split(" - ")[0]} - {profile}')
#Radius vs Length
axes = plt.gca()
plot_1.axes.grid(color='b', linestyle='--', linewidth=0.5)
plot_1.plot(mat[0:i,10], mat[0:i,11], color='r',label=f'{profile} Profile')
plot_1.plot(mat[0:i,10], -mat[0:i,11], color='r')
plot_1.legend()
#Mach vs Length
plot_2.axes.grid(color='b', linestyle='--', linewidth=0.5)
plot_2.plot(mat[0:i,10], mat[0:i,8], color='g',label='Mach Number')
plot_2.legend()
#Pressure vs Length
plot_3.axes.grid(color='b', linestyle='--', linewidth=0.5)
plot_3.plot(mat[0:i,10], mat[0:i,3]/1e3, color='b', label='Pressure (kPa)')
plot_3.legend()
fig.text(0.5, 0.04, 'Length (m)', ha='center', va='center')
fig.text(0.045, 0.7, 'Radius (m)', ha='center', rotation='vertical')
fig.text(0.045, 0.41, 'Mach Number', ha='center', rotation='vertical')
fig.text(0.045, 0.12, 'Pressure (kPa)', ha='center', rotation='vertical')
plt.show()

#T-S Diagram
plotting = "T - S Diagram"
plt.figure(figsize=(10, 8))
axes = plt.gca()
axes.grid(color='b', linestyle='--', linewidth=0.5)

T_triple = PropsSI("T_triple", w_fluid)
T_crit = PropsSI("Tcrit", w_fluid)

num_points = 200
T_range = np.linspace(T_triple * 1.0001, T_crit * 0.9999, num_points)

s_sat_l = np.zeros_like(T_range)
s_sat_v = np.zeros_like(T_range)

for idx, T_val in enumerate(T_range):
    try:
        s_sat_l[idx] = PropsSI('S', 'T', T_val, 'Q', 0, w_fluid)
        s_sat_v[idx] = PropsSI('S', 'T', T_val, 'Q', 1, w_fluid)
    except:
        s_sat_l[idx] = np.nan
        s_sat_v[idx] = np.nan

plt.fill_betweenx(T_range, s_sat_l, s_sat_v, color='lightblue', alpha=0.5, label='Saturation Dome')
plt.plot(s_sat_l, T_range, 'b-', linewidth=2, label='Saturated Liquid Line')
plt.plot(s_sat_v, T_range, 'r-', linewidth=2, label='Saturated Vapor Line')

plt.plot(mat[:i,2], mat[:i,1], 'g-', linewidth=3, label='Isentropic Process')

s_crit = PropsSI('S', 'T', T_crit, 'Q', 1, w_fluid)
plt.plot(s_crit, T_crit, 'ko', markersize=8, label='Critical Point')

plt.title(f'Case {case}: {case_descriptions[case-1].split(" - ")[0]} - {profile}')
plt.xlabel("Entropy (J / kg-K)")
plt.ylabel("Temperature (K)")

all_s_values = np.concatenate([s_sat_l, s_sat_v, mat[:i,2]])
s_min = np.nanmin(all_s_values) * 0.95
s_max = np.nanmax(all_s_values) * 1.05

all_T_values = np.concatenate([T_range, mat[:i,1]])
T_min = np.nanmin(all_T_values) * 0.95
T_max = np.nanmax(all_T_values) * 1.05

plt.xlim(s_min, s_max)
plt.ylim(T_min, T_max)

plt.legend(loc='best')
plt.tight_layout()

plt.show()

# anon, when this code was built in 2024, only God and I understood how it works
# today, only God knows how it works
