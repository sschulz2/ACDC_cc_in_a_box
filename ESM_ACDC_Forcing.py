import numpy as np
import matplotlib.pyplot as plt

def e_s(T): # saturation vapor pressure
	TC = T - 273.15
	# Function to calculate saturation mixing ratio. Returns pressure in mbar
	e_s = 6.11*10**(7.5*TC/(237.5+TC))  # TEMPERATURES IN CELSIUS!!!
	return(e_s)

def q_s(T,P): #clausius clapeyron
	es = e_s(T)
	return(0.622*es/(P - 0.378*es))

def year_sum(Array): #sum over a year
    the_sum = np.zeros(int(len(Array)/4))
    i = 0
    while i < len(the_sum):
        the_sum[i] = np.sum(Array[i*4:(i+1)*4])
        i+=1
    return(the_sum)

def C3(c_L,F):
    ###### CONSTANTS #################################################
    O = 210			# O2 concentration mbar
    k_c = 2.5 		# turnover time of RuP2 carboxylase at 25˚C (1/s) 
    k_o = 0.21*k_c 		# turnover time of RuP2 oxygenase at 25˚C (1/s)
    tau_c = 1/k_c
    tau_o = 1/k_o
    K_C = .46		# Michaelis constant for CO2 (mbar)
    K_O = 330		# Michaelis constant for O2 (mbar) 
    E_t = 87		# Total concentration of enzyme sites in chloroplast (umol/(gChl))
    R_p = 300		# pool size of potential acceptors [umol]	
    j_max = 467		# light saturated electron transport (micro mol electrons/gCHl/s)
    m = 2*k_c*E_t		# maximum rate of reduction of PGA (micro mol/gChl/s)
    f_RCHL = .41            # sunlight absorbed by chloroplasts
    lam_v = 600e-09		# wavelength of visible light [m]
    nu_v = (3e8)/lam_v	# frequency of visible light
    h = 6.626e-34		# Planck's constant [Js]
    Avo = 6.022e23		# Avogadro's number [#/mol]	
    rho_CHL = 0.45		# g CHL/m^2 Leaf
    M_CO2 = 44.009e-03	# molar mass of CO2 [kg/mol]
    M_H2O = 18.01e-03	# molar mass of water [kg/mol]
    R_d = 0			# "dark" respiration [umol/m^2/s]
    ################################################################
    phi = tau_c*(O/K_O)/(tau_o*(c_L/K_C)) # ratio of oxygenation to carboxylation
    omega_c = (c_L)/(tau_c*(c_L + (K_C*(1 + (O/K_O)))))
    I = (F/(h*nu_v*Avo))*1e6# photon flux density (micro mol photons/m^2/s)
    j = 0.5*f_RCHL*I/rho_CHL # electron transport rate (micro mol e-/g CHL/s)
    if j > j_max: # saturation value
        j = j_max
    jp = j/(4 + (4*phi))  
    mp = m/(2 + (1.5*phi))
    p = omega_c*R_p
    b = jp + p + (jp*p/mp)
    c = jp*p
    vc_Rlimit = 0.5*(b - (((b**2) - (4*c))**0.5))
    vc = min(vc_Rlimit,omega_c*E_t)                         # umol CO2/gCHl/s 
    A = M_CO2*(1e-6)*((rho_CHL*(1 - (0.5*phi))*vc) - R_d)   # kg CO2/m^2/s
    Q_P = M_H2O*(1e-06)*rho_CHL*j	                    # kg H2O/m^2/s
    return(A,Q_P)

def calc_Dc(c_A,c_L,W):
    ######### CONSTANTS ##################################
    M_CO2 = 44.009e-03		# molar mass of CO2 [kg/mol]
    M_air = 28.96e-03		# molar mass of air [kg/mol]
    rho_a = 1.25			# density of air [kg/m^3]
    P_a = 1013
    ####### DIFFUSION OF CO_2 into leaf [kg CO2/m^2 leaf area /s]
    g_V = 1/200			# Stomatal conductance of veg [m/s]
    D_c = (M_CO2/M_air)*(1/P_a)*rho_a*g_V*W*(c_A - c_L)
    return(D_c)

#def C3_leaf_module(F,w_V,theta_A,theta_F,T_S,q_A,ep_A,ep_F):
def GPP_Module(F,W,c_A):
######## BASED ON QUICK EQUILIBRATION BETWEEN LEAF AND ATMOS.	
    N = 100
    c_V = np.linspace(.01,c_A,N) # range of possible C_L values
    i = 0
    D = np.zeros(N)
    A = np.zeros(N)
    QP = np.zeros(N)
    while i < N:
        D[i] = calc_Dc(c_A,c_V[i],W)
        A[i],QP[i] = C3(c_V[i],F)
        i+=1
    Cidx = np.argmin(abs(D - A))
    A = A*(12/44.) # Converting to carbon mass [kg C/m^2/s]
    return(c_V[Cidx],A[Cidx],QP[Cidx])

def rate_constants(T,S):
    # From 1 The carbon dioxide system in seawater: equilibrium chemistry and measurements // Andrew G. Dickson  
    # T in Kelvin
    # S in parts per thousand
    Tcomp = (93.4517*(100/T)) - 60.2409 + (23.3585*np.log(T/100.))
    Scomp = S*(0.023517 - (0.023656*(T/100.)) + (0.0047036*((T/100)**2)))

    K_zed = np.exp(Tcomp + Scomp) # mol/kg

    one = (-3633.86/T) + 61.2172 - (9.67770*np.log(T)) + (0.011555*S) - (0.0001152*(S**2))
    K_1 = 10**(one) # mol/kg

    two = (-471.78/T) - 25.9290 + (3.16967*np.log(T)) + (0.01781*S) - (0.0001122*(S**2))
    K_2 = 10**(two) # mol/kg

    return(K_zed,K_1,K_2)

def the_model(nyear = 3000,E_ = lambda t: 0, F_ext = lambda t: 0):
    # the 0*t is a hacky way to make sure the function returns an array
    nyear = int(nyear)
    ###### GLOBAL PARAMETERS
    dt = 365*24*60*60/4. #time step of 3 months
    N = nyear * 4 # number of time steps
    time = np.linspace(0,nyear-1,N)
    MCO2 = 44 # MOLAR MASS OF CO2 g/mol
    MC = 12     # MOLAR MASS OF C
    MW = 18     # MOLAR MASS OF WATER
    MAIR = 29 # MOLAR MASS OF AIR

    F_vec = np.vectorize(F_ext) #vectorize so that the function can be specified as scalar or not
    E_vec = np.vectorize(E_)

    F_ext = F_vec(time) #W/m^2? 
    #E = np.zeros(N)
    E = E_vec(time) #kgC 

    ################ ATMOS PARAMS
    c_base = 280    # base CO2 concentration [ppm]
    T_base = 275.15 # Baseline temperature [K]
    alpha = 5.35    # CO2 forcing parameter [W/m^2]
    lam = 1.4       # Climate Sensitivity [W/m^2/K]
    M_A = 5.2e18    # Mass of atmosphere [kg]
    A_A = 5.1e14    # area of atmosphere [m^2]
    c_pa = 1e3      # specific enthalpy of air [J/kg/K]    
    C_A = (c_pa*M_A/A_A) # heat capacity of atmosphere [Ws/m^2/K]
    POOL_ATMOS = c_base*(MCO2/MAIR)*(1e-06)*(M_A)*(MC/MCO2)*(1e-12) #ppmv to PgC

    ################ OCEAN PARAMS
    DO = 4000                   # depth of deep ocean [m]
    kappa = 4e-4                # turbulent diffusivity [m^2/s] (citation?)
    rho_L = 1030                # Density of sea water [kg/m^3]
    ALK = 2322                  # Average Alkalnity [umol/kg]
    S = 35                      # Average Salinity [parts per thousand]
    f_O = 0.7                   # fraction of earth's surface occupied by ocean
    A_O = f_O*A_A               # area of the ocean
    O_layers = 50               # Number of ocean layers
    H_O = DO/O_layers           # Layer Thickness [m]    
    Tau_ML = (np.pi*1e8)/10.    # Timescale of air-sea mixing for energy and carbon

    ############## LAND PARAMS
    C_VEG = 450e12              # Mass of vegetation (kg C)
    C_SOIL = 1700e12            # Mass of soil carbon (kg C)
    f_B = 0.14                  # Fraction of earth that's composed of biosphere
    P_A = 1000                  # Surface Pressure on the land surface
    PRECIP = 715/(np.pi*1e7)    # Precipitation rate, annual mean [kg/m^2/s]
    G_S = 1/200                 # Surface conductance [m/s]
    rho_A = 1.25                # Air density at land surface [kg air/m^3]
    RH = 0.5                    # Average land surface Relative Humidity [fraction]
    rho_W = 1000                # density of pure liquid water
    theta_X = 0.5               # maximum soil porosity
    a_o = 0.75                  # minimum respiration
    a_1 = 0.001                 # respiration sensitivity to temperature
    beta = 0.25                 # fraction of GPP that goes into soil

    # ORIGINAL PARAMS
    k_o = 2.9/(np.pi*1e9)       # timescale of heterotropic respiration [1/s]
    k_1 = 1/(np.pi*1e11)        # sensitivity of het. respiration to temperature [1/s/K] 
    # Testing
    #k_o = 2.9/(np.pi*1e10)       # timescale of heterotropic respiration [1/s]
    #k_1 = 1/(np.pi*1e12)        # sensitivity of het. respiration to temperature [1/s/K] 
    ################# ATMOS STATE VARIABLES
    c_A = np.zeros(N)                       #ATMOSPHERIC CO2 CONCENTRATION [ppmv]
    T_A = np.zeros(N)                       #ATMOSPHERIC TEMPERATURE ANOMALY [˚C]
    ################## OCEAN STATE VARIABLES
    T_O = np.zeros(shape=(N,O_layers))      # DEEP OCEAN TEMPERATURE ANOMALY PROFILE [˚C] 
    DIC = np.ones(shape=(N,O_layers))       # Dissolved Inorganic Carbon [umol/kg] preset so pco2 = 280 in the ocean
    pco2 = np.zeros(N-1)
    ################## LAND STATE VARIABLES
    W_X = np.zeros(N)                       # SOIL SATURATION [-]
    C_VEG = np.zeros(N)                     # Vegetation carbon stock [kg]
    C_SOIL = np.zeros(N)                    # Soil carbon stock [kg]
    ################## FLUXES TO KEEP TRACK OF
    F_AS = np.zeros(shape=(N))
    F_LAND = np.zeros(shape=N)
    dummy = np.zeros(shape=N)
    F_rad = np.zeros(shape=N)
    ################# INITIAL CONDITIONS
    i = 0
    c_A[i] = c_base
    W_X[i] = 0.5
    DIC = DIC*2111
    C_VEG[i] = 450e12                   # kg C
    C_SOIL[i] = 3100e12                 # kg C

    while i < N-1:
        # Emissions Flux
        E_flux = (E[i]/M_A)*(MCO2/MC)*(MAIR/MCO2)*(1e6)/dt # in ppmv CO2 /s
        # RADIATIVE FORCING AT TOA
        F_G = alpha*np.log(c_A[i]/c_base)                   # Greenhouse effect via CO2
        F = F_G + F_ext[i]                                # Total radiative forcing [W/m^2]
        F_rad[i+1] = F

        SWX = 200                                            # Summertime max shortwave [W/m^2]
        #Calculate pco2 at surface
        K_0,K_1,K_2 = rate_constants(T_O[i,:] + T_base,S) # all in mol/kg except K0 in mol/kg/atmos
        pco2[i] = (MCO2/MAIR)*(K_2[0])*(((2*DIC[i,0]) - ALK)**2)/((K_1[0]*K_0[0])*(ALK - DIC[i,0])) #ppmv
        # Air sea energy mixing
        E_airsea = (T_A[i] - T_O[i,0])/Tau_ML                  	# in K/s
        C_airsea = K_0[0]*(c_A[i] - pco2[i])/Tau_ML 	    	# in umol CO2/kg/s
        # Diffusive ocean energy mixing
        DE_surf = kappa*(T_O[i,0] - T_O[i,1])/(H_O**2)                      # Finite difference K/s
        DOE     = kappa*(T_O[i,:-2] - 2*T_O[i,1:-1] + T_O[i,2:])/(H_O**2)   # Finite differene K/s
        DC_surf = kappa*(DIC[i,0] - DIC[i,1])/(H_O**2)                      # Finite difference umol C/kg/s
        DOC     = kappa*(DIC[i,:-2] - 2*DIC[i,1:-1] + DIC[i,2:])/(H_O**2)   # Finite difference umol C/kg/s
        # Primary Productivity on the land surface
        c_v,GPP,QP = GPP_Module(SWX,W_X[i],P_A*c_A[i]*1e-06)                # GPP Calculation kg C/m^2/s
        # Respiration
        R_VEG = (a_o + (a_1*(T_A[i])/10))*GPP                               # kg C /m^2/s
        Soil_in = GPP*beta                                                  # kg C /m^2/s
        R_Soil = (k_o + (T_A[i]*k_1))*C_SOIL[i]                             # kg C/s

        # Land Surface Water Flux
        ET = G_S*rho_A*W_X[i]*(1-RH)*q_s(T_A[i] + T_base + 10,P_A)          # ET at peak summer conditions [kg/m^2/s]

        # TEMPERATURE TENDENCY EQUATIONS
        dTA_dt = ((1/C_A)*(F - (lam*T_A[i]))) - E_airsea                 # K/s
        dTOsurf_dt = E_airsea - DE_surf                                     # K/s
        # CARBON TENDENCY EQUTIONS
        F_LAND[i+1] = (GPP - R_VEG - (R_Soil/(f_B*A_A)))*f_B*A_A            # Net land flux kg C/s
        dummy[i+1] = R_Soil/(f_B*A_A)
        F_AS[i+1] = C_airsea*(A_O*H_O*rho_L)*(44e-09)*(12/44)               # Air sea flux kg C/s
        dDIC_S_dt = (F_AS[i+1]/(12e-09)/(A_O*H_O*rho_L)) - DC_surf          # Surface ocean carbon tendency
        
        atmos_ocean = 1e6*(F_AS[i+1]/M_A)*(MAIR/MC)
        atmos_land  = 1e6*(F_LAND[i+1]/M_A)*(MAIR/MC)
        #print(atmos_ocean)
        #print(atmos_land)
        #if i==10:
        #    f = breakhere

        #dcA_dt = E_flux - (1e6*(F_AS[i+1]/M_A)*(MCO2/MC)*(MAIR/MCO2))       # Atmospheric co2 tendency
        dcA_dt = E_flux - atmos_ocean - atmos_land

        # Land Surface Moisture Tendency
        dWX_dt = (PRECIP*(1-W_X[i]) - ET)/(2*rho_W*theta_X)                 # Surface water (1m depth with 0.5 field capacity)
        # INTEGRATING
        T_A[i+1] = T_A[i] + dTA_dt*dt                                       # Surface temperature
        T_O[i+1,0] = T_O[i,0] + (dTOsurf_dt)*dt                             # Ocean surface temperature
        T_O[i+1,1:-1] = T_O[i,1:-1] + (dt*DOE)                              # Interior Ocean Temperature
        T_O[i+1,-1] = T_O[i+1,-2]                                           # No flux bottom B.C.
        c_A[i+1] = c_A[i] + (dcA_dt*dt)                                     # Atmospheric CO2
        DIC[i+1,0] = DIC[i,0] + dDIC_S_dt*dt                                # Oceanic carbon at surface
        DIC[i+1,1:-1] = DIC[i,1:-1] + (dt*DOC)                              # Oceanic interior
        DIC[i+1,-1] = DIC[i+1,-2]                                           # No flux bottom B.C.
        W_X[i+1] = W_X[i] + dWX_dt*dt 

        C_VEG[i+1] = C_VEG[i] + (GPP - R_VEG - Soil_in)*f_B*A_A*dt
        C_SOIL[i+1] =C_SOIL[i] +  (Soil_in)*f_B*A_A*dt - (R_Soil*dt)

        # Catching exceptions of bucket parameterization
        if W_X[i+1] > 1:
            W_X[i+1] = 1
        if W_X[i+1] < 0:
            W_X[i+1] = 0

        #STEP
        i+=1
    return(time,T_A,c_A,T_O,pco2,DIC,F_AS*dt,F_LAND*dt,W_X,F_rad)



