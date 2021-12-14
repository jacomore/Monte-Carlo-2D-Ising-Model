# input libraries and dependences
from Ising_script import*
import seaborn as sns

#graphics
sns.set_theme()


# defining the inputs of the computation

N = 10      # number of sites per size
n_mc = 500  # number of Monte Carlo sweep
init = 0       # random initial configuration
B = 0         # null magnetic field 
J = 1         # exchange interaction

print("Performing the computation of the 2D Ising model with the following input parameters:")
print("lattice sites per size: ", N)
print("Monte Carlo steps:", n_mc)
if init == 0: 
    print("init = 0 --> spins randomly initialiazed")
elif init == 1: 
    print("init = 1 --> all spin initialized pointing up")
else: 
    print("init = -1 --> all spin initialized pointing down")
print("External magnetic field:",B)
print("Exchange constant:", J)


# defining the temperature range for which physical quantities will be computed.
T_min = 0.1
T_max = 6
T_dim = 100
T_ar = np.linspace(T_min,T_max,T_dim,endpoint = True)

print("The temperature range is: [",T_min,",",T_max,"]")

# creating the instance of the class IsingSystem_2D
system = IsingSystem_2D(size = N,
                        exchange = J,
                        temperature = T_min,
                        external_field = B,
                        m_steps = n_mc,
                        init_config = init)


# Task1: showing the increase of statistical noise when approaching the critic temperature 

print("Executing task 1...")
print("Showing the increase of statistical noise when approaching the critic temperature")

# T_list contains the temperature for which the magnetization is calculated
T_list = [1,1.5,2,2.5,3,4]
T_str = str(T_list)
# M_tau is a matrix that contains the magnetization at each MC step on the rows
# and each column is associated with a certain temperature
M_tau = np.empty((n_mc+1,len(T_list)))
  
# Creating the visualization
figMt,axMt = plt.subplots(nrows=6, ncols=1, sharex=True)

#creating the x-data array
time = np.linspace(1,n_mc + 1, n_mc + 1)
    
for k,T in enumerate(T_list):
    
    print("Temperature in task 1:",T)
    system.T = T                                          # update temperature
    system.metropolis_MC()                                   # perform metropolis algorithm
       
    M_tau[:,k] = system.magnetisation_list[:]/N**2           # store data inside M_tau
    axMt[k].plot(time,M_tau[:,k], label = "T = "+str(T))     # plot
    axMt[k].yaxis.set_ticklabels([])
    axMt[k].legend(loc = "center right")
    axMt[k].set_ylim(-1.2,1.2)

print("Task 1 executed!")
print("Storing data in M-time.txt")
figMt.suptitle("Magnetization as a function of MC-time")    
figMt.tight_layout()
np.savetxt("M-time.txt",M_tau,fmt = "%10.5f", header = T_str)
figMt.savefig("noise.pdf")        


#Task2: Magnetization as a function of temperature at different values of external field.

print("Executing task 2:")
print("Showing Magnetization as a function of temperature at different values of external field.")

# B_list contains the values of the external magnetic field for which the magnetization is computed
B_list = [0,0.1,0.3,0.6,1]
B_str = str(B_list)
# M_b is a matrix that contains the magnetization at each temperature on the rows
# and each column is associated with a certain B

M_b = system.MvsB(T_ar, B_list)

# creating visualization 
figMB, axMB = plt.subplots()

for k, B in enumerate(B_list):    

    print("value of B in task 2:",B)
    axMB.scatter(T_ar[:]/2.45,M_b[:,k], s = 10, label = "B = "+str(B) )   # plot
    axMB.set_xlabel(r"$\frac{T}{T_c}$ [a.u]", fontsize = 14)
    axMB.set_ylabel(r"$\frac{M}{M_{sat}}$ [a.u]", fontsize = 14)
    axMB.tick_params(axis='both', which='major', labelsize=14)
    axMB.axhline()
    axMB.legend()

figMB.tight_layout()
print("Task 2 executed! Storing data in M-temperature.txt")
np.savetxt("M-temperature.txt",M_b,fmt = "%10.5f", header = B_str)
np.savetxt("T-array.txt", T_ar,fmt = "%10.5f")
figMB.savefig("magnetization_temperature.pdf")


# Task 3: inverse of the magnetic susceptibility as a function of temperature.

print("Executing task 3:")
print("Computing magnetic susceptibility, energy and heat capacity per spin as a function of temperature.")

# computing energy, magnetization, susceptibility and specific heat per spin
system.T = T_min
system.equilibration()
system.physics_val(T_ar) 
    
print("Task 3 executed: storing data in X.txt, Energy.txt and C.txt")
# selecting the temperature above T_c    
Tc = 2.26

tmp = T_ar >= Tc

# creating the visualization for the 1/Chi vs T
figX,axX = plt.subplots()
axX.scatter(T_ar[tmp]/Tc,1/system.Chi[tmp], marker = "+", s =40, c = "r")
axX.set_xlabel(r"$\frac{T}{T_c}$ [a.u]", fontsize = 14)
axX.set_ylabel(r"$\frac{1}{\chi}$ [a.u]", fontsize = 14)
axX.tick_params(axis='both', which='major', labelsize=14)

np.savetxt("X.txt",system.Chi[tmp],fmt = "%10.5f")
np.savetxt("temperature-X.txt",T_ar[tmp],fmt = "%10.5f")
np.savetxt("Magnetization.txt",system.M_xspin,fmt = "%10.5f")
axX.set_xlim(0,T_max/Tc)   

figX.tight_layout()
figX.savefig("chi_temperature.pdf")

# creating the visualization for energy vs temperature
figE,axE = plt.subplots()
axE.scatter(T_ar/Tc,system.E_xspin, marker = "^", c = "r")
axE.set_xlabel(r"$\frac{T}{T_c}$ [a.u]", fontsize = 14)
axE.set_ylabel("Energy per spin [a.u]", fontsize = 14)
axE.tick_params(axis='both', which='major', labelsize=14)   

np.savetxt("C.txt",system.C,fmt = "%10.5f")
np.savetxt("Energy.txt",system.E_xspin,fmt = "%10.5f")

figE.tight_layout()
figE.savefig("energy_temperature.pdf")

# creating the visualization for heat capacity vs temperature
figC,axC = plt.subplots()
axC.scatter(T_ar/Tc,system.C,s = 13, c = "r", marker = "s")
axC.set_xlabel(r"$\frac{T}{T_c}$ [a.u]", fontsize = 14)
axC.set_ylabel("Heat capacity [a.u]", fontsize = 14)
axC.tick_params(axis='both', which='major', labelsize=14)   

figC.tight_layout()
figC.savefig("heat-capacity.pdf")

#Task4: calculate the correlation function as a function of the distance from the reference spin site.

print("Executing task 4:")
print("Calculate the correlation function as a function of the distance from the reference spin site.")


# T_list contains the temperature for which the correlation function is computed
T_list = [1,2,2.2,2.4,2.6,2.8,3,4]


# G is a matrix that contains the value of the correlation function at a certain temperature (column)
# at a certain distance (row)
G = np.empty((int(N/2),len(T_list)))

# creating the visualization
figf,axf = plt.subplots()
for i,T in enumerate(T_list):

    print("Temperature in task 4:",T)

    system.T = T
    system.equilibration()
    system.correlation()      # call correlation > metropolis_MC > flip
    G[:,i] = system.G_glob    # store G_glob inside G

    axf.plot(np.linspace(1,N/2,int(N/2)),G[:,i], 's--',label = "T = "+str(T))
    axf.set_xlabel("lattice distance", fontsize = 14)
    axf.set_ylabel("Correlation function", fontsize = 14)
    axf.tick_params(axis='both', which='major', labelsize=14) 
    axf.legend()
    

print("Task 4 executed! Storing data in Corr.txt")
np.savetxt("Corr.txt",G,fmt = "%10.5f")
np.savetxt("position-Corr.txt",np.linspace(1,N/2,int(N/2)),fmt = "%10.5f")

figf.tight_layout()
figf.savefig("correlation.pdf")
# Good bye!

print("The program has executed all the basic tasks. However, if you still want to find out something more, take a look at the Ising_script.py file!")
print("Goodbye!")

plt.show()
