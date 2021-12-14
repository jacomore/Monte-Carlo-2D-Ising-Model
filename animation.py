from tkinter import*
import numpy as np

window=Tk()
window.title("Simulation of 2D Ising Model")
window.geometry('+50+50')

class IsingSystem_2D:

    def __init__(self, size, exchange, temperature, external_field, m_steps, init_config,size_w):
        self.N = size
        self.J = exchange
        self.B = external_field
        self.MC_steps = m_steps
        self.square_w = size_w
        
        self.Kb = 1   # Botzmann constant
        self.u_b = 1  # bohr magneton
        
        self.T = temperature
        self.generate_lattice(init_config)
        

	#defining a function that will assign a color to the pixel for the corrisponding site      
    def pixel(self,i,j):
        if self.lattice[i,j]==1:
            temp="#cc0000"#spin up
        else:
            temp= "#fffafa"#spin down
        image1.put(temp,to=(i*self.square_w,j*self.square_w,(i+1)*self.square_w,(j+1)*self.square_w)) 

    def generate_lattice(self, init_config):
    
        """
        Generating a 2D lattice with a random spin configuration, 
        
        INPUT:   init_config == 0: random initial lattice
                 init_config == 1: hot initial lattice
                 init_config == -1: cold initial lattice
                 
        OUTPUT:  lattice: NxN array with each entry in the array being +1 or -1,
        representing the direction of the spin of the particle on that lattice site
        
        """
        if init_config == 0:
            self.lattice = np.random.choice([-1, 1], (self.N, self.N))
            self.equilibration()
            
        if init_config == 1:
            self.lattice = np.empty((self.N,self.N))
            self.lattice[:,:] = 1
            self.equilibration()
        
        if init_config == -1:
            self.lattice = np.empty((self.N,self.N))
            self.lattice[:,:] = -1
            self.equilibration()
            
    
    def site_E_int(self, site):
        """
        Finding the contribution to the total energy from a single lattice site
        
        Argument: 
        site: a list that contains the positions in x and y of the lattice site.
        
        """
        #Extracting x and y indices
        x = site[0]
        y = site[1]
        
        #Interaction term  (using % operator to apply periodic boundary conditions)
        interaction_E = - self.J * self.lattice[x, y] *(
            self.lattice[(x + 1)% self.N, y] + 
            self.lattice[x, (y + 1) % self.N] + 
            self.lattice[(x -  1)% self.N, y] + 
            self.lattice[x, (y - 1)% self.N]
            )
        
        return interaction_E
    
    def site_E_ext(self,site):
        #Extracting x and y indices
        x = site[0]
        y = site[1]
        
        external_E = -self.B* self.u_b * self.lattice[x, y] 
        return external_E
    

    def init_en(self):
        """
        Finding the total lattice energy energy associated 
        with the lattice in a specific microstate
        
        """    
        E = 0
        for i in range(self.N):
            for j in range(self.N):
                E += self.site_E_int([i,j])/2 + self.site_E_ext([i,j])
        return E
    
    
    def init_magn(self):
        """
        Finding the overall magnetisation associated
        with the lattice in a specific microstate
        
        """
        M = np.sum(self.lattice)
        return M
    
                
    def equilibration(self):
        """
        Performing the initial equilibration at temperature T.
        """
        self.Ene = self.init_en()
        self.Magn = self.init_magn()
        for n_steps in range(self.MC_steps):
            self.flip()
    
    def flip(self):
            ix = int(np.random.random()*self.N)    # choose a random row and column
            iy = int(np.random.random()*self.N)
            delta_E = -2 * (self.site_E_int([ix, iy])+self.site_E_ext([ix,iy])) #Energy difference between flipped 
                                                                                #and unflipped spin    
            if (delta_E <= 0) or (np.exp(-(delta_E)/(self.Kb * tempscale.get())) > np.random.rand()):                                               
                self.lattice[ix,iy] *= -1
                self.pixel(ix,iy)        
                

    def metropolis_MC(self):

        """
        Implementing the Metropolis algorithm (Monte Carlo), finding energy and
        magnetisation of the system after each sweep and finding the average
        energy and magnetisation over all sweeps

        """

        #Array containing lattice configuration after each Metropolis sweep
        #(dimensions: (num_steps + 1) x N x N)
        self.lattice_list = np.zeros([self.MC_steps + 1, self.N, self.N])   
        self.lattice_list[0] = self.lattice  

        if running:
            T=tempscale.get()
            for j in range(1, self.MC_steps + 1):
                self.flip()


            #Assigning updated lattice to lattice_output variable
                self.lattice_list[j] = self.lattice
        window.after(1,self.metropolis_MC)
        
        
#Lattice sizes
N =40

Tc = 2.26
#Number of sweeps of Metropolis algorithm
num_steps = 1000

w = 10#input('cell width of each pixels:')#width of each sites
canv_w= N*w
running=False 


#creating the canvas to simulate the phase transitions
canvas=Canvas(window,width=canv_w,height=canv_w)
canvas.pack()
image1=PhotoImage(width=canv_w,height=canv_w)
canvas.create_image((3, 3),image=image1,anchor="nw",state="normal")

#defining a functiond that starts and stops the simulation
def start_stop():
    global running
    running=not running
    if running:
        startbutton.config(text="stop")
    else:
        startbutton.config(text="start")

controlFrame=Frame(window)#creats a frame for evrerything
controlFrame.pack()
tempscale=Scale(controlFrame,from_=0.01,to=10.,resolution=0.01,length=120,orient='horizontal')
tempscale.pack(side="left")
tempscale.set(2.26)
templabel=Label(controlFrame,text="temperature:")
templabel.pack(side="left")
spacer=Frame(controlFrame,width=40)
spacer.pack(side="left")
startbutton=Button(controlFrame, text="start",width=8,command=start_stop)
startbutton.pack(side="left")



    #Creating the system
system = IsingSystem_2D(size = N,
                      temperature = tempscale.get(),
                      exchange = 1,
                      external_field = 0,m_steps=num_steps, init_config=0, size_w = 10)

    #Running Metropolis
system.metropolis_MC()
window.mainloop()#runs the interface that simulates the lattice
