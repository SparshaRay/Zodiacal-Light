
''' Libraries _______________________________________'''

import numpy as np                                  #   Numpy for array handling and manipulation
import matplotlib.pyplot as plt                     #   Matplotlib for plotting
from matplotlib.animation import FuncAnimation      #   Animation libraries

''' Params   ________________________________________'''

sunpos  = np.array([0.0, 0.0, 0.0])         # Position of Sun, taken as origin
marspos = np.array([0.0, 249.2E9, 0.0])     # Position of Mars, at perigee, in km
marsvel = np.array([26.5E3, 0.0, 0.0])      # Velocity of Mars, at perigee, in ms-1

marsmass = 6.39E23              # Mass of mars in kg
sunmass  = 1.989E30             # Mass of sun in kg
timestep = 1000                 # Timestep, in seconds
l = 0                           # Angular momentum
e = 0                           # Total energy
time = [i for i in range(100)]  # time array
larr = [0 for i in range(100)]  # angular momentum array
earr = [0 for i in range(100)]  # total energy array
G = 6.67E-11                    # Gravitational constant in SI

moredust = True         # If true, generate more dust particles
muvel = 5E3             # Mean velocity of dust particles
sigmavel = 0.5E3        # Std. deviation of velocity of dust particles
mumass = 1E-5           # Mean mass of dust particles
sigmamass = 1E-7        # Std. deviation of mass of dust particles
dust = []               # Array to store dust particles

fps = 30                    # Target value of frames per second

graph = plt.figure(2)       # plot l and e values
linel, = plt.plot([], [])   # angular momentum (l)
linee, = plt.plot([], [])   # total energy (e)
plt.xlim(0,99)              # timescale plotting range
plt.ylim(0,1E40)            # l and e plotting range
def graphplot(frame0):      # call function
    larr.pop(0)         
    larr.append(l)          # add new l value
    earr.pop(0)
    earr.append(-e*1E7)     # add new e value
    linel.set_data(time,larr)   # pack l data
    linee.set_data(time,earr)   # pack e data
    return linel, linee         # return packed data

plotanim = FuncAnimation(graph, graphplot, interval=1000/fps)       # plot l and e graphs

fig = plt.figure(1)                 # Def fig
ax = plt.axes(projection='3d')      # Def ax
ax.view_init(0, 0)                  # Initial elevation and azimuth
iters = 1000                        # Euler-cromer intergrations per datastamp
disp = 2                            # Full cycles of calculations per render

bodiesplot = plt.plot([], [], [], 'ro'           )[0]       # Sun and Mars
dustplot   = plt.plot([], [], [], 'b.', alpha=1.0)[0]       # Dust particles

extent = np.linalg.norm(marspos) * 2    #
ext = -extent,extent                    #
ax.set_xlim(ext)                        #   Setting plotting range
ax.set_ylim(ext)                        #
ax.set_zlim(ext)                        #

''' Dust particle class _____________________________'''

class Dust :

    def __init__(self, pos, vel, mass, trajec, trj):
        self.pos  = pos         # position
        self.vel  = vel         # velocity
        self.mass = mass        # mass
        self.trajec = trajec    # trajectory
        self.trj = trj          # plotting object

    def eulercromer(self):
        sunvec  = sunpos  - self.pos        # vector to sun
        marsvec = marspos - self.pos        # vector to mars

        sungrav  = (G * sunmass * self.mass * sunvec) / (np.linalg.norm(sunvec) ** 3)       # gravity of sun
        marsgrav = (G * marsmass * self.mass * marsvec) / (np.linalg.norm(marsvec) ** 3)    # gravity of mars

        sunacc  = sungrav / self.mass           # acceleration due to sun
        marsacc = marsgrav / self.mass          # acceleration due to mars
        solwind = np.array([0.0, 0.0, 0.0])     # acceleration due to solar wind
        netacc = sunacc + marsacc + solwind     # total acceleration
        self.vel += ( timestep * netacc )       # update velocity
        self.pos += ( timestep * self.vel )     # update position

    def position(self):
        return self.pos     # return position

    def velocity(self):
        return self.vel     # return velocity

    def get_mass(self):
        return self.mass    # return mass

    def update_trajectory(self) :
        self.trajec.append([list(self.pos)])    # append postion point

    def get_trajectory(self):
        self.trj.set_data_3d(np.concatenate(self.trajec).T)     # make trajectory plotting dataset
        return self.trj                                         # return plotting object

''' Euler-cromer on Mars due to Sun _________________'''

def sunmars() :

    global marspos, marsvel     # fetch global variables

    rvec = sunpos - marspos                                                 # vector to sun
    grav = (G * sunmass * marsmass * rvec) / (np.linalg.norm(rvec) ** 3)    # calculate force
    acc = grav / marsmass                                                   # calculate acceleration
    marsvel = marsvel + ( timestep * acc )                                  # update velocity
    marspos = marspos + ( timestep * marsvel )                              # update position

''' Dust generating, data packing and rendering _____'''

def update(frame):

    global dust, moredust, l, e                 # fetch global variables

    ax.view_init(ax.elev+0.05, ax.azim-0.1)     # rotate view
    if len(dust)>100 : moredust=False           # stop dust generation if there is too much dust

    for j in range(disp):

        if moredust :
            theta = np.random.uniform(0, 2*np.pi)                                       #
            z = np.random.uniform(-1, 1)                                                #
            x = ((1-z**2)**0.5)*np.cos(theta)                                           #   Generate random initial velocity vector for dust particles and half the z component
            y = ((1-z**2)**0.5)*np.sin(theta)                                           #
            initvel = np.array([x, y, z/2]) * abs(np.random.normal(muvel, sigmavel))    #
            dust.append(Dust(marspos, marsvel+initvel, abs(np.random.normal(mumass, sigmamass)), [], plt.plot([], [], [], 'g-', alpha=0.2, linewidth=0.2)[0]))      # create new dust particle

        for i in range(iters) :
            sunmars()                                           # euler-cromer on mars
            for particle in dust : particle.eulercromer()       # euler-cromer on dust particles
        
        l = np.linalg.norm(marsmass*np.cross(marspos,marsvel)) + sum([np.linalg.norm(particle.get_mass()*np.cross(particle.position(),particle.velocity())) for particle in dust])      # angular momentum
        k = 0.5*marsmass*np.linalg.norm(marsvel)**2 + sum([0.5*particle.get_mass()*np.linalg.norm(particle.velocity())**2 for particle in dust])                                        # kinetic energy
        p = G*marsmass*sunmass/np.linalg.norm(marspos) + sum([G*particle.get_mass()*sunmass/np.linalg.norm(particle.position()) for particle in dust])                                  # potential energy
        e = k - p                                                                                                                                                                       # total energy

        for particle in dust : particle.update_trajectory()     # update trajectory datastamp

    bodiesplot.set_data_3d(np.stack((sunpos, marspos)).T)                       # pack sun and mars for plotting
    arr = np.concatenate(list([particle.position()] for particle in dust))      # prepare dust particle positions for plotting
    dustplot.set_data_3d(arr.T)                                                 # pack dust particle positions

    return [particle.get_trajectory() for particle in dust], dustplot, bodiesplot       # pack and send data for plotting

ani = FuncAnimation(fig, update, interval=1000/fps)     # animation of system
plt.show()                                              # show final plot

''' ___ Sparsha Ray, MS21256, evac, ver 1.2 dated 17/07/2022, rtr.iiserm ___ '''