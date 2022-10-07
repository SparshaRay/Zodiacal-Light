
''' Libraries __________________________________________'''

import numpy as np                          # Numpy for array handling and manipulation
from scipy.stats import truncnorm           # Truncated normal distribution, can be done without scipy too
import matplotlib.pyplot as plt             # Matplotlib for plotting


''' Parameters and databases (SI units) ________________'''

G = 6.67E-11                        # Universal gravitational constant
C = 3E8                             # Speed of light in vacuum
timestep = 1500                     # Timestep for integration
runduration = 20 * (3.154E7)        # Time for which the simulation will run for each size of dust. The value outside the paranthesis is the number of earth-years.
max_dust = 61                       # Maximum number of dust particles for each size of dust
muspeed = 5E3                       # Mean speed of ejected dust particles
sigmaspeed = 5E2                    # Standard deviation of speed of ejected dust particles
density = 3000                      # Density of dust particles
reflectivity = 0.25                 # Reflectivity of dust particles, value must be in the interval [0, 1]
sizerange = np.array([1E-7, 1E-5])  # Minimum and maximum diameter of ejected dust particles
datapoints = 20                     # Datapoints of size of dust particles
dustmasses = np.e ** np.linspace(*np.log(density*4*np.pi*(sizerange**3)/24), num=datapoints)        # Dust mass sample points array
sorteddists = []                    # Sorted distance array


''' Functions __________________________________________'''

# Integration on all particles by Semi-implicit Euler method ...
def semi_implicit_euler(iters):
    global pos, vel, time               # Fetch global variables
    for i in range(iters):              # Do integration repeatedly
        time += timestep                # Increment time 
        for body in interactions :      # Select each body
            radprsracc = 0              # Variable for acceleration due to radiation pressure
            if body[:4] == 'dust' : radprsracc = pos[body] * (1+reflectivity) * 3.8E26 * ((mass[body]/density)**(2/3)) / ( 2 * C * mass[body] * (np.linalg.norm(pos[body])**3) )    # Calculate acceleration of dust particles due to radiation pressure
            acceleration = radprsracc + ( G * sum([( mass[elem] / (( np.linalg.norm(pos[elem]-pos[body]) )**3) ) * (pos[elem]-pos[body]) for elem in interactions[body]]) )         # Calculate total acceleration of selected body
            vel[body] = vel[body] + (timestep * acceleration)       # Update velocity
            pos[body] = pos[body] + (timestep * vel[body])          # Update position

# Function to create an ejected dust particle ...
def add_dust(probability, originbody, selfvel, selfmass):
    if np.random.rand() < probability :                         # Probability of adding dust particle
        global num_dust, pos, vel, mass, interactions           # Fetch global variables
        pos[f'dust{num_dust}']  = pos[originbody]               # Add initial position of dust particle to position dictionary
        vel[f'dust{num_dust}']  = vel[originbody] + selfvel     # Add initial velocity of dust particle to velocity dictionary
        mass[f'dust{num_dust}'] = selfmass                      # Add mass of dust particle to mass dictionary
        interactions[f'dust{num_dust}'] = ['sun', 'mars']       # Add list of gravitationally influencing bodies to interactions dictionary
        num_dust += 1                                           # Increment number of dust particles

# Create a random 3D vector with specified magnitude ...
def initvel(mag):
    theta = np.random.uniform(0, 2*np.pi)
    z = np.random.uniform(-1, 1)
    x = ((1-z**2)**0.5)*np.cos(theta)
    y = ((1-z**2)**0.5)*np.sin(theta)
    return mag * np.array([x, y, z])


''' Run simulation and generate data ___________________'''

for selfmass in dustmasses :

    pos = {                                                 # Position dictionary
        'sun'  : np.array([0.0, 0.0, 0.0]),
        'mars' : np.array([0.0, 249.261E9, 0.0])
        }

    vel = {                                                 # Velocity dictionary
        'sun'  : np.array([0.0, 0.0, 0.0]),
        'mars' : np.array([26.50E3, 0.0, 0.0])
        }

    mass = {                                                # Mass dictionary
        'sun'  : 1.989E30,
        'mars' : 6.39E23
        }

    interactions = {                                        # Gravitational interactions dictionary
        'mars' : ['sun']
    }

    time = 0                # Current time (tracking variable)
    num_dust = 0            # Number of dust particles (tracking variable)
    finaldists = []         # List of distances from sun to dust particles

    while time < runduration :          # Run simulation for the specified time
        if num_dust < max_dust : add_dust(1, 'mars', initvel(truncnorm.rvs(-muspeed/sigmaspeed, 2*muspeed/sigmaspeed, loc=muspeed, scale=sigmaspeed)), selfmass)    # Add dust particles
        semi_implicit_euler(1500)       # Do integration

    for num in range(num_dust) : finaldists.append(np.linalg.norm(pos[f'dust{num}']))   # Make the list of distances
    sorteddists.append(np.sort(finaldists)/1.5E11)                                      # Sort list of distances and convert to AU
    datapoints-=1; print(f'{datapoints} datapoints remaining')                          # Progress update


''' Plot data __________________________________________'''

sizelist = 2 * 1E6 * ((3*dustmasses/(4*np.pi*density))**(1/3))      # List of size of dust particles in datapoints

# Extract and plot datapoints for distribution bands
for i in range(1, 6) :
    yu = []
    yl = []
    for distribiution in sorteddists :
        yu.append(distribiution[-i*5])
        yl.append(distribiution[-1+i*5])
    plt.fill_between(sizelist, yu, yl, color='blue', alpha=0.1)

# Plot the median line
y  = []
for distribiution in sorteddists : y.append(distribiution[30])
plt.plot(sizelist, y, 'b-')

# Configure plot and display plot
plt.xscale("log")
plt.xlabel('Ejecta Size in micron')
plt.ylabel('Distance from sun in AU')
plt.show()