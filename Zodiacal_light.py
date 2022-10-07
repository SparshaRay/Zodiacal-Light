
''' Libraries __________________________________________'''

import numpy as np                          # Numpy for array handling and manipulation
from scipy.stats import truncnorm           # Truncated normal distribution, can be done without scipy too
import matplotlib.pyplot as plt             # Matplotlib for plotting
import matplotlib.animation as animation    # Animation libraries
import mpl_toolkits.axisartist as AA        # For subplots
from matplotlib.colors import hsv_to_rgb    # For coloring trajectories


''' Initialize subplots ________________________________'''

fig = plt.figure(1, figsize=(15, 10))       # Parent window

# Trajectory plot ...
sunsysax = fig.add_subplot(2, 2, 1, projection='3d')
sunsysax.title.set_text('Trajectories')
ext = 500E9
sunsysax.set_xlim(-ext, ext); sunsysax.set_ylim(-ext, ext); sunsysax.set_zlim(-ext, ext)

# Distribution plot ...
distribution = fig.add_subplot(2, 2, 2)
distribution.title.set_text('Dust Distribution')

# Angular Momentum and Total Energy plot of mars ...
marsstats = fig.add_subplot(2, 2, 3, axes_class=AA.Axes)
marsstats.title.set_text('L and E of Mars')
marsstats_ = marsstats.twinx()
pltmarsl = marsstats .plot([], [], 'r-')[0]; pltmarst = marsstats_.plot([], [], 'g-')[0]
marsstats.axis["left"].set_label('Angular momentum'); marsstats.axis["left"].label.set_color('red')
marsstats.axis["right"].label.set_visible(True)
marsstats.axis["right"].set_label('\n\n\n\nKE + PE'); marsstats.axis["right"].label.set_color('green')

# Plot of average Angular Momentum and average Total Energy of dust particles ...
duststats = fig.add_subplot(2, 2, 4, axes_class=AA.Axes)
duststats.title.set_text('Average L and E of Dust Particles')
duststats_ = duststats.twinx()
pltdustl = duststats .plot([], [], 'r-')[0]; pltdustt = duststats_.plot([], [], 'g-')[0]
duststats.axis["left"].set_label('Angular momentum'); duststats.axis["left"].label.set_color('red')
duststats.axis["right"].label.set_visible(True)
duststats.axis["right"].set_label('\n\n\n\nKE + PE'); duststats.axis["right"].label.set_color('green')

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)       # Adjust spacings


''' Parameters and databases (SI units) ________________'''

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

sunsys = {                                              # Plotting array dictionary
    'sunmars'    :  sunsysax.plot([], [], [], 'ro')[0],
    'marstrajec' : [sunsysax.plot([], [], [], 'g-')[0], [[], [], []]],
    'dusts'      :  sunsysax.plot([], [], [], 'b.')[0]
}

G = 6.67E-11            # Universal gravitational constant
C = 3E8                 # Speed of light in vacuum
time = 0                # Current time (tracking variable)
timestep = 1500         # Timestep for integration
iters = 1500            # Number of times integration is done for one datapoint
num_dust = 0            # Number of dust particles (tracking variable)
max_dust = 100          # Maximum number of dust particles generated
muspeed = 5E3           # Mean speed of ejected dust particles
sigmaspeed = 5E2        # Standard deviation of speed of ejected dust particles
density = 3000          # Density of dust particles
reflectivity = 0.25     # Reflectivity of dust particles, value must be in the interval [0, 1]
history = 200           # Maximum number of datapoints to show in trajectory
span = history          # Number of datapoints to show in angular momentum and total energy plot
sizerange = np.array([1E-5, 1E-3])                                      # Minimum and maximum diameter of ejected dust particles 
massrange = density * 4 * np.pi * (sizerange**3) / 24                   # Mass range of ejected dust particles
marsTE = [0 for _ in range(span)]; marsl = [0 for _ in range(span)]     # Angular momentum and total energy of mars, plotting database
dustTE = [0 for _ in range(span)]; dustl = [0 for _ in range(span)]     # Angular momentum and total energy of dust, plotting database
timeline = [0 for _ in range(span)]                                     # Timestamps, plotting database
x, y = [], []                                                           # Dust distribution database


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
        hue = 0.05 + 0.9 * (np.log10(selfmass) - np.log10(massrange[0])) / (np.log10(massrange[1]) - np.log10(massrange[0]))                                    # Set color to dust particle's trajectory according to its mass
        sunsys[f'trajec{num_dust}'] = [sunsysax.plot([], [], [], linestyle='-', linewidth=0.60, color=hsv_to_rgb([hue,1,0.5]), alpha=0.30)[0], [[], [], []]]    # Add to plotting and position array to plotting array dictionary
        num_dust += 1                                           # Increment number of dust particles

# Create a random 3D vector with specified magnitude ...
def initvel(mag):
    theta = np.random.uniform(0, 2*np.pi)
    z = np.random.uniform(-1, 1)
    x = ((1-z**2)**0.5)*np.cos(theta)
    y = ((1-z**2)**0.5)*np.sin(theta)
    return mag * np.array([x, y, z])

# Angular momentum and total energy plot of mars and dust particles ...
def l_m_plot():
    timeline.pop(0)                     # Remove last time
    timeline.append(time)               # Append new time
    marsTE.pop(0)                       # Remove last total energy of mars
    marsTE.append(0.5 * mass['mars'] * np.dot(vel['mars'], vel['mars']) - G * mass['sun'] * mass['mars'] / np.linalg.norm(pos['mars']))     # Append new total energy of mars
    marsl.pop(0)                        # Remove last angular momentum of mars
    marsl.append(mass['mars'] * np.linalg.norm(np.cross(pos['mars'], vel['mars'])))     # Append new angular momentum of mars
    if num_dust == 0 : return None      # If there are no dust particles then exit
    dustl.pop(0)                        # Remove last average angular momentum of dust particles
    dustl.append(sum([mass[f'dust{i}']*np.linalg.norm(np.cross(pos[f'dust{i}'], vel[f'dust{i}'])) for i in range(num_dust)])/num_dust)      # Append new average angular momentum of dust particles
    dustTE.pop(0)                       # Remove last average total energy of dust particles
    dustTE.append(sum([0.5 * mass[f'dust{i}'] * np.dot(vel[f'dust{i}'], vel[f'dust{i}']) - G * mass['sun'] * mass[f'dust{i}'] / np.linalg.norm(pos[f'dust{i}']) for i in range(num_dust)])/num_dust)    # Append new average total energy of dust particles
    # Set plot ranges ...
    marsstats .set_xlim(time-span*timestep*iters, time)
    marsstats .set_ylim(min(marsl), max(marsl))
    marsstats_.set_ylim(min(marsTE), max(marsTE))
    duststats .set_xlim(time-span*timestep*iters, time)
    duststats .set_ylim(min(dustl), max(dustl))
    duststats_.set_ylim(min(dustTE), max(dustTE))
    # Pack plotting arrays ...
    pltmarsl.set_data(timeline, marsl)
    pltmarst.set_data(timeline, marsTE)
    pltdustl.set_data(timeline, dustl)
    pltdustt.set_data(timeline, dustTE)

# Plot distribution of dust particles ...
def distributionplot():
    for num in range(num_dust) : 
        if len(sunsys[f'trajec{num}'][1][0]) > 1 :
            if (sunsys[f'trajec{num}'][1][0][-1] * sunsys[f'trajec{num}'][1][0][-2]) < 0 :      # When it crosses the xz plane
                x.append(sunsys[f'trajec{num}'][1][1][-1])                                      # Get x value
                y.append(sunsys[f'trajec{num}'][1][2][-1])                                      # Get y value

    distribution.hexbin(x, y, gridsize = (35,15), extent = (-ext*2, ext*2, -ext/1.5, ext/1.5))  # Plot 2D histogram

# Pack the position arrays and send for plotting ...
def packsunsys(): 
    for i in (0, 1, 2) :
            sunsys['marstrajec'][1][i].append(pos['mars'][i])
            if len(sunsys['marstrajec'][1][i])>history : del sunsys['marstrajec'][1][i][0]
            sunsys['marstrajec'][0].set_data_3d(sunsys['marstrajec'][1])

    for num in range(num_dust) : 
        for i in (0, 1, 2) :
            sunsys[f'trajec{num}'][1][i].append(pos[f'dust{num}'][i])
            if len(sunsys[f'trajec{num}'][1][i])>history : del sunsys[f'trajec{num}'][1][i][0]
            sunsys[f'trajec{num}'][0].set_data_3d(sunsys[f'trajec{num}'][1])

    sunsys['sunmars'].set_data_3d( [pos['sun'][i],pos['mars'][i]] for i in (0, 1, 2) )
    sunsys['dusts']  .set_data_3d( [pos[f'dust{num}'][i] for num in range(num_dust)] for i in (0, 1, 2) )

# Draw surface of the torus ...
def torus():
    num = 100
    theta = np.linspace(0, 2.*np.pi, num)
    phi = np.linspace(0, 2.*np.pi, num)
    theta, phi = np.meshgrid(theta, phi)
    cu, ar = 2*249.261E9, 249.261E9
    xt = (cu + ar*np.cos(theta)) * np.cos(phi)
    yt = (cu + ar*np.cos(theta)) * np.sin(phi)
    zt = ar * np.sin(theta)
    sunsysax.plot_wireframe(xt, yt, zt, rstride=0, cstride=5, alpha=0.1, linewidth=0.5)

# Display function ...
def disp(f):
    if num_dust < max_dust : add_dust(0.50, 'mars', initvel(truncnorm.rvs(-muspeed/sigmaspeed, 2*muspeed/sigmaspeed, loc=muspeed, scale=sigmaspeed)), 10 ** truncnorm.rvs(-2, 2, loc = (np.log10(massrange[0]) + np.log10(massrange[1])) / 2, scale = (np.log10(massrange[1]) - np.log10(massrange[0])) / 4))       # tbh idk what i wrote in this line, it basically creates dust particles with 0.5 probability and with a nice truncated normal distribution of mass and initial speed 
    semi_implicit_euler(iters)      # Do integration
    packsunsys()                    # Pack arrays, plot bodies and trajectories
    distributionplot()              # Plot dust particle distribution
    l_m_plot()                      # Plot angular momentum and total energy


''' Animate and display ________________________________'''

torus()                                                         # Draw the torus
anim = animation.FuncAnimation(fig, disp, interval=1000/60)     # Animate everything
plt.show()                                                      # Show animation