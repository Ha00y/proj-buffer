# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 10:25:35 2024

@author: cawle
"""

import numpy as np
import matplotlib.pyplot as plt

# CHOOSE PARAMETERS #
lamb = 1
mu = 1
B = 3
T = 100
a = [1, 2, 1]
p = [0.3, 0.1, 0.6]  # mixture of exponential arrays, p must sum to 1
p_cum = np.cumsum(p)  # cumulative sum of mixture probabilities

# SET PARAMETERS FOR LOOP #
elapsing = True  # while loop parameter
t_elapsed = 0  # time elapsed since t=0 (starts at 0)
arrivals = np.array([])  # array for time between data arrivals
buff_state = 0  # current buffer state (starts at 0)
buff_array = np.zeros(1)  # array to store buffer state at every data arrival (first entry is 0 for state at t=0)

# Arrays to store buffer occupancy and corresponding time points
buff_combined = np.zeros(1)
time_points = np.zeros(1)

# SIMULATE BUFFER WITH WHILE LOOP #
while elapsing:

    t_arrival = np.random.exponential(lamb)  # sample time between data arrivals from exp dist

    if t_elapsed + t_arrival < T:  # make sure arrivals fall within the time range t specified
        
        p_finder = np.random.uniform()  # select a random value in range [0,1]
        p_index = np.searchsorted(p_cum, p_finder, 'right')  # use p_finder to find which exponential in mixture to sample from
        job_size = np.random.exponential(a[p_index])  # sample job size from chosen distribution

        if buff_state - mu * t_arrival < 0 : 
            t_interp = t_elapsed + buff_state/mu
            time_points = np.append(time_points, t_interp)
            buff_combined = np.append(buff_combined, 0)
            
        t_elapsed += t_arrival  # update current time
        
        time_points = np.append(time_points, t_elapsed)  
        buff_combined = np.append(buff_combined, max(buff_state - mu * t_arrival, 0))

        arrivals = np.append(arrivals, t_arrival)
            
        buff_state = min(B, max(buff_state - mu * t_arrival, 0) + job_size)  # find buffer state
        buff_combined = np.append(buff_combined, buff_state)
        time_points = np.append(time_points, t_elapsed)  # Record the time point
    else:

        elapsing = False  # end loop when time elapsed exceeds final time T
        final_buff_state = min(B, max(buff_state - mu * (T - t_elapsed), 0))  # find buffer state at final time t=T
        buff_array = np.append(buff_array, final_buff_state)

# Buffers arrays normalization
buff_combined /= B

# PLOT BUFFER OCCUPANCY OVER TIME #
plt.plot(time_points, buff_combined)
plt.xlabel('Time')
plt.ylabel('Buffer Occupancy')
plt.title(f'Buffer Occupancy Simulation for $\lambda = {lamb}$ & $\mu = {mu}$')
#plt.xlim(0,10)
plt.show()