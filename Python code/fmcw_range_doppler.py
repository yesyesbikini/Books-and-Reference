# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 19:42:41 2021

@author: Yefeng Cai
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

c0 = 3e8 # [m/s]

r0 = 5 # [m]
v0 = 14 # [m/s]

ts = 1e-6 # sampling time [sec]

f0 = 76e9 # [Hz]
B = 0.5e9 # [Hz]
N1 = 128 # no. of samples in fast time
z_fast = 4*N1
N2 = 128 # no. of chirps in slow time
z_slow = 4*N2
T1 = ts*N1 # chirp interval
mu = B/T1

#%% total time of n2 in slow time, constant r * v
# for each n1 in fast time and each n2 in slow time
s = np.zeros((N2, N1), dtype=np.complex64)
for n2 in range(N2):
    for n1 in range(N1):
        tau = 2*(r0 + v0*T1*n2 + v0*n1*ts) / c0
        s[n2, n1] = np.exp(
            1j* 2* np.pi * (
                f0*tau + mu*tau*n1*ts - 0.5*mu*(tau**2)
                )
            )

r_res = c0/(2*B)
r_vec = np.linspace(0, 1-1/z_fast, z_fast-1)*(N1-1)*r_res
v_res = (c0/f0)/(2*N2*T1)
v_vec = np.linspace(0, 1-1/z_slow, z_slow-1)*(N2-1)*v_res

win_r = np.hanning(N1)
win_r = np.tile(win_r, (N2,1))
rp = np.fft.fft(s*win_r, n=z_fast, axis=1)
rp_abs = np.abs(rp)

# fig, ax = plt.subplots(figsize=(6,6))
# ax.imshow(rp_abs/np.max(rp_abs), 
#           cmap=plt.cm.Reds, 
#           interpolation='none', 
#           extent=[r_vec[0], r_vec[-1], v_vec[0], v_vec[-1]],
#           aspect='auto',
#           origin='lower')

win_d = np.hanning(N2)
win_d = np.tile(win_d, (z_fast,1)).T
rd = np.fft.fft(rp*win_d, n=z_slow, axis=0)
rd_abs = np.abs(rd)

fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(rd_abs/np.max(rd_abs), 
          cmap=plt.cm.Reds, 
          interpolation='none', 
          extent=[r_vec[0], r_vec[-1], v_vec[0], v_vec[-1]],
          aspect='auto',
          origin='lower')

# for each n2 in slow time, but all n1 in fast time once
t_list_chirp = ts * np.linspace(0, N1-1, N1)
for n2 in range(N2):
    tau = 2*(r0 + v0*T1*n2 + v0*t_list_chirp) / c0
    s[n2, :] = np.exp(
        1j* 2* np.pi * (
            f0*tau + mu*tau*t_list_chirp - 0.5*mu*(tau**2)
            )
        )
rp = np.fft.fft(s*win_r, n=z_fast, axis=1)
rp_abs = np.abs(rp)
rd = np.fft.fft(rp*win_d, n=z_slow, axis=0)
rd_abs = np.abs(rd)

fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(rd_abs/np.max(rd_abs), 
          cmap=plt.cm.Reds, 
          interpolation='none', 
          extent=[r_vec[0], r_vec[-1], v_vec[0], v_vec[-1]],
          aspect='auto',
          origin='lower')

#%% total time over multiple N2, r & v changing
# for each n1 in fast time and each i_T1 in slow time
t_list = ts*np.linspace(0, N2*4-1, N2*4)

v = np.zeros(t_list.shape)
r = np.zeros(t_list.shape)

s1 = np.zeros((t_list.shape[0], N1), dtype=np.complex64)
for i_T1 in range(t_list.shape[0]):
    # one r & one v for each T1 chirp
    v[i_T1] = v0 + 0.01*T1*i_T1
    r[i_T1] = r0 + v[i_T1]*T1*i_T1
    
    for n1 in range(N1):
        tau = 2*(r[i_T1] + v[i_T1]*n1*ts) / c0
        s1[i_T1, n1] = np.exp(
            1j* 2* np.pi * (
                f0*tau + mu*tau*n1*ts - 0.5*mu*(tau**2)
                )
            )

win_r = np.hanning(N1)
win_r = np.tile(win_r, (t_list.shape[0],1))
rp1 = np.fft.fft(s1*win_r, n=z_fast, axis=1)
rp1_abs = np.abs(rp1)

# fig, ax = plt.subplots(figsize=(6,6))
# ax.imshow(rp_abs/np.max(rp_abs), 
#           cmap=plt.cm.Reds, 
#           interpolation='none', 
#           extent=[r_vec[0], r_vec[-1], v_vec[0], v_vec[-1]],
#           aspect='auto',
#           origin='lower')

rd1 = np.fft.fft(rp1[0*N2:1*N2,:]*win_d, n=z_slow, axis=0)
rd1_abs = np.abs(rd1)

fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(rd1_abs/np.max(rd1_abs), 
          cmap=plt.cm.Reds, 
          interpolation='none', 
          extent=[r_vec[0], r_vec[-1], v_vec[0], v_vec[-1]],
          aspect='auto',
          origin='lower')

# rd1 = np.fft.fft(rp1[int(0.5*N2):int(1.5*N2),:]*win_d, n=z_slow, axis=0)
# rd1_abs = np.abs(rd1)

# fig, ax = plt.subplots(figsize=(6,6))
# ax.imshow(rd1_abs/np.max(rd1_abs), 
#           cmap=plt.cm.Reds, 
#           interpolation='none', 
#           extent=[r_vec[0], r_vec[-1], v_vec[0], v_vec[-1]],
#           aspect='auto',
#           origin='lower')

# for each i_T1 in slow time, but all n1 in fast time once
for i_T1 in range(t_list.shape[0]):
    # one r & one v for each T1 chirp
    v[i_T1] = v0 + 0.01*T1*i_T1
    r[i_T1] = r0 + v[i_T1]*T1*i_T1
    
    tau = 2*(r[i_T1] + v[i_T1]*t_list_chirp) / c0
    s[n2, :] = np.exp(
        1j* 2* np.pi * (
            f0*tau + mu*tau*t_list_chirp - 0.5*mu*(tau**2)
            )
        )

rd1 = np.fft.fft(rp1[0*N2:1*N2,:]*win_d, n=z_slow, axis=0)
rd1_abs = np.abs(rd1)

fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(rd1_abs/np.max(rd1_abs), 
          cmap=plt.cm.Reds, 
          interpolation='none', 
          extent=[r_vec[0], r_vec[-1], v_vec[0], v_vec[-1]],
          aspect='auto',
          origin='lower')
