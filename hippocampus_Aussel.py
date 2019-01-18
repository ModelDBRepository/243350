#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#To use this code you can execute an instruction of the form :
#main_process(range(8), 60*psiemens,600*psiemens,0.1,0.06)

#Don't forget to change the paths line 500-506 to use your own input files
#for these simulations three sets of 8-minute-long input signals were used

import brian2
from brian2 import * 

from topology_Aussel import topology
from Event_detection_Aussel import event_detection_and_analysis
import scipy        
from scipy import signal
import ast


from joblib import Parallel, delayed
import multiprocessing

import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'


import time


def read_file(filename):
    data_array=[]
    file=open(filename,'r')
    for data in file :
        d=ast.literal_eval(data[:-1])
        data_array.append(d)
    file.close()
    return data_array
  

def write_file(simutype,data,tmin,dur):
    start_ind=int(tmin/record_dt)
    end_ind=int(start_ind+dur/record_dt)
    
    time_series=open('time_series_'+simutype+'.txt','w')
    time_series.write('[')
    for n in data[start_ind:end_ind]:
        time_series.write('%.2E,'%n)
    time_series.write(']\n')
    time_series.close()
    
    

def preparation(num_simu,g_max_e,g_max_i,p_co,p_co_CA3):
    global inh_eqs, py_eqs, py_CAN_eqs, py_stim_eqs
    global V_th, N1, N2, N3, p_CAN, asleep, grid, elec_ind, scale, runtime, timestep, sigma, coeff, scale, scale_str
    global noise_amp
    global stim,co, gain_stim
    global record_dt
    global elec_pos, p_in
    global runtime
    global version
    noise_amp=500*pamp
    V_th=-20*mvolt
    p_in=0.01 
    global num_in

    
    if num_simu in [0,1,4,5]:
        stim='sleep'
    else :
        stim='wake'
        
    if num_simu in [0,2,4,6]:
        co='sleep'
    else :
        co='wake'
    print('input='+stim)
    print('connectivity='+co)
    asleep=int(stim=='sleep')
    var_coeff=3 #3
       
    timestep=defaultclock.dt
    
    sigma= 0.3*siemens/meter
    scale=150*umetre 
    scale_str='150*umetre'
    

    
    #########Définition de toutes les équations :################# 
    
    # Inhibitory
    inh_eqs = '''
    dv/dt = ( - I_leak - I_K - I_Na - I_SynE - I_SynExt - I_SynI - randn()*noise_amp) / ((1 * ufarad * cm ** -2) * (14e3 * umetre ** 2)) : volt 
    Vm = (- I_leak - I_K - I_Na) / ((1 * ufarad * cm ** -2) * (14e3 * umetre ** 2))*timestep : volt 
    I_leak = ((0.1e-3 * siemens * cm ** -2) * (14e3 * umetre ** 2)) * (v - (-65 * mV)) : amp 
    I_K = ((9e-3 * siemens * cm ** -2) * (14e3 * umetre ** 2)) * (n ** 4) * (v - (-90 * mV)) : amp
        dn/dt = (n_inf - n) / tau_n : 1
        n_inf = alphan / (alphan + betan) : 1
        tau_n = 0.2 / (alphan + betan) : second
        alphan = 0.01 * (mV ** -1) * (v  + 34 * mV) / (1. - exp(- 0.1 * (mV ** -1) * (v + 34 * mV))) / ms : Hz
        betan = 0.125 * exp( - (v + 44 * mV) / (80 * mV)) / ms : Hz
    I_Na = ((35e-3 * siemens * cm ** -2) * (14e3 * umetre ** 2)) * (m ** 3) * h * (v - (55 * mV)) : amp
        dm/dt = (m_inf - m) / tau_m : 1
        dh/dt = (h_inf - h) / tau_h : 1
        m_inf = alpham / (alpham + betam)  : 1
        tau_m = 0.2 / (alpham + betam) : second
        h_inf = alphah / (alphah + betah) : 1
        tau_h = 0.2 / (alphah + betah) : second
        alpham = 0.1 * (mV ** -1) * (v + 35 * mV) / (1. - exp(- (v + 35 * mV) / (10 * mV))) / ms : Hz
        betam = 4 * exp(- (v + 60 * mV) / (18 * mV)) / ms : Hz
        alphah = 0.07 * exp(- (v + 58 * mV) / (20 * mV)) / ms : Hz
        betah = 1. / (exp((- 0.1 * (mV ** -1)) * (v + 28 * mV)) + 1.) / ms : Hz
    I_SynE = + ge * (v - (0 * mV)) : amp
        dge/dt = (-ge+he) * (1. / (0.3 * ms)) : siemens
        dhe/dt=-he/(5*ms) : siemens
    I_SynExt = + ge_ext * (v - (0 * mV)) : amp
        dge_ext/dt = (-ge_ext+he_ext) * (1. / (0.3 * ms)) : siemens
        dhe_ext/dt=-he_ext/(5*ms) : siemens    
    I_SynI = + gi * (v - (-80 * mV)) : amp
        dgi/dt = (-gi+hi) * (1. / (1 * ms)) : siemens
        dhi/dt=-hi/(10*ms) : siemens
    x_Sa:metre
    y_Sa:metre
    z_Sa:metre
    '''
    
    # Pyramidal CAN
    global gCAN, gM
    if co=='sleep':
        gCAN=0.5* usiemens*cmetre**-2 #50 si wake, 0.5 si sleep
    else:
        gCAN=25* usiemens*cmetre**-2 #50 si wake, 0.5 si sleep
        
    if num_simu in [4,5,6,7]:  
        if co=='sleep':
            gCAN=25* usiemens*cmetre**-2 #0.5 si wake, 50 si sleep
        else:
            gCAN=0.5* usiemens*cmetre**-2 #0.5 si wake, 50 si sleep
    gM=90 * usiemens*cmetre**-2
    
    py_CAN_eqs = '''
    dv/dt = ( - I_CAN - I_M - I_leak - I_K - I_Na - I_Ca - I_SynE - I_SynExt - I_SynI - randn()*noise_amp) / ((1 * ufarad * cm ** -2) * (29e3 * umetre ** 2)) : volt 
    Vm =( - I_CAN - I_M - I_leak - I_K - I_Na - I_Ca) / ((1 * ufarad * cm ** -2) * (29e3 * umetre ** 2))*timestep : volt 
    I_CAN =  ((gCAN) * (29e3 * umetre ** 2)) * mCAN ** 2 * (v - (-20 * mV)) : amp
        dmCAN/dt = (mCANInf - mCAN) / mCANTau : 1
        mCANInf = alpha2 / (alpha2 + (0.0002 * ms ** -1)) : 1
        mCANTau = 1. / (alpha2 + (0.0002 * ms ** -1)) / (3.0 ** ((36. -22) / 10)) : second
        alpha2 = (0.0002 * ms ** -1) * (Ca_i / (5e-4 * mole * metre ** -3)) ** 2 : Hz 
    I_M = ((gM) * (29e3 * umetre ** 2)) * p * (v - (-100 * mV)) : amp
        dp/dt = (pInf - p) / pTau : 1
        pInf = 1. / (1 + exp(- (v + (35 * mV)) / (10 * mV))) : 1
        pTau = (1000 * ms) / (3.3 * exp((v + (35 * mV)) / (20 * mV)) + exp(- (v + (35 * mV)) / (20 * mV))) : second
    I_leak = ((1e-5 * siemens * cm ** -2) * (29e3 * umetre ** 2)) * (v - (-70 * mV)) : amp 
    I_K = ((5 * msiemens * cm ** -2) * (29e3 * umetre ** 2)) * (n ** 4) * (v - (-100 * mV)) : amp
        dn/dt = alphan * (1 - n) - betan * n : 1
        alphan =  - 0.032 * (mV ** -1) * (v  - (-55 * mV) - 15 * mV) / (exp(- (v - (-55 * mV) - 15 * mV) / (5 * mV)) - 1.) / ms : Hz
        betan = 0.5 * exp( - (v - (-55 * mV) - 10 * mV) / (40 * mV)) / ms : Hz
    I_Na = ((50 * msiemens * cm ** -2) * (29e3 * umetre ** 2)) * (m ** 3) * h * (v - (50 * mV)) : amp
        dm/dt = alpham * (1 - m) - betam * m : 1
        dh/dt = alphah * (1 - h) - betah * h : 1
        alpham = - 0.32 * (mV ** -1) * (v - (-55 * mV) - 13 * mV) / (exp(- (v - (-55 * mV) - 13 * mV) / (4 * mV)) - 1.) / ms : Hz
        betam = 0.28 * (mV ** -1) * (v - (-55 * mV) - 40 * mV) / (exp((v - (-55 * mV) - 40 * mV) / (5 * mV)) - 1.) / ms : Hz
        alphah = 0.128 * exp(- (v - (-55 * mV) - 17 * mV) / (18 * mV)) / ms : Hz
        betah = 4. / (1 + exp(- (v - (-55 * mV) - 40 * mV) / (5 * mV))) / ms : Hz
    I_Ca = ((1e-4 * siemens * cm ** -2) * (29e3 * umetre ** 2)) * (mCaL ** 2) * hCaL * (v - (120 * mV)) : amp
        dmCaL/dt = (alphamCaL * (1 - mCaL)) - (betamCaL * mCaL) : 1
        dhCaL/dt = (alphahCaL * (1 - hCaL)) - (betahCaL * hCaL) : 1
        alphamCaL = (0.055 * mV ** -1) * ((-27 * mV) - v) / (exp(((-27 * mV) - v) / (3.8 * mV)) - 1.) / ms : Hz
        betamCaL = 0.94 * exp(((-75 * mV) - v) / (17 * mV)) / ms : Hz
        alphahCaL = 0.000457 * exp(((-13 * mV) - v) / (50 * mV)) / ms : Hz
        betahCaL = 0.0065 / (exp(((-15 * mV) - v) / (28 * mV)) + 1.) / ms : Hz
        dCa_i/dt = driveChannel + ((2.4e-4 * mole * metre**-3) - Ca_i) /  (200 * ms) : mole * meter**-3
        driveChannel = (-(1e4) * I_Ca / (cm ** 2)) / (2 * (96489 * coulomb * mole ** -1) * (1 * umetre)) : mole * meter ** -3 * Hz
    I_SynE = + ge * (v - (0 * mV)) : amp
        dge/dt = (-ge+he) * (1. / (0.3 * ms)) : siemens
        dhe/dt=-he/(5*ms) : siemens
    I_SynExt = + ge_ext * (v - (0 * mV)) : amp
        dge_ext/dt = (-ge_ext+he_ext) * (1. / (0.3 * ms)) : siemens
        dhe_ext/dt=-he_ext/(5*ms) : siemens        
    I_SynI = + gi * (v - (-80 * mV)) : amp
        dgi/dt = (-gi+hi) * (1. / (1 * ms)) : siemens
        dhi/dt=-hi/(10*ms) : siemens
    x_Sa:metre
    y_Sa:metre
    z_Sa:metre
    x_dendrite:metre
    y_dendrite:metre
    z_dendrite:metre
    x_inh:metre
    y_inh:metre
    z_inh:metre
    dir_x:1
    dir_y:1
    dir_z:1
    '''
    
    
    #Pyramidal non CAN :
    py_eqs = '''
    dv/dt = ( - I_M - I_leak - I_K - I_Na - I_Ca - I_SynE - I_SynExt - I_SynI- randn()*noise_amp) / ((1 * ufarad * cm ** -2) * (29e3 * umetre ** 2)) : volt 
    Vm=( - I_M - I_leak - I_K - I_Na - I_Ca) / ((1 * ufarad * cm ** -2) * (29e3 * umetre ** 2))*timestep : volt 
    I_M = ((gM) * (29e3 * umetre ** 2)) * p * (v - (-100 * mV)) : amp
        dp/dt = (pInf - p) / pTau : 1
        pInf = 1. / (1 + exp(- (v + (35 * mV)) / (10 * mV))) : 1
        pTau = (1000 * ms) / (3.3 * exp((v + (35 * mV)) / (20 * mV)) + exp(- (v + (35 * mV)) / (20 * mV))) : second
    I_leak = ((1e-5 * siemens * cm ** -2) * (29e3 * umetre ** 2)) * (v - (-70 * mV)) : amp 
    I_K = ((5 * msiemens * cm ** -2) * (29e3 * umetre ** 2)) * (n ** 4) * (v - (-100 * mV)) : amp
        dn/dt = alphan * (1 - n) - betan * n : 1
        alphan =  - 0.032 * (mV ** -1) * (v  - (-55 * mV) - 15 * mV) / (exp(- (v - (-55 * mV) - 15 * mV) / (5 * mV)) - 1.) / ms : Hz
        betan = 0.5 * exp( - (v - (-55 * mV) - 10 * mV) / (40 * mV)) / ms : Hz
    I_Na = ((50 * msiemens * cm ** -2) * (29e3 * umetre ** 2)) * (m ** 3) * h * (v - (50 * mV)) : amp
        dm/dt = alpham * (1 - m) - betam * m : 1
        dh/dt = alphah * (1 - h) - betah * h : 1
        alpham = - 0.32 * (mV ** -1) * (v - (-55 * mV) - 13 * mV) / (exp(- (v - (-55 * mV) - 13 * mV) / (4 * mV)) - 1.) / ms : Hz
        betam = 0.28 * (mV ** -1) * (v - (-55 * mV) - 40 * mV) / (exp((v - (-55 * mV) - 40 * mV) / (5 * mV)) - 1.) / ms : Hz
        alphah = 0.128 * exp(- (v - (-55 * mV) - 17 * mV) / (18 * mV)) / ms : Hz
        betah = 4. / (1 + exp(- (v - (-55 * mV) - 40 * mV) / (5 * mV))) / ms : Hz
    I_Ca = ((1e-4 * siemens * cm ** -2) * (29e3 * umetre ** 2)) * (mCaL ** 2) * hCaL * (v - (120 * mV)) : amp
        dmCaL/dt = (alphamCaL * (1 - mCaL)) - (betamCaL * mCaL) : 1
        dhCaL/dt = (alphahCaL * (1 - hCaL)) - (betahCaL * hCaL) : 1
        alphamCaL = (0.055 * mV ** -1) * ((-27 * mV) - v) / (exp(((-27 * mV) - v) / (3.8 * mV)) - 1.) / ms : Hz
        betamCaL = 0.94 * exp(((-75 * mV) - v) / (17 * mV)) / ms : Hz
        alphahCaL = 0.000457 * exp(((-13 * mV) - v) / (50 * mV)) / ms : Hz
        betahCaL = 0.0065 / (exp(((-15 * mV) - v) / (28 * mV)) + 1.) / ms : Hz
        dCa_i/dt = driveChannel + ((2.4e-4 * mole * metre**-3) - Ca_i) /  (200 * ms) : mole * meter**-3
        driveChannel = (-(1e4) * I_Ca / (cm ** 2)) / (2 * (96489 * coulomb * mole ** -1) * (1 * umetre)) : mole * meter ** -3 * Hz
    I_SynE = + ge * (v - (0 * mV)) : amp
        dge/dt = (-ge+he) * (1. / (0.3 * ms)) : siemens
        dhe/dt=-he/(5*ms) : siemens
    I_SynExt = + ge_ext * (v - (0 * mV)) : amp
        dge_ext/dt = (-ge_ext+he_ext) * (1. / (0.3 * ms)) : siemens
        dhe_ext/dt=-he_ext/(5*ms) : siemens        
    I_SynI = + gi * (v - (-80 * mV)) : amp
        dgi/dt = (-gi+hi) * (1. / (1 * ms)) : siemens
        dhi/dt=-hi/(5*ms) : siemens
    x_Sa:metre
    y_Sa:metre
    z_Sa:metre
    x_dendrite:metre
    y_dendrite:metre
    z_dendrite:metre
    x_inh:metre
    y_inh:metre
    z_inh:metre
    dir_x:1
    dir_y:1
    dir_z:1
    '''
    
    
    ## Functions defining neurn groups ##
    
    
    def create_group_py(zone_name):
        G_exc=zone_name+'_py'
        G_exc_coords='coord_'+G_exc
        exec("Nexc=len("+G_exc_coords+"[:,0])",globals())
        G_exc_Dcoords='Dcoord_'+G_exc
        G_exc_Icoords='Icoord_'+G_exc
        G_exc_dir='dir_'+zone_name
        exec(G_exc+"=NeuronGroup("+str(Nexc)+",py_eqs,threshold='v>V_th',refractory=3*ms,method='exponential_euler')", globals())
        exec(G_exc+".v = '-60*mvolt-rand()*10*mvolt'", globals()) 
        exec(G_exc+".x_Sa="+G_exc_coords+"[:,0]*scale", globals())
        exec(G_exc+".y_Sa="+G_exc_coords+"[:,1]*scale", globals())
        exec(G_exc+".z_Sa="+G_exc_coords+"[:,2]*scale", globals())
        exec(G_exc+".x_dendrite="+G_exc_Dcoords+"[:,0]*scale", globals())
        exec(G_exc+".y_dendrite="+G_exc_Dcoords+"[:,1]*scale", globals())
        exec(G_exc+".z_dendrite="+G_exc_Dcoords+"[:,2]*scale", globals())
        exec(G_exc+".x_inh="+G_exc_Icoords+"[:,0]*scale", globals())
        exec(G_exc+".y_inh="+G_exc_Icoords+"[:,1]*scale", globals())
        exec(G_exc+".z_inh="+G_exc_Icoords+"[:,2]*scale", globals())
        exec(G_exc+".dir_x ="+G_exc_dir+"[:,0]", globals())
        exec(G_exc+".dir_y ="+G_exc_dir+"[:,1]", globals())
        exec(G_exc+".dir_z ="+G_exc_dir+"[:,2]", globals())
       
        
    def create_group_pyCAN(zone_name):
        G_exc=zone_name+'_py_CAN'
        G_exc_coords='coord_'+G_exc
        exec("NCAN=len("+G_exc_coords+"[:,0])",globals())
        G_exc_Dcoords='Dcoord_'+G_exc
        G_exc_Icoords='Icoord_'+G_exc
        G_exc_dir='dir_'+zone_name+'_CAN'
        exec(G_exc+"=NeuronGroup("+str(NCAN)+",py_CAN_eqs,threshold='v>V_th',refractory=3*ms,method='exponential_euler')", globals())
        exec(G_exc+".v = '-60*mvolt-rand()*10*mvolt'", globals())
        exec(G_exc+".x_Sa="+G_exc_coords+"[:,0]*scale", globals())
        exec(G_exc+".y_Sa="+G_exc_coords+"[:,1]*scale", globals())
        exec(G_exc+".z_Sa="+G_exc_coords+"[:,2]*scale", globals())
        exec(G_exc+".x_dendrite="+G_exc_Dcoords+"[:,0]*scale", globals())
        exec(G_exc+".y_dendrite="+G_exc_Dcoords+"[:,1]*scale", globals())
        exec(G_exc+".z_dendrite="+G_exc_Dcoords+"[:,2]*scale", globals()) 
        exec(G_exc+".x_inh="+G_exc_Icoords+"[:,0]*scale", globals())
        exec(G_exc+".y_inh="+G_exc_Icoords+"[:,1]*scale", globals())
        exec(G_exc+".z_inh="+G_exc_Icoords+"[:,2]*scale", globals())
        exec(G_exc+".dir_x ="+G_exc_dir+"[:,0]", globals())
        exec(G_exc+".dir_y ="+G_exc_dir+"[:,1]", globals())
        exec(G_exc+".dir_z ="+G_exc_dir+"[:,2]", globals())
          
        
    def create_group_inh(zone_name):
        G_inh=zone_name+'_inh'
        G_inh_coords='coord_'+G_inh
        exec("Ninh=len("+G_inh_coords+"[:,0])",globals())
        exec(G_inh+"=NeuronGroup("+str(Ninh)+",inh_eqs,threshold='v>V_th',refractory=3*ms,method='exponential_euler')", globals())
        exec(G_inh+".v = '-60*mvolt-rand()*10*mvolt'", globals())
        exec(G_inh+".x_Sa="+G_inh_coords+"[:,0]*scale", globals())
        exec(G_inh+".y_Sa="+G_inh_coords+"[:,1]*scale", globals())
        exec(G_inh+".z_Sa="+G_inh_coords+"[:,2]*scale", globals()) 
        
    print('Creating the neurons')    
    #Pour CA3    
    create_group_pyCAN('CA3')
    create_group_inh('CA3')
        
    #Pour CA1 :    
    create_group_pyCAN('CA1')
    create_group_inh('CA1')    
    
    #Pour le gyrus denté :
    create_group_py('DG')
    create_group_inh('DG')
      
    #Pour le cortex entorhinal :    
    create_group_pyCAN('EC')
    create_group_inh('EC')
        
    
    if pCAN!=1:
        create_group_py('CA3')
        create_group_py('CA1')
        create_group_py('EC')
    
    
    print('Adding synapses')
    ## Definition of the synaptic connections within each region ##
    def create_syn(zone_name, has_CAN, has_noCAN, p_EE, sigEE, p_EI, sigEI, p_IE, sigIE, p_II, sigII, var_E, var_I):
        if p_EE!=0:
            if has_noCAN:
                exec(zone_name+"_EE=Synapses("+zone_name+"_py,"+zone_name+"_py,on_pre='''he_post+="+var_E+"*g_max_e''')", globals())
                exec(zone_name+"_EE.connect(condition='i!=j',p='"+p_EE+"*exp(-((x_Sa_pre-x_Sa_post)**2+(y_Sa_pre-y_Sa_post)**2+(z_Sa_pre-z_Sa_post)**2)/(2*"+sigEE+"**2))')", globals())
            if has_CAN :
                exec(zone_name+"_EcEc=Synapses("+zone_name+"_py_CAN,"+zone_name+"_py_CAN,on_pre='''he_post+="+var_E+"*g_max_e''')", globals())
                exec(zone_name+"_EcEc.connect(condition='i!=j',p='"+p_EE+"*exp(-((x_Sa_pre-x_Sa_post)**2+(y_Sa_pre-y_Sa_post)**2+(z_Sa_pre-z_Sa_post)**2)/(2*"+sigEE+"**2))')", globals())
                if has_noCAN:
                    exec(zone_name+"_EEc=Synapses("+zone_name+"_py,"+zone_name+"_py_CAN,on_pre='''he_post+="+var_E+"*g_max_e''')", globals())
                    exec(zone_name+"_EEc.connect(condition='i!=j',p='"+p_EE+"*exp(-((x_Sa_pre-x_Sa_post)**2+(y_Sa_pre-y_Sa_post)**2+(z_Sa_pre-z_Sa_post)**2)/(2*"+sigEE+"**2))')", globals()) 
                    exec(zone_name+"_EcE=Synapses("+zone_name+"_py_CAN,"+zone_name+"_py,on_pre='''he_post+="+var_E+"*g_max_e''')", globals())
                    exec(zone_name+"_EcE.connect(condition='i!=j',p='"+p_EE+"*exp(-((x_Sa_pre-x_Sa_post)**2+(y_Sa_pre-y_Sa_post)**2+(z_Sa_pre-z_Sa_post)**2)/(2*"+sigEE+"**2))')", globals())

               
        if p_EI!=0:
            if has_noCAN:
                exec(zone_name+"_EI=Synapses("+zone_name+"_py,"+zone_name+"_inh,on_pre='''he_post+="+var_E+"*g_max_e''')", globals())
                exec(zone_name+"_EI.connect(p='"+p_EI+"*exp(-((x_Sa_pre-x_Sa_post)**2+(y_Sa_pre-y_Sa_post)**2+(z_Sa_pre-z_Sa_post)**2)/(2*"+sigEI+"**2))')", globals())
            if has_CAN :
                exec(zone_name+"_EcI=Synapses("+zone_name+"_py_CAN,"+zone_name+"_inh,on_pre='''he_post+="+var_E+"*g_max_e''')", globals())
                exec(zone_name+"_EcI.connect(p='"+p_EI+"*exp(-((x_Sa_pre-x_Sa_post)**2+(y_Sa_pre-y_Sa_post)**2+(z_Sa_pre-z_Sa_post)**2)/(2*"+sigEI+"**2))')", globals()) 
                
        if p_IE!=0: 
            if has_noCAN:
                exec(zone_name+"_IE=Synapses("+zone_name+"_inh,"+zone_name+"_py,on_pre='''hi_post+="+var_I+"*g_max_i''')", globals())
                exec(zone_name+"_IE.connect(p='"+p_IE+"*exp(-((x_Sa_pre-x_inh_post)**2+(y_Sa_pre-y_inh_post)**2+(z_Sa_pre-z_inh_post)**2)/(2*"+sigIE+"**2))')", globals())
            if has_CAN :
                exec(zone_name+"_IEc=Synapses("+zone_name+"_inh,"+zone_name+"_py_CAN,on_pre='''hi_post+="+var_I+"*g_max_i''')", globals())
                exec(zone_name+"_IEc.connect(p='"+p_IE+"*exp(-((x_Sa_pre-x_inh_post)**2+(y_Sa_pre-y_inh_post)**2+(z_Sa_pre-z_inh_post)**2)/(2*"+sigIE+"**2))')", globals())   
                
        if p_II!=0:        
            exec(zone_name+"_II=Synapses("+zone_name+"_inh,"+zone_name+"_inh,on_pre='''hi_post+="+var_I+"*g_max_i''')", globals())
            exec(zone_name+"_II.connect(p='"+p_II+"*exp(-((x_Sa_pre-x_Sa_post)**2+(y_Sa_pre-y_Sa_post)**2+(z_Sa_pre-z_Sa_post)**2)/(2*"+sigII+"**2))')", globals())
               
                             
          

    sigE, sigI='(2500*umetre)', '(350*umetre)'  #350  
    #In CA3
    var_E_CA3=int(co=='sleep')+int(co=='wake')/var_coeff
    var_I_CA3=1
    p_CA3_EI, p_CA3_EE, p_CA3_IE=0.75, 0.56, 0.75 #0.1, 0.1, 0.7 #0.1 0.1 0.6
    create_syn('CA3', True, pCAN<1, str(p_CA3_EE), sigE, str(p_CA3_EI), sigE, str(p_CA3_IE), sigI,0, 0, str(var_E_CA3), str(var_I_CA3))

    #In CA1
    p_CA1_EI, p_CA1_IE, p_CA1_II=0.28, 0.3, 0.7 #0.3, 0.8, 0.9 #0.3 0.5 0.9
    var_E_CA1=1
    var_I_CA1=int(co=='sleep')+int(co=='wake')*var_coeff
    create_syn('CA1', True,pCAN<1, 0, 0, str(p_CA1_EI), sigE, str(p_CA1_IE), sigI, str(p_CA1_II), sigI, str(var_E_CA1), str(var_I_CA1))

    #In the dentate gyrus
    p_DG_EE, p_DG_EI, p_DG_IE= 0, 0.06, 0.14                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    var_E_DG=int(co=='sleep')+int(co=='wake')*var_coeff
    var_I_DG=int(co=='sleep')+int(co=='wake')*var_coeff
    create_syn('DG', False, True, str(p_DG_EE), sigE, str(p_DG_EI), sigE, str(p_DG_IE), sigI, 0, sigI, str(var_E_DG), str(var_I_DG))

    #In the entorhinal cortex
    p_EC_EI, p_EC_IE=0.37, 0.54
    var_E_EC=int(co=='sleep')+int(co=='wake')/var_coeff
    var_I_EC=1
    create_syn('EC', True,pCAN<1, 0, 0, str(p_EC_EI), sigE, str(p_EC_IE), sigI, 0,0, str(var_E_EC), str(var_I_EC))

 
    ## Definition of the synaptic connections between different regions ##
                                 
    def connect_2zones(zone_name1, has_CAN1, has_noCAN1 ,zone_name2, has_CAN2, has_noCAN2, pE, pI, sig_E, var_E):
        if has_noCAN1:
            exec(zone_name1+"_"+zone_name2+"_I = Synapses("+zone_name1+"_py,"+zone_name2+"_inh,on_pre='''he_ext_post+="+var_E+"*g_max_e''')", globals())
            exec(zone_name1+"_"+zone_name2+"_I.connect(p="+pI+")", globals())
            if has_noCAN2 :
                exec(zone_name1+"_"+zone_name2+"_E = Synapses("+zone_name1+"_py,"+zone_name2+"_py,on_pre='''he_ext_post+="+var_E+"*g_max_e''')", globals())
                exec(zone_name1+"_"+zone_name2+"_E.connect(p="+pE+")", globals())
            if has_CAN2:
                exec(zone_name1+"_"+zone_name2+"c_E = Synapses("+zone_name1+"_py,"+zone_name2+"_py_CAN,on_pre='''he_ext_post+="+var_E+"*g_max_e''')", globals())
                exec(zone_name1+"_"+zone_name2+"c_E.connect(p="+pE+")", globals())
                
        if has_CAN1:
            exec(zone_name1+"c_"+zone_name2+"_I = Synapses("+zone_name1+"_py_CAN,"+zone_name2+"_inh,on_pre='''he_ext_post+="+var_E+"*g_max_e''')", globals())
            exec(zone_name1+"c_"+zone_name2+"_I.connect(p="+pI+")", globals()) 
            
            if has_noCAN2:
                exec(zone_name1+"c_"+zone_name2+"_E = Synapses("+zone_name1+"_py_CAN,"+zone_name2+"_py,on_pre='''he_ext_post+="+var_E+"*g_max_e''')", globals())
                exec(zone_name1+"c_"+zone_name2+"_E.connect(p="+pE+")", globals())

            if has_CAN2:
                exec(zone_name1+"c_"+zone_name2+"c_E = Synapses("+zone_name1+"_py_CAN,"+zone_name2+"_py_CAN,on_pre='''he_ext_post+="+var_E+"*g_max_e''')", globals())
                exec(zone_name1+"c_"+zone_name2+"c_E.connect(p="+pE+")", globals())
                              
      
    ## From the entorhinal cortex to the dentate gyrus ##
    sig_E='(2500*umetre)'
    p_EC_DG_I=p_co
    p_EC_DG_E=p_co
    connect_2zones('EC', True, pCAN<1,'DG', False, True, str(p_EC_DG_E), str(p_EC_DG_I),sig_E,str(var_E_EC))
     
    ## From the dentate gyrus to CA3 ##
    p_DG_CA3_I=p_co
    p_DG_CA3_E=p_co
    connect_2zones('DG', False, True,'CA3', True, pCAN<1, str(p_DG_CA3_E), str(p_DG_CA3_I), sig_E,str(var_E_DG)) 

    ## From the entorhinal cortex to CA3 ##
    p_EC_CA3_I=p_co_CA3
    p_EC_CA3_E=p_co_CA3
    connect_2zones('EC', True, pCAN<1,'CA3', True, pCAN<1, str(p_EC_CA3_E), str(p_EC_CA3_I), sig_E,str(var_E_EC))

    ## From the entorhinal cortex to CA1
    p_EC_CA1_I=p_co_CA3
    p_EC_CA1_E=p_co_CA3
    connect_2zones('EC', True, pCAN<1,'CA1', True, pCAN<1, str(p_EC_CA1_E), str(p_EC_CA1_I), sig_E,str(var_E_EC))

    ## From CA3 to CA1 ##
    p_CA3_CA1_I=p_co
    p_CA3_CA1_E=p_co
    connect_2zones('CA3', True, pCAN<1, 'CA1', True, pCAN<1, str(p_CA3_CA1_E), str(p_CA3_CA1_I), sig_E,str(var_E_CA3))
    
    ## From CA1 to the entorhinal cortex ##
    p_CA1_EC_I=p_co
    p_CA1_EC_E=p_co
    connect_2zones('CA1', True, pCAN<1,'EC', True, pCAN<1, str(p_CA1_EC_E), str(p_CA1_EC_I), sig_E,str(var_E_CA1))


def process(num_simu,g_max_e,g_max_i,p_co,p_co_CA3) :
    print('Simulation n°'+str(num_simu+1)+'/80')
    global all_CA1_t, all_CA1_i,all_EC_t,all_EC_i
    nb_runs=int(10*runtime/second)

    start_scope()
    prefs.codegen.target = 'numpy'

    n,i,m=0,0,0
    del n
    del i  
    del m

    ver=((num_simu%64)//8)+1
    input_num=chr(num_simu//64+65)
    version='_'+str(ver)+input_num
    type_simu=(num_simu%64)%8
    input_num=ord(input_num)-64
    
    print('Building the network')    
              
    preparation(type_simu,g_max_e,g_max_i,p_co,p_co_CA3)
    print('Adding the inputs')
    
    all_simu_types=['S_S','S_W','W_S','W_W','S_S_CAN','S_W_noCAN','W_S_CAN','W_W_noCAN']
    simu=all_simu_types[type_simu]+version
    print(simu)
    
    if type_simu in [0,1,4,5]:
        stim='sleep'
    else :
        stim='wake'
        
    if type_simu in [0,2,4,6]:
        co='sleep'
    else :
        co='wake'
    
    
    input_S_1=read_file('input_files/input_sleep_B12_B11_480_'+str(input_num)+'.txt')
    input_S_2=read_file('input_files/input_sleep_O9_O8_480_'+str(input_num)+'.txt')
    input_S_3=read_file('input_files/input_sleep_P7_P6_480_'+str(input_num)+'.txt')
    
    input_W_1=read_file('input_files/input_wake_B12_B11_480_'+str(input_num)+'.txt')
    input_W_2=read_file('input_files/input_wake_O9_O8_480_'+str(input_num)+'.txt')
    input_W_3=read_file('input_files/input_wake_P7_P6_480_'+str(input_num)+'.txt')
    
    N=2
    Fc=40
    fs=1024
    nyq = 0.5 * fs
    low = 3 / nyq
    high=50/nyq
    b, a = scipy.signal.butter(N, high, btype='low') 
    
    record_dt=1./1024 *second
    #[120000:] 
    
    debut=(int(ver)-1)*60000
    fin=int(ver)*60000
    print(debut,fin)
    inputs_filt_S_1=scipy.signal.filtfilt(b,a,input_S_1[debut:fin])
    inputs_filt_S_2=scipy.signal.filtfilt(b,a,input_S_2[debut:fin])
    inputs_filt_S_3=scipy.signal.filtfilt(b,a,input_S_3[debut:fin])

    inputs_envelope_S_1=abs(inputs_filt_S_1)
    inputs_envelope_S_2=abs(inputs_filt_S_2)
    inputs_envelope_S_3=abs(inputs_filt_S_3)
    
    inputs_envelope_S_1=5/6*inputs_envelope_S_1+max(inputs_envelope_S_1)/6*rand(len(inputs_envelope_S_1))
    inputs_envelope_S_2=5/6*inputs_envelope_S_2+max(inputs_envelope_S_2)/6*rand(len(inputs_envelope_S_2))
    inputs_envelope_S_3=5/6*inputs_envelope_S_3+max(inputs_envelope_S_3)/6*rand(len(inputs_envelope_S_3))

    MMM=max((max(inputs_envelope_S_1),max(inputs_envelope_S_2),max(inputs_envelope_S_3)))
#    print(MMM)
    inputs_envelope_S_1=200*inputs_envelope_S_1/MMM
    inputs_envelope_S_2=200*inputs_envelope_S_2/MMM
    inputs_envelope_S_3=200*inputs_envelope_S_3/MMM


    input_S_1=TimedArray(inputs_envelope_S_1*Hz,dt=record_dt)
    input_S_2=TimedArray(inputs_envelope_S_2*Hz,dt=record_dt)
    input_S_3=TimedArray(inputs_envelope_S_3*Hz,dt=record_dt) 
    
    print(inputs_envelope_S_1)
    
#    figure()
#    plot(inputs_envelope_S_1)
#    plot(inputs_envelope_S_2)
#    plot(inputs_envelope_S_3)

#    bruit=0.8*mean(array([max(inputs_envelope_S_1),max(inputs_envelope_S_2),max(inputs_envelope_S_3)]))*Hz*(rand(int(runtime/record_dt)))
#    print(bruit)
#    #inputs_bruit=TimedArray(max(max(input_S_1),max(input_S_2),max(input_S_3))*rand(int(runtime/record_dt)),dt=record_dt)
#    input_bruit=TimedArray(bruit,dt=record_dt)
      
    inputs_filt_W_1=scipy.signal.filtfilt(b,a,input_W_1[debut:fin])
    inputs_envelope_W_1=abs(inputs_filt_W_1)
    
    inputs_filt_W_2=scipy.signal.filtfilt(b,a,input_W_2[debut:fin])
    inputs_envelope_W_2=abs(inputs_filt_W_2)
    
    inputs_filt_W_3=scipy.signal.filtfilt(b,a,input_W_3[debut:fin])
    inputs_envelope_W_3=abs(inputs_filt_W_3)
    
    MMM=max((max(inputs_envelope_W_1),max(inputs_envelope_W_2),max(inputs_envelope_W_3)))
    inputs_envelope_W_1=200*inputs_envelope_W_1/MMM
    inputs_envelope_W_2=200*inputs_envelope_W_2/MMM
    inputs_envelope_W_3=200*inputs_envelope_W_3/MMM  
    
    input_W_1=TimedArray(inputs_envelope_W_1*Hz,dt=record_dt)
    input_W_2=TimedArray(inputs_envelope_W_2*Hz,dt=record_dt)
    input_W_3=TimedArray(inputs_envelope_W_3*Hz,dt=record_dt) 
    
#    sum_S=inputs_envelope_S_1+inputs_envelope_S_2+inputs_envelope_S_3
#    sum_ve=inputs_envelope_ve_1+inputs_envelope_ve_2+inputs_envelope_ve_3
    
#    print('Preparation du réseau')    
#    n,i,m=0,0,0
#    del n
#    del i  
#    del m              
#    preparation(num_simu,g_max_e,g_max_i,p_co,p_co_CA3)
    #print('Adding the inputs')
    
    
    #ajout des inputs
    if stim=='sleep':
        #print('test')
        inputs1=input_S_1
        inputs2=input_S_2
        inputs3=input_S_3
    else :
        inputs1=input_W_1
        inputs2=input_W_2
        inputs3=input_W_3
    

    In_exc1=NeuronGroup(10000, 'rates : Hz', threshold='rand()<inputs1(t)*timestep')    #dt ? record_dt ?
    S11 = Synapses(In_exc1, EC_py_CAN, on_pre='he_post+=g_max_e')
    S11.connect(p='p_in')
    S13 = Synapses(In_exc1, EC_inh, on_pre='he_post+=g_max_e')
    S13.connect(p='p_in')

    In_exc2=NeuronGroup(10000, 'rates : Hz', threshold='rand()<inputs2(t)*timestep')    #dt ? record_dt ?
    S21 = Synapses(In_exc2, EC_py_CAN, on_pre='he_post+=g_max_e')
    S21.connect(p='p_in')
    S23 = Synapses(In_exc2, EC_inh, on_pre='he_post+=g_max_e')
    S23.connect(p='p_in')

    In_exc3=NeuronGroup(10000, 'rates : Hz', threshold='rand()<inputs3(t)*timestep')    #dt ? record_dt ?
    S31 = Synapses(In_exc3, EC_py_CAN, on_pre='he_post+=g_max_e')
    S31.connect(p='p_in')
    S33 = Synapses(In_exc3, EC_inh, on_pre='he_post+=g_max_e')
    S33.connect(p='p_in') 

    
    
    if pCAN<1:
            S12 = Synapses(In_exc1, EC_py, on_pre='he_post+=g_max_e')
            S12.connect(p=p_in)
            S22 = Synapses(In_exc2, EC_py, on_pre='he_post+=g_max_e')
            S22.connect(p=p_in)
            S32 = Synapses(In_exc3, EC_py, on_pre='he_post+=g_max_e')
            S32.connect(p=p_in)
    
    #### Simultation #######
    print('Changing compilation method')
    prefs.codegen.target = 'cython' 
    
    single_runtime=runtime/nb_runs
    signal_principal=zeros(int(runtime/timestep))

    
    for test_ind in range(nb_runs):
        
        syn_CA3_py_CAN_E = StateMonitor(CA3_py_CAN,'I_SynE',record=True,dt=timestep)
        syn_CA1_py_CAN_E = StateMonitor(CA1_py_CAN,'I_SynE',record=True,dt=timestep)
        syn_DG_py_E = StateMonitor(DG_py,'I_SynE',record=True,dt=timestep)
        syn_EC_py_CAN_E = StateMonitor(EC_py_CAN,'I_SynE',record=True,dt=timestep)
        
        syn_CA3_py_CAN_Ext = StateMonitor(CA3_py_CAN,'I_SynExt',record=True,dt=timestep)
        syn_CA1_py_CAN_Ext = StateMonitor(CA1_py_CAN,'I_SynExt',record=True,dt=timestep)
        syn_DG_py_Ext = StateMonitor(DG_py,'I_SynExt',record=True,dt=timestep)
        syn_EC_py_CAN_Ext = StateMonitor(EC_py_CAN,'I_SynExt',record=True,dt=timestep)
        
        syn_CA3_py_CAN_I = StateMonitor(CA3_py_CAN,'I_SynI',record=True,dt=timestep)
        syn_CA1_py_CAN_I = StateMonitor(CA1_py_CAN,'I_SynI',record=True,dt=timestep)
        syn_DG_py_I = StateMonitor(DG_py,'I_SynI',record=True,dt=timestep)
        syn_EC_py_CAN_I = StateMonitor(EC_py_CAN,'I_SynI',record=True,dt=timestep)
        
        
        run(single_runtime,report='text',report_period=300*second)
        
        
        ###Calcul du LFP
        print('Calcul du LFP')  
        start_plot_time=500*msecond
        start_ind=int(start_plot_time/record_dt)      
        

        all_isyn=zeros((len(elec_pos),int(single_runtime/timestep)))
        lfp_dg_e=zeros((len(elec_pos),int(single_runtime/timestep)))
        lfp_dg_i=zeros((len(elec_pos),int(single_runtime/timestep)))
        lfp_ec_e=zeros((len(elec_pos),int(single_runtime/timestep)))
        lfp_ec_i=zeros((len(elec_pos),int(single_runtime/timestep)))
        lfp_ca1_e=zeros((len(elec_pos),int(single_runtime/timestep)))
        lfp_ca1_i=zeros((len(elec_pos),int(single_runtime/timestep)))
        lfp_ca3_e=zeros((len(elec_pos),int(single_runtime/timestep)))
        lfp_ca3_i=zeros((len(elec_pos),int(single_runtime/timestep)))
        
        xx=array(elec_pos)[:,0]*scale
        yy=array(elec_pos)[:,1]*scale
        zz=array(elec_pos)[:,2]*scale
        
        ##For DG:
        x=tile(xx,(len(DG_py.x_Sa),1)).T
        y=tile(yy,(len(DG_py.x_Sa),1)).T
        z=tile(zz,(len(DG_py.x_Sa),1)).T   
        dx=x-(DG_py.x_Sa+DG_py.x_dendrite)*0.5
        dy=y-(DG_py.y_Sa+DG_py.y_dendrite)*0.5
        dz=z-(DG_py.z_Sa+DG_py.z_dendrite)*0.5
        dist=(dx**2+dy**2+dz**2)**0.5
        w=1/(4*pi*sigma*dist**2)*((DG_py.x_Sa-DG_py.x_dendrite)**2+(DG_py.y_Sa-DG_py.y_dendrite)**2+(DG_py.z_Sa-DG_py.z_dendrite)**2)**0.5
        cos_angle=(DG_py.dir_x*dx+DG_py.dir_y*dy+DG_py.dir_z*dz)/dist
        lfp_dg_e+=w*cos_angle@syn_DG_py_E.I_SynE
        
        dx=x-(DG_py.x_Sa+DG_py.x_inh)*0.5
        dy=y-(DG_py.y_Sa+DG_py.y_inh)*0.5
        dz=z-(DG_py.z_Sa+DG_py.z_inh)*0.5
        dist=(dx**2+dy**2+dz**2)**0.5
        w=1/(4*pi*sigma*dist**2)*((DG_py.x_Sa-DG_py.x_inh)**2+(DG_py.y_Sa-DG_py.y_inh)**2+(DG_py.z_Sa-DG_py.z_inh)**2)**0.5
        cos_angle=(DG_py.dir_x*dx+DG_py.dir_y*dy+DG_py.dir_z*dz)/dist
        lfp_dg_i+=w*cos_angle@syn_DG_py_I.I_SynI
        
        lfp_dg_e+=w*cos_angle@syn_DG_py_Ext.I_SynExt #de DG vers EC : basal
                
        ##For EC :
        
        x=tile(xx,(len(EC_py_CAN.x_Sa),1)).T
        y=tile(yy,(len(EC_py_CAN.x_Sa),1)).T
        z=tile(zz,(len(EC_py_CAN.x_Sa),1)).T    
        dx=x-(EC_py_CAN.x_Sa+EC_py_CAN.x_dendrite)*0.5
        dy=y-(EC_py_CAN.y_Sa+EC_py_CAN.y_dendrite)*0.5
        dz=z-(EC_py_CAN.z_Sa+EC_py_CAN.z_dendrite)*0.5
        dist=(dx**2+dy**2+dz**2)**0.5
        w=1/(4*pi*sigma*dist**2)*((EC_py_CAN.x_Sa-EC_py_CAN.x_dendrite)**2+(EC_py_CAN.y_Sa-EC_py_CAN.y_dendrite)**2+(EC_py_CAN.z_Sa-EC_py_CAN.z_dendrite)**2)**0.5
        cos_angle=(EC_py_CAN.dir_x*dx+EC_py_CAN.dir_y*dy+EC_py_CAN.dir_z*dz)/dist
        lfp_ec_e+=w*cos_angle@syn_EC_py_CAN_E.I_SynE 
        
        dx=x-(EC_py_CAN.x_Sa+EC_py_CAN.x_inh)*0.5
        dy=y-(EC_py_CAN.y_Sa+EC_py_CAN.y_inh)*0.5
        dz=z-(EC_py_CAN.z_Sa+EC_py_CAN.z_inh)*0.5
        dist=(dx**2+dy**2+dz**2)**0.5
        w=1/(4*pi*sigma*dist**2)*((EC_py_CAN.x_Sa-EC_py_CAN.x_inh)**2+(EC_py_CAN.y_Sa-EC_py_CAN.y_inh)**2+(EC_py_CAN.z_Sa-EC_py_CAN.z_inh)**2)**0.5
        cos_angle=(EC_py_CAN.dir_x*dx+EC_py_CAN.dir_y*dy+EC_py_CAN.dir_z*dz)/dist
        lfp_ec_i+=w*cos_angle@syn_EC_py_CAN_I.I_SynI 
        
        lfp_ec_e+=w*cos_angle@syn_EC_py_CAN_Ext.I_SynExt #de l'extérieur vers l'EC ? basal ?
        
        
        x=tile(xx,(len(CA1_py_CAN.x_Sa),1)).T
        y=tile(yy,(len(CA1_py_CAN.x_Sa),1)).T
        z=tile(zz,(len(CA1_py_CAN.x_Sa),1)).T    
        dx=x-(CA1_py_CAN.x_Sa+CA1_py_CAN.x_dendrite)*0.5
        dy=y-(CA1_py_CAN.y_Sa+CA1_py_CAN.y_dendrite)*0.5
        dz=z-(CA1_py_CAN.z_Sa+CA1_py_CAN.z_dendrite)*0.5
        dist=(dx**2+dy**2+dz**2)**0.5
        w=1/(4*pi*sigma*dist**2)*((CA1_py_CAN.x_Sa-CA1_py_CAN.x_dendrite)**2+(CA1_py_CAN.y_Sa-CA1_py_CAN.y_dendrite)**2+(CA1_py_CAN.z_Sa-CA1_py_CAN.z_dendrite)**2)**0.5
        cos_angle=(CA1_py_CAN.dir_x*dx+CA1_py_CAN.dir_y*dy+CA1_py_CAN.dir_z*dz)/dist
        lfp_ca1_e+=w*cos_angle@syn_CA1_py_CAN_E.I_SynE
        
        lfp_ca1_e+=w*cos_angle@syn_CA1_py_CAN_Ext.I_SynExt #de CA3 à CA1 et de EC à CA1 : apical
        
        dx=x-(CA1_py_CAN.x_Sa+CA1_py_CAN.x_inh)*0.5
        dy=y-(CA1_py_CAN.y_Sa+CA1_py_CAN.y_inh)*0.5
        dz=z-(CA1_py_CAN.z_Sa+CA1_py_CAN.z_inh)*0.5
        dist=(dx**2+dy**2+dz**2)**0.5
        w=1/(4*pi*sigma*dist**2)*((CA1_py_CAN.x_Sa-CA1_py_CAN.x_inh)**2+(CA1_py_CAN.y_Sa-CA1_py_CAN.y_inh)**2+(CA1_py_CAN.z_Sa-CA1_py_CAN.z_inh)**2)**0.5
        cos_angle=(CA1_py_CAN.dir_x*dx+CA1_py_CAN.dir_y*dy+CA1_py_CAN.dir_z*dz)/dist
        lfp_ca1_i+=w*cos_angle@syn_CA1_py_CAN_I.I_SynI

        
        x=tile(xx,(len(CA3_py_CAN.x_Sa),1)).T
        y=tile(yy,(len(CA3_py_CAN.x_Sa),1)).T
        z=tile(zz,(len(CA3_py_CAN.x_Sa),1)).T   
        dx=x-(CA3_py_CAN.x_Sa+CA3_py_CAN.x_dendrite)*0.5
        dy=y-(CA3_py_CAN.y_Sa+CA3_py_CAN.y_dendrite)*0.5
        dz=z-(CA3_py_CAN.z_Sa+CA3_py_CAN.z_dendrite)*0.5
        dist=(dx**2+dy**2+dz**2)**0.5
        w=1/(4*pi*sigma*dist**2)*((CA3_py_CAN.x_Sa-CA3_py_CAN.x_dendrite)**2+(CA3_py_CAN.y_Sa-CA3_py_CAN.y_dendrite)**2+(CA3_py_CAN.z_Sa-CA3_py_CAN.z_dendrite)**2)**0.5
        cos_angle=(CA3_py_CAN.dir_x*dx+CA3_py_CAN.dir_y*dy+CA3_py_CAN.dir_z*dz)/dist
        lfp_ca3_e+=w*cos_angle@syn_CA3_py_CAN_E.I_SynE 
        
        lfp_ca3_e+=w*cos_angle@syn_CA3_py_CAN_Ext.I_SynExt # De DG à CA3 et de EC à CA3 : apical
        
        dx=x-(CA3_py_CAN.x_Sa+CA3_py_CAN.x_inh)*0.5
        dy=y-(CA3_py_CAN.y_Sa+CA3_py_CAN.y_inh)*0.5
        dz=z-(CA3_py_CAN.z_Sa+CA3_py_CAN.z_inh)*0.5
        dist=(dx**2+dy**2+dz**2)**0.5
        w=1/(4*pi*sigma*dist**2)*((CA3_py_CAN.x_Sa-CA3_py_CAN.x_inh)**2+(CA3_py_CAN.y_Sa-CA3_py_CAN.y_inh)**2+(CA3_py_CAN.z_Sa-CA3_py_CAN.z_inh)**2)**0.5
        cos_angle=(CA3_py_CAN.dir_x*dx+CA3_py_CAN.dir_y*dy+CA3_py_CAN.dir_z*dz)/dist
        lfp_ca3_i+=w*cos_angle@syn_CA3_py_CAN_I.I_SynI 
        
        all_isyn=lfp_dg_e+lfp_dg_i+lfp_ec_e+lfp_ec_i+lfp_ca1_e+lfp_ca1_i+lfp_ca3_e+lfp_ca3_i
        
        isyn_principal_1=sum(all_isyn[:144,:],axis=0)/144
        isyn_principal_2=sum(all_isyn[144:288,:],axis=0)/144
        isyn_principal=isyn_principal_2-isyn_principal_1


        del w
        del cos_angle
        del dx,dy,dz,dist
        signal_principal[test_ind*int(single_runtime/timestep):(test_ind+1)*int(single_runtime/timestep)]=isyn_principal

    print('This simulation has ended.')
    
    N=3
    fs=1/timestep*second
    nyq = 0.5 * fs
    low=0.15 / nyq
    high = 480 / nyq
    b, a = scipy.signal.butter(N, high, btype='low')
    res_filt=scipy.signal.filtfilt(b,a,signal_principal)  
    b, a = scipy.signal.butter(N, low, btype='high')
    res_filt=scipy.signal.filtfilt(b,a,res_filt) 
    step=int(1/1024/timestep*second)
    res_1024=res_filt[::step]
    
    all_simu_types=['S_S','S_W','W_S','W_W','S_S_CAN','S_W_noCAN','W_S_CAN','W_W_noCAN']

    simu=all_simu_types[type_simu]+version
    write_file(simu,res_1024,0*second,runtime)
    
    event_detection_and_analysis(res_1024,simu,1024*Hz)
    
    return res_1024

def main_process(simu_range,g_max_e,g_max_i,p_co,p_co_CA3):
    t1=time.time()    
    #close("all")
    
    global runtime,record_dt,start_ind,simu,version,epilepsy,raster,pCAN
    runtime =60*second  
    print(runtime)
    record_dt=1./1024 *second
    start_ind=int(500*msecond/record_dt)
    timestep=defaultclock.dt
    
    all_simu_types=['S_S','S_W','W_S','W_W','S_S_CAN','S_W_noCAN','W_S_CAN','W_W_noCAN']   
    pCAN=1
    
    global coord_CA1_py,Dcoord_CA1_py,Icoord_CA1_py,coord_CA1_py_CAN,Dcoord_CA1_py_CAN,Icoord_CA1_py_CAN,coord_CA1_inh,coord_CA3_py,Dcoord_CA3_py,Icoord_CA3_py,coord_CA3_py_CAN,Dcoord_CA3_py_CAN,Icoord_CA3_py_CAN,coord_CA3_inh,coord_DG_py,Dcoord_DG_py,Icoord_DG_py,coord_DG_inh,coord_EC_py,Dcoord_EC_py,Icoord_EC_py,coord_EC_py_CAN,Dcoord_EC_py_CAN,Icoord_EC_py_CAN,coord_EC_inh, elec_pos
    (coord_CA1_py,Dcoord_CA1_py,Icoord_CA1_py,coord_CA1_py_CAN,Dcoord_CA1_py_CAN,Icoord_CA1_py_CAN,coord_CA1_inh,coord_CA3_py,Dcoord_CA3_py,Icoord_CA3_py,coord_CA3_py_CAN,Dcoord_CA3_py_CAN,Icoord_CA3_py_CAN,coord_CA3_inh,coord_DG_py,Dcoord_DG_py,Icoord_DG_py,coord_DG_inh,coord_EC_py,Dcoord_EC_py,Icoord_EC_py,coord_EC_py_CAN,Dcoord_EC_py_CAN,Icoord_EC_py_CAN,coord_EC_inh, elec_pos)  =topology(10000,1000,100,pCAN)
       
    global dir_CA1, dir_CA1_CAN, dir_CA3, dir_CA3_CAN, dir_DG, dir_EC, dir_EC_CAN, dir_NC
    
    dir_CA1_CAN=(coord_CA1_py_CAN-Dcoord_CA1_py_CAN)/norm(coord_CA1_py_CAN-Dcoord_CA1_py_CAN,2,1).reshape(-1,1)
    dir_CA3_CAN=(coord_CA3_py_CAN-Dcoord_CA3_py_CAN)/norm(coord_CA3_py_CAN-Dcoord_CA3_py_CAN,2,1).reshape(-1,1)
    dir_DG=(coord_DG_py-Dcoord_DG_py)/norm(coord_DG_py-Dcoord_DG_py,2,1).reshape(-1,1)
    dir_EC_CAN=(coord_EC_py_CAN-Dcoord_EC_py_CAN)/norm(coord_EC_py_CAN-Dcoord_EC_py_CAN,2,1).reshape(-1,1)
    
    if pCAN!=1:
        dir_CA1=(coord_CA1_py-Dcoord_CA1_py)/norm(coord_CA1_py-Dcoord_CA1_py,2,1).reshape(-1,1)
        dir_CA3=(coord_CA3_py-Dcoord_CA3_py)/norm(coord_CA3_py-Dcoord_CA3_py,2,1).reshape(-1,1)
        dir_EC=(coord_EC_py-Dcoord_EC_py)/norm(coord_EC_py-Dcoord_EC_py,2,1).reshape(-1,1)
  
    
    all_results=[]
    num_cores = multiprocessing.cpu_count()
       
    all_results += Parallel(n_jobs=num_cores)(delayed(process)(num_simu,g_max_e,g_max_i,p_co,p_co_CA3) for num_simu in simu_range)

    print('All the simulations have ended')
        
    t2=time.time()
    print('Total simulation time = '+str(int((t2-t1)/60))+' minutes') 
        
        
