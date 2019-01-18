#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from brian2 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
import datetime

def topology(N1,N2,N3,p_CAN):
    
    #Creation of the electrode
    elec_start_point=array([-21, 0, 50])
    elec_end_point=array([15, 0, 50])
    elec_dir=(elec_end_point-elec_start_point)/norm(elec_end_point-elec_start_point)
    elec=[]
    psi = arccos(dot(elec_dir,array([0,1,0])))
    radius = 400/150
    for t in linspace(0,1,33):
        centre=(1-t)*elec_start_point+t*elec_end_point
        for theta in arange(0,2*pi,pi/6):
            point=[centre[0]+radius*cos(theta)*cos(psi),centre[1]-radius*cos(theta)*sin(psi),centre[2]+radius*sin(theta)]
            elec.append(point)
    elec_array=array(elec[:144]+elec[252:]) #removing the inter contact points
    
    def topo_one_pop(init_segs,end_segs,N,i_soma): 
        seg=randint(0,len(init_segs))
        t=uniform(i_soma[0],i_soma[1])
        all_t=zeros(int(N))
        all_z=zeros(int(N))
        all_t[0]=seg
        z=100*random()
        all_z[0]=z
        topo=append((1-t)*init_segs[seg]  + t * end_segs[seg],z)
        topo_end=append(end_segs[seg],z)
        topo_inh=append(init_segs[seg],z)
        
        for i in range(int(N-1)):
            seg=randint(0,len(init_segs)-1)
            t=random()
            init=t*init_segs[seg]+(1-t)*init_segs[seg+1]
            all_t[i]=seg+t
            end=t*end_segs[seg]+(1-t)*end_segs[seg+1] 
            t2=uniform(i_soma[0],i_soma[1]) 
            z=100*random()
            all_z[i]=z
            coords=append((1 - t2)*init  + t2 * end,z)
            topo=vstack((topo,coords)) 
            topo_end=vstack((topo_end,append(end,z))) 
            topo_inh=vstack((topo_inh,append(0.9*init+0.1*end,z))) 
        sort_index=argsort(all_t)
        topo=topo[sort_index]
        topo_end=topo_end[sort_index]
        topo_inh=topo_inh[sort_index]
        all_z=all_z[sort_index]
        sort_index2=argsort(all_z)
        topo=topo[sort_index2]
        topo_end=topo_end[sort_index2]
        topo_inh=topo_inh[sort_index2]
        
        #Points inside the electrode are pushed away :
        for i in range(int(N)):
            x=topo[i,0]
            y=topo[i,1]
            z=topo[i,2]
            dist_elec=norm(cross((array([x,y,50])-elec_start_point),elec_dir))
            if dist_elec<radius/2:
                if z<50:
                    topo[i,2]-=radius*(1-(dist_elec/radius)**2)
                    topo_end[i,2]-=radius*(1-(dist_elec/radius)**2)
                    topo_inh[i,2]-=radius*(1-(dist_elec/radius)**2)
                else :
                    topo[i,2]+=radius*(1-(dist_elec/radius)**2) 
                    topo_end[i,2]+=radius*(1-(dist_elec/radius)**2) 
                    topo_inh[i,2]+=radius*(1-(dist_elec/radius)**2)
                #print(topo[i,2])    
        
        return topo,topo_end,topo_inh
    
    ###CA1
    init_CA1=[[0,16],[-3.5,16],[-8,15.5],[-12,14],[-15,12],[-19,9],[-21.5,4.6],[-22,-0.15],[-21,-4],[-19,-9],[-17,-12],[-13.8,-15],[-9,-17],[-6,-18]]
    end_CA1=[[0,9],[-2,8.5],[-4,7],[-5.8,5.6],[-7,4],[-7.9,2],[-8,0.25],[-8,-1],[-7.75,-2],[-7.5,-2.5],[-6,-4],[-4.5,-5.5],[-2,-7.25],[0,-8.5]]
    init_CA1=array(init_CA1)
    end_CA1=array(end_CA1)
    
    N_CA1_py=int(N1*(1-p_CAN))
    N_CA1_py_CAN=int(N1*p_CAN)
    if N_CA1_py>0:
        CA1_py,CA1_py_end,CA1_py_inh=topo_one_pop(init_CA1,end_CA1,N_CA1_py,(0.1,0.7))
    else :
        CA1_py,CA1_py_end,CA1_py_inh=0,0,0
    CA1_pyCAN,CA1_pyCAN_end,CA1_pyCAN_inh=topo_one_pop(init_CA1,end_CA1,N_CA1_py_CAN,(0.1,0.7))
    CA1_inh,CA1_inh_end,CA1_inh_inh=topo_one_pop(init_CA1,end_CA1,N2,(0,0.1))   #Ca1_inh_end is not used
    
    ###DG
    
    end_DG=[[4.5,3.6],[4.75,3],[5,2.5],[5.7,1.75],[6,1.4],[7.3,0.6],[9,0.5],[10,0.4],[10.9,0.6],[11.6,1.4],[12.5,2.25],[13,3],[12.75,3.5],[12.5,4]]
    init_DG=[[0.5,7],[-1.5,5.5],[-4,3],[-3.7,0],[-1.5,-4],[2,-6],[5.4,-7],[10,-7.2],[13.5,-6],[16,-2.8],[17,1],[18,4],[16.5,6.5],[13.5,7.5]]
    init_DG=array(init_DG)
    end_DG=array(end_DG)
    
    N_DG_py=int(N1)
    DG_py,DG_py_end,DG_py_inh=topo_one_pop(init_DG,end_DG,N_DG_py,(0.1,0.6))
    DG_inh,DG_inh_end,DG_inh_inh=topo_one_pop(init_DG,end_DG,N3,(0,0.1))
    
    ###CA3
    
    init_CA3=[[3,15.5],[5,14.75],[6.5,14],[8,12.8],[9.5,11],[10.5,7.5],[10.8,4.5],[10,2]]
    end_CA3=[[2,9],[3,8.9],[4,8.5],[4.75,8],[5.5,7.5],[6,6],[6.5,5.25],[7,4.5]]
    init_CA3=array(init_CA3)
    end_CA3=array(end_CA3)

    N_CA3_py=int(N2*(1-p_CAN))
    N_CA3_py_CAN=int(N2*p_CAN) 
    if N_CA3_py>0:
        CA3_py,CA3_py_end,CA3_py_inh=topo_one_pop(init_CA3,end_CA3,N_CA3_py,(0.1,0.6))
    else :
        CA3_py,CA3_py_end,CA3_py_inh=0,0,0
    CA3_pyCAN,CA3_pyCAN_end,CA3_pyCAN_inh=topo_one_pop(init_CA3,end_CA3,N_CA3_py_CAN,(0.1,0.6))
    CA3_inh,CA3_inh_end,CA3_inh_inh=topo_one_pop(init_CA3,end_CA3,N3,(0,0.1))
    
    ###EC
    
    init_EC=[[5,-21],[6.6,-21],[8.3,-21],[10,-21],[11,-21.8],[12,-22.8],[13,-25],[13.3,-27],[13.6,-29],[14,-32],[13.6,-35],[13.3,-37],[13,-40],[12,-42.5],[11,-44],[10,-45],[8,-45],[6,-45],[4,-45],[2,-45],[0,-45],[-2,-45],[-4,-45],[-6,-45],[-8,-45],[-10,-45]]
    end_EC=[[6,-10.5],[7.3,-10.5],[8.6,-10.5],[10,-10.5],[13.5,-11],[16.5,-13],[19,-16],[21.5,-20],[23,-25],[24,-32],[23.2,-38],[22.2,-44],[20.4,-49],[17,-52.5],[14,-54],[11,-55],[8,-55],[6,-55],[4,-55],[2,-55],[0,-55],[-2,-55],[-4,-55],[-6,-55],[-8,-55],[-10,-55]]
    init_EC=array(init_EC)
    end_EC=array(end_EC)

    N_EC_py=int(N1*(1-p_CAN))
    N_EC_py_CAN=int(N1*p_CAN) 
    if N_EC_py>0:
        EC_py,EC_py_end,EC_py_inh=topo_one_pop(init_EC,end_EC,N_EC_py,(0.1,0.6))
    else :
        EC_py,EC_py_end,EC_py_inh=0,0,0
    EC_pyCAN,EC_pyCAN_end,EC_pyCAN_inh=topo_one_pop(init_EC,end_EC,N_EC_py_CAN,(0.1,0.6))
    EC_inh,EC_inh_end,EC_inh_inh=topo_one_pop(init_EC,end_EC,N2,(0,0.1))      

    #3D figure of the network    
    figure(figsize=(10,8))
    subplot(111, projection='3d')
    if N_CA1_py>0:
        plot(CA1_py[:,0],CA1_py[:,1],CA1_py[:,2],'bo')
        plot(CA1_py_end[:,0],CA1_py_end[:,1],CA1_py_end[:,2],'bo')
    plot(CA1_pyCAN[:,0],CA1_pyCAN[:,1],CA1_pyCAN[:,2],'bo')
    plot(CA1_pyCAN_end[:,0],CA1_pyCAN_end[:,1],CA1_pyCAN_end[:,2],'bo')
    plot(CA1_inh[:,0],CA1_inh[:,1],CA1_inh[:,2],'co')
    
    if N_CA3_py>0:
        plot(CA3_py[:,0],CA3_py[:,1],CA3_py[:,2],'go')
        plot(CA3_py_end[:,0],CA3_py_end[:,1],CA3_py_end[:,2],'go')
    plot(CA3_pyCAN[:,0],CA3_pyCAN[:,1],CA3_pyCAN[:,2],'go')
    plot(CA3_pyCAN_end[:,0],CA3_pyCAN_end[:,1],CA3_pyCAN_end[:,2],'go')
    plot(CA3_inh[:,0],CA3_inh[:,1],CA3_inh[:,2],'yo')
    
    plot(DG_py[:,0],DG_py[:,1],DG_py[:,2],'ro')
    plot(DG_py_end[:,0],DG_py_end[:,1],DG_py_end[:,2],'ro')
    plot(DG_inh[:,0],DG_inh[:,1],DG_inh[:,2],'mo')

    if N_EC_py>0:    
        plot(EC_py[:,0],EC_py[:,1],EC_py[:,2],'ko')
        plot(EC_py_end[:,0],EC_py_end[:,1],EC_py_end[:,2],'ko')
    plot(EC_pyCAN[:,0],EC_pyCAN[:,1],EC_pyCAN[:,2],'ko')
    plot(EC_pyCAN_end[:,0],EC_pyCAN_end[:,1],EC_pyCAN_end[:,2],'ko')
    plot(EC_inh[:,0],EC_inh[:,1],EC_inh[:,2],'wo') 
    
    plot(elec_array[:144,0],elec_array[:144,1],elec_array[:144,2],'yo') 
    plot(elec_array[144:,0],elec_array[144:,1],elec_array[144:,2],'yo')

    uniq_filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
    savefig('3D_'+uniq_filename+'.png')

 
#Black and white 2D figure od the network
    C1=array([16.5,-45,82])
    C2=array([16.5,-45,50])
    C3=array([16.5,-45,16])
    figure(figsize=(10,8))
    if N_CA1_py>0:
        plot(CA1_py[:,0],CA1_py[:,1],'o', color='0.5')
    plot(CA1_pyCAN[:,0],CA1_pyCAN[:,1],'o', color='0.5')
    if N_CA1_py>0:
        plot(CA1_py_end[:,0],CA1_py_end[:,1],'o', color='0.8')
    plot(CA1_pyCAN_end[:,0],CA1_pyCAN_end[:,1],'o', color='0.8')
    plot(CA1_inh[:,0],CA1_inh[:,1],'o', color='0.3')
    
    if N_CA3_py>0:
        plot(CA3_py[:,0],CA3_py[:,1],'o', color='0.5')
    plot(CA3_pyCAN[:,0],CA3_pyCAN[:,1],'o', color='0.5')
    if N_CA3_py>0:
        plot(CA3_py_end[:,0],CA3_py_end[:,1],'o', color='0.8')
    plot(CA3_pyCAN_end[:,0],CA3_pyCAN_end[:,1],'o', color='0.8')
    plot(CA3_inh[:,0],CA3_inh[:,1],'o', color='0.3')
    
    plot(DG_py[:,0],DG_py[:,1],'o', color='0.5')
    plot(DG_py_end[:,0],DG_py_end[:,1],'o', color='0.8')
    plot(DG_inh[:,0],DG_inh[:,1],'o', color='0.3')
    
    if N_EC_py>0:
        plot(EC_py[:,0],EC_py[:,1],'o', color='0.5')
    plot(EC_pyCAN[:,0],EC_pyCAN[:,1],'o', color='0.5')
    if N_EC_py>0:
        plot(EC_py_end[:,0],EC_py_end[:,1],'o', color='0.8')
    plot(EC_pyCAN_end[:,0],EC_pyCAN_end[:,1],'o', color='0.8')
    plot(EC_inh[:,0],EC_inh[:,1],'o', color='0.3')  
    
    scalebar_len=1000/150
    plot(array([25,25+scalebar_len]),array([15,15]),'k',linewidth=3.5)
    
    xlim([-40,40])
    ylim([-60,20])
    uniq_filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
    savefig('BW2D_'+uniq_filename+'.png')
    
    return(CA1_py,CA1_py_end,CA1_py_inh,CA1_pyCAN,CA1_pyCAN_end,CA1_pyCAN_inh,CA1_inh,CA3_py,CA3_py_end,CA3_py_inh,CA3_pyCAN,CA3_pyCAN_end,CA3_pyCAN_inh,CA3_inh,DG_py,DG_py_end,DG_py_inh,DG_inh,EC_py,EC_py_end,EC_py_inh,EC_pyCAN,EC_pyCAN_end,EC_pyCAN_inh,EC_inh, elec) 
    
    
#temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8,temp9,temp10,temp11,temp12,temp13,temp14,temp15,temp16,temp17,temp18,temp19,temp20,temp21,temp22,temp23,temp24,temp25,elec_pos=topologie(10000,1000,100,1)
