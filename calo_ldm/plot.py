import numpy as np

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.antialiased']=False
matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def _postpro(fig, virtual=False):
    if virtual:
        fig.canvas.draw()
        r = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).astype(float)
        ret = 2.*(r-r.min())/(r.max()-r.min()) - 1.
        plt.close()
        return ret.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    else:
        return fig

class CaloPlotter:
    def __init__(self,ds="2"):
        matplotlib.use('Agg')
        matplotlib.rcParams['text.antialiased']=False
        matplotlib.rcParams['text.usetex']=False

        #self.colornorm=LogNorm(vmin=0.01, vmax=10000) # 10KeV - 10GeV
        self.rad = None #np.linspace(0, 1, 9)
        self.ang = None # np.linspace(0, 2*np.pi, 16)
        self.ncol = 9
        self.nrow = 5
        self.sz = 1.5
        # fig.tight_layout(pad=0)

    def draw(self,x, _y, ievt=0, virtual=False): # x in is NZAR
        if self.rad is None:
            self.rad = np.linspace(0, 1, x.shape[-3])
            self.ang = np.linspace(0, 2*np.pi, x.shape[-1])
            self.r, self.th = np.meshgrid(self.rad, self.ang)

        fig,axs = plt.subplots(self.nrow, self.ncol, subplot_kw={"projection":'polar'},figsize=self.sz*plt.figaspect(self.nrow/self.ncol))
        fig.suptitle(f"E={_y[ievt,0]/1000.:.2f} GeV")
        # print("x",x.shape)
        # print("th,r",self.th.shape,self.r.shape)
        min_cut = 1e-5
        max_cut = 2
        colornorm = LogNorm(vmin = min_cut, vmax=max_cut)
        for j in range(self.nrow):
            for i in range(self.ncol):
                ax = axs[j][i]
                il = i+j*self.ncol # Z channel 
                if il >= x.shape[2]: break
                
                ax.pcolormesh(self.th, self.r, (x[ievt,:,il]*(x[ievt,:,il]>min_cut)).T, shading='auto', norm=colornorm)
                ax.axis('off')
                ax.text(4*np.pi/5, 1.3, str(il))

        return _postpro(fig,virtual)

# rewrite the official calo metric
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm as LN
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# from XMLHandler import XMLHandler
import xml.etree.ElementTree as ET
DS1_pho='''
<Bins>
   <Bin pid="22" etaMin="0" etaMax="130" name="photon">
      <Layer id="0" r_edges="0,5,10,30,50,100,200,400,600" n_bin_alpha="1" /> 
      <Layer id="1" r_edges="0,2,4,6,8,10,12,15,20,30,40,50,70,90,120,150,200" n_bin_alpha="10"/> 
      <Layer id="2" r_edges="0,2,5,10,15,20,25,30,40,50,60,80,100,130,160,200,250,300,350,400" n_bin_alpha="10"/>
      <Layer id="3" r_edges="0,50,100,200,400,600" n_bin_alpha="1" /> 
      <Layer id="4" r_edges="0" n_bin_alpha="1" /> 
      <Layer id="5" r_edges="0" n_bin_alpha="1" /> 
      <Layer id="6" r_edges="0" n_bin_alpha="1" /> 
      <Layer id="7" r_edges="0" n_bin_alpha="1" /> 
      <Layer id="8" r_edges="0" n_bin_alpha="1" /> 
      <Layer id="9" r_edges="0" n_bin_alpha="1" /> 
      <Layer id="10" r_edges="0" n_bin_alpha="1" /> 
      <Layer id="11" r_edges="0" n_bin_alpha="1" /> 
      <Layer id="12" r_edges="0,100,200,400,1000,2000" n_bin_alpha="1" /> 
      <Layer id="13" r_edges="0" n_bin_alpha="1" /> 
      <Layer id="14" r_edges="0" n_bin_alpha="1" /> 
      <Layer id="15" r_edges="0" n_bin_alpha="1" /> 
      <Layer id="16" r_edges="0" n_bin_alpha="1" /> 
      <Layer id="17" r_edges="0" n_bin_alpha="1" /> 
      <Layer id="18" r_edges="0" n_bin_alpha="1" /> 
      <Layer id="19" r_edges="0" n_bin_alpha="1" /> 
      <Layer id="20" r_edges="0" n_bin_alpha="1" /> 
      <Layer id="21" r_edges="0" n_bin_alpha="1" /> 
      <Layer id="22" r_edges="0" n_bin_alpha="1" /> 
      <Layer id="23" r_edges="0" n_bin_alpha="1" /> 
   </Bin>
</Bins>
'''

DS1_pion='''
<Bins>
   <Bin pid="211" etaMin="0" etaMax="80" name="pion">
      <Layer id="0"  r_edges="0,5,10,30,50,100,200,400,600" n_bin_alpha="1" />
      <Layer id="1"  r_edges="0,1,4,7,10,15,30,50,90,150,200" n_bin_alpha="10"/>
      <Layer id="2"  r_edges="0,5,10,20,30,50,80,130,200,300,400" n_bin_alpha="10"/>
      <Layer id="3"  r_edges="0,50,100,200,400,600" n_bin_alpha="1" />
      <Layer id="4"  r_edges="0" n_bin_alpha="1" />
      <Layer id="5"  r_edges="0" n_bin_alpha="1" />
      <Layer id="6"  r_edges="0" n_bin_alpha="1" />
      <Layer id="7"  r_edges="0" n_bin_alpha="1" />
      <Layer id="8"  r_edges="0" n_bin_alpha="1" />
      <Layer id="9"  r_edges="0" n_bin_alpha="1" />
      <Layer id="10" r_edges="0" n_bin_alpha="1" />
      <Layer id="11" r_edges="0" n_bin_alpha="1" />
      <Layer id="12" r_edges="0,10,20,30,50,80,100,130,160,200,250,300,350,400,1000,2000" n_bin_alpha="10" />
      <Layer id="13" r_edges="0,10,20,30,50,80,100,130,160,200,250,300,350,400,600,1000,2000" n_bin_alpha="10" />
      <Layer id="14" r_edges="0,50,100,150,200,250,300,400,600,1000,2000" n_bin_alpha="1" />
      <Layer id="15" r_edges="0" n_bin_alpha="1" />
      <Layer id="16" r_edges="0" n_bin_alpha="1" />
      <Layer id="17" r_edges="0" n_bin_alpha="1" />
      <Layer id="18" r_edges="0" n_bin_alpha="1" />
      <Layer id="19" r_edges="0" n_bin_alpha="1" />
      <Layer id="20" r_edges="0" n_bin_alpha="1" />
      <Layer id="21" r_edges="0" n_bin_alpha="1" />
      <Layer id="22" r_edges="0" n_bin_alpha="1" />
      <Layer id="23" r_edges="0" n_bin_alpha="1" />
   </Bin> 
</Bins>
'''

DS2='''
<Bins>
   <Bin pid="11" name="electron">
      <Layer id="0" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="1" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="2" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="3" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="4" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="5" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="6" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="7" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="8" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="9" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="10" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="11" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="12" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="13" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="14" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="15" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="16" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="17" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="18" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="19" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="20" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="21" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="22" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="23" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="24" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="25" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="26" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="27" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="28" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="29" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="30" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="31" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="32" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="33" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="34" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="35" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="36" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="37" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="38" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="39" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="40" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="41" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="42" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="43" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
      <Layer id="44" r_edges="0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85" n_bin_alpha="16" />
   </Bin>
</Bins>
'''

DS3='''
<Bins>
   <Bin pid="11" name="electron">
      <Layer id="0" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="1" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="2" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="3" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="4" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="5" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="6" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="7" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="8" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="9" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="10" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="11" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="12" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="13" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="14" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="15" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="16" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="17" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="18" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="19" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="20" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="21" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="22" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="23" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="24" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="25" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="26" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="27" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="28" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="29" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="30" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="31" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="32" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="33" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="34" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="35" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="36" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="37" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="38" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="39" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="40" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="41" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="42" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="43" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
      <Layer id="44" r_edges="0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85" n_bin_alpha="50" />
   </Bin>
</Bins>
'''

class XMLHandler:

    def __init__(self, particle_name, xml_string): # filename='binning.xml'

        # tree = ET.parse(filename)
        # root = tree.getroot()
        root=ET.fromstring(xml_string)

        self.r_bins = []
        self.a_bins = []
        self.nBinAlphaPerlayer = []
        self.alphaListPerLayer = []

        self.r_edges = []
        self.r_midvalue = []
        self.r_midvalueCorrected = []
        self.relevantlayers = []
        self.layerWithBinningInAlpha = []

        self.eta_edges = []
        self.phi_edges = []
        self.eta_bins = []
        self.phi_bins = []

        self.etaRegion = 0

        found_particle = False
        for particle in root:
            if particle.attrib["name"] == particle_name:
                found_particle = True
                for layer in particle:
                    self.ReadPolarCoordinates(layer)
        if not found_particle:
            raise ValueError('Particle {} not found in {}'.format(particle_name, "string"))

        self.totalBins = 0
        self.bin_number = []

        self.eta_all_layers = []
        self.phi_all_layers = []

        self.SetEtaAndPhiFromPolar()
        self.bin_edges = [0]
        for i in range(len(self.bin_number)):
            self.bin_edges.append(self.bin_number[i] + self.bin_edges[i])

    def ReadPolarCoordinates(self, subelem):
        bins = 0
        r_list = []
        str_r = subelem.attrib.get('r_edges')
        r_list = [float(s) for s in str_r.split(',')]
        bins = len(r_list) - 1

        self.r_edges.append(r_list)
        self.r_bins.append(bins)
        layer = subelem.attrib.get('id')

        bins_in_alpha = int(subelem.attrib.get('n_bin_alpha'))
        self.a_bins.append(bins_in_alpha)
        self.r_midvalue.append(self.get_midpoint(r_list))
        if bins_in_alpha > 1:
            self.layerWithBinningInAlpha.append(int(layer))

    def fill_r_a_lists(self, layer):
        no_of_rbins = self.r_bins[layer]
        list_mid_values = self.r_midvalue[layer]
        list_a_values = self.alphaListPerLayer[layer]
        r_list = []
        a_list = []
        actual_no_alpha_bins = self.nBinAlphaPerlayer[layer][0]
        for j0 in range(0, actual_no_alpha_bins):
            for i0 in range(0, no_of_rbins):
                r_list.append(list_mid_values[i0])
                a_list.append(list_a_values[i0][j0])
        return r_list, a_list

    def get_midpoint(self, arr):
        middle_points = []
        for i in range(len(arr)-1):
            middle_value = arr[i] + float((arr[i+1] - arr[i]))/2
            middle_points.append(middle_value)
        return middle_points

    def SetEtaAndPhiFromPolar(self):
        self.minAlpha = -math.pi
        self.SetNumberOfBins()

        r_all_layers = []
        alpha_all_layers = []

        for layer in range(len(self.r_bins)):
            r_list, a_list = self.fill_r_a_lists(layer)
            r_all_layers.append(r_list)
            alpha_all_layers.append(a_list)

        for layer in range(len(self.r_bins)):
            eta = r_all_layers[layer] * np.cos(alpha_all_layers[layer])
            self.eta_all_layers.append(eta)
            phi = r_all_layers[layer] * np.sin(alpha_all_layers[layer])
            self.phi_all_layers.append(phi)

    def SetNumberOfBins(self):
        for layer in range(len(self.r_bins)):
            bin_no = 0
            alphaBinList = []
            nBinAlpha = []

            bin_no = self.r_bins[layer]*self.a_bins[layer]
            centres_alpha = self.get_midpoint(np.linspace(self.minAlpha,
                                                          math.pi, self.a_bins[layer]+1))
            for _ in range(self.r_bins[layer]):
                alphaBinList.append(centres_alpha)
                nBinAlpha.append(self.a_bins[layer])

            self.totalBins += bin_no
            self.bin_number.append(bin_no)
            if self.r_bins[layer] > 0:
                self.relevantlayers.append(layer)
                self.alphaListPerLayer.append(alphaBinList)
                self.nBinAlphaPerlayer.append(nBinAlpha)
            else:
                self.alphaListPerLayer.append([0])
                self.nBinAlphaPerlayer.append([0])

    def GetTotalNumberOfBins(self):
        return self.totalBins

    def GetBinEdges(self):
        return self.bin_edges

    def GetEtaPhiAllLayers(self):
        return self.eta_all_layers, self.phi_all_layers

    def GetRelevantLayers(self):
        return self.relevantlayers

    def GetLayersWithBinningInAlpha(self):
        return self.layerWithBinningInAlpha

    def GetEtaRegion(self):
        return self.etaRegion

class HighLevelFeatures2:
    """ Computes all high-level features based on the specific geometry stored in the binning file
    """
    def __init__(self, virtual=True, ds="2"): # filename='binning.xml'
        """ particle (str): particle to be considered
            filename (str): path/to/binning.xml of the specific detector geometry.
            particle is redundant, as it is also part of the binning file, however, it serves as a
            crosscheck to ensure the correct binning is used.
        """
        matplotlib.use('Agg')
        matplotlib.rcParams['text.antialiased']=False
        matplotlib.rcParams['text.usetex']=False
        self.virtual=virtual

        if ds=="2":
            particle="electron"
            xml_source = DS2
        elif ds=="3":
            particle="electron"
            xml_source = DS3
        elif ds=="1_photon":
            particle="photon"
            xml_source = DS1_pho
        elif ds=="1_pion":
            particle="pion"
            xml_source = DS1_pion

        xml = XMLHandler(particle, xml_source)
        self.bin_edges = xml.GetBinEdges()
        self.eta_all_layers, self.phi_all_layers = xml.GetEtaPhiAllLayers()
        # print("DEBUG grid",self.eta_all_layers) # ds1 pion 8 100 100 5 150 160 10
        self.relevantLayers = xml.GetRelevantLayers()
        self.layersBinnedInAlpha = xml.GetLayersWithBinningInAlpha()
        self.r_edges = [redge for redge in xml.r_edges if len(redge) > 1]
        self.num_alpha = [len(xml.alphaListPerLayer[idx][0]) for idx, redge in \
                          enumerate(xml.r_edges) if len(redge) > 1]
        self.E_tot = None
        self.E_layers = {}
        self.EC_etas = {}
        self.EC_phis = {}
        self.width_etas = {}
        self.width_phis = {}
        self.particle = particle

        self.num_voxel = []
        for idx, r_values in enumerate(self.r_edges):
            self.num_voxel.append((len(r_values)-1)*self.num_alpha[idx])
    
    def reset(self):
        self.E_tot = None
        self.E_layers = {}
        self.EC_etas = {}
        self.EC_phis = {}
        self.width_etas = {}
        self.width_phis = {}

    def _calculate_EC(self, eta, phi, energy):
        eta_EC = (eta * energy).sum(axis=-1)/(energy.sum(axis=-1)+1e-16)
        phi_EC = (phi * energy).sum(axis=-1)/(energy.sum(axis=-1)+1e-16)
        return eta_EC, phi_EC

    def _calculate_Widths(self, eta, phi, energy):
        eta_width = (eta * eta * energy).sum(axis=-1)/(energy.sum(axis=-1)+1e-16)
        phi_width = (phi * phi * energy).sum(axis=-1)/(energy.sum(axis=-1)+1e-16)
        return eta_width, phi_width

    def GetECandWidths(self, eta_layer, phi_layer, energy_layer):
        """ Computes center of energy in eta and phi as well as their widths """
        eta_EC, phi_EC = self._calculate_EC(eta_layer, phi_layer, energy_layer)
        eta_width, phi_width = self._calculate_Widths(eta_layer, phi_layer, energy_layer)
        # The following checks are needed to assure a positive argument to the sqrt,
        # if there is very little energy things can go wrong
        eta_width = np.sqrt((eta_width - eta_EC**2).clip(min=0.))
        phi_width = np.sqrt((phi_width - phi_EC**2).clip(min=0.))
        return eta_EC, phi_EC, eta_width, phi_width

    def CalculateFeatures(self, data):
        """ Computes all high-level features for the given data """
        self.E_tot = data.sum(axis=-1)

        for l in self.relevantLayers:
            E_layer = data[:, self.bin_edges[l]:self.bin_edges[l+1]].sum(axis=-1)
            self.E_layers[l] = E_layer

        for l in self.relevantLayers:
            if l in self.layersBinnedInAlpha:
                self.EC_etas[l], self.EC_phis[l], self.width_etas[l], \
                    self.width_phis[l] = self.GetECandWidths(
                        self.eta_all_layers[l],
                        self.phi_all_layers[l],
                        data[:, self.bin_edges[l]:self.bin_edges[l+1]])

    def _DrawSingleLayer(self, data, layer_nr, filename=None, title=None, fig=None, subplot=(1, 1, 1),
                         vmax=None, colbar='alone',virtual=False):
        """ draws the shower in layer_nr only """
        if fig is None:
            fig = plt.figure(figsize=(2, 2), dpi=200)
        num_splits = 400
        max_r = 0
        for radii in self.r_edges:
            if radii[-1] > max_r:
                max_r = radii[-1]
        radii = np.array(self.r_edges[layer_nr])
        if self.particle != 'electron':
            radii[1:] = np.log(radii[1:])
        theta, rad = np.meshgrid(2.*np.pi*np.arange(num_splits+1)/ num_splits, radii)
        pts_per_angular_bin = int(num_splits / self.num_alpha[layer_nr])
        data_reshaped = data.reshape(int(self.num_alpha[layer_nr]), -1)
        data_repeated = np.repeat(data_reshaped, (pts_per_angular_bin), axis=0)
        ax = fig.add_subplot(*subplot, polar=True)
        ax.grid(False)
        if vmax is None:
            vmax = data.max()
        pcm = ax.pcolormesh(theta, rad, data_repeated.T+1e-16, norm=LN(vmin=1e-2, vmax=vmax))
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if self.particle == 'electron':
            ax.set_rmax(max_r)
        else:
            ax.set_rmax(np.log(max_r))
        if title is not None:
            ax.set_title(title)
        #wdth = str(len(self.r_edges)*100)+'%'
        if colbar == 'alone':
            axins = inset_axes(fig.get_axes()[-1], width='100%',
                               height="15%", loc='lower center', bbox_to_anchor=(0., -0.2, 1, 1),
                               bbox_transform=fig.get_axes()[-1].transAxes,
                               borderpad=0)
            cbar = plt.colorbar(pcm, cax=axins, fraction=0.2, orientation="horizontal")
            cbar.set_label(r'Energy (MeV)', y=0.83, fontsize=12)
        elif colbar == 'both':
            axins = inset_axes(fig.get_axes()[-1], width='200%',
                               height="15%", loc='lower center',
                               bbox_to_anchor=(-0.625, -0.2, 1, 1),
                               bbox_transform=fig.get_axes()[-1].transAxes,
                               borderpad=0)
            cbar = plt.colorbar(pcm, cax=axins, fraction=0.2, orientation="horizontal")
            cbar.set_label(r'Energy (MeV)', y=0.83, fontsize=12)
        elif colbar == 'None':
            pass
        #if title is not None:
        #    plt.gcf().suptitle(title)
        # if filename is not None:
        #     plt.savefig(filename, facecolor='white')
        return _postpro(fig,self.virtual)

    def _DrawShower(self, data, filename=None, title=None, virtual=False):
        """ Draws the shower in all layers """
        if self.particle == 'electron':
            figsize = (10, 20)
        else:
            figsize = (len(self.relevantLayers)*2, 3)
        fig = plt.figure(figsize=figsize, dpi=200)
        # to smoothen the angular bins (must be multiple of self.num_alpha):
        num_splits = 400
        layer_boundaries = np.unique(self.bin_edges)
        max_r = 0
        for radii in self.r_edges:
            if radii[-1] > max_r:
                max_r = radii[-1]
        vmax = data.max()
        for idx, layer in enumerate(self.relevantLayers):
            radii = np.array(self.r_edges[idx])
            if self.particle != 'electron':
                radii[1:] = np.log(radii[1:])
            theta, rad = np.meshgrid(2.*np.pi*np.arange(num_splits+1)/ num_splits, radii)
            pts_per_angular_bin = int(num_splits / self.num_alpha[idx])
            data_reshaped = data[layer_boundaries[idx]:layer_boundaries[idx+1]].reshape(
                int(self.num_alpha[idx]), -1)
            data_repeated = np.repeat(data_reshaped, (pts_per_angular_bin), axis=0)
            if self.particle == 'electron':
                ax = plt.subplot(9, 5, idx+1, polar=True)
            else:
                ax = plt.subplot(1, len(self.r_edges), idx+1, polar=True)
            ax.grid(False)
            pcm = ax.pcolormesh(theta, rad, data_repeated.T+1e-16, norm=LN(vmin=1e-2, vmax=vmax))
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            if self.particle == 'electron':
                ax.set_rmax(max_r)
            else:
                ax.set_rmax(np.log(max_r))
            ax.set_title('Layer '+str(layer))
        if self.particle == 'electron':
            axins = inset_axes(fig.get_axes()[-3], width="500%",
                               height="15%", loc='lower center', bbox_to_anchor=(0., -0.2, 1, 1),
                               bbox_transform=fig.get_axes()[-3].transAxes,
                               borderpad=0)
        else:
            wdth = str(len(self.r_edges)*100)+'%'
            axins = inset_axes(fig.get_axes()[len(self.r_edges)//2], width=wdth,
                               height="15%", loc='lower center', bbox_to_anchor=(0., -0.2, 1, 1),
                               bbox_transform=fig.get_axes()[len(self.r_edges)//2].transAxes,
                               borderpad=0)
        cbar = plt.colorbar(pcm, cax=axins, fraction=0.2, orientation="horizontal")
        cbar.set_label(r'Energy (MeV)', y=0.83, fontsize=12)
        if title is not None:
            plt.gcf().suptitle(title)
        # if filename is not None:
        #     plt.savefig(filename, facecolor='white')
        # else:
        #     plt.show()
        return _postpro(fig,self.virtual)

    def GetEtot(self):
        """ returns total energy of the showers """
        return self.E_tot

    def GetElayers(self):
        """ returns energies of the showers deposited in each layer """
        return self.E_layers

    def GetECEtas(self):
        """ returns dictionary of centers of energy in eta for each layer """
        return self.EC_etas

    def GetECPhis(self):
        """ returns dictionary of centers of energy in phi for each layer """
        return self.EC_phis

    def GetWidthEtas(self):
        """ returns dictionary of widths of centers of energy in eta for each layer """
        return self.width_etas

    def GetWidthPhis(self):
        """ returns dictionary of widths of centers of energy in phi for each layer """
        return self.width_phis

    def DrawAverageShower(self, data, filename=None, title=None, virtual=False):
        """ plots average of provided showers """
        return self._DrawShower(data.mean(axis=0), title=title, virtual=virtual)

    def DrawSingleShower(self, data, filename=None, title=None, virtual=False):
        """ plots all provided showers after each other """
        ret = []
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        for num, shower in enumerate(data):
            if filename is not None:
                local_name, local_ext = os.path.splitext(filename)
                local_name += '_{}'.format(num) + local_ext
            else:
                local_name = None
            ret.append(self._DrawShower(shower, title=title, virtual=virtual))
        return ret

class phyPlotter():
    def __init__(self, virtual=False, ds="2"): # filename='binning.xml'
        self.virtual=virtual
        self.x_scale="log"
        # minimal readout per voxel, follow official, aka
        # ds1: from Michele (10MeV), ds2/3: 0.5 keV / 0.033 (~15MeV) scaling factor
        self.min_energy=0.5e-3/0.033 if ds in ["2","3"] else 10 # 
        self.dataset="electron"
        matplotlib.use('Agg')
        matplotlib.rcParams['text.antialiased']=False
        matplotlib.rcParams['text.usetex']=False
        self.ds=ds
        print("plotter ds",ds)

    def plot_Etot_Einc_discrete(self, hlf_class, reference_class): # only ds1
        """ plots Etot normalized to Einc histograms for each Einc in ds1 """
        if self.ds in ["2","3"]:
            return None,None
        ret2=[]
        # hardcode boundaries?
        bins = np.linspace(0.4, 1.4, 21)
        fig=plt.figure(figsize=(10, 10))
        target_energies = 2**np.linspace(8, 23, 16)
        for i in range(len(target_energies)-1):
            if i > 3 and 'photon' in self.ds:
                bins = np.linspace(0.9, 1.1, 21)
            energy = target_energies[i]
            which_showers_ref = ((reference_class.Einc.squeeze() >= target_energies[i]) & \
                                (reference_class.Einc.squeeze() < target_energies[i+1])).squeeze()
            which_showers_hlf = ((hlf_class.Einc.squeeze() >= target_energies[i]) & \
                                (hlf_class.Einc.squeeze() < target_energies[i+1])).squeeze()
            ax = plt.subplot(4, 4, i+1)
            counts_ref, _, _ = ax.hist(reference_class.GetEtot()[which_showers_ref] /\
                                    reference_class.Einc.squeeze()[which_showers_ref],
                                    bins=bins, label='reference', density=True,
                                    histtype='stepfilled', alpha=0.2, linewidth=2.)
            counts_data, _, _ = ax.hist(hlf_class.GetEtot()[which_showers_hlf] /\
                                        hlf_class.Einc.squeeze()[which_showers_hlf], bins=bins,
                                        label='generated', histtype='step', linewidth=3., alpha=1.,
                                        density=True)
            if i in [0, 1, 2]:
                energy_label = 'E = {:.0f} MeV'.format(energy)
            elif i in np.arange(3, 12):
                energy_label = 'E = {:.1f} GeV'.format(energy/1e3)
            else:
                energy_label = 'E = {:.1f} TeV'.format(energy/1e6)
            ax.text(0.95, 0.95, energy_label, ha='right', va='top',
                    transform=ax.transAxes)
            ax.set_xlabel(r'$E_{\mathrm{tot}} / E_{\mathrm{inc}}$')
            ax.xaxis.set_label_coords(1., -0.15)
            ax.set_ylabel('counts')
            ax.yaxis.set_ticklabels([])
            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            seps = _separation_power(counts_ref, counts_data, bins)
            ret2.append(seps)
            # print("Separation power of Etot / Einc at E = {} histogram: {}".format(energy, seps))
            # with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(self.dataset)),
            #         'a') as f:
            #     f.write('Etot / Einc at E = {}: \n'.format(energy))
            #     f.write(str(seps))
            #     f.write('\n\n')
            h, l = ax.get_legend_handles_labels()
        ax = plt.subplot(4, 4, 16)
        ax.legend(h, l, loc='center', fontsize=20)
        ax.axis('off')
        # filename = os.path.join(arg.output_dir, 'Etot_Einc_dataset_{}_E_i.png'.format(self.dataset))
        # plt.savefig(filename, dpi=300)
        # plt.close()
        return _postpro(fig,self.virtual),rets2

    def plot_Etot_Einc(self, hlf_class, reference_class):
        """ plots Etot normalized to Einc histogram """

        bins = np.linspace(0, 3.0 , 101)
        fig=plt.figure(figsize=(6, 6))
        counts_ref, _, _ = plt.hist(reference_class.GetEtot() / reference_class.Einc.squeeze(),
                                    bins=bins, label='reference', density=True,
                                    histtype='stepfilled', alpha=0.2, linewidth=2.)
        counts_data, _, _ = plt.hist(hlf_class.GetEtot() / hlf_class.Einc.squeeze(), bins=bins,
                                    label='generated', histtype='step', linewidth=3., alpha=1.,
                                    density=True)
        plt.xlim(0, 3.0)
        plt.xlabel(r'$E_{\mathrm{tot}} / E_{\mathrm{inc}}$')
        plt.legend(fontsize=20)
        plt.tight_layout()
        seps = self._separation_power(counts_ref, counts_data, bins)
        return _postpro(fig,self.virtual),seps
        # if arg.mode in ['all', 'hist-p', 'hist']:
        #     filename = os.path.join(arg.output_dir, 'Etot_Einc_dataset_{}.png'.format(self.dataset))
        #     plt.savefig(filename, dpi=300)
        # if arg.mode in ['all', 'hist-chi', 'hist']:
        #     seps = _separation_power(counts_ref, counts_data, bins)
        #     print("Separation power of Etot / Einc histogram: {}".format(seps))
        #     with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(self.dataset)),
        #             'a') as f:
        #         f.write('Etot / Einc: \n')
        #         f.write(str(seps))
        #         f.write('\n\n')
        # plt.close()

    def plot_E_layers(self, hlf_class, reference_class):
        """ plots energy deposited in each layer """
        ret=[]
        ret2=[]
        for key in hlf_class.GetElayers().keys():
            fig=plt.figure(figsize=(6, 6))
            if self.x_scale == 'log':
                bins = np.logspace(np.log10(self.min_energy),
                                np.log10(reference_class.GetElayers()[key].max()),
                                40)
            else:
                bins = 40
            counts_ref, bins, _ = plt.hist(reference_class.GetElayers()[key], bins=bins,
                                        label='reference', density=True, histtype='stepfilled',
                                        alpha=0.2, linewidth=2.)
            counts_data, _, _ = plt.hist(hlf_class.GetElayers()[key], label='generated', bins=bins,
                                        histtype='step', linewidth=3., alpha=1., density=True)
            plt.title("Energy deposited in layer {}".format(key))
            plt.xlabel(r'$E$ [MeV]')
            plt.yscale('log')
            plt.xscale('log')
            plt.legend(fontsize=20)
            plt.tight_layout()
            ret.append(_postpro(fig,self.virtual))
            ret2.append(self._separation_power(counts_ref, counts_data, bins))
            # if arg.mode in ['all', 'hist-p', 'hist']:
            #     filename = os.path.join(arg.output_dir, 'E_layer_{}_dataset_{}.png'.format(
            #         key,
            #         self.dataset))
            #     plt.savefig(filename, dpi=300)
            # if arg.mode in ['all', 'hist-chi', 'hist']:
            #     seps = _separation_power(counts_ref, counts_data, bins)
            #     print("Separation power of E layer {} histogram: {}".format(key, seps))
            #     with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(self.dataset)),
            #             'a') as f:
            #         f.write('E layer {}: \n'.format(key))
            #         f.write(str(seps))
            #         f.write('\n\n')
            # plt.close()
        return ret,ret2

    def plot_Enorm_layers(self, hlf_class, reference_class):
        """ plots energy deposited in each layer """
        ret=[]
        ret2=[]
        for key in hlf_class.GetElayers().keys():
            fig=plt.figure(figsize=(6, 6))
            if self.x_scale == 'log':
                bins = np.logspace(-9,1,
                                40)
            else:
                bins = 40
            _Elayer_ref=reference_class.GetElayers()[key]
            Elayer_ref=(_Elayer_ref/reference_class.GetEtot()).clip(min=1e-9)
            _Elayer=hlf_class.GetElayers()[key]
            Elayer=(_Elayer/hlf_class.GetEtot()).clip(min=1e-9)

            counts_ref, bins, _ = plt.hist(Elayer_ref, bins=bins,
                                        label='reference', density=True, histtype='stepfilled',
                                        alpha=0.2, linewidth=2.)
            counts_data, _, _ = plt.hist(Elayer, label='generated', bins=bins,
                                        histtype='step', linewidth=3., alpha=1., density=True)
            plt.title("Energy deposited in layer {}".format(key))
            plt.xlabel(r'$E_{normed}$')
            plt.yscale('log')
            plt.xscale('log')
            plt.legend(fontsize=20)
            plt.tight_layout()
            ret.append(_postpro(fig,self.virtual))
            ret2.append(self._separation_power(counts_ref, counts_data, bins))
        return ret,ret2

    def plot_ECEtas(self, hlf_class, reference_class):
        """ plots center of energy in eta """
        ret=[]
        ret2=[]
        for key in hlf_class.GetECEtas().keys():
            if self.dataset in ['2', '3']:
                lim = (-30., 30.)
            elif key in [12, 13]:
                lim = (-500., 500.)
            else:
                lim = (-100., 100.)
            fig=plt.figure(figsize=(6, 6))
            bins = np.linspace(*lim, 101)
            counts_ref, _, _ = plt.hist(reference_class.GetECEtas()[key], bins=bins,
                                        label='reference', density=True, histtype='stepfilled',
                                        alpha=0.2, linewidth=2.)
            counts_data, _, _ = plt.hist(hlf_class.GetECEtas()[key], label='generated', bins=bins,
                                        histtype='step', linewidth=3., alpha=1., density=True)
            plt.title(r"Center of Energy in $\Delta\eta$ in layer {}".format(key))
            plt.xlabel(r'[mm]')
            plt.xlim(*lim)
            plt.yscale('log')
            plt.legend(fontsize=20)
            plt.tight_layout()
            ret.append(_postpro(fig,self.virtual))
            ret2.append(self._separation_power(counts_ref, counts_data, bins))
            # if arg.mode in ['all', 'hist-p', 'hist']:
            #     filename = os.path.join(arg.output_dir,
            #                             'ECEta_layer_{}_dataset_{}.png'.format(key,
            #                                                                 self.dataset))
            #     plt.savefig(filename, dpi=300)
            # if arg.mode in ['all', 'hist-chi', 'hist']:
            #     seps = _separation_power(counts_ref, counts_data, bins)
            #     print("Separation power of EC Eta layer {} histogram: {}".format(key, seps))
            #     with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(self.dataset)),
            #             'a') as f:
            #         f.write('EC Eta layer {}: \n'.format(key))
            #         f.write(str(seps))
            #         f.write('\n\n')
            # plt.close()
        return ret,ret2

    def plot_ECPhis(self, hlf_class, reference_class):
        """ plots center of energy in phi """
        ret=[]
        ret2=[]
        for key in hlf_class.GetECPhis().keys():
            if self.dataset in ['2', '3']:
                lim = (-30., 30.)
            elif key in [12, 13]:
                lim = (-500., 500.)
            else:
                lim = (-100., 100.)
            fig=plt.figure(figsize=(6, 6))
            bins = np.linspace(*lim, 101)
            counts_ref, _, _ = plt.hist(reference_class.GetECPhis()[key], bins=bins,
                                        label='reference', density=True, histtype='stepfilled',
                                        alpha=0.2, linewidth=2.)
            counts_data, _, _ = plt.hist(hlf_class.GetECPhis()[key], label='generated', bins=bins,
                                        histtype='step', linewidth=3., alpha=1., density=True)
            plt.title(r"Center of Energy in $\Delta\phi$ in layer {}".format(key))
            plt.xlabel(r'[mm]')
            plt.xlim(*lim)
            plt.yscale('log')
            plt.legend(fontsize=20)
            plt.tight_layout()
            ret.append(_postpro(fig,self.virtual))
            ret2.append(self._separation_power(counts_ref, counts_data, bins))
            # if arg.mode in ['all', 'hist-p', 'hist']:
            #     filename = os.path.join(arg.output_dir,
            #                             'ECPhi_layer_{}_dataset_{}.png'.format(key,
            #                                                                 self.dataset))
            #     plt.savefig(filename, dpi=300)
            # if arg.mode in ['all', 'hist-chi', 'hist']:
            #     seps = _separation_power(counts_ref, counts_data, bins)
            #     print("Separation power of EC Phi layer {} histogram: {}".format(key, seps))
            #     with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(self.dataset)),
            #             'a') as f:
            #         f.write('EC Phi layer {}: \n'.format(key))
            #         f.write(str(seps))
            #         f.write('\n\n')
            # plt.close()
        return ret,ret2

    def plot_ECWidthEtas(self, hlf_class, reference_class):
        """ plots width of center of energy in eta """
        ret=[]
        ret2=[]
        for key in hlf_class.GetWidthEtas().keys():
            if self.dataset in ['2', '3']:
                lim = (0., 30.)
            elif key in [12, 13]:
                lim = (0., 400.)
            else:
                lim = (0., 100.)
            fig=plt.figure(figsize=(6, 6))
            bins = np.linspace(*lim, 101)
            counts_ref, _, _ = plt.hist(reference_class.GetWidthEtas()[key], bins=bins,
                                        label='reference', density=True, histtype='stepfilled',
                                        alpha=0.2, linewidth=2.)
            counts_data, _, _ = plt.hist(hlf_class.GetWidthEtas()[key], label='generated', bins=bins,
                                        histtype='step', linewidth=3., alpha=1., density=True)
            plt.title(r"Width of Center of Energy in $\Delta\eta$ in layer {}".format(key))
            plt.xlabel(r'[mm]')
            plt.xlim(*lim)
            plt.yscale('log')
            plt.legend(fontsize=20)
            plt.tight_layout()
            ret.append(_postpro(fig,self.virtual))
            ret2.append(self._separation_power(counts_ref, counts_data, bins))
            # if arg.mode in ['all', 'hist-p', 'hist']:
            #     filename = os.path.join(arg.output_dir,
            #                             'WidthEta_layer_{}_dataset_{}.png'.format(key,
            #                                                                     self.dataset))
            #     plt.savefig(filename, dpi=300)
            # if arg.mode in ['all', 'hist-chi', 'hist']:
            #     seps = self._separation_power(counts_ref, counts_data, bins)
            #     print("Separation power of Width Eta layer {} histogram: {}".format(key, seps))
            #     with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(self.dataset)),
            #             'a') as f:
            #         f.write('Width Eta layer {}: \n'.format(key))
            #         f.write(str(seps))
            #         f.write('\n\n')
            # plt.close()
        return ret,ret2

    def plot_ECWidthPhis(self, hlf_class, reference_class):
        """ plots width of center of energy in phi """
        ret=[]
        ret2=[]
        for key in hlf_class.GetWidthPhis().keys():
            if self.dataset in ['2', '3']:
                lim = (0., 30.)
            elif key in [12, 13]:
                lim = (0., 400.)
            else:
                lim = (0., 100.)
            fig=plt.figure(figsize=(6, 6))
            bins = np.linspace(*lim, 101)
            counts_ref, _, _ = plt.hist(reference_class.GetWidthPhis()[key], bins=bins,
                                        label='reference', density=True, histtype='stepfilled',
                                        alpha=0.2, linewidth=2.)
            counts_data, _, _ = plt.hist(hlf_class.GetWidthPhis()[key], label='generated', bins=bins,
                                        histtype='step', linewidth=3., alpha=1., density=True)
            plt.title(r"Width of Center of Energy in $\Delta\phi$ in layer {}".format(key))
            plt.xlabel(r'[mm]')
            plt.yscale('log')
            plt.xlim(*lim)
            plt.legend(fontsize=20)
            plt.tight_layout()
            ret.append(_postpro(fig,self.virtual))
            ret2.append(self._separation_power(counts_ref, counts_data, bins))
            # if arg.mode in ['all', 'hist-p', 'hist']:
            #     filename = os.path.join(arg.output_dir,
            #                             'WidthPhi_layer_{}_dataset_{}.png'.format(key,
            #                                                                     self.dataset))
            #     plt.savefig(filename, dpi=300)
            # if arg.mode in ['all', 'hist-chi', 'hist']:
            #     seps = _separation_power(counts_ref, counts_data, bins)
            #     print("Separation power of Width Phi layer {} histogram: {}".format(key, seps))
            #     with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(self.dataset)),
            #             'a') as f:
            #         f.write('Width Phi layer {}: \n'.format(key))
            #         f.write(str(seps))
            #         f.write('\n\n')
            # plt.close()
        return ret,ret2

    def plot_cell_dist(self, shower_arr, ref_shower_arr):
        """ plots voxel energies across all layers """
        fig=plt.figure(figsize=(6, 6))
        if self.x_scale == 'log':
            bins = np.logspace(np.log10(self.min_energy),
                            np.log10(ref_shower_arr.max()),
                            50)
        else:
            bins = 50

        counts_ref, bins, _ = plt.hist(ref_shower_arr.flatten(), bins=bins,
                                    label='reference', density=True, histtype='stepfilled',
                                    alpha=0.2, linewidth=2.)
        counts_data, _, _ = plt.hist(shower_arr.flatten(), label='generated', bins=bins,
                                    histtype='step', linewidth=3., alpha=1., density=True)
        plt.title(r"Voxel energy distribution")
        plt.xlabel(r'$E$ [MeV]')
        plt.yscale('log')
        if self.x_scale == 'log':
            plt.xscale('log')
        #plt.xlim(*lim)
        plt.legend(fontsize=20)
        plt.tight_layout()
        seps = self._separation_power(counts_ref, counts_data, bins)
        return _postpro(fig,self.virtual),seps
        # if arg.mode in ['all', 'hist-p', 'hist']:
        #     filename = os.path.join(arg.output_dir,
        #                             'voxel_energy_dataset_{}.png'.format(self.dataset))
        #     plt.savefig(filename, dpi=300)
        # if arg.mode in ['all', 'hist-chi', 'hist']:
        #     seps = _separation_power(counts_ref, counts_data, bins)
        #     print("Separation power of voxel distribution histogram: {}".format(seps))
        #     with open(os.path.join(arg.output_dir,
        #                         'histogram_chi2_{}.txt'.format(self.dataset)), 'a') as f:
        #         f.write('Voxel distribution: \n')
        #         f.write(str(seps))
        #         f.write('\n\n')
        # plt.close()
    
    def plot_cellnorm_dist(self, _shower_arr, _ref_shower_arr): # this is not official, for debug only
        """ plots voxel energies normalized across all layers """
        shower_arr=_shower_arr.reshape(_shower_arr.shape[0],-1)
        shower_arr=shower_arr/shower_arr.sum(axis=(-1,),keepdims=True).clip(min=1e-15)
        ref_shower_arr=_ref_shower_arr.reshape(_ref_shower_arr.shape[0],-1)
        ref_shower_arr=ref_shower_arr/ref_shower_arr.sum(axis=(-1,),keepdims=True).clip(min=1e-15)
        fig=plt.figure(figsize=(6, 6))
        if self.x_scale == 'log':
            bins = np.logspace(-15,
                            1,
                            50)
        else:
            bins = 50

        counts_ref, bins, _ = plt.hist(ref_shower_arr.flatten(), bins=bins,
                                    label='reference', density=False, histtype='stepfilled',
                                    alpha=0.2, linewidth=2.)
        counts_data, _, _ = plt.hist(shower_arr.flatten(), label='generated', bins=bins,
                                    histtype='step', linewidth=3., alpha=1., density=False)
        plt.title(r"Voxel energy (normed) distribution [non-official metric]")
        plt.xlabel(r'$E_{norm}$')
        plt.yscale('log')
        if self.x_scale == 'log':
            plt.xscale('log')
        #plt.xlim(*lim)
        plt.legend(fontsize=20)
        plt.tight_layout()
        seps = self._separation_power(counts_ref, counts_data, bins)
        return _postpro(fig,self.virtual),seps

    def plot_cellnormlin_dist(self, _shower_arr, _ref_shower_arr): # this is not official, for debug only
        """ plots voxel energies normalized across all layers """
        shower_arr=_shower_arr.reshape(_shower_arr.shape[0],-1)
        shower_arr=shower_arr/shower_arr.sum(axis=(-1,),keepdims=True)
        ref_shower_arr=_ref_shower_arr.reshape(_ref_shower_arr.shape[0],-1)
        ref_shower_arr=ref_shower_arr/ref_shower_arr.sum(axis=(-1,),keepdims=True)
        fig=plt.figure(figsize=(6, 6))
        bins = np.linspace(-0.05,1.05,50)

        counts_ref, bins, _ = plt.hist(ref_shower_arr.flatten(), bins=bins,
                                    label='reference', density=False, histtype='stepfilled',
                                    alpha=0.2, linewidth=2.)
        counts_data, _, _ = plt.hist(shower_arr.flatten(), label='generated', bins=bins,
                                    histtype='step', linewidth=3., alpha=1., density=False)
        plt.title(r"Voxel energy (normed) distribution [non-official metric]")
        plt.xlabel(r'$E_{norm}$')
        plt.yscale('log')
        plt.legend(fontsize=20)
        plt.tight_layout()
        seps = self._separation_power(counts_ref, counts_data, bins)
        return _postpro(fig,self.virtual),seps
    
    def _separation_power(self, hist1, hist2, bins):
        """ computes the separation power aka triangular discrimination (cf eq. 15 of 2009.03796)
            Note: the definition requires Sum (hist_i) = 1, so if hist1 and hist2 come from
            plt.hist(..., density=True), we need to multiply hist_i by the bin widhts
        """
        hist1, hist2 = hist1*np.diff(bins), hist2*np.diff(bins)
        ret = (hist1 - hist2)**2
        ret /= hist1 + hist2 + 1e-16
        return 0.5 * ret.sum()
