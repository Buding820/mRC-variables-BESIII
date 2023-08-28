#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import pickle
import scipy
import vector
import math 
import matplotlib.pyplot as plt 
import matplotlib
import shapely as sp 
# from shapely.plotting import plot_polygon
from shapely.geometry import Point, LineString
from time import sleep, time


class hepmc():
    def __init__(self) -> None:
        self.file = None
        self.base = {}
        self.pdgID = {
            "-15":  "tau+",
            "15":   "tau-",
            "-16":  "vtt",
            "16":   "vt",
            "-13":  "mu+",
            "13":   "mu-",
            "-14":  "vmt",
            "14":   "vm",
            "100443":   "Psi2S"
        }

    def read_file_to_pkl(self, fname):
        self.file = fname
        event_id = 0
        ent = {}

        with open(self.file, 'r') as f1:
            for ii, line in enumerate(f1):
                # print(len(line))
                if len(line) not in [10, 45]:
                    print(line, len(line))
                else:
                    if len(line) == 10:
                        # print(len(ent))
                        event_id = int(line.strip())
                        print("Storing Event ID -> {}".format(event_id))
                        # sleep(1)
                        ent[event_id] = {}
                    else:
                        line = line.strip().split()
                        ent[event_id][self.pdgID[line[0]]] = np.array(
                            list(map(float, line[-4:])))

        with open("muon.pkl", 'wb') as fp:
            pickle.dump(ent, fp)

    def read_pkl_events(self, fname):
        mmin = []
        mmax = []
        with open(fname, 'rb') as fp:
            self.events = pickle.load(fp)
        # print(self.events[0])
        # print(type(self.events))
        # print(len(self.events))
        itern = 1
        iterf = 0 
        ptsav = os.path.abspath("./image/{}/".format(iterf))
        if not os.path.exists(ptsav):
            os.makedirs(os.path.abspath("./image/{}/".format(iterf)))

        tstart = time()
        for ii, env in self.events.items():
            # print(ii, env)
            if itern % 100 == 0:
                iterf = itern//100
                ptsav = os.path.abspath("./image/{}/".format(iterf))
                print("{}th 100 events cost {:.2f} sec".format(iterf, time() - tstart))
                tstart = time()

                if not os.path.exists(ptsav):
                    os.makedirs(os.path.abspath("./image/{}/".format(iterf)))
            entry = analysis()
            entry.env = env
            entry.obs['ID'] = ii 
            entry.obs['path'] = ptsav
            entry.set()
            if entry.obs['rec']:
                mmin.append(entry.obs['mRC_min'])
                mmax.append(entry.obs['mRC_max'])
            del entry
                
            itern += 1
            # if itern > 200:
                # break

        df = pd.DataFrame({
            "mRC_min":  mmin,
            "mRC_max":  mmax
        })
        df.to_csv("mRC.csv", index=False)

        #     break

        # entry = analysis()
        # entry.env = self.events[9]
        # entry.obs['ID'] = 9
        # entry.set()
        # if entry.obs['rec']:
        #     mmin.append(entry.obs['mRC_min'])
        # del entry
        
        # self.plot(mmin)

    def plot_csv(self, dff):
        df = pd.read_csv(dff)
        self.plot(df)

    def plot(self, hdata):
        dmax = max(hdata['mRC_max'])
        from matplotlib.colors import LinearSegmentedColormap
        cmap_name = "self"
        collist = ["#02004f", "#000b57", "#00165e", "#002166", "#012c6e", "#003875", '#00437d', "#004e85", "#00598c", "#026494", "#006f9c", "#007aa3", "#0085ab", "#0090b3", "#009cba", "#27a699", "#4bb178", "#70bc56", "#97c637", "#d2c70d", "#e8bd08", "#ffb300", "#ffbf2a", "#fecc55", "#ffd980", "#ffe6aa", "#ffe6aa"]
        cmap = LinearSegmentedColormap.from_list(cmap_name, collist, N=256)
        # pwd = os.path.abspath(os.path.dirname(__file__))

        fig = plt.figure(figsize=(10, 9))
        axy = fig.add_axes([0.1026, 0.21, 0.09, 0.73 ])
        axx = fig.add_axes([0.198, 0.104, 0.657, 0.1 ])
        ax  = fig.add_axes([0.198, 0.21, 0.657, 0.73])
        axc = fig.add_axes([0.86, 0.235, 0.02, 0.7])

        cc = ax.hist2d(hdata['mRC_min'], hdata['mRC_max'], 180, range=[[0, dmax], [0, dmax]], cmap=cmap, cmin=1, norm=matplotlib.colors.LogNorm(vmin=1), zorder=10)
        axx.hist(hdata['mRC_min'], 180, range=(0, dmax), color='#2e66ff', histtype="stepfilled", alpha=0.5, orientation='vertical', zorder=10)
        axx.hist(hdata['mRC_min'], 180, range=(0, dmax), color='#231aa5', histtype="step", alpha=1, orientation='vertical', zorder=10)
        axy.hist(hdata['mRC_max'], 180, range=(0, dmax), color='#2e66ff', histtype="stepfilled", alpha=0.5, orientation='horizontal', zorder=10)
        axy.hist(hdata['mRC_max'], 180, range=(0, dmax), color='#231aa5', histtype="step", alpha=1, orientation='horizontal', zorder=10)

        # ax.hist(hdata['mRC_min'], 10, range=(0., 2.), color='#2e66ff', histtype="stepfilled", alpha=0.5, orientation='vertical', zorder=10)
        # ax.hist(hdata['mRC_min'], 10, range=(0., 2.), color='#231aa5', histtype="step", alpha=1, orientation='vertical', zorder=10)

        # ax.hist(hdata['mRC_max'], 100, range=(0., 2.), color='#2e66ff', histtype="stepfilled", alpha=0.5, orientation='vertical', zorder=10)
        # ax.hist(hdata['mRC_max'], 100, range=(0., 2.), color='#231aa5', histtype="step", alpha=1, orientation='vertical', zorder=10)

        axy.grid(axis="both", which='major', alpha=0.8, zorder=1)
        ax.grid(axis="both", which='major', alpha=0.8, zorder=1)
        axx.grid(axis="both", which='major', alpha=0.8, zorder=1)
        axy.grid(axis="both", which='minor', alpha=0.2, zorder=1)
        ax.grid(axis="both", which='minor', alpha=0.2, zorder=1)
        axx.grid(axis="both", which='minor', alpha=0.2, zorder=1)
        fig.colorbar(cc[3], cax=axc)

        axx.set_xlim(0, dmax)
        axy.set_ylim(0, dmax)
        ax.set_xlim(0, dmax)
        ax.set_ylim(0, dmax)

        from matplotlib.ticker import FixedLocator, AutoMinorLocator, AutoLocator, MaxNLocator


        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(
            which='both',
            direction="in",
            left=True,
            right=True,
            bottom=True,
            top=True
        )
        ax.tick_params(which="major", length=10, width=1.2)
        ax.tick_params(which="minor", length=4, width=1.2)
        ax.set_xticklabels([])
        ax.set_yticklabels([])



        axc.tick_params(
            which='both',
            direction="in",
            labelsize=12,
            left=False,
            right=True,
            bottom=False,
            top=False
        )
        axc.tick_params(which="major", length=7, width=2.0, color='w')
        axc.tick_params(which="minor", length=4, width=1.2, color='w')


        axx.xaxis.set_minor_locator(AutoMinorLocator())
        axx.yaxis.set_major_locator(MaxNLocator(2))
        axx.yaxis.set_minor_locator(AutoMinorLocator())
        axx.yaxis.tick_right()
        # axx.yaxis.set_major_formatter(CustomTicker())

        axx.tick_params(
            which='both',
            direction="in",
            labelsize=12,
            left=True,
            right=True,
            bottom=True,
            top=True
        )
        axx.tick_params(which="major", length=10, width=1.2)
        axx.tick_params(which="minor", length=4, width=1.2)
        # axx.set_xticklabels([])
        # axx.set_yticklabels([])


        axy.yaxis.set_minor_locator(AutoMinorLocator())
        axy.xaxis.set_major_locator(MaxNLocator(2))
        axy.xaxis.set_minor_locator(AutoMinorLocator())
        axy.xaxis.tick_top()
        axy.tick_params(
            which='both',
            direction="in",
            labelsize=12,
            left=True,
            right=True,
            bottom=True,
            top=True
        )
        axy.tick_params(which="major", length=6, width=1.2)
        axy.tick_params(which="minor", length=3.6, width=1.2)

        axy.set_ylabel(r"$m_{\rm RC}^{\rm max}~[{\rm GeV}]$", fontsize=30, loc='top')
        axx.set_xlabel(r"$m_{\rm RC}^{\rm min}~[{\rm GeV}]$", fontsize=30, loc='right')
        axc.set_ylabel(r"Events", fontsize=30, loc='top')

        # plt.show()
        # plt.savefig("./mRC.pdf")
        plt.savefig("./mRC.png", dpi=300)


            


class analysis():
    def __init__(self) -> None:
        self.env = None
        self.obs = {
            "sqrt_s":  0.,
            "rec":  False
        }
        self.ax = None

    def get_4d_vector(self, kid):
        return vector.obj(
            px=self.env[kid][0],
            py=self.env[kid][1],
            pz=self.env[kid][2],
            energy=self.env[kid][3]
        )

    def set(self):
        def solveXY(p1, p2, pMiss):
            x           = 0.5 / pMiss * ( p1**2 - p2**2 + pMiss ** 2)
            na = math.sqrt(2.0 * ((p1 * p2)**2 + (p1  * pMiss)**2 + (p2 * pMiss)**2 ))
            nb = math.sqrt((p1 ** 4 + p2 ** 4 + pMiss ** 4))
            numinator   = (na - nb) * (na + nb)
            # if numinator <= 0.:
            #     print(p1, p2, pMiss, na, nb, numinator)
            y           = 0.5 / pMiss * math.sqrt(numinator)
            return vector.obj(x=x, y=y)
 

        
        try:
            self.obs['sqrt_s'] = self.env['Psi2S'][-1]
            #set the event information 
            self.pPP    = self.get_4d_vector("Psi2S")
            self.pPa    = self.get_4d_vector("tau+")
            self.pVa    = self.get_4d_vector("mu+")
            self.pIa1   = self.get_4d_vector("vm")
            self.pIa2   = self.get_4d_vector("vtt")
            self.pPb    = self.get_4d_vector("tau-")
            self.pVb    = self.get_4d_vector("mu-")
            self.pIb1   = self.get_4d_vector("vmt")
            self.pIb2   = self.get_4d_vector("vt")
            self.pInv   = self.pIa1 + self.pIa2 + self.pIb1 + self.pIb2

            self.pIa    = self.pIa1 + self.pIa2 
            self.pIb    = self.pIb1 + self.pIb2 

            # Define the vectors in CM 
            vCMbeta = - self.pPP.to_beta3()
            self.pInv_PP    = self.pInv.boost_beta3(vCMbeta)
            self.pVa_PP     = self.pVa.boost_beta3(vCMbeta)
            self.pVb_PP     = self.pVb.boost_beta3(vCMbeta)

            self.pIa_PP     = self.pIa.boost_beta3(vCMbeta)
            self.pIb_PP     = self.pIb.boost_beta3(vCMbeta)

            self.pPa_PP     = self.pPa.boost_beta3(vCMbeta)
            self.pPb_PP     = self.pPb.boost_beta3(vCMbeta)

            # self.pPP_PP     = self.pInv_PP + self.pVa_PP + self.pVb_PP
            self.pPP_PP     = self.pPP.boost_beta3(vCMbeta)

            # print("Lab Frame:", self.pPP)
            # print("PP Frame: ", self.pPP_PP)
            self.pMiss      = self.pPP_PP - self.pVa_PP - self.pVb_PP

            ss      = self.pPP_PP.E / 2.0 
            pVa     = self.pVa_PP.p 
            pVb     = self.pVb_PP.p 
            pIamax  = ss - self.pVa_PP.p
            pIbmax  = ss - self.pVb_PP.p 
            # pMiss   = self.pInv_PP.p 
            pMiss   = self.pMiss.p 
            pIa     = self.pIa_PP.E 
            pIb     = self.pIb_PP.E 
            # print("pPa, \t pPb   \t ->", self.pPa_PP.p, self.pPb_PP.p)
            # print(self.pPa_PP, self.pPb_PP)
            # print("pIa, \t pIb   \t ->", pIa, pIb)
            # print("pIa_max, pIb_max ->", pIamax, pIbmax, self.pMiss.p )
            # print("pVa, pVb, pMiss ->", pVa, pVb, pMiss, self.pMiss.p  )
            pos = {
                "M" : vector.obj(x=0., y=0.),
                "N" : vector.obj(x=pMiss, y=0.),
                "P" : solveXY(pVa, pVb, pMiss),
                "B" : solveXY(pIamax, pIbmax, pMiss)
            }
            pos.update({
                    "A": vector.obj(x=pos["B"].x, y=-pos["B"].y),
                    "O": vector.obj(x=pos["B"].x, y=0.)
            })
            self.obs['pos'] = pos 
            self.obs['pVa'] = pVa
            self.obs['pVb'] = pVb 
            self.obs['pIamax'] = pIamax
            self.obs['pIbmax'] = pIbmax
            self.obs['ss'] = ss 

            self.draw_geo(pos)
            self.calculate_mRC()
            # elif self.obs['P_in_D']:

 
            self.obs['rec'] = True
            print("Successed", self.obs['ID'])
        except:
            print(self.obs['ID'], "\tcan not be reconstructed!")
            # pass             

    def calculate_mRC(self):
        self.obs["mRC_min"] = math.sqrt(self.obs['ss']**2 - (self.obs['pos']["P"] - self.obs['pos']["A"]).rho ** 2 )
        self.obs['mRC_max'] = math.sqrt(self.obs['ss']**2 - (self.obs['pos']['P'] - self.obs['pos']['Z']).rho ** 2 )

    
    def draw_geo(self, pos):
        def draw_line(A, B, kwags):
            # print(kwags)
            self.ax.plot(
                [A.x, B.x],
                [A.y, B.y],
                '-', **kwags
            )
            # a = LineString(A, B)
            # print(a)
            # print(A, B)

        def draw_arrow(A, B, kwags):
            self.ax.scatter([A.x, B.x], [A.y, B.y], marker="o", s=10, c='#123456', zorder=80)
            self.ax.arrow(
                A.x, A.y,
                (B - A).x, (B - A).y, 
                length_includes_head = True,
                width=0.0001, head_width=0.01,
                head_length=0.02, 
                **kwags
            )

        def draw_geoms():
            M = pos['M']
            N = pos['N']
            B = pos['B']

            rIa = (M - B).rho 
            rIb = (N - B).rho   

            pointM = Point((M.x, M.y)).buffer(rIa)
            pointN = Point((N.x, N.y)).buffer(rIb)

            # print(B-M, (B-M).phi, rIa * math.cos((B-M).phi))            
            Phi_BMN = np.linspace((B-M).phi, (N-M).phi, 1000)
            Cone_BMN = [[M.x + self.obs['ss']* np.cos(phi), M.y + self.obs['ss']*np.sin(phi)] for phi in Phi_BMN]
            Cone_BMN = sp.geometry.Polygon([[M.x, M.y]] + Cone_BMN + [[M.x, M.y]])

            xm, ym = Cone_BMN.exterior.xy 
            # self.ax.fill(xm, ym, c="#C69BF9", alpha=0.3, zorder=20)
            
            Phi_BNM = np.linspace((B-N).phi, (M-N).phi, 1000)
            Cone_BNM = [[N.x + self.obs['ss']* np.cos(phi), N.y + self.obs['ss']*np.sin(phi)] for phi in Phi_BNM]
            Cone_BNM = sp.geometry.Polygon([[N.x, N.y]] + Cone_BNM + [[N.x, N.y]])

            xn, yn = Cone_BNM.exterior.xy 
            # self.ax.fill(xn, yn, c="#C69BF9", alpha=0.3, zorder=20)

            c = pointM.intersection(pointN)
            d = pointM.difference(pointN)
            e = pointN.difference(pointM)

            xc, yc = c.exterior.xy 
            xd, yd = d.exterior.xy 
            xe, ye = e.exterior.xy 

            # self.ax.fill(xc, yc, c="orange", alpha=0.3, zorder=7)
            # self.ax.fill(xd, yd, color="gray", alpha=0.2, zorder=1)
            # self.ax.fill(xe, ye, color="gray", alpha=0.2, zorder=1)
                
            self.obs['P_in_C'] =  Point((pos['P'].x, pos["P"].y)).within(c)
            self.obs['P_in_D'] =  Point((pos['P'].x, pos["P"].y)).within(d)
            self.obs['P_in_E'] =  Point((pos['P'].x, pos["P"].y)).within(e) 

            self.obs['M_in_C'] =  Point((M.x, M.y)).within(c)
            self.obs['N_in_C'] =  Point((N.x, N.y)).within(c)

            self.obs['P_in_Mcone'] = Point((pos['P'].x, pos['P'].y)).within(Cone_BMN) 
            self.obs['P_in_Ncone'] = Point((pos['P'].x, pos['P'].y)).within(Cone_BNM) 

            # print(pos['P'], pos['B'])

            self.obs['P_in_InvL'] = (pos['P'].x < pos['B'].x)
            # print(self.obs['P_in_Mcone'], self.obs['P_in_Ncone'], self.obs['P_in_InvL'])
            if self.obs['P_in_C']:
                self.obs['pos']['Z'] = pos['P']
            elif self.obs['P_in_InvL'] and self.obs['P_in_Ncone']:
                self.obs['pos']['Z'] = vector.obj(x= N.x + rIb * math.cos((pos['P'] - N).phi) , y=N.y + rIb * math.sin((pos['P'] - N).phi))
            elif (not self.obs["P_in_InvL"]) and self.obs['P_in_Mcone']:
                self.obs['pos']['Z'] = vector.obj(x= M.x + rIa * math.cos((pos['P'] - M).phi) , y=M.y + rIa * math.sin((pos['P'] - M).phi))
            else:
                self.obs['pos']['Z'] = pos['B']
            # self.ax.scatter(self.obs['pos']['Z'].x, self.obs['pos']['Z'].y, marker='o', s=9, color="#33ACFF", zorder=100)
            # self.ax.scatter(self.obs['pos']['P'].x, self.obs['pos']['P'].y, marker='o', s=18, color="#DA33FF", zorder=99)

        # fig = plt.figure(figsize=(5, 5))
        # self.ax  = fig.add_axes([0., 0., 1., 1. ])

        # draw_arrow(pos['M'], pos["N"], {"linewidth":1.8, "color":"black", "zorder":5})
        # draw_arrow(pos['P'], pos['M'], {"linewidth":1.8, "color":"lightgreen", "zorder":5})
        # draw_arrow(pos['N'], pos['P'], {"linewidth":1.8, "color":"lightgreen", "zorder":5})
        # draw_line(pos['B'], pos['A'], {"linewidth":1.8, "c":"orange", "zorder":5})


        # xlim = self.ax.get_xlim()
        # ylim = self.ax.get_ylim()
        # xxl  = xlim[1] - xlim[0]
        # yyl  = ylim[1] - ylim[0]
        # print(xxl, yyl)
        # llm  = max(xxl, yyl) * 1.5
        # self.ax.set_xlim(xlim[0] - (llm - xxl)/2., xlim[1] + (llm - xxl)/2.)
        # self.ax.set_ylim(ylim[0] - (llm - yyl)/2., ylim[1] + (llm - yyl)/2.)
        # print(self.ax.get_xlim(), self.ax.get_ylim())
        draw_geoms()


        # plt.show()
        # plt.savefig("{}/{}_geo.png".format(self.obs['path'], self.obs['ID']), dpi=300)
        # plt.close()

        # print(self.obs)
