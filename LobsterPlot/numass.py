"""
Code written by Jeremy Cushman and updated by Jorge Torres.

Todo:
    * Document what each function does.
    * Update sensitivities, etc.
    * Make sure to add word "Preliminary if using CUORE's unpublished results"

"""

from numpy import cos, sin, exp, pi, sqrt, arcsin
import numpy as np
import matplotlib.pylab as plt
from random import uniform, betavariate
import cmath


def nuM_range(m, hiearchy,includeSterile, p12, p13, p14, nSamples= 1000):
    """
    The function calculates the effective Majorana mass,
    bete decay kinetic mass, and sum mass of all neutrino flavors
    given the lightest neutrino mass. 
    The calculation depends on the hiearchy specified (IH, or NH)
    and can possibly include the simple 3+1 sterile neutrino model if desired.
    
    Mixing parameters provided in the parameter list must follow:
    p12 = [[angle_bestfit, angle_min, angle_max],
           [deltaMassSquared_bestfit, deltaMassSqaured_min, deltaMassSqaured_max]
           ]
    
    If the boolean flag 'includeSterile' is False, P14 is ignored.
    
    In the calculation, CP-violating phase, two majorana phases (3 if sterile)
    are randomly drawn from (0, 2pi). 
    Mixing parameters are drawn randomly from (min, max) using a beta function 
    that strongly favors the end points.
    
    nSamples specifies how many random sampling points are used.
    Any number on the order of 100K would take about hours. 
    
    Suffix notation:
    _0: best fit values
    _3sigma: 3sigma related
    _U: upper boundary
    _L: lower boundary
    """
    
    T12_0 = p12[0][0]
    C12_0= cos(T12_0)
    S12_0= sin(T12_0)

    T14_0 = p14[0][0]
    S14_0 = sin(T14_0)

    T13_0 = p13[0][0]
    C13_0 = cos(T13_0)
    S13_0 = sin(T13_0)
    
    dM2_12_0 = p12[1][0]
    dM2_13_0 = p13[1][0]
    dM2_14_0 = p14[1][0]

    mbb_U_0= 1E-3
    mbb_L_0= 1E6
    mb_0 =0
    sumM_0 =0
    
    mbb_U_3sigma= 1E-3
    mbb_L_3sigma= 1E6
    mb_U_3sigma =1E-3
    mb_L_3sigma =1E6
    sumM_U_3sigma =1E-3
    sumM_L_3sigma =1E6

    if hiearchy == 'NH':
        m1_0 = m
        m2_0 = sqrt(m**2 + dM2_12_0)
        m3_0 = sqrt(m**2 + dM2_13_0)
        m4_0 = sqrt(m**2 + dM2_14_0)
        
    else:
        m1_0 = sqrt(m**2+ dM2_13_0)
        m2_0 = sqrt(m**2 + dM2_13_0 + dM2_12_0)
        m3_0 = m
        m4_0 = sqrt(m**2 + dM2_13_0 + dM2_14_0)
    if not includeSterile:
        m4_0 = 0
        
    for ii in range(nSamples):
        alpha = uniform(0, 2*pi)
        beta  = uniform(0, 2*pi)
        gamma = uniform(0, 2*pi)
        
        ### Does this probe the whole phase space correctly? ###
        ### Beta distribution is used under the assumption that
        ### extreme mixing parameters bring exteme effective majorana masses
        
        T12 = p12[0][1] + betavariate(0.1,0.1) * (p12[0][2]-p12[0][1])
        T13 = p13[0][1] + betavariate(0.1,0.1) * (p13[0][2]-p13[0][1])
        T14 = p14[0][1] + betavariate(0.1,0.1) * (p14[0][2]-p14[0][1])

        dM2_12 = p12[1][1] + betavariate(0.1,0.1) * (p12[1][2]-p12[1][1])
        dM2_13 = p13[1][1] + betavariate(0.1,0.1) * (p13[1][2]-p13[1][1])
        dM2_14 = p14[1][1] + betavariate(0.1,0.1) * (p14[1][2]-p14[1][1])

        C12 = cos(T12); S12 = sin(T12)
        C13 = cos(T13); S13 = sin(T13)
        C14 = cos(T14); S14 = sin(T14)
        
        if hiearchy == 'NH':
            m1 = m
            m2 = sqrt(m**2 + dM2_12)
            m3 = sqrt(m**2 + dM2_13)
            m4 = sqrt(m**2 + dM2_14)

        else:
            m1 = sqrt(m**2+ dM2_13)
            m2 = sqrt(m**2 + dM2_13 + dM2_12)
            m3 = m
            m4 = sqrt(m**2 + dM2_13 + dM2_14)
        if  not includeSterile:
            m4 = 0
            
        m_bb_0 = abs(\
                m1_0 * C12_0**2 * C13_0**2\
                + m2_0 * S12_0**2 * C13_0**2 * cmath.exp(complex(0,alpha)) \
                + m3_0 * S13_0**2 * cmath.exp(complex(0,beta))\
                + m4_0 * S14_0**2 * cmath.exp(complex(0,gamma))\
                )         
        
        if m_bb_0 > mbb_U_0: mbb_U_0 = m_bb_0 
        if m_bb_0 < mbb_L_0: mbb_L_0 = m_bb_0 

        m_bb_3sigma = abs(\
                m1 * C12**2 * C13**2 \
                + m2 * S12**2 * C13 **2 * cmath.exp(complex(0,alpha)) \
                + m3 * S13**2 * cmath.exp(complex(0,beta)) \
                + m4 * S14**2 * cmath.exp(complex(0,gamma)) \
                )
                         
        if m_bb_3sigma > mbb_U_3sigma: mbb_U_3sigma = m_bb_3sigma 
        if m_bb_3sigma < mbb_L_3sigma: mbb_L_3sigma = m_bb_3sigma 
        
        #### kinetic neutrino mass from beta Decay         
        mb_0 = sqrt(\
                    m1_0**2 * C12_0**2 * C13_0**2 \
                    + m2_0**2 * S12_0**2 * C13_0**2 \
                    + m3_0**2 * S13_0**2 \
                    + m4_0**2 * S14_0**2 \
                    )

        mb_3sigma = sqrt(\
                         m1**2 * C12**2 * C13**2 \
                         + m2**2 * S12**2 * C13**2 \
                         + m3**2 * S13**2 \
                         + m4**2 * S14**2 \
                         )

        if mb_3sigma > mb_U_3sigma: mb_U_3sigma = mb_3sigma
        if mb_3sigma < mb_L_3sigma: mb_L_3sigma = mb_3sigma

        #### Sum of neutrino mass ############
        sumM_0 = m1_0 + m2_0 + m3_0 + m4_0
        sumM_3sigma = m1 + m2 + m3 + m4

        if sumM_3sigma > sumM_U_3sigma: sumM_U_3sigma = sumM_3sigma
        if sumM_3sigma < sumM_L_3sigma: sumM_L_3sigma = sumM_3sigma
    
    return [mbb_L_0, mbb_U_0, mbb_L_3sigma, mbb_U_3sigma], \
           [mb_0, mb_L_3sigma, mb_U_3sigma], \
           [sumM_0, sumM_L_3sigma, sumM_U_3sigma]


def boloLimits(IH, NH, xMin, xMax, yMin=-1, yMax=-1):
    
    arrowXscale = 1.2
    
    IH.axhspan(270, 650, lw=0,fc='#2B4970',ec='#2B4970', alpha=0.3)
    Teline = IH.axhline(270, color='#2B4970', label='$^{130}$Te limit (CUORE-0 + Cuoricino)')
    IH.errorbar(xMin*arrowXscale, 270,yerr = 270, lolims=True,  color='#2B4970')
    IH.text(xMin*pow(arrowXscale,2), 270*1.2, '$^{130}$Te Limit', color='#2B4970',fontsize='small')
    
    IH.axhspan(200, 400,lw=0, ec='#AA7F39',fill=None, alpha=0.3, hatch='\\\\\\')
    Geline = IH.axhline(200, color='#AA7F39', label='$^{76}$Ge limit (GERDA + HdM + IGEX)', linestyle='--')
    IH.errorbar(xMin*6, 200, yerr=200, lolims=True,  color='#AA7F39')
    
    IH.axhspan(120, 250,lw=0, ec='#AA7F39',fill=None, alpha=0.3, hatch='///')
    Xeline = IH.axhline(120, color='#AA7F39', label='$^{136}$Xe limit (KamLAND-Zen + EXO-200)', linestyle='-')
    IH.errorbar(xMin*6*arrowXscale, 120, yerr=120, lolims=True,  color='#AA7F39')
    
    IH.axhspan(50, 130,lw=0, fc='#2B4970',ec='#2B4970', alpha=0.3)
    IH.axhline(50, color='#2B4970', label='CUORE Sensitivity')
    IH.errorbar(xMin*arrowXscale, 50, yerr=50, lolims=True,  color='#2B4970')
    IH.text(xMin*pow(arrowXscale,2), 50*1.2, 'CUORE Sensitivity', color='#2B4970',fontsize='small')
    
    IH.legend(handles=[Teline, Geline, Xeline], loc=3,prop={'size':9})
        
    # ON the right panel
    NH.axhspan(270, 650, lw=0,fc='#2B4970',ec='#AA9B39', alpha=0.3, label='Te-130 Limit')
    #NH.axhspan(270, 650, lw=0,ec='#AA9B39',fill=None, alpha=0.3, hatch='///')
    NH.axhline(270, color='#2B4970', label='Te-130 Limit')
    
    NH.axhspan(200, 400, lw=0,ec='#AA7F39',fill=None, alpha=0.3, hatch='\\\\\\')
    NH.axhline(200, color='#AA7F39', label='Ge-76 Limit', linestyle='--')
    
    NH.axhspan(120, 250,lw=0, ec='#AA7F39',fill=None, alpha=0.3, hatch='///')
    NH.axhline(120, color='#AA7F39', label='Xe-136 Limit', linestyle='-')
    
    NH.axhspan(50, 130, lw=0,fc='#2B4970',ec='#452F74', alpha=0.3, label='CUORE Sensitivity')
    #NH.axhspan(50, 130,lw=0, ec='#452F74',fill=None, alpha=0.3, hatch='\\\\\\')
    NH.axhline(50, color='#2B4970', label='CUORE Sensitivity')
        
    IH.set_title('3 Flavors, Inverted Hierarchy')
    NH.set_title('3 Flavors, Normal Hierarchy')
    
    IH.set_xlim(xMin, xMax)
    NH.set_xlim(xMin, xMax)
    
    if yMin>0 and yMax >0:
        IH.set_ylim(yMin, yMax)
    else:
        IH.set_ylim(xMin, xMax)


def massContour(ax, xArray, y2DArray, col, xlab, ylab=''):
    if y2DArray.shape[1]==3:
        ax.plot(xArray, y2DArray[:,0], color=col)
        ax.fill_between(xArray,y2DArray[:,1], y2DArray[:,2], color=col, alpha=0.2)
    elif y2DArray.shape[1]==4:
        ax.fill_between(xArray,y2DArray[:,0], y2DArray[:,1], color=col, alpha=0.4)
        ax.fill_between(xArray,y2DArray[:,2], y2DArray[:,3], color=col, alpha=0.2)
    else:
        return
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)