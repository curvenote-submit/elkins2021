import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['lines.linewidth']=2
mpl.rcParams['font.size']=16

def plot_inputs(df,figsize=(8,6)):
    '''
        pretty plots input data from pandas dataframe df
    '''

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True,figsize=figsize)
    ax1.plot(df['F'],df['P'])
    ax1.invert_yaxis()
    ax1.set_xlabel('F')
    ax1.set_ylabel('Pressure (kbar)')
    xticks = np.linspace(0,max(df['F']),10)
    ax1.grid()
    ax1.set_xticks(xticks,minor=True)
    ax2.plot(df['Kr'],df['P'])
    ax2.set_xlabel(r'$K_r$')
    for s in ['DU','DTh','DRa','DPa']:
        ax3.semilogx(df[s],df['P'],label=s)
    ax3.set_xlabel(r'$D_i$')
    ax3.legend(loc='best',bbox_to_anchor=(1.1,1))

def plot_1Dcolumn(df,figsize=(8,6)):
    '''
        pretty plots output data from dataframe of output
    '''

    fig, (ax1, ax2) = plt.subplots(1,2, sharey=True,figsize=figsize)
    ax1.plot(df['phi'],df['P'],'r',label='$\phi$')
    ax1.set_xlabel('Porosity',color='r')
    ax1.set_ylabel('Pressure (kbar)')
    ax1.invert_yaxis()

    ax1a = ax1.twiny()
    ax1a.plot(df['F'],df['P'],'b',label='$F$')
    ax1a.set_xlabel('Degree of melting',color='b')

    for s in ['(230Th/238U)','(226Ra/230Th)','(231Pa/235U)']:
        ax2.plot(df[s],df['P'],label=s)
    ax2.set_xlabel('Activity Ratios')
    ax2.set_xlim(0,5)
    ax2.set_xticks(range(5))
    ax2.grid()
    ax2.legend(loc='best',bbox_to_anchor=(1.1,1))
    return fig,(ax1,ax1a,ax2)

def plot_contours(phi0,W0,act,figsize=(12,12)):
    '''
    pretty plot activity contour plots
    '''

    Nplots = act.shape[0]
    if Nplots == 2:
        labels = ['$(^{230}Th/^{238}U)$', '$(^{226}Ra/^{230}Th)$']
    else:
        labels = ['$(^{231}Pa/^{235}U)$']


    if Nplots == 1:
        plt.figure(figsize=figsize)
        cf = plt.contourf(phi0,W0,act[0])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Porosity ($\phi$)')
        plt.ylabel('Upwelling Rate (cm/yr)')
        plt.gca().set_aspect('auto')
        plt.title(labels[0])
        plt.colorbar(cf, ax=plt.gca(), orientation='horizontal',shrink=1.)
    else:
        fig, axes = plt.subplots(Nplots,1,sharey=True,figsize=(10,24))
        for i,ax in enumerate(axes):
            cf = ax.contourf(phi0,W0,act[i])
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_aspect('auto')
            ax.set_xlabel('Porosity ($\phi$)')
            ax.set_ylabel('Upwelling Rate (cm/yr)')
            ax.set_title(labels[i])
            fig.colorbar(cf, ax=ax, orientation='horizontal',shrink=1.)

def plot_mesh_Ra(Th,Ra,W0,phi0,figsize=(12,12)):
    '''
    activity mesh plot for Ra vs. Th, from gridded data
    '''
    mW,nphi=Th.shape
    plt.figure()
    for m in range(mW):
        plt.plot(Th[m],Ra[m],label='W = {} cm/yr'.format(W0[m]))
    for n in range(nphi):
        plt.plot(Th.T[n],Ra.T[n],label='$\phi$ = {}'.format(phi0[n]))

    plt.legend(loc='upper left',bbox_to_anchor=(0,-0.3))
    plt.axis([0.9,2.0,0.7,10.0])
    plt.grid()
    plt.xlabel('($^{230}$Th/$^{238}$U)')
    plt.ylabel('($^{226}$Ra/$^{230}$Th)')
#    plt.axes().set_aspect('equal',)

def plot_mesh_Pa(Th,Pa,W0,phi0,figsize=(12,12)):
    '''
    activity mesh plot for Pa vs. Th, from gridded data
    '''
    mW,nphi = Th.shape
    plt.figure()
    for m in range(mW):
        plt.plot(Th[m],Pa[m],label='W = {} cm/yr'.format(W0[m]))
    for n in range(nphi):
        plt.plot(Th.T[n],Pa.T[n],label='$\phi$ = {}'.format(phi0[n]))

    plt.legend(loc='upper left',bbox_to_anchor=(0,-0.3))
    plt.axis([0.9,2.0,0.7,10.0])
    plt.grid()
    plt.xlabel('($^{230}$Th/$^{238}$U)')
    plt.ylabel('($^{226}$Pa/$^{230}$Th)')

    #plt.show()

from scipy.optimize import brentq
from scipy.interpolate import interp1d, pchip

class UserCalc:
    ''' A class for constructing solutions for Equilibrium Transport U-series calculations
        ala Spiegelman, 2000, G-cubed

        Usage:
        us = UserCalc(model,df,dPdz = 0.32373, n = 2., tol=1.e-6, phi0 = 0.008, W0 = 3.)

        with
        model : a class of a Useries decaychain model
        df   :  a pandas-dataframe with columns ['P','Kr','DU','DTh','DRa','DPa']
        dPdz :  Pressure gradient to convert P to z
        n    :  permeability exponent
        tol  :  tolerance for ODE solver
        phi0 :  initial porosity
        W0   : upwelling velocity (cm/yr)

    '''
    def __init__(self, model, df, dPdz = 0.32373, n = 2., tol = 1.e-6, phi0 = 0.008, W0 = 3.):
        self.model = model
        self.df = df
        self.dPdz = dPdz
        self.n = n
        self.tol = 1.e-6
        self.phi0 = phi0
        self.W0 = W0/1.e5

        # set depth scale h
        self.zmin = df['P'].min()/dPdz
        self.zmax = df['P'].max()/dPdz
        self.h = self.zmax - self.zmin

        # lambda function to define scaled column height zprime
        self.zp = lambda P: (self.zmax - P/dPdz)/self.h

        # set interpolants for F and Kr and pressure
        self.F = pchip(self.zp(df['P']),df['F'])
        self.dFdz = self.F.derivative(nu=1)
        self.Kr = interp1d(self.zp(df['P']),df['Kr'],kind='cubic')
        self.P = interp1d(self.zp(df['P']),df['P'],kind='cubic')

        # set maximum degree of melting
        self.Fmax = self.df['F'].max()

        # set reference densities (assuming a mantle composition)
        self.rho_s = 3300.
        self.rho_f = 2800.

        # set  decay constants for [ 238U, 230Th, 226Ra] and [ 235U, 231Pa ]
        t_half_238 = np.array([4.468e9, 7.54e4, 1600.])
        t_half_235 = np.array([7.03e8, 3.276e4])
        self.lambdas_238 = np.log(2.)/t_half_238
        self.lambdas_235 = np.log(2.)/t_half_235

        # set interpolation functions for Partition coefficients for each chain
        self.D_238 = [ interp1d(self.zp(df['P']),df['DU'],kind='cubic'),
                       interp1d(self.zp(df['P']),df['DTh'],kind='cubic'),
                       interp1d(self.zp(df['P']),df['DRa'],kind='cubic') ]
        self.D_235 = [ interp1d(self.zp(df['P']),df['DU'],kind='cubic'),
                       interp1d(self.zp(df['P']),df['DPa'],kind='cubic')]

        # lambda function to get partition coefficients at zprime = 0
        self.get_D0 = lambda D: np.array([ D[i](0) for i in range(len(D))])

        # initialize reference permeability
        self.setAd(self.phi0,n=n)

    # initialize porosity function
    def setAd(self,phi0,n):
        '''
            sets the reference permeability given the maximum porosity
        '''
        Fmax = self.Fmax
        self.phi0 = phi0
        self.n = n
        self.Ad =  (self.rho_s/self.rho_f*Fmax - phi0*(1. - Fmax)/(1. - phi0))/(phi0**n*(1.-phi0))

    def phi(self,zp):
        '''
        returns porosity as function of dimensionless column height zp
        '''
        # effective permeability
        K = self.Kr(zp)*self.Ad

        # degree of melting
        F = self.F(zp)

        # density ratio
        rs_rf = self.rho_s/self.rho_f

        # rootfinding function to define porosity such that f(phi) = 0

        # check if scalar else loop
        if np.isscalar(zp):
            f = lambda phi: K*phi**self.n*(1. - phi)**2 + phi*(1. + F*(rs_rf - 1.)) - F*rs_rf
            upper_bracket= 1.05*self.phi0
            try:
                phi = brentq(f,0.,upper_bracket)
            except ValueError:
                phi_test = np.linspace(0,upper_bracket)
                print('Error in brentq: brackets={}, {}'.format(f(0.),f(upper_bracket)))
                print('zp={},F={}, K={}'.format(zp,F,K))
        else: # loop over length of zp
            phi = np.zeros(zp.shape)
            for i,z in enumerate(zp):
                f = lambda phi: K[i]*phi**self.n*(1. - phi)**2 + phi*(1. + F[i]*(rs_rf - 1.)) - F[i]*rs_rf
                phi[i] = brentq(f,0.,1.)
        return phi

    def set_column_params(self, phi0, n, W0):
        '''
        set porosity/permeability and upwelling rate parameters for a single column

        phi0: porosity at Fmax
        n   : permeability exponent
        W0  : upwelling rate (cm/yr)
        '''

        self.setAd(phi0,n)
        self.W0 = W0/1.e5  # upwelling in km/yr.

    def solve_1D(self,D,lambdas,alpha0 = None, z_eval = None):
        '''
        Solves 1-D decay problem (assumes column parameters have been set

        Usage:  z, a, Us, Uf = solve_1D(D,lambdas,alpha0,z_eval)

        Input:
        D      :  function that returns bulk partition coefficients for each nuclide
        lambdas:  decay coefficients of each nuclide
        alpha0 :  initial activities of the nuclide in the unmelted solid (defaults to 1)
        z_eval :  dimensionless column heights where solution is returned

        Output:
        z:   coordinates where evaluated
        a:   activities of each nuclide
        Us:  solid nuclide concentration
        Uf:  fluid nuclide concentration
        '''

        # if z_eval is not set, use initial Input values
        if z_eval is None:
            z_eval = self.zp(self.df['P'])
        elif np.isscalar(z_eval):
            z_eval = np.array([z_eval])

        # if alpha is not set, use 1
        if alpha0 is None:
            alpha0 = np.ones(len(lambdas))

        # scaled decay constants and initial partition coefficients
        lambdap = self.h*lambdas/self.W0

        us = self.model(alpha0,lambdap,D,self.W0,self.F,self.dFdz,self.phi,self.rho_f,self.rho_s)

        D0 = self.get_D0(D)
        z, Us, Uf = us.solve(z_eval)

        # calculate activities
        act =  [ alpha0[i]/D0[i]*np.exp(Uf[i]) for i in range(len(D0)) ]
        return z, act, Us, Uf

    def solve_all_1D(self,phi0 ,n , W0, alphas = np.ones(4), z_eval = None):
        '''
        Sets up and solves the 1-D column model for a given phi0,n, and upwelling rate W0 (in cm/yr).
        Solves for both the U238 decay chain and the U235 decay chain.

        Returns a pandas dataframe
        '''
        self.set_column_params(phi0,n,W0)

        # evaluate at input depths if not specified
        if z_eval is None:
            z_eval = self.zp(self.df['P'])

        # set initial alpha values for [ 238U, 230Th, 226Ra] and [ 235U, 231Pa ]
        self.alphas_238 = alphas[0:3]
        self.alphas_235 = alphas[3:5]

        # solve for the U238 model
        z238, a238, Us238, Uf238 = self.solve_1D(self.D_238, self.lambdas_238, self.alphas_238, z_eval = z_eval)

        # solve for the U235 model
        z235, a235, Us235, Uf235 = self.solve_1D(self.D_235, self.lambdas_235, self.alphas_235, z_eval = z_eval)

        # start building output dataframe
        z = z_eval

        df = pd.DataFrame()
        df['P'] = self.P(z)
        df['z'] = self.zmax - self.h*z
        df['F'] = self.F(z)
        df['phi'] = self.phi(z)
        names = ['(230Th/238U)','(226Ra/230Th)']
        for i,name in enumerate(names):
            df[name] = a238[i+1]/a238[i]

        df['(231Pa/235U)'] = a235[1]/a235[0]

        names = ['Uf_238U','Uf_230Th', 'Uf_226Ra']
        for i,name in enumerate(names):
            df[name] = Uf238[i]

        names = ['Us_238U','Us_230Th', 'Us_226Ra']
        for i,name in enumerate(names):
            df[name] = Us238[i]

        names = ['Uf_235U','Uf_231Pa']
        for i,name in enumerate(names):
            df[name] = Uf235[i]

        names = ['Us_235U','Us_231Pa']
        for i,name in enumerate(names):
            df[name] = Us235[i]

        return df

    def solve_grid(self, phi0, n, W0, D, lambdas, alpha0 = None, z = 1.):
        '''
        solves of activity ratios at the height z in the column for a mesh grid of porosites phi0 and upwelling
        velocities W0 (slow, not vectorized)

        '''
        # number of nuclides in chain
        Nchain = len(lambdas)

        # if alpha is not set, use 1
        if alpha0 is None:
            alpha0 = np.ones(Nchain)

        act = np.zeros((Nchain - 1,len(W0),len(phi0)))

        for j, W in enumerate(W0):
            print('\nW = {}'.format(W), end=" ")
            for i, phi in enumerate(phi0):
                print('.', end=" ")
                self.set_column_params(phi,n,W)
                z, a, Us, Uf = self.solve_1D(D,lambdas,alpha0,z_eval = z)
                for k in range(1,Nchain):
                    act[k-1,j,i] = a[k]/a[k-1]

        return act
