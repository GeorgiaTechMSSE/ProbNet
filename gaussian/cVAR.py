####################################################################
### ProbNet: A probabilistic graph modeling toolkit that enables the design of cyber-physical-social systems 
###	     with the considerations of trust, resilience, and information dynamics',
###   author: Yan Wang,
###   author_email: yan-wang@gatech.edu
####################################################################
################################################################################
##### constrained least-square error fitting with constraint for VAR model #####
################################################################################
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar import util
from statsmodels.tsa.vector_ar.var_model import VARResults, VARResultsWrapper

import numpy as np
from scipy.optimize import minimize, Bounds



class cVAR(VAR):

    def __init__(self, endog, bnds=None, adj=None, exog=None, dates=None, freq=None, missing='none'):
        super(cVAR, self).__init__(endog, exog, dates, freq, missing=missing)
        if bnds is not None:
            self.bounds = bnds
        else:
            self.bounds = None
        if adj is not None:
            self.adjacency = np.array(adj,dtype=int) + np.identity(adj.shape[0],dtype=int)
        else:
            self.adjacency = None




	### overload the parameter estimation by constrained minimization method
    def _estimate_var(self, lags, offset=0, trend='c'):
        """
        lags : int
            Lags of the endogenous variable.
        offset : int
            Periods to drop from beginning-- for order selection so it's an
            apples-to-apples comparison
        trend : string or None
            As per above
        """
        # have to do this again because select_order doesn't call fit
        self.k_trend = k_trend = util.get_trendorder(trend)

        if offset < 0:  # pragma: no cover
            raise ValueError('offset must be >= 0')

        nobs = self.n_totobs - lags - offset
        endog = self.endog[offset:]
        exog = None if self.exog is None else self.exog[offset:]
        z = util.get_var_endog(endog, lags, trend=trend,
                               has_constant='raise')
        if exog is not None:
            # TODO: currently only deterministic terms supported (exoglags==0)
            # and since exoglags==0, x will be an array of size 0.
            x = util.get_var_endog(exog[-nobs:], 0, trend="nc",
                                   has_constant="raise")
            x_inst = exog[-nobs:]
            x = np.column_stack((x, x_inst))
            del x_inst  # free memory
            temp_z = z
            z = np.empty((x.shape[0], x.shape[1]+z.shape[1]))
            z[:, :self.k_trend] = temp_z[:, :self.k_trend]
            z[:, self.k_trend:self.k_trend+x.shape[1]] = x
            z[:, self.k_trend+x.shape[1]:] = temp_z[:, self.k_trend:]
            del temp_z, x  # free memory
        # the following modification of z is necessary to get the same results
        # as JMulTi for the constant-term-parameter...
        for i in range(self.k_trend):
            if (np.diff(z[:, i]) == 1).all():  # modify the trend-column
                z[:, i] += lags
            # make the same adjustment for the quadratic term
            if (np.diff(np.sqrt(z[:, i])) == 1).all():
                z[:, i] = (np.sqrt(z[:, i]) + lags)**2

        y_sample = endog[lags:]

        #################################################################################
        ### TOPOLOGY CONSTRAINED VAR MODEL FITTING
            ### retrieve sizes associated with coefficient matrix c
        num_rows = z.shape[1]
        num_nodes = y_sample.shape[1]
        ### the loss function to minimize difference between [Z]*[C] and [Y], element-wise square-sum of [Z]*[C]-[Y]
        def loss(c):
            c=c.reshape((num_rows, num_nodes))        #convert c from 1D array to 2D
            return np.sum(np.square((np.dot(z, c) - y_sample)))
        ### initial value of variable
        c0 = np.zeros((num_rows, num_nodes))
        ### bounds of variables to be optimized
        if self.bounds is not None:
            bnds = np.tensordot(np.ones(num_rows*num_nodes), self.bounds, axes=0)
        else:
            bnds = None
        ### if topology is used as constraints
        if self.adjacency is not None:
            ### Index matrix (with the same size of c) to identify the zero coefficients corresponding to no-connection in graph
##            H = np.ones((1,num_nodes),dtype=int)   # First row of coefficient (bias) should be zero
            H = np.zeros((1,num_nodes),dtype=int)
            for i in range(0,lags):
                H=np.append(H, (1-self.adjacency.T), axis=0)

            ### constraints:
            ### based on adjacency matrix, non-adjacent coefficients indicated by H are zeros
            cons = ({'type': 'eq',
                   'fun' : lambda c: np.sum(np.square(H*c.reshape((num_rows, num_nodes))))  })

            if self.adjacency.all():    ## if with full connections, no constraint should be imposed
                res = minimize(loss, c0, method='SLSQP', constraints=(),
                        bounds=bnds, options={'disp': True})
            else:
                res = minimize(loss, c0, method='SLSQP', constraints=cons,
                        bounds=bnds, options={'disp': True})
        ### only Coefficients VALUE RANGE CONSTRAINED VAR MODEL FITTING
        else:
##            H = np.ones((1,num_nodes),dtype=int)   # First row of coefficient (bias) should be zero
##            for i in range(0,lags):
##                H=np.append(H, np.zeros((num_nodes,num_nodes)), axis=0)
##            cons = ({'type': 'eq',
##                   'fun' : lambda c: np.sum(np.square(H*c.reshape((num_rows, num_nodes))))  })
            res = minimize(loss, c0, method='SLSQP', constraints=(),
                        bounds=bnds, options={'disp': True})

        params=res.x.reshape((num_rows, num_nodes))
        ###
        ###################################################################################

        # L�tkepohl p75, about 5x faster than stated formula
#        params = np.linalg.lstsq(z, y_sample, rcond=1e-15)[0]
        resid = y_sample - np.dot(z, params)

        # Unbiased estimate of covariance matrix $\Sigma_u$ of the white noise
        # process $u$
        # equivalent definition
        # .. math:: \frac{1}{T - Kp - 1} Y^\prime (I_T - Z (Z^\prime Z)^{-1}
        # Z^\prime) Y
        # Ref: L�tkepohl p.75
        # df_resid right now is T - Kp - 1, which is a suggested correction

        avobs = len(y_sample)
        if exog is not None:
            k_trend += exog.shape[1]
        df_resid = avobs - (self.neqs * lags + k_trend)

        sse = np.dot(resid.T, resid)
        omega = sse / df_resid

        varfit = VARResults(endog, z, params, omega, lags,
                            names=self.endog_names, trend=trend,
                            dates=self.data.dates, model=self, exog=self.exog)
        return VARResultsWrapper(varfit)
