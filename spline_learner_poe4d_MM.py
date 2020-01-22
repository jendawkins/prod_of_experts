from helper import *
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import os
from datetime import datetime

class SplineLearnerPOE_4D():
    def __init__(self, use_mm = 1, a = 1, b = 0.5, num_bact=3, MEAS_VAR=.01, PROC_VAR=1, THETA_VAR=1, AVAR=1, BVAR=1, POE_VAR=1, NSAMPS=2, NPTSPERSAMP=5,DT=.01,outdir =  'outdir'):
        self.num_bugs = num_bact
        aval = a

        if not os.path.exists(outdir):
            os.mkdir(outdir)

        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
        self.outdir = outdir + '/' + date_time

        self.true_a = a*np.ones((num_bact,num_bact))
        np.fill_diagonal(self.true_a, -3*np.diag(self.true_a))
        self.true_a[1, 2] = -self.true_a[1, 2]
        self.true_a[2, 1] = -self.true_a[2, 1]

        # self.gr = gr*np.ones(num_bact)
        if use_mm:
            self.true_b = b*np.ones((num_bact, num_bact))
        else:
            self.true_b = np.ones((num_bact, num_bact))
        a2 = 0
        b2 = np.sum(self.true_b,1)
        np.random.seed(4)
        self.xin = [st.truncnorm(a2, b2).rvs(size=(1, num_bact)) for i in range(NSAMPS)]
        self.gr = [-(self.true_a@xo.T).squeeze()/(b2 + xo.squeeze()) for xo in self.xin]

        self.use_mm = use_mm
        self.mvar = MEAS_VAR
        self.pvar = PROC_VAR
        self.avar = AVAR
        self.bvar = BVAR
        self.dt = DT
        self.plot_iter = 10
        self.num_mice = NSAMPS
        self.num_states = NPTSPERSAMP
        
        self.theta_var = THETA_VAR
        self.gibbs_var = 1

        self.alpha = 2
        
        self.k = 3
        
        self.pvar = 0
        self.states, self.observations= generate_data_MM(
            self.true_a, self.true_b, self.gr, self.xin, self.dt, self.use_mm, self.mvar, self.pvar*self.dt, self.num_mice, self.num_states)

        self.pvar = PROC_VAR
        self.X1 = np.concatenate(self.states,0)

        # self.X = diag_block_mat(self.X1, len(self.Y))
        self.num_states = len(self.states[0])
        
        self.num_knots = self.num_states-self.k 

        # Maybe redo this:
        self.mu_betas = np.mean(np.array([[[self.xin[n][0,i]*self.xin[n][0,j]*np.ones(self.num_knots)
                                   for i in range(self.num_bugs)] for j in range(self.num_bugs)] for n in range(self.num_mice)]),0)

        self.knots = np.array([[np.linspace(min(self.X1[:,i]*self.X1[:,j]) + .01, max(self.X1[:,i]*self.X1[:,j])-.01, self.num_knots) for i in range(self.num_bugs)] for j in range(self.num_bugs)])

        self.poe_var = POE_VAR*np.eye(num_bact*(self.num_states-1))
        self.beta_poevar = self.poe_var / (self.alpha - 1)

        self.pvar = self.pvar*np.eye(num_bact*(self.num_states-1))
        self.mvar = self.mvar*np.eye(num_bact*(self.num_states))

        self.beta_mvar = self.mvar / (self.alpha - 1)
        self.beta_pvar = self.pvar / (self.alpha - 1)

        self.states = np.transpose(self.states, (1,2,0))
        self.observations = np.transpose(self.observations, (1,2,0))
        self.true_bmat1 = np.concatenate([self.calc_bmat(self.states[:-1,:,i]) for i in range(self.num_mice)],0)
        # self.true_betas = np.linalg.lstsq(self.true_bmat1, self.Y)

        Y_est = [(self.states[1:, :, i]-self.states[:-1, :, i] - self.states[:-1, :, i]*self.dt*self.gr[i])/(self.dt) for i in range(self.num_mice)]
        Y_est = np.concatenate([Y_est[i].flatten(order='F') for i in range(self.num_mice)],0)
        
        self.true_betas = np.linalg.lstsq(self.true_bmat1, Y_est,rcond=None)[0]


    def bsplines(self, xi, x, bug1, bug2, k=3):
        # knots = np.linspace(x.min() + .01, x.max()-.01, num_knots)
        bmat = np.zeros((len(self.knots[bug1,bug2]), k))
        for ki in range(k):
            for i in np.arange(len(self.knots[bug1,bug2])-ki-2, 0, -1):
                if ki == 0:
                    if self.knots[bug1, bug2][i] <= xi <= self.knots[bug1, bug2][i+1]:
                        bmat[i, ki] = 1
                    else:
                        bmat[i, ki] = 0
                else:
                    bmat[i, ki] = ((xi-self.knots[bug1, bug2][i])*bmat[i, ki-1])/(self.knots[bug1, bug2][i+ki+1-1]-self.knots[bug1, bug2][i]) + (
                        (self.knots[bug1, bug2][i+ki+1] - xi)*bmat[i+1, ki-1])/(self.knots[bug1, bug2][i+ki+1]-self.knots[bug1, bug2][i+1])
        return bmat

    def calc_bmat(self,X,k=3):
        bmat = np.zeros((X.shape[0],self.num_bugs,self.num_bugs,self.num_knots))
        for t in range(X.shape[0]):
            bmat_mini = np.array([[self.bsplines(X[t,i]*X[t,j], np.array(X),i,j)[:,k-1] for i in range(X.shape[1])] for j in range(X.shape[1])])
            bmat[t,:,:,:] = bmat_mini
        bmat_full = []
        for i in range(self.num_bugs):
            r1 = np.array([bmat[t, i, :, :].flatten()
                           for t in range(X.shape[0])])
            bmat_full.append(r1)
        bmat_fin = diag_mat(bmat_full)

        return bmat_fin
    
    def calc_func_vals(self,x,betas,theta_2,ob):
        bmat = self.calc_bmat(x[:-1, :])
        g1 = betas@bmat.T
        g1 = np.reshape(g1, (x.shape[0]-1, x.shape[1]), order='F')

        g2 = michaelis_menten(x[:-1, :], theta_2[0], theta_2[1],self.use_mm)

        f1 = (x[:-1, :] + x[:-1, :]*self.dt * self.gr[ob]) + self.dt*g1
        f2 = x[:-1, :] + x[:-1, :]*self.dt*self.gr[ob] + self.dt * \
            michaelis_menten(x[:-1, :], theta_2[0], theta_2[1],self.use_mm)
        xy = ((x[1:, :] - x[:-1, :] - x[:-1, :]*self.gr[ob]*self.dt)/self.dt)

        return g1,g2,f1,f2,xy


    # Redo overleaf by keeping f1 and f2
    def update_theta(self, states, theta2,ob):
        x = states
        bmat = self.calc_bmat(x[:-1])
        mu_thetas = self.mu_betas[ob].flatten(order='F')
        sig_post = np.linalg.inv((1/self.theta_var)*np.eye(bmat.shape[1]) + self.dt*bmat.T@np.linalg.inv(
            self.pvar*self.dt)@bmat*self.dt + self.dt*bmat.T@np.linalg.inv(self.poe_var)@bmat*self.dt)

        Y_est = (x[1:, :]-x[:-1, :] - self.gr[ob]*x[:-1, :]*self.dt)
        Y_est = Y_est.flatten(order='F')

        f2_flat = michaelis_menten(states[:-1], theta2[0],theta2[1],self.use_mm).flatten(order = 'F')

        mu_post =  ((bmat.T@np.linalg.inv(self.pvar*self.dt))@(
            Y_est*self.dt) + (bmat.T@np.linalg.inv(self.poe_var))@((self.dt**2)*f2_flat))@sig_post
        return mu_post, sig_post

    def px(self, x, y, x0, betas, theta_2, ob, prior_var=.5, k=3):
        bmat = self.calc_bmat(x[:-1,:])
###############################################################################
        g1 = betas@bmat.T
        g1 = np.reshape(g1,(x.shape[0]-1,x.shape[1]),order = 'F')

        f1 = (x[:-1, :] + x[:-1, :]*self.dt * self.gr[ob]) + self.dt*g1
        f2 = x[:-1, :] + x[:-1, :]*self.dt*self.gr[ob] + self.dt * \
            michaelis_menten(x[:-1, :], theta_2[0], theta_2[1],self.use_mm)
        xy = ((x[1:, :] - x[:-1, :] - x[:-1, :]*self.gr[ob]*self.dt)/self.dt)
        part1 = -.5*((x[0, :]-x0)**2)*(1/prior_var)
        pvar=[self.pvar[(xy.shape[0])*i: (xy.shape[0])*(i+1), (xy.shape[0])*i: (xy.shape[0])*(i+1)] for i in range(self.num_bugs)]
        part2 = np.array([(-.5)*(xy[:, i]-g1[:, i]).T@(np.linalg.inv(self.dt * pvar[i]))@(xy[:, i]-g1[:, i]) for i in range(self.num_bugs)])
        mvar = [self.mvar[(x.shape[0])*i: (x.shape[0])*(i+1), (x.shape[0])
                          * i: (x.shape[0])*(i+1)] for i in range(self.num_bugs)]
        part3 = np.array([-0.5*((y[:,i]-x[:,i]).T@(np.linalg.inv(mvar[i]))@(y[:,i]-x[:,i])) for i in range(self.num_bugs)])
        poevar = [self.poe_var[(xy.shape[0])*i: (xy.shape[0])*(i+1), (xy.shape[0])
                          * i: (xy.shape[0])*(i+1)] for i in range(self.num_bugs)]
        part4 = np.array([(-.5)*(f1[:, i]-f2[:, i]).T@(np.linalg.inv(self.dt * poevar[i]))
                 @(f1[:, i]-f2[:, i]) for i in range(self.num_bugs)])

        try1 = np.array([part1, part2, part3, part4])
##################################################################################

        f1 = (x[:-1,:] + x[:-1,:]*self.dt*self.gr[ob]).flatten(order = 'F') + self.dt*(betas@bmat.T)
        f2 = x[:-1, :] + x[:-1, :]*self.dt*self.gr[ob] + self.dt * \
            michaelis_menten(x[:-1,:], theta_2[0],theta_2[1], self.use_mm)
        f2 = f2.flatten(order = 'F')
        xy = ((x[1:, :] - x[:-1, :] - x[:-1, :]*self.gr[ob]
               * self.dt)/self.dt).flatten(order='F')
        # import pdb
        # pdb.set_trace()
        part1 = -.5*((x[0,:]-x0)@(x[0,:]-x0))*(1/prior_var)
        # import pdb; pdb.set_trace()
        part2 = (-.5)*(xy-betas@bmat.T)@(np.linalg.inv(self.dt*self.pvar))@(xy-betas@bmat.T)
        y = y.flatten(order = 'F')
        x = x.flatten(order = 'F')
        part3 = -0.5*((y-x).T@(np.linalg.inv(self.mvar))@(y-x))
        part4 = -.5*((f1-f2).T@np.linalg.inv(self.poe_var)@(f1-f2))

        try2 = np.array([part1, part2, part3, part4])

        return try1,try2

    def update_x(self, x, y, x0, betas, theta_2, ob):
        x = x.astype(float)
        xp = x.copy()

        # import pdb; pdb.set_trace()
        mvar = np.delete(self.mvar, list(range(self.num_states-1,self.mvar.shape[0],self.num_states-1)),axis=0)
        mvar = np.delete(mvar, list(
            range(self.num_states-1, self.mvar.shape[1], self.num_states-1)), axis=1)

        sig = np.linalg.inv(np.linalg.inv(self.pvar*self.dt) +  np.linalg.inv(mvar))
        # sig = 1/(1/(self.pvar*self.dt) + 1/self.mvar)

        x1next = np.random.normal(x[0,:], np.sqrt(self.gibbs_var))
        xp[0,:] = x1next

        num1,num = self.px(xp, y, x0, betas, theta_2, ob)
        dem1,dem = self.px(x, y, x0, betas, theta_2, ob)

        prob_keep = np.exp(np.sum(num,0) - np.sum(dem,0))
        # idxs = np.where(prob_keep > 1)
        # x[0,idxs] = xp[0,idxs]

        if prob_keep > 1:
            x[0,:] = xp[0,:]

        # import pdb; pdb.set_trace()
        proposed_x = np.zeros(x.shape)
        proposed_x[0,:] = x1next
        for i in range(1, x.shape[0]):
            xp = x.copy()
            mx = np.expand_dims(x[i-1,:],0)
            mvar1 = self.mvar[i-1:self.mvar.shape[0]:self.num_states,i-1:self.mvar.shape[1]:self.num_states]
            pvar1 = self.pvar[i-1:self.pvar.shape[0]:self.num_states -1, i-1:self.pvar.shape[1]:self.num_states-1]
            
            sig1 = sig[i-1:sig.shape[0]:self.num_states -1, i-1:sig.shape[1]:self.num_states-1]
            mu_xi = (np.linalg.inv(pvar1*self.dt)@(x[i-1, :] + self.dt *(x[i-1, :]*self.gr[ob] + betas@self.calc_bmat(mx).T)) + (np.linalg.inv(mvar1)@y[i]))@sig1

            # xnext = x[i] + np.random.normal(0,np.sqrt(self.gvar))
            xnext = st.multivariate_normal(mu_xi, sig1).rvs()
            # xnext = (np.random.normal(x[i-1],np.sqrt(self.pvar)) + np.random.normal(y[i],np.sqrt(self.mvar)))/2
            xp[i,:] = xnext
            
            num1,num = self.px(xp, y, x0, betas, theta_2, ob)
            dem1,dem= self.px(x, y, x0, betas, theta_2, ob)
            prob_keep = np.exp(np.sum(num, 0) - np.sum(dem, 0))

            g1p, g2p, f1p, f2p, xyp = self.calc_func_vals(xp,betas,theta_2,ob)
            g1, g2, f1, f2, xy = self.calc_func_vals(
                x, betas, theta_2, ob)

            if prob_keep > 1:
                x[i,:] = xp[i,:]
            # else:
                # import pdb; pdb.set_trace()
            proposed_x[i, :] = xnext
        return x, proposed_x
    
    def px2(self,states,theta,f1,ob):

        xin = states[:-1,:]
        xingr = xin*self.gr[ob]
        xin_f = xin.flatten(order = 'F')
        f1a = ((f1 - xin_f)/self.dt - xingr.flatten(order = 'F'))
        a=theta[0]
        b=theta[1]
        part1=-.5*(michaelis_menten(xin, a, b,self.use_mm).flatten(order = 'F')-f1a).T@np.linalg.inv(
            self.poe_var)@(michaelis_menten(xin, a, b, self.use_mm).flatten(order='F')-f1a)

        part2 = 0
        #######################################
        # xin = states[:-1,:]
        # f1b = np.reshape(f1,xin.shape,order = 'F')
        # f1a = ((f1b - xin)/self.dt - xin*self.gr)

        # a=theta[0]
        # b=theta[1]
        # poevar = [self.poe_var[(xin.shape[0])*i: (xin.shape[0])*(i+1), (xin.shape[0])
        #                     * i: (xin.shape[0])*(i+1)] for i in range(self.num_bugs)]
        # import pdb; pdb.set_trace()
        # part1=[-.5*(sigmoid2d(xin, a, b)-f1a)[:,i].T@np.linalg.inv(
        #     poevar[i])@(sigmoid2d(xin, a, b)-f1a)[:,i] for i in range(self.num_bugs)]

        # part2 = 0
        return part1 + part2

    def update_f2(self,states,theta,f1, ob):
        if not self.use_mm:
            xin=states[:-1, :]
            f1_a = np.reshape(f1, (self.num_states-1, self.num_bugs), order='F')
            f1_a = (f1_a - xin - xin*self.gr[ob]*self.dt)/(self.dt)
            xbig = [xin for i in range(self.num_bugs)]
            X = diag_mat(xbig)

            f1_aa = f1_a.flatten(order = 'F')            
            sig_new = np.linalg.inv(np.linalg.inv(self.avar*np.eye(self.num_bugs**2)) + X.T@np.linalg.inv(self.poe_var)@X)
            
            mu_new = ((X**2).T@np.linalg.inv(self.poe_var)@f1_aa)@sig_new

            return mu_new, sig_new

        else:
            anew = theta[0] + np.random.normal(0,self.avar, size = (self.num_bugs,self.num_bugs))
            pold = self.px2(states,theta,f1, ob)
            pnew = self.px2(states,[anew, theta[1]], f1, ob)
            prob_keep = np.exp(pnew - pold)
            if prob_keep > 1:
                theta[0] = anew
            bnew = theta[1] + \
                np.random.normal(0, self.bvar, size=(self.num_bugs, self.num_bugs))
            pold=self.px2(states, theta, f1, ob)
            pnew=self.px2(states, [theta[0], bnew], f1, ob)
            prob_keep = np.exp(pnew-pold)

            if prob_keep > 1:
                theta[1] = bnew
            return theta

    def update_poe(self,f1, f2):
        # import pdb; pdb.set_trace()
        f1 = np.expand_dims(f1,0)
        f2 = np.expand_dims(f2,0)
        beta_post = self.beta_poevar + 0.5*(f1-f2).T@(f1-f2)
        alpha_post = self.alpha + 1/2
        return st.invgamma(alpha_post, beta_post).rvs()*np.eye(len(f1))

    def update_pvar(self,states, f1):

        f1 = np.expand_dims(f1,0)
        states = np.expand_dims(states.flatten(order = 'F'),0)
        beta_post = self.beta_pvar + 0.5*(states-f1).T@(states-f1)
        alpha_post = self.alpha + 1/2
        return st.invgamma(alpha_post, beta_post).rvs()

    def update_mvar(self, states, obs):
        beta_m_post = self.beta_mvar + 0.5*(obs - states).T@(obs - states)
        alpha_m_post = self.alpha + 1/2
        return st.invgamma(alpha_m_post, beta_m_post).rvs()

    
    def run(self,gibbs_steps=500, train_x = True, train_f1 = True, train_f2 = True, train_var = True, plot = True):
        self.trace_a = []
        self.trace_b =[]
        self.trace_x=[]
        self.trace_f1=[]
        self.trace_f2=[]
        self.trace_beta=[]

        f1 = np.random.normal(0, np.sqrt(self.theta_var),
                              size=((self.num_states-1)*self.num_bugs))
        f2 = np.random.normal(0, np.sqrt(self.theta_var),
                              size=((self.num_states-1)*self.num_bugs))

        betas = st.multivariate_normal(self.mu_betas.flatten(order = 'F'),self.theta_var*np.eye(np.prod(self.mu_betas.shape))).rvs()

        theta2 = [np.random.normal(0, self.avar, size=(self.num_bugs, self.num_bugs)),
                  np.random.normal(0, self.bvar, size=(self.num_bugs, self.num_bugs))]
        if not self.use_mm:
                theta2[1] = np.ones(theta2[1].shape)
        if plot:
            plot_orig(self.outdir, self.states, self.observations)

        for s in range(gibbs_steps):
                    
            self.betavec = []
            self.xxvec = []
            self.f2vec = []
            self.avec=[]
            self.bvec=[]
            self.f1vec=[]
            for i in range(self.observations.shape[-1]):
                
                y = self.observations[:, :, i]
                x0 = np.random.normal(self.observations[0, :, i], np.sqrt(100))
                x = np.random.normal(np.mean(self.observations[:, :, i], 0), 1, size=(
                    self.num_states, self.num_bugs))

                # Train X
                if train_x:
                    xnew,proposed_xnew = self.update_x(x, y, x0, betas, theta2,i)
                    x = xnew
                    x0 = x[0,:]
                    if s % self.plot_iter == 0 and plot:
                        if s == 0:
                            xold = x
                        print('observation ' + str(i))
                        plot_states(self.outdir, xnew, self.states[:, :, i], self.observations[:, :, i],
                                    xold, proposed_xnew)
                        xold = xnew
                else:
                    x = self.states[:,:,i]

                # Train F1
                xplot = self.states[:-1, :, i]
                bmat_plot = self.calc_bmat(xplot)
                if train_f1:
                    mu_theta, sig_theta = self.update_theta(x, theta2, i)
                    betas = st.multivariate_normal(
                        mu_theta.squeeze(), sig_theta).rvs()
                    if s % self.plot_iter == 0 and plot:
                        plot_f1(self.outdir, xplot, bmat_plot, mu_theta, sig_theta, self.dt, self.gr[i], self.true_betas, self.true_bmat1[i*(
                            self.num_bugs*(self.num_states-1)):(i+1)*(self.num_bugs*(self.num_states-1)), :])
                else:
                    mu_theta = self.true_betas
                    betas = mu_theta

                # Re-define f1
                f1 = (x[:-1, :] + x[:-1, :]*self.dt *self.gr[i]).flatten(order='F') + self.dt*(betas@self.calc_bmat(x[:-1, :]).T)
                
                if train_f2:
                    # xin = x[:-1,:]
                    if not self.use_mm:
                        mu2, sig2 = self.update_f2(x,theta2,f1,i)
                        if s % self.plot_iter == 0 and plot:
                            plot_f2_linear(self.outdir,
                                xplot, mu2, sig2, [self.true_a, self.true_b], self.use_mm, self.dt, self.gr[i])

                        theta2 = [np.reshape(st.multivariate_normal(mu2, sig2).rvs(),(self.num_bugs, self.num_bugs),order ='F'),\
                            np.ones((self.num_bugs,self.num_bugs))]
                        # import pdb; pdb.set_trace()
                    else:
                        theta2 = self.update_f2(x,theta2,f1,i)
                        if s % self.plot_iter == 0 and plot:
                            plot_f2(self.outdir, xplot, theta2, [
                                    self.true_a, self.true_b], self.use_mm, self.dt, self.gr[i])

                else:
                    theta2 = [self.true_a, self.true_b]
                f2 = xplot + self.dt * \
                    (self.gr[i]*xplot + michaelis_menten(xplot, theta2[0], theta2[1],self.use_mm))

                f2 = f2.flatten(order='F')
                if s % self.plot_iter == 0 and train_var:
                    print('Step ' + str(s))
                    print('Meas Var:' + str(np.mean(self.mvar)))
                    print('Proc Var:' + str(np.mean(self.pvar)))
                    print('POE Var:' + str(np.mean(self.poe_var)))
                if train_var:
                    self.pvar = self.update_pvar(x[1:,:], f1)
                    self.poe_var = self.update_poe(f1, f2)

                self.betavec.append(betas)
                self.f1vec.append(f1)
                self.avec.append(theta2[0])
                self.bvec.append(theta2[1])
                self.xxvec.append(x)
                self.f2vec.append(f2)

            self.trace_a.append(self.avec)
            self.trace_b.append(self.bvec)
            self.trace_x.append(self.xxvec)
            self.trace_f1.append(self.f1vec)
            self.trace_f2.append(self.f2vec)
            self.trace_beta.append(self.betavec)
            print('Step ' + str(s) + ' Complete')

            if s % 100 == 0:
                with open(outdir + '_data_' + str(s), 'wb') as f:
                    pickle.dump([self.trace_a, self.trace_b, self.trace_x, self.trace_f1,\
                        self.trace_f2, self.trace_beta], f)

