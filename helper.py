import numpy as np
def sigmoid(t, yscale=20, xscale=.1):
    y = yscale/(1+np.exp(-xscale*(t)))
    return y


def generate_data(a, b, mvar, pvar, xrange, nsamples=25, nptspersample=20, resolution=.1):
    t = np.linspace( - np.abs(xrange)/2, np.abs(xrange)/2,np.abs(xrange)/resolution)
    f = sigmoid(t, a, b) - xrange/2
    xvec = []
    yvec = []
    ns = 0
    while ns < nsamples:
        xin = [(f.max()-f.min())*np.random.rand() + f.min()]
        # xout = []
        for iterate in range(nptspersample):
            xin.append(xin[-1] + resolution*sigmoid(xin[-1],a,b) + np.random.normal(0,np.sqrt(pvar)))
        #     if iterate < 99:
        #         xin.append(xout[-1])
            # import pdb; pdb.set_trace()

        xin = np.array(xin)
        y = xin + np.random.normal(0, np.sqrt(mvar), size=xin.shape)
        if np.var(xin)/np.var(y) > .5:
            ns += 1
            xvec.append(xin)
            yvec.append(y)

    return xvec, yvec, t, f

def spline_change_pts(states):
    nvec = []
    for s in states:
        nn = np.where([np.abs(s[i] - s[i+1]) < .0001 for i in range(len(s)-1)])[0]
        if len(nn) == 0:
            n = len(s)
        elif len(nn) > 1:
            #         import pdb; pdb.set_trace()
            n = nn[0]
        else:
            n = nn
        nvec.append(n)
    return np.mean(nvec)/len(states)


             
