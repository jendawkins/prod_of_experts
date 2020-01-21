from spline_learner_poe4d_MM import *
import sys, getopt
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-MM", "--use_mm", help="use Michaelis Menten",type=int)
    parser.add_argument("-a","--aval",help = "a value",type=float)
    parser.add_argument("-b","--bval", help = "b value",type=float)
    parser.add_argument("-nb", "--num_bact", help="number OTUS", type=int)
    parser.add_argument("-mv", "--mvar", help="meas var", type=float)
    parser.add_argument("-pv", "--pvar", help="proc var", type=float)
    parser.add_argument("-poe", "--poe_var", help="poe var", type=float)
    parser.add_argument("-no", "--nsamps", help="number of sample", type=int)
    parser.add_argument("-ns", "--nstates",
                        help="number of states", type=int)
    parser.add_argument("-dt", "--delta_t", help="delta t", type=float)
    parser.add_argument("-gr", "--growth_rate", help="growth rate", type=float)
    parser.add_argument("-av", "--avar", help="a variance", type=float)
    parser.add_argument("-bv", "--bvar", help="b variance", type=float)
    parser.add_argument("-tv", "--theta_var",
                        help="theta variance", type=float)
    
    parser.add_argument("-o", "--outdir", help="out directory", type=str)

    args = parser.parse_args()

    curr = os.getcwd()
    if args.outdir == None:
        print('Specifiy Outdir')
        sys.exit(0)
    elif not os.path.exists(curr  + '/' + args.outdir):
        os.mkdir(curr + '/' + args.outdir)
    
    spl = SplineLearnerPOE_4D(use_mm=args.use_mm, a=args.aval, b=args.bval, num_bact=args.num_bact,
                        MEAS_VAR=args.mvar, PROC_VAR=args.pvar, THETA_VAR=args.theta_var, AVAR=args.avar,
                        BVAR=args.bvar, POE_VAR=args.poe_var, NSAMPS=args.nsamps, NPTSPERSAMP=args.nstates,
                        DT=args.delta_t,outdir = args.outdir)
    spl.run(gibbs_steps=500)

if __name__ == "__main__":
   main()
