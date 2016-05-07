import numpy as np
import matplotlib.pyplot as plt

def comparisons_matched_cdf(values, values2=None, values3=None, values4=None):
    fig = plt.figure(figsize=(6,3))

    plt.xlabel("Fraction Matched")
    plt.ylabel("CCDF of Origins")

    plt.ylim(0, 1.1)
    plt.xlim(0, 1)

    sorted_values = np.sort(values)
    yvals = 1 - np.arange(len(sorted_values)) / float(len(sorted_values))

    if values2:
        sorted_2 = np.sort(values2)
        yvals2 = 1 - np.arange(len(sorted_2)) / float(len(sorted_2))

    if values3:
        sorted_3 = np.sort(values3)
        yvals3 = 1 - np.arange(len(sorted_3)) / float(len(sorted_3))

    if values4:
        sorted_4 = np.sort(values4)
        yvals4 = 1 - np.arange(len(sorted_4)) / float(len(sorted_4))


    #plt.axvline(.5, color='k', ls='--')
    #plt.axhline(.5, color='k', ls='--')
    med = np.median(values4)
    #plt.axvline(med, color='k', ls=':')
    print med

    plt.plot([0] + list(sorted_values), [1] +list(yvals), lw=2, ls=':',
    color='k', label="All")
    if values2:
        plt.plot([0] + list(sorted_2), [1] +list(yvals2), lw=2, ls="-.",
                 color='k', label="$\delta$>25")

    if values3:
        plt.plot([0] + list(sorted_3), [1] +list(yvals3), lw=2, ls="--",
                 color='k', label="$\delta$>50")

    if values4:
        plt.plot([0] + list(sorted_4), [1] +list(yvals4), lw=2, ls="-",
                 color='k', label="$\delta$>100")


    plt.legend(loc='lower left', frameon=False, numpoints=2)

    plt.savefig("../plots/fraction_matched.eps", bbox_inches='tight')
    plt.show()


def correlations_cdf(values, values2=None, values3=None, values4=None):
    fig = plt.figure(figsize=(6,3))

    plt.xlabel("Correlation Coefficient")
    plt.ylabel("CCDF of Origins")

    plt.ylim(0, 1.1)
    plt.xlim(-1, 1)

    sorted_values = np.sort(values)
    yvals = np.arange(len(sorted_values)) / float(len(sorted_values))

    if values2:
        sorted_2 = np.sort(values2)
        yvals2 = np.arange(len(sorted_2)) / float(len(sorted_2))

    if values3:
        sorted_3 = np.sort(values3)
        yvals3 = np.arange(len(sorted_3)) / float(len(sorted_3))

    if values4:
        sorted_4 = np.sort(values4)
        yvals4 = np.arange(len(sorted_4)) / float(len(sorted_4))


    plt.plot(sorted_values, yvals, lw=2, color='k', ls="-", label="All")

    #if values2:
    #    plt.plot(list(sorted_2), list(yvals2), lw=2, ls="-.",
    #             color='k', label="$\delta$>25")

    #if values3:
    #    plt.plot(list(sorted_3), list(yvals3), lw=2, ls="--",
    #             color='k', label="$\delta$>50")

    #if values4:
    #    plt.plot(list(sorted_4), list(yvals4), lw=2, ls="-",
    #             color='k', label="$\delta$>100")


    #plt.legend(loc='upper left', frameon=False, numpoints=2)


    plt.savefig("../plots/correlation.eps", bbox_inches='tight')
    plt.show()


def total_rank_change_cdf(values, val2):
    fig = plt.figure(figsize=(6,3))

    plt.xlabel("Normalized Rank Change")
    plt.ylabel("CCDF of Chains")

    plt.ylim(0, 1.1)
    plt.xlim(-0.05, 1)

    sorted_values = np.sort(values)
    yvals = 1 - np.arange(len(sorted_values)) / float(len(sorted_values))

    rand_values = np.sort(val2)
    rand_yvals = 1- np.arange(len(rand_values)) / float(len(rand_values))


    plt.plot(list(sorted_values), list(yvals), lw=2, color='k', label="Fury Route")
    plt.plot(list(rand_values), list(rand_yvals), lw=2, color='k',ls='--', label="Noise")
    plt.legend(loc='lower right', frameon=False, numpoints=2)
    plt.savefig("../plots/rank_change.png", bbox_inches='tight')
    plt.show()


def completion_cdf(values):
    fig = plt.figure(figsize=(6,3))
   
    plt.xlabel("Fraction of Complete Chains")
    plt.ylabel("CCDF of Origins")
   
    plt.ylim(0, 1.1)
    plt.xlim(0, 1.05)

    sorted_values = np.sort(values)
    yvals = 1 - np.arange(len(sorted_values)) / float(len(sorted_values))
    plt.plot([0] + list(sorted_values) + [1], [1] + list(yvals) + [0], lw=2, color='k')
    plt.savefig("../plots/completion.eps", bbox_inches='tight')
    plt.show()


def query_cdf(values):
    fig = plt.figure(figsize=(6,3))
  

    print "MED:", np.median(values)
    print "MAX:", max(values)
    print "Min:", min(values)

    plt.xlabel("Total Number of Queries")
    plt.ylabel("CDF of Chains")
   
    plt.ylim(0, 1.1)
    plt.xlim(0, 900)

    sorted_values = np.sort(values)
    yvals = np.arange(len(sorted_values)) / float(len(sorted_values))
    plt.plot(sorted_values, yvals, lw=2, color='k')
    plt.savefig("../plots/query_overhead.eps", bbox_inches='tight')
    plt.show()

#def query_cache(yvals, error_bars):
def query_cache(yvals, error_bars):
    
    #yvals = [511, 728, 441, 610, 135, 618, 4, 297, 441, 133, 2193, 621, 440, 4, 196] #ucla 
    #yvals2 = [390, 420, 323, 612, 4, 563, 550, 287, 364, 4, 2095, 483, 560, 105, 0] # rutgers
    #yvals3 = [471, 385, 322, 750, 4, 567, 532, 365, 591, 4, 2033, 553, 553, 91, 0] # eth
    
    xvals = range(len(yvals))
   
  
    yvals = np.asarray(yvals)
    error_band = np.asarray(error_bars)
    
    fig = plt.figure(figsize=(6,3))
   
    plt.xlabel("Query Index")
    plt.ylabel("Number of Queries")
   
    plt.ylim(0, 350)
    plt.xlim(-1, 100)

    plt.plot(xvals, yvals, lw=2, color='k')
    #plt.plot(xvals, yvals2, lw=2, color='k', ls="--")
    #plt.plot(xvals, yvals3, lw=2, color='k', ls=":")
    plt.errorbar(xvals, yvals, yerr=error_bars, color='k', lw=2, errorevery=4, elinewidth=.75)
    #plt.fill_between(xvals, yvals - error_band, yvals + error_band,
    #                 facecolor='lightgray', lw=0)

    #plt.plot(xvals, yvals - error_band, lw=1, ls="--", color='k')
    #plt.plot(xvals, yvals + error_band, lw=1, ls="--", color='k')


    plt.savefig("../plots/query_cache.eps", bbox_inches='tight')
    plt.show()
   
def absolute_plot(ping, fr):
  
    fig = plt.figure(figsize=(6,3))


    xvals = range(len(ping))
   
    plt.xlabel("Sorted Pair Index")
    plt.ylabel("Normalized Distance")
   
    #plt.ylim(0, 1.1)
    #plt.xlim(0, 1)

    #sorted_ping = np.sort(ping)
    #ping_yvals = np.arange(len(sorted_ping)) / float(len(sorted_ping))
    #plt.plot(sorted_ping, ping_yvals, lw=2, color='k', ls="--", label="RTT")
    plt.plot(xvals, ping, lw=2, color='k', ls="--", label="RTT")


    #fr_values = np.sort(fr)
    #fr_yvals = np.arange(len(fr_values)) / float(len(fr_values))
    #plt.plot(fr_values, fr_yvals, lw=2, color='k', ls=":")
    plt.plot(xvals, fr, lw=2, color='k', ls=":", label="Fury Route")

    plt.legend(loc='lower right', frameon=False, numpoints=2)
  
    plt.savefig("../plots/absolute.eps", bbox_inches='tight')
    plt.show()

def sensitivity_plot(yvals, error_bars):
  
    fig = plt.figure(figsize=(6,3))

    xvals = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
  

    yvals = np.asarray(yvals)
    error_band = np.asarray(error_bars)

    plt.xlabel("Approximate RTT Difference")
    plt.ylabel("Fraction Correct")
   
    plt.ylim(0, 1.1)
    plt.xlim(20, 200)

    plt.errorbar(xvals, yvals, yerr=error_bars, color='k', lw=2)
    plt.fill_between(xvals, yvals - error_band, yvals + error_band,
                     facecolor='lightgray', lw=0)

    plt.plot(xvals, yvals - error_band, lw=1, ls="--", color='k')
    plt.plot(xvals, yvals + error_band, lw=1, ls="--", color='k')


    #plt.legend(loc='lower right', frameon=False, numpoints=2)
  
    plt.savefig("../plots/sensitivity.eps", bbox_inches='tight')
    plt.show()


def overtime_plot(yvals, error_bars):
#def overtime_plot(yvals, error_bars_low, error_bars_high):
  
    fig = plt.figure(figsize=(6,3))

    xvals = range(len(yvals))
    yvals = np.asarray(yvals)
    
    error_band = np.asarray(error_bars)
   
    #error_band_low = np.asarray(error_bars_low)
    #error_band_high = np.asarray(error_bars_high)
    #relative_low = yvals - error_band_low
    #relative_high = error_band_high - yvals
    #error = [relative_low, relative_high]

    plt.xlabel("Hour")
    plt.ylabel("Mean Relative Length")
   
    plt.ylim(0.0, 2)
    #plt.xlim(20, 200)

    plt.errorbar(xvals, yvals, yerr=error_bars, color='k', lw=2)
    plt.fill_between(xvals, yvals - error_band, yvals + error_band,
                     facecolor='lightgray', lw=0)
    plt.plot(xvals, yvals - error_band, lw=1, ls="--", color='k')
    plt.plot(xvals, yvals + error_band, lw=1, ls="--", color='k')

    #plt.errorbar(xvals, yvals, yerr=error, color='k', lw=2)
    ###plt.plot(xvals, yvals, color='k', lw=2)
    #plt.fill_between(xvals, error_band_low, error_band_high,
    #                 facecolor='lightgray', lw=0)
    #plt.plot(xvals, error_band_low, lw=1, ls="--", color='k')
    #plt.plot(xvals, error_band_high, lw=1, ls="--", color='k')
    

    #plt.legend(loc='lower right', frameon=False, numpoints=2)
  
    plt.savefig("../plots/overtime.eps", bbox_inches='tight')
    plt.show()

