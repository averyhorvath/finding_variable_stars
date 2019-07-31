import numpy as np
import glob
import pdb
import os.path
import math
import random
import argparse
import multiprocessing
from argparse import RawTextHelpFormatter
from functools import partial
import csv
import matplotlib
import pandas as pd

matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.style.use('ggplot')

DEBUG       = 0

MJD_2016    = 57388
MJD_2017    = 57754
MJD_2018    = 58119

EPOCH_NUM         = int(3*525600/2)
TEN_HOUR_PERIOD   = 60*10.0
ECLIPSE_LENGTH    = 2.0*60.0
TWENTY_FOUR_HOURS = 24*60.0

PERIOD_SIM        = int(10*60)

median_mags = []
stds        = []

def manipulate_data(data):
    epoch   = data[:,0]
    mag     = data[:,1]
    mag_err = data[:,2]
    flag    = data[:,3]

    CONDITION_NOT_NAN = np.array(np.where((epoch == epoch) & (mag == mag) & (mag_err == mag_err) & (flag == flag))).flatten()
    epoch   = epoch[CONDITION_NOT_NAN]
    mag     = mag[CONDITION_NOT_NAN]
    mag_err = mag_err[CONDITION_NOT_NAN]
    flag    = flag[CONDITION_NOT_NAN]

    epoch = epoch[flag == 0]
    mag   = mag[flag == 0]
    

    DATE_LIMIT = np.array(np.where((epoch<MJD_2016) | (epoch>=MJD_2017))).flatten()
    epoch  = epoch[DATE_LIMIT]
    mag    = mag[DATE_LIMIT]
    mag_err= mag_err[DATE_LIMIT]
    flag   = flag[DATE_LIMIT]

    epoch = epoch - min(epoch)

    SIGMA_NUM      = 3
    CONDITION_CLIP = np.array(np.where(np.abs(mag-np.mean(mag)) < SIGMA_NUM * np.std(mag))).flatten()
    epoch          = epoch[CONDITION_CLIP]
    mag            = mag[CONDITION_CLIP]
    mag_err        = mag_err[CONDITION_CLIP]
    flag           = flag[CONDITION_CLIP]

    CONDITION_CLIP = np.array(np.where(np.abs(mag-np.mean(mag)) < SIGMA_NUM * np.std(mag))).flatten()
    epoch          = epoch[CONDITION_CLIP]
    mag            = mag[CONDITION_CLIP]
    mag_err        = mag_err[CONDITION_CLIP]
    flag           = flag[CONDITION_CLIP]


    median_mags.append(np.median(mag))
    stds.append(np.std(mag))

    return epoch, mag, mag_err, flag, median_mags, stds

def find_best_periods(period_arange,delta_mags,num):
    
    mags_sort_ind = np.argsort(delta_mags)
    test_period = period_arange[mags_sort_ind][0:num]

    return test_period

def supersmoother(epoch_arr, mag_arr,period=None,period_min=None,period_max=None,stepsize=None): 

    if period:
        "list of size 1 so it works with loop below"
        period_arange = np.arange(period,period+1,1) 
    if stepsize:
        period_arange = np.arange(period_min,period_max,stepsize)
    
    delta_mags    = []
    for period in period_arange:
        period *= TWENTY_FOUR_HOURS
        phased_epoch        = np.fmod(epoch_arr,period)/period
        phase_sorted_ind    = np.argsort(phased_epoch)
        mag_sorted          = mag_arr[phase_sorted_ind]
        phased_epoch_sorted = phased_epoch[phase_sorted_ind]
        delta_mags.append(np.mean(np.fabs(mag_sorted[1:] - mag_sorted[0:-1])))

    return period_arange, delta_mags, phased_epoch_sorted, mag_sorted

def BLS(epoch_arr,mag_arr,period_min=None,period_max=None,stepsize=None):
    
    if stepsize:
        period_arange = np.arange(period_min,period_max,stepsize) 
    
    period_arange = day_mask(period_arange, period_arange)

    mag_arr -= np.median(mag_arr)
    
    best_bins = []
    for period in period_arange:
        period *= TWENTY_FOUR_HOURS

        best_bin     = -1e9
        phased_epoch = np.fmod(epoch_arr,period)/(period) #fraction through phase 

        bin_width    = ECLIPSE_LENGTH / (period) # bin width
        num_bins     = int(1.0 / bin_width)

        #print("bin width = %s" % bin_width)
        #print("num bins = %s" % num_bins)
        for bin in range(num_bins):
            bin_phase_time_start = bin_width * bin
            bin_phase_time_end   = bin_width * (bin+1)

            bin_pts              = mag_arr[(phased_epoch >= bin_phase_time_start) & (phased_epoch < bin_phase_time_end)]

            if len(bin_pts) > 10:
               # Metric
                weighted_mean = np.sum(bin_pts)/(len(bin_pts)*(len(mag_arr)-len(bin_pts)))
                if weighted_mean > best_bin:
                    best_bin = weighted_mean
        
        best_bins.append(best_bin)
    best_bins = np.array(best_bins)

    return period_arange, best_bins

def create_lightcurve():
    DEBUG_SIM    = 1
    
    MIN_STEP     = 2
    STEADY_MAG   = 12
    MAG_DROP     = 0.5

    SNR_ARR      = [0.001,2,10]
    SNR_ARR      = [0.3]
    
    

    for snr in SNR_ARR:
        noise        = 0.5/snr
        epoch_arr    = np.arange(0,EPOCH_NUM,MIN_STEP)
        mag_arr      = np.ones(EPOCH_NUM // MIN_STEP)*STEADY_MAG
        
 
        for i in range(0,len(mag_arr),int(TEN_HOUR_PERIOD // MIN_STEP)):
            mag_arr[i:i + int(ECLIPSE_LENGTH // MIN_STEP)] += MAG_DROP
        
        mag_lst   = [value for value in mag_arr]
        epoch_lst = [value for value in epoch_arr]
        for i in range(0,len(mag_arr),int(TWENTY_FOUR_HOURS // MIN_STEP)):
            del mag_lst[i:i + int(TEN_HOUR_PERIOD // MIN_STEP)]
            del epoch_lst[i:i + int(TEN_HOUR_PERIOD // MIN_STEP)]

        if DEBUG_SIM:
            mag_arr = np.array(mag_lst)
            epoch_arr = np.array(epoch_lst)
            plt.plot(epoch_arr,mag_arr)
            plt.xlim(0,6000)
            plt.xlabel("Time (minutes)")
            plt.ylabel("Magnitude")
            plt.title("Magnitude Over 100 Hours Without Any Noise")
            #plt.savefig("./eclipse_magnitudes_day_night_without_noise.png")
            plt.show()

        mag_arr             = np.array([i + np.random.normal(scale = noise) for i in mag_arr])
        #mag_arr += np.random.normal(scale = noise)

    
        if DEBUG_SIM:
            period_arange, delta_mags, phased_epoch_sorted,mag_sorted = supersmoother(epoch_arr,mag_arr,period_min=0.2,period_max=10.0,stepsize=0.01)

            plt.scatter(phased_epoch_sorted,mag_sorted, marker='o',edgecolors='blue')
            plt.ylabel("Magnitude")
            plt.xlabel("Fraction Through Period")
            ax = plt.gca()
            ax.set_ylim(ax.get_ylim()[::-1])
            plt.title("Mock Data Light Curve with SNR = %s " % snr)
            #plt.savefig("./phased_up_ten_hour_period_%s_SNR.png" % snr)
            plt.show()
    

    return epoch_arr, mag_arr #, phased_epoch_sorted,mag_sorted

def delete_nans(arr_cut_nans, arr_to_shorten):
    arr_cut_nans   = np.array(arr_cut_nans)
    arr_to_shorten = np.array(arr_to_shorten)

    cond = np.array(np.where(arr_cut_nans==arr_cut_nans)).flatten()
    return arr_to_shorten[cond]

def direct_measurement(period_arange, BLS_bins):
    DIRECT_DEBUG = 0
    
    stepsize = period_arange[1]-period_arange[0]
    
    # Find max peak
    peaks_sort_ind = np.argsort(BLS_bins)
    max_peak_ind = peaks_sort_ind[-1]
    max_peak = BLS_bins[max_peak_ind]
  
    # Getting local region of bin values
    max_peak_period = period_arange[peaks_sort_ind][-1]

    # Isolating periods +/- 10% around maximum peak
    period_limit                = max_peak_period * 0.1
    period_arange_cut           = np.arange(max_peak_period - period_limit, max_peak_period + period_limit,stepsize)
    
    # Make number of periods even: Assure same number of periods are grabbed to the left and right of max peak
    if len(period_arange_cut) % 2 != 0:
        period_arange_cut = period_arange[(max_peak_ind - len(period_arange_cut)//2): max_peak_ind + len(period_arange_cut)//2]
    BLS_bins_cut = BLS_bins[(max_peak_ind - len(period_arange_cut)//2): max_peak_ind + len(period_arange_cut)//2]
    
    if DIRECT_DEBUG:
        print("len bins = ", len(BLS_bins_cut))
        print("len period = ", len(period_arange_cut))
        
    # If max peak is near beginning and cannot grab periods -10% of peak
    if (max_peak_ind - len(period_arange_cut)//2) < 0:
        period_arange_cut = period_arange[:max_peak_ind + len(period_arange_cut)//2]
        BLS_bins_cut      = BLS_bins[     :max_peak_ind + len(period_arange_cut)//2]

    BLS_bins_cut      = delete_nans(BLS_bins_cut,BLS_bins_cut)
    period_arange_cut = delete_nans(BLS_bins_cut, period_arange_cut)

    if DIRECT_DEBUG:
        print("len bins = ", len(BLS_bins_cut))
        print("len period = ", len(period_arange_cut))
        plt.plot(period_arange_cut,BLS_bins_cut)
        plt.title("Local Region Around Peak Extracted")
        plt.xlabel("Period/Days")
        plt.ylabel("Periodogram Power")
        plt.savefig("./local_region_direct_measure.png")
        plt.show()
    local_rms = np.sqrt(np.mean(BLS_bins_cut))
    local_avg = np.mean(BLS_bins_cut)
    

    peak_snr = (max_peak - local_avg)/local_rms
    return max_peak_period,peak_snr,period_arange_cut, BLS_bins_cut

def monte_carlo_simulations(epoch_arr, mag_arr):
    NUM_TRIALS    = 500
    PERIOD_MIN    = 0.133
    PERIOD_MAX    = 20
    STEPSIZE      = 0.01
    SNR_THRESHOLD = 0.05

    # Original Result for non-shuffled mag_arr
    sim_period_arange, sim_best_bins =  BLS(epoch_arr,mag_arr,period_min=PERIOD_MIN,period_max=PERIOD_MAX,stepsize=STEPSIZE)
    peak_snr = direct_measurement(sim_period_arange,sim_best_bins)

    reproduced_results     = []

    for i in range(NUM_TRIALS):
        mag_shuffled              = np.array(random.sample(list(mag_arr), len(mag_arr)))
        _, sim_best_bins_shuffled =  BLS(epoch_arr,mag_shuffled,period_min=PERIOD_MIN,period_max=PERIOD_MAX,stepsize=STEPSIZE)
        shuffle_peak_snr          = direct_measurement(sim_period_arange,sim_best_bins_shuffled)
        reproduced_results.append(shuffle_peak_snr)

        if i % 50 == 0:
            print("%s Complete" % (i/NUM_TRIALS * 100))
    plt.hist(reproduced_results)
    plt.show()

def flatten_BLS(period_arange, best_bins): 
    poly_coeffs = np.polyfit(period_arange, best_bins, 4)
    
    # Sigma Clipping
    rms = np.std(best_bins)                     # measure the standard deviation of the periodogram
                                                # this includes the slope, but we'll assume that's ok
    clip_mask =    best_bins <  rms * 3.0       # make a mask for the points we want to keep
    periods_polyfit = period_arange[clip_mask]  # apply the mask to periods
    pgram_polyfit = best_bins[clip_mask]        # and the periodogram

    polynomial_func = np.poly1d(poly_coeffs)
    y_fit_values = polynomial_func(period_arange)

    return y_fit_values

def day_mask(cond_arr, masking_arr):
    percentage_cut = 0.02
    masking_fracs  = [0.5,1.]
    for fraction in masking_fracs:
        remainder            = np.remainder(cond_arr,fraction)
        period_arange_masked = np.array(np.where((remainder < (1 - percentage_cut)) & (remainder > percentage_cut))).flatten()
        array_masked = masking_arr[period_arange_masked]
    return array_masked


def do_work(TEST_DATA,args,filepath):
    SAVE_FIG       = True
    PLOT_SHOW      = False
    SIMULATING     = False
    MEDIAN_MAGS    = False

    csv_params = {"Star": None,"RMS" : None, "Median Mag": None, "SS Best Metric" : None, "SNR at Peak": None, "Peak Period": None }

    if SIMULATING:
        BLS_METHOD     = False
        SIGMA_CLIPPING = False
        SUPERSMOOTHING = False
        MEDIAN_MAGS    = False

        period_min = 0.2
        period_max = 1.0
        stepsize   = 0.001
        simulated_epoch, simulated_mag  = create_lightcurve()
        
        #monte_carlo_simulations(simulated_epoch, simulated_mag)
        
        if SUPERSMOOTHING:
            _, sim_delta_mags,_,_ = supersmoother(simulated_epoch,simulated_mag,period_min=period_min,period_max=period_max,stepsize=stepsize)

            sim_test_period = find_best_periods(sim_period_arange,sim_delta_mags,1)
            sim_test_period = 10*60
            #pdb.set_trace()
            sim_phased_epoch=np.fmod(simulated_epoch,sim_test_period)/sim_test_period

            plt.figure()
            plt.scatter(sim_phased_epoch,simulated_mag)
            plt.title("Simulated Data \n Period = %s Days" % (sim_test_period/TWENTY_FOUR_HOURS))
            plt.xlabel("Phased Times")
            plt.ylabel("Magnitude")
            if SAVE_FIG:
                plt.savefig("./images/%s_imgs/%s/%s_phased_period_%s.png" % (TEST_DATA,img,img,test_period[0]))
            if PLOT_SHOW:
                plt.show()

            plt.plot(sim_period_arange, sim_delta_mags)
            plt.show()

        if BLS_METHOD:
            sim_period_arange, sim_best_bins =  BLS(simulated_epoch,simulated_mag,period_min=period_min,period_max=period_max,stepsize=stepsize)
            peak_snr = direct_measurement(sim_period_arange,sim_best_bins)
            print("Peak SNR = %s" % peak_snr)
            
            
            plt.plot(sim_period_arange,sim_best_bins)
        plt.xlabel("Period/Days")
        plt.ylabel("Periodogram Power")
        plt.title("Simulated LC with BLS and Supersmoother")
        if SAVE_FIG:
            plt.savefig("./images/simulated_LC_with_BLS")
        if PLOT_SHOW:
            plt.show()
    
    elif not SIMULATING:
        PERIOD_MIN      = 0.2
        PERIOD_MAX      = 10.0
        NUM_TEST_POINTS = 10000
        STEPSIZE        = (PERIOD_MAX - PERIOD_MIN)/NUM_TEST_POINTS

        data  = np.loadtxt(filepath)
        epoch_days, mag, mag_err, flag, median_mags, stds = manipulate_data(data)
        epoch = epoch_days * TWENTY_FOUR_HOURS
        
        csv_params["RMS"]        = np.std(mag)
        csv_params["Median Mag"] = np.median(mag)

        img = (filepath.split("/")[-1]).split(".txt")[0]
        print(img)
        """
        if len(img.split("_")) > 2:
        ra        = img.split('_')[-2]
        dec       = img.split('_')[-1]

        csv_params["RA"]  = ra
        csv_params["DEC"] = dec
        """

        if os.path.isdir(args.data):
            data_type = filepath.split('/')[-2]
        else:
            data_type = "test"
        
        csv_params['Star'] = img.split("_")[1]

     
        if not os.path.exists("./images/%s_imgs/%s" % (TEST_DATA,img)):
            os.makedirs("./images/%s_imgs/%s" % (TEST_DATA,img))

        if "supersmoother" in args.m:
            period_arange, delta_mags,_,_ = supersmoother(epoch,mag,period_min=PERIOD_MIN,period_max=PERIOD_MAX,stepsize=STEPSIZE)

            
            test_period                   = find_best_periods(period_arange,delta_mags,1)
            csv_params['SS Best Metric']  = test_period[0]

            phased_epoch=np.fmod(epoch,test_period)/test_period

            plt.figure()
            plt.scatter(phased_epoch,mag)
            plt.title("%s Phased Up Light Curve \n Period = %s" % (img,test_period))
            plt.xlabel("Phased Times")
            plt.ylabel("Magnitude")
            plt.ylim(plt.ylim()[::-1])
            if SAVE_FIG:
                plt.savefig("./images/%s_imgs/%s/%s_phased_period_%s.png" % (TEST_DATA,img,img,test_period[0]))
            if PLOT_SHOW:
                plt.show()
            
            plt.figure()
            plt.plot(period_arange, delta_mags)
            plt.xlabel('Period (Days)')
            plt.ylabel('Supersmoother Power')
            plt.ylim(plt.ylim()[::-1])
            plt.title('%s Supersmoother Method' % img)
            if SAVE_FIG:
                plt.savefig("./images/%s_imgs/%s/SS_%s.png" % (TEST_DATA,img,img))
            if PLOT_SHOW:
                plt.show()

        if "phased_up" in args.m:
            if args.tp:
                print(args.tp)
                test_period = float(args.tp[0])
                phased_epoch=np.fmod(epoch_days,test_period)/test_period
                print(test_period)
                plt.figure()
                plt.scatter(phased_epoch,mag, s=5)
                plt.title("%s Phased Up Light Curve \n Period = %s" % (img,test_period))
                plt.xlabel("Phased Times")
                plt.ylabel("Magnitude")
                if SAVE_FIG:
                     plt.savefig("./images/%s_imgs/%s/%s_phased_period_%s.png" % (TEST_DATA,img,img,test_period))
                
                plt.show()
            else:
                test_periods                   = find_best_periods(period_arange,delta_mags,5)
                for test_period in test_periods:
                    phased_epoch=np.fmod(epoch_days,test_period)/test_period
                    print(test_period)
                    plt.figure()
                    plt.scatter(phased_epoch,mag, s=5, color = 'blue')
                    plt.title("%s Phased Up Light Curve \n Period = %s" % (img.split('_')[1],test_period))
                    plt.xlabel("Phased Times")
                    plt.ylabel("Magnitude")
                    if SAVE_FIG:
                        plt.savefig("./images/%s_imgs/%s/%s_phased_period_%s.png" % (TEST_DATA,img,img,test_period))
                    
                    plt.show()
        if MEDIAN_MAGS:
            plt.scatter(median_mags,stds)
            plt.xlabel('Median Magnitudes')
            plt.ylabel('Standard Deviation Magnitudes')
            plt.savefig('median_mags_std.png')
            plt.legend()
            if PLOT_SHOW:
                plt.show()
        
        
        if "bls" in args.m:    
            period_arange, best_bins = BLS(epoch,mag,period_min=PERIOD_MIN,period_max=PERIOD_MAX,stepsize=STEPSIZE)
            

            y_fit_values = flatten_BLS(period_arange, best_bins)

            best_bins_flat = best_bins - y_fit_values

            if args.snr is not None and "direct" in args.snr:
                peak_period,peak_snr,period_arange_cut, best_bins_flat_cut = direct_measurement(period_arange,best_bins_flat)
                csv_params["Peak Period"] = peak_period
                csv_params["SNR at Peak"] = peak_snr

            plt.figure()
            plt.plot(period_arange,best_bins_flat)
            plt.xlabel("Period")
            plt.ylabel("Maximum Average Bin Value (Dimmest Magnitude)")
            plt.title("%s \n Using BLS" % img)
            if SAVE_FIG:
                plt.savefig("./images/%s_imgs/%s/BLS_%s.png" % (TEST_DATA,img,img))
            if PLOT_SHOW:
                plt.show()
    
    csv_name = "./csv_files/%s_data.csv" % data_type
    outfile  = open(csv_name, "a")
    w        = csv.DictWriter(outfile,csv_params.keys())
    w.writerow(csv_params)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("data", type=str,
                            help = "directory to data .txt files or single .txt file to be evaluated: ex) bls_test_lc.txt or evr_samples")
    parser.add_argument("-m", type=str, nargs = "+",
                            help = "method, ex) '-m supersmoother bls' or just '-m supersmoother' or '-m bls'  ")
    parser.add_argument("-snr", type=str, nargs = "+", action ='store',
                            help = "method, ex) '-snr direct mc' or just '-m direct' or '-m mc'  ")
    parser.add_argument("-tp", type=str, nargs = "+", action ='store',
                            help = "test period to phase up light curve")
    args   = parser.parse_args()

    if not os.path.isdir(args.data):
        filelist = glob.glob('%s' % args.data)
        print(filelist)
        TEST_DATA = (filelist[0].split('.txt')[0]).split('/')[-1]
        data_type = "test"
        filename  = filelist[0].split('/')[-1]
        
        if filename   == "bls_test_lc.txt" or filename == "supersmoother_test_lc.txt":
            TEST_DATA = "debug"
            data_type = filename.split('_')[0]
    else:
        data = args.data
        filelist  = glob.glob('%s/*.txt' % args.data)
        print(filelist[0:5])
        TEST_DATA  = filelist[0].split('/')[-2]
        data_type  = data.split('/')[-1]
    print("data_type", data_type)
    
    csv_file = "./csv_files/%s_data.csv" % data_type
    headers  = ["Star","RMS", "Median Mag", "SS Best Metric", "SNR at Peak", "Peak Period"]
    with open(csv_file, "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(headers)
  

    if DEBUG:
        for filepath in filelist:
            if not os.path.isdir(args.data):
                do_work(TEST_DATA,args,filepath)
            else:
                ra = np.float((filepath.split('/')[-1]).split('_')[2])
                if ra < 125.:
                    do_work(TEST_DATA,args,filepath)
    else:
        pool = multiprocessing.Pool(32)
        input_list = []
        for filepath in filelist:
            if not os.path.isdir(args.data):
                input_list.append([TEST_DATA,args,filepath])
            else:
                ra = np.float((filepath.split('/')[-1]).split('_')[2])
                if ra < 125.:
                    input_list.append([TEST_DATA,args,filepath])
                

        pool.starmap(do_work,input_list)
        pool.close()
        pool.join()

