# https://vizier.u-strasbg.fr/viz-bin/VizieR-3?-source=J/A%2bA/620/A172/ums&-out.max=50&-out.form=HTML%20Table&-out.add=_r&-out.add=_RAJ,_DEJ&-sort=_r&-oc.form=sexa
# skip until 95th row
import matplotlib.pyplot as plt
import argparse
from argparse import RawTextHelpFormatter
import pandas as pd
from collections import OrderedDict


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def get_data(filename):

    ra_lst,dec_lst = [],[]
    with open(filename,"r") as f:
        for line in f:
            
            first_isnum_bool = is_number(line[0])
            if first_isnum_bool:
                    
                    line     = line.split(";")
                    line[-1] = line[-1].strip("\n")
                    
                    last_isnum_bool = is_number(line[-1])
                    if last_isnum_bool:
                        ra_val  = float(line[0])
                        dec_val = float(line[1])
                        if ra_val < 125.:
                            ra_lst.append(ra_val)
                            dec_lst.append(dec_val)
                    
    return ra_lst, dec_lst

def create_dataframe(ra_arr,dec_arr,filename):
    df = pd.DataFrame(OrderedDict( (('RA',pd.Series(ra_arr)), ('DEC',pd.Series(dec_arr)) )))
    filename=filename.split(".tsv")[0]
    print(filename)
    df.to_csv("./%s_ra_dec_coords.csv" % filename, index=False)
    return df



if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("file", type=str,
                        help = "data to be evaluated: pre_main.tsv or upper_main.tsv")
    args     = parser.parse_args()
    filename = args.file


    ra_lst, dec_lst = get_data(filename)

    plt.figure(1)
    plt.scatter(ra_lst,dec_lst, s = 0.1)
    plt.title("%s Detections" % filename)
    plt.ylabel("DEC/deg")
    plt.xlabel("RA/deg")
    plt.savefig("%s_0_to_125_deg_RA.png" % filename)
    plt.show()

    df= create_dataframe(ra_lst,dec_lst,filename)
    print(df)

