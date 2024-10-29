import argparse
import os
from Clever.utils import *


parser = argparse.ArgumentParser(description='create the h5ad input file for Clever')
parser.add_argument("-data", type=str, help="path to the marker expression data", default=None)
parser.add_argument("-meta", type=str, help="path to the cell metadata", default=None)
parser.add_argument("-outdir", type=str, help="path to output file", default=None)
parser.add_argument("-transform", help="whether to apply arcsinh transformation or not", default='False', action='store_true')
parser.add_argument("-cofactor", type=int, help="cofactor for the arcsinh transformation", default=5)


def main():
    args = parser.parse_args()
    datafile = args.data
    metafile = args.meta
    
    if os.path.exists(args.outdir)==False:
        os.mkdir(args.outdir)

    pre, ext = os.path.splitext(datafile)
    dataname = pre.split('/')[-1]

    if ext == '.fcs':
        adata = read_fcs(path=datafile, meta_path=metafile, transform=args.transform, cofactor=args.cofactor)
    elif ext == '.csv':
        adata = read_CSV(path=datafile, meta_path=metafile, transform=args.transform, cofactor=args.cofactor)
    else:
        print(f"Unsupported file format: {ext}")

    adata.write_h5ad(args.outdir+dataname+'.h5ad')

if __name__ == "__main__":
    main() 