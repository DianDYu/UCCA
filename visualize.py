import os
from argparse import ArgumentParser

from ucca import visualization, layer0
from ucca.ioutil import get_passages_with_progress_bar, external_write_mode

if __name__ == "__main__":
    argparser = ArgumentParser(description="Visualize the given passages as graphs.")
    argparser.add_argument("passages", nargs="+", help="UCCA passages, given as xml/pickle file names")
    argparser.add_argument("-t", "--tikz", action="store_true", help="print tikz code rather than showing plots")
    argparser.add_argument("-o", "--out-dir", help="directory to save figures in (otherwise displayed immediately)")
    argparser.add_argument("-i", "--node-ids", action="store_true", help="print tikz code rather than showing plots")
    argparser.add_argument("-f", "--format", choices=("png", "svg"), default="png", help="image format")
    args = argparser.parse_args()

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        if not args.tikz:
            import matplotlib
            matplotlib.use('Agg')
    for passage in get_passages_with_progress_bar(args.passages, desc="Visualizing"):
        if args.tikz:
            tikz = visualization.tikz(passage)
            if args.out_dir:
                with open(os.path.join(args.out_dir, passage.ID + ".tikz.txt"), "w") as f:
                    print(tikz, file=f)
            else:
                with external_write_mode():
                    print(tikz)
        else:
            import matplotlib.pyplot as plt
            width = len(passage.layer(layer0.LAYER_ID).all) * 19/27
            plt.figure(figsize=(width, width * 10/19))
            visualization.draw(passage, node_ids=args.node_ids)
            if args.out_dir:
                plt.savefig(os.path.join(args.out_dir, passage.ID + "." + args.format))
                plt.close()
            else:
                plt.show()
