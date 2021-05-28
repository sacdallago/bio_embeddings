from pathlib import Path

import pandas
import plotly

from bio_embeddings.mutagenesis_visualize.pipeline import plot


def main():
    cwd = Path(__file__).resolve().parent
    probabilities = pandas.read_csv(cwd.joinpath("probabilities.csv"))
    plotly.offline.plot(plot(probabilities), filename=str(cwd.joinpath("plot.html")))


if __name__ == "__main__":
    main()
