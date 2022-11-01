import argparse

from utils.network_graph import save_cell_dag_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='PATH', type=str, help="save folder path", required=True)
    parser.add_argument('-spec', metavar='JSON_PATH', type=str, required=True)
    args = parser.parse_args()

    save_cell_dag_image(args.spec, args.p)
