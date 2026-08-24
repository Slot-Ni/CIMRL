import numpy as np
import os
from pathlib import Path


data = "Algebra"
# data = "Bars-Rev"
# data = "Restaurants-Rev"
# data = "Geometry"
# data = "Music-Rev"







PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_GRAPH_DIR = PROJECT_ROOT / "HyperGraphMatrix"
LEGACY_GRAPH_DIR = Path(r"D:\study\Papers\data\HyperGraph")


def _resolve_graph_path(filename):
    local_npy = LOCAL_GRAPH_DIR / f"{filename}.npy"
    if local_npy.exists():
        return local_npy

    local_txt = LOCAL_GRAPH_DIR / f"{filename}.txt"
    if local_txt.exists():
        return local_txt

    return LEGACY_GRAPH_DIR / f"{filename}.txt"


def get_Graph(filename):
    graph_path = _resolve_graph_path(filename)
    if graph_path.suffix == ".npy":
        G_matrix = np.load(graph_path, allow_pickle=True)
        nodelist_path = graph_path.with_name(f"{filename}_nodelist.npy")
        if nodelist_path.exists():
            NodeList = np.load(nodelist_path, allow_pickle=True).tolist()
            return G_matrix, NodeList
        return G_matrix, list(range(G_matrix.shape[0]))

    with graph_path.open("r", encoding="utf-8") as f:
        arrayOfLines = f.readlines()
    HyperEdges = []
    for line in arrayOfLines:
        line = line.strip().split(' ')
        array = []
        for i in line:
            array.append(int(i))
        HyperEdges.append(array)

    NodeList = list(set(sum(HyperEdges, [])))

    #矩阵的行代表节点，列代表超边
    G_matrix = np.zeros([len(NodeList), len(HyperEdges)])

    for i in range(len(HyperEdges)):
        for j in HyperEdges[i]:
            G_matrix[NodeList.index(j)][i] = 1

    return G_matrix, NodeList

if __name__ == "__main__":
    G_matrix, NodeList = get_Graph(data)

