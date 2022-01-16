def test_state_tf():

    from tsp_env import TspEnv, heuristic
    from utils import state_to_pyg_data

    env = TspEnv(n_nodes=20, dim=2, graph_type="ba")
    state = env.reset(1)

    for i in range(5):
        action = heuristic(state)
        state, reward, done, info = env.step(action)
        env.render("./tmp_rnder")
    print(state)
    pyg_data = state_to_pyg_data(state)

    # plot the graph

    print(pyg_data)

    from torch_geometric.utils import to_networkx

    G = to_networkx(pyg_data, to_undirected=True)

    import networkx as nx
    import matplotlib.pyplot as plt

    nx.draw(G, pos=pyg_data.pos, with_labels=True)

    # save
    plt.savefig("test_state_tf.png")


def test_gcn_state():
    from dqn.model import GraphEdgeConvEmb
    from tsp_env import TspEnv, heuristic
    from utils import state_to_pyg_data

    model = GraphEdgeConvEmb(
        hidden_channels=31,
        input_vert_channels=2,
        input_vert_n_vocab=4,
    )

    env = TspEnv(n_nodes=20, dim=2, graph_type="ba")
    state = env.reset(1)

    pyg_data = state_to_pyg_data(state)

    y = model.forward(
        x=pyg_data.x, edge_index=pyg_data.edge_index, x_emb=pyg_data.x_occ
    )

    print(y)


if __name__ == "__main__":
    test_gcn_state()
