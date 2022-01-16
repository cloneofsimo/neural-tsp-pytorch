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


if __name__ == "__main__":
    test_state_tf()
