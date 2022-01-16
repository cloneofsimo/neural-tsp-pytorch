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
    from utils import state_to_pyg_data, batching_behavior, g_argmax, g_gather_index
    import torch
    from torch_geometric import seed_everything

    seed_everything(0)

    model = GraphEdgeConvEmb(
        hidden_channels=31,
        input_vert_channels=2,
        # input_vert_n_vocab=4,
    )

    env = TspEnv(n_nodes=5, dim=2, graph_type="ba")
    state = env.reset(5)
    states = []
    torch.no_grad()
    for i in range(5):

        env.render("./tmp_rnder")

        pyg_data1 = state_to_pyg_data(state)

        pyg_batched = batching_behavior([pyg_data1])
        y1 = model.forward(
            x=pyg_batched.x,
            edge_index=pyg_batched.edge_index,
            # x_emb=pyg_batched.x_occ,
            batch=pyg_batched.batch,
        )
        print("GCN Result", y1)
        print("batchidx, occ", pyg_batched.batch, pyg_batched.x_occ)
        act = g_argmax(y1, pyg_batched.batch)
        print("Chosen action", act)
        states.append(pyg_data1)
        state, reward, done, info = env.step(act.item())

    pyg_data_all = batching_behavior(states)

    print(
        pyg_data_all.x, pyg_data_all.edge_index, pyg_data_all.x_occ, pyg_data_all.batch
    )

    y = model.forward(
        x=pyg_data_all.x,
        edge_index=pyg_data_all.edge_index,
        # x_emb=pyg_data_all.x_occ,
        batch=pyg_data_all.batch,
    )

    print("Total Y", y)
    print(pyg_data_all.batch)

    batch_argmax = g_argmax(y, pyg_data_all.batch)
    print(batch_argmax)

    batch_gather_index = g_gather_index(y, pyg_data_all.batch, batch_argmax)
    print(batch_gather_index)

    print(y.shape)


if __name__ == "__main__":
    test_gcn_state()
