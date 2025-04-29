import torch.cuda.graphs as cg

def capture_graph(model_fn, x):
    g = torch.cuda.CUDAGraph()
    static_out = torch.empty_like(x)

    with torch.cuda.graph(g):
        static_out.copy_(model_fn(x))

    return g, static_out
