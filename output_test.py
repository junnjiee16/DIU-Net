import torch
from diunet.inception import InceptionResBlock, WideInceptionResBlock
from diunet.dense_inception import DenseInceptionBlock
from diunet.sampling_blocks import DownsamplingBlock, UpsamplingBlock
from torchview import draw_graph

model = InceptionResBlock(64, 128)
architecture = "InceptionResBlock"
model_graph = draw_graph(
    model,
    input_size=(1, 64, 28, 28),
    graph_dir="TB",
    roll=True,
    expand_nested=True,
    graph_name=f"self_{architecture}",
    save_graph=True,
    directory="assets/self",
    filename=f"self_{architecture}",
)


model = WideInceptionResBlock(64, 128)
architecture = "WideInceptionResBlock"
model_graph = draw_graph(
    model,
    input_size=(1, 64, 28, 28),
    graph_dir="TB",
    roll=True,
    expand_nested=True,
    graph_name=f"self_{architecture}",
    save_graph=True,
    directory="assets/self",
    filename=f"self_{architecture}",
)


model = DenseInceptionBlock(64, 128, 4)
architecture = "DenseInceptionBlock"
model_graph = draw_graph(
    model,
    input_size=(1, 64, 28, 28),
    graph_dir="TB",
    roll=True,
    expand_nested=True,
    graph_name=f"self_{architecture}",
    save_graph=True,
    directory="assets/self",
    filename=f"self_{architecture}",
)


model = DownsamplingBlock(64)
architecture = "DownsamplingBlock"
model_graph = draw_graph(
    model,
    input_size=(1, 64, 28, 28),
    graph_dir="TB",
    roll=True,
    expand_nested=True,
    graph_name=f"self_{architecture}",
    save_graph=True,
    directory="assets/self",
    filename=f"self_{architecture}",
)

model = UpsamplingBlock(64)
architecture = "UpsamplingBlock"
model_graph = draw_graph(
    model,
    input_size=(1, 64, 28, 28),
    graph_dir="TB",
    roll=True,
    expand_nested=True,
    graph_name=f"self_{architecture}",
    save_graph=True,
    directory="assets/self",
    filename=f"self_{architecture}",
)
