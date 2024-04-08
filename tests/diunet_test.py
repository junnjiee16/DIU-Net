from diunet import DIUNet
from torchview import draw_graph

model = DIUNet(0.25, 0.25)
architecture = "DIU-Net"
model_graph = draw_graph(
    model,
    input_size=(1, 1, 512, 512),
    graph_dir="TB",
    roll=True,
    expand_nested=True,
    graph_name=f"self_{architecture}",
    save_graph=True,
    directory="assets/self",
    filename=f"self_{architecture}",
)
