import torch
from models.vcn_8 import VCN8

if __name__ == "__main__":
    shapes = [(1, 3, 112, 112), (1, 3, 945, 673),
              (1, 3, 448, 448), (1, 3, 567, 345), (1, 3, 452, 224)]

    model = VCN8()
    for shape in shapes:
        inp = torch.rand(shape)
        out = model(inp)

        assert out.shape[2:] == inp.shape[2:]
