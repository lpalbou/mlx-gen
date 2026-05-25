from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE


class ErnieImageVAE(Flux2VAE):
    def __init__(self):
        super().__init__()
        self.bn.eps = 1e-5
