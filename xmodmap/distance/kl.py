import torch


class PiRegularizationSystem:
    """
        pi regularization (KL divergence)
    """
    def __init__(self, zeta_S, zeta_T, w_T, norm=True):
        self.zeta_S = zeta_S
        self.zeta_T = zeta_T
        self.w_T = w_T
        self.norm = norm

        nu_T = zeta_T * w_T

        self.nuTconst = torch.sum(nu_T)
        # two alternatives for distribution to compare to
        if not self.norm: # compare to overall distribution of features
            self.nu_Tprob = torch.sum(nu_T, dim=0) / self.nuTconst
        else: # compare to uniform distribution on features
            self.nu_Tprob = torch.ones((1, nu_T.shape[-1])) / nu_T.shape[-1]

        self.weight = 1.0 #TODO: put normalizing cste here ?1.0 / torch.log(torch.tensor(nu_T.shape[-1]))
    
    def Pi_ST_init(self, w_S):
        """
        This function computes the initial transfer function Pi_ST based on the input data Source and Target.

        Parameters:
        - zeta_S: Feature distribution for source data (tensor)
        - zeta_T: Feature distribution for target data (tensor)
        - w_T: Weights for target data (tensor)
        - w_S: Weights for source data (tensor)
        - Csqpi: Scaling constant for Pi_ST (default is 1.0)

        Returns:
        - pi_STinit: Initial value of transfer function Pi_ST (tensor)
        """
        # two alternatives (linked to variation of KL divergence norm that consider)
        pi_STinit = torch.zeros((self.zeta_S.shape[-1], self.zeta_T.shape[-1]))
        if not self.norm:
            # feature distribution of target scaled by total mass in source
            pi_STinit[:, :] = (self.zeta_T * self.w_T).sum(axis=0) / torch.sum(w_S)
        else:
            pi_STinit[:, :] = torch.ones((1, self.zeta_T.shape[-1])) / self.zeta_T.shape[-1]
            pi_STinit = pi_STinit *  torch.sum(self.w_T) / torch.sum(w_S)

        return torch.sqrt(pi_STinit)

    def get_params(self):
        params_dict = {
            "weight": self.weight,
            "norm": True,
            "nuTconst": self.nuTconst
        }
        return params_dict

    def __call__(self, qw, pi_est):

        mass_S = torch.sum(qw * self.zeta_S, dim=0)
        pi_ST = pi_est ** 2
        pi_S = torch.sum(pi_ST, dim=-1)
        pi_STprob = pi_ST / pi_S[..., None]
        numer = pi_STprob / self.nu_Tprob
        di = mass_S @ (pi_ST * (torch.log(numer)))
        return (1.0 / self.nuTconst) * di.sum()
