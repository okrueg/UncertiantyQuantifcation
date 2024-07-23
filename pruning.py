import torch
import numpy as np
from torch.nn.utils import prune
import plotly.express as px
from bayesArchetectures import BNN
import bayesUtils
from datasets import loadData

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

class PruneByMU():
    def __init__(self):
        pass

    def __call__(self, amount, mu_tensor, rho_tensor):
        return self.calc_threshhold(amount, mu_tensor, rho_tensor)


    def get_scores(self, mu_tensor: torch.Tensor, rho_tensor: torch.Tensor) -> torch.Tensor:

        return torch.abs(mu_tensor)


    def calc_threshhold(self, amount: float, mu_tensor: torch.Tensor, rho_tensor: torch.Tensor) -> int:

        abs_score = self.get_scores(mu_tensor, None)

        num_elements = mu_tensor.numel()
        
        num_prune = int(amount * num_elements)
        
        topk_values, _ = torch.topk(abs_score.view(-1), num_prune, largest=False)

        threshold = topk_values.max()

        return threshold


class PruneByRho():
    def __init__(self):
        pass

    def __call__(self, amount, mu_tensor, rho_tensor):
        return self.calc_threshhold(amount, mu_tensor, rho_tensor)


    def get_scores(self, mu_tensor: torch.Tensor, rho_tensor: torch.Tensor) -> torch.Tensor:

        return rho_tensor


    def calc_threshhold(self, amount: float, mu_tensor: torch.Tensor, rho_tensor: torch.Tensor) -> int:


        num_elements = rho_tensor.numel()
        
        num_prune = int(amount * num_elements)
        
        topk_values, _ = torch.topk(rho_tensor.view(-1), num_prune, largest=True)

        threshold = topk_values.min()

        return threshold


class PruneByHyper():

    @staticmethod
    def minMaxNorm(t):

        minimum = torch.min(t)
        maximum = torch.max(t)

        return (t - minimum) / (maximum - minimum)


    def __init__(self, mu_weight= 0.75):
        super().__init__()
        #Check To see if mu weight is valid
        assert (mu_weight <= 1 and mu_weight >= 0)
        print(f'mu_weight: {mu_weight}')

        self.mu_weight = mu_weight


    def __call__(self, amount, mu_tensor, rho_tensor):
        return self.calc_threshhold(amount, mu_tensor, rho_tensor)


    def get_scores(self, mu_tensor: torch.Tensor, rho_tensor: torch.Tensor):
        mu_abs = torch.abs(mu_tensor)

        mu_norm = PruneByHyper.minMaxNorm(mu_abs)

        mu_scores = mu_norm * self.mu_weight

        rho_norm = PruneByHyper.minMaxNorm(rho_tensor)

        rho_scores = 1 - (rho_norm * (1- self.mu_weight))

        scores = mu_scores + rho_scores

        return scores


    def calc_threshhold(self, amount: float, mu_tensor: torch.Tensor, rho_tensor: torch.Tensor) -> int:

        hyper_scores = self.get_scores(mu_tensor, rho_tensor)

        num_elements = mu_tensor.numel()
        
        num_prune = int(amount * num_elements)
        
        topk_values, _ = torch.topk(hyper_scores, num_prune, largest=False)

        threshold = topk_values.max()

        return threshold


class PruneByKL():

    @staticmethod
    def indv_kl(mu_q, sigma_q, mu_p, sigma_p):

        #https://github.com/IntelLabs/bayesian-torch/blob/main/bayesian_torch/layers/base_variational_layer.py#L53

        kl = torch.log(sigma_p) - torch.log(
            sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 *
                                                          (sigma_p**2)) - 0.5
        return kl


    @staticmethod
    def sigma_calc(rho_tensor):
        return torch.log1p(torch.exp(rho_tensor))


    def __init__(self):
        pass


    def __call__(self, amount, mu_tensor, rho_tensor):
        return self.calc_threshhold(amount, mu_tensor, rho_tensor)


    def get_scores(self, mu_tensor: torch.Tensor, rho_tensor: torch.Tensor) -> torch.Tensor:
        sigma_tensor = PruneByKL.sigma_calc(rho_tensor)

        mu_prior = torch.zeros_like(mu_tensor)

        sigma_prior = torch.ones_like(rho_tensor)

        kl_scores = PruneByKL.indv_kl(mu_tensor, sigma_tensor, mu_prior, sigma_prior )

        return kl_scores


    def calc_threshhold(self, amount: float, mu_tensor: torch.Tensor, rho_tensor: torch.Tensor) -> int:

        kl_scores = self.get_scores(mu_tensor, rho_tensor)

        num_elements = mu_tensor.numel()
        
        num_prune = int(amount * num_elements)
        
        topk_values, _ = torch.topk(kl_scores, num_prune, largest=False)

        threshold = topk_values.max()

        return threshold


class BaysianPruning():
    def __init__(self, model: BNN):
        self.model = model


    def check_rho_zeros(self, all_rho):

        for module, name in all_rho:
            #print(model.)
            try:
                if torch.max(module.get_parameter(name)).item() > 0:
                    raise ValueError(f'There is a postive rho found: {torch.max(module.get_parameter(name)).item()}')
            except:
                for name, param in module.named_parameters():
                    print(name)
            

    def collect_all_parameters(self, collect_by: str):
        all_mu= []
        all_rho= []

        match collect_by:
            case 'mu':
                for module in self.model.modules():
                    if hasattr(module, 'mu_weight'):
                        all_mu.append((module, 'mu_weight'))
                        all_rho.append((module, 'rho_weight'))

                    elif hasattr(module, 'mu_kernel'):
                        all_mu.append((module, 'mu_kernel'))
                        all_rho.append((module, 'rho_kernel'))
            
            case 'rho':
                for module in self.model.modules():
                    if hasattr(module, 'rho_weight'):
                        all_rho.append((module, 'rho_weight'))
                        all_mu.append((module, 'mu_weight'))
                        
                    elif hasattr(module, 'rho_kernel'):
                        all_rho.append((module, 'rho_kernel'))
                        all_mu.append((module, 'mu_kernel'))
            
            case _:
                raise ValueError(f'Invalid collection type of {collect_by}')

    # Test collecting just 1 layers  
        # all_mu.clear()
        # all_mu.append((self.model.fc1, 'mu_weight'))

        # all_rho.clear()
        # all_rho.append((self.model.fc1, 'rho_weight'))
        # print(all_mu, all_rho)
        return all_mu, all_rho
        

    def global_just_mu_prune(self, amount: float):

        all_mu, _ = self.collect_all_parameters('mu')

        prune.global_unstructured(
            all_mu,
            pruning_method=prune.L1Unstructured,
            amount=amount)
        
    
    def global_just_rho_prune(self, amount: float):
        _, all_rho = self.collect_all_parameters('rho')

        prune.global_unstructured(
            all_rho,
            pruning_method=prune.L1Unstructured,
            amount=amount)
        return


class GlobalUnstructuredPrune():

    class ThreshholdPrune(prune.BasePruningMethod):
        PRUNING_TYPE = 'unstructured'

        @classmethod
        def apply(cls, module, name: str, is_rho: bool, symbol_flip: bool, threshhold: int, scores: torch.Tensor | None):
            super().apply(module, name, importance_scores= None, is_rho=is_rho, symbol_flip=symbol_flip, threshhold=threshhold, scores=scores)



        def __init__(self, is_rho:bool, symbol_flip: bool, threshhold:int, scores: torch.Tensor | None):
            super().__init__()

            self.is_rho = is_rho
            self.symbol_flip = symbol_flip
            self.threshhold = threshhold
            self.scores = scores


        def compute_mask(self, t, default_mask):

            self.scores = t if self.scores is None else self.scores
            mask = default_mask.clone()
            # Filter Large Variences for prune by Rho
            if self.symbol_flip:
                if self.is_rho:
                    mask[self.scores >= self.threshhold] = torch.inf
                else:
                    mask[self.scores >= self.threshhold] = 0
            # For all other Cases
            else:
                if self.is_rho:
                    mask[self.scores <= self.threshhold] = torch.inf
                else:
                    mask[self.scores <= self.threshhold] = 0

            return mask
    

    def __init__(self, amount: float, pruner: BaysianPruning, method, **kwargs) -> None:
        assert (amount <= 1 and amount >= 0), "Prune range must be [0,1]"

        self.amount = amount

        self.mu_list, self.rho_list = pruner.collect_all_parameters('mu')

        self.method = method(**kwargs)

        self.threshhold = 0


    def collect_theshhold(self,):

        assert hasattr(self.method, 'calc_threshhold')

        mu_params_list = [ getattr(mu_param, name) for mu_param, name in self.mu_list]

        rho_params_list = [ getattr(rho_param, name) for rho_param, name in self.rho_list]

        mu_vector = torch.nn.utils.parameters_to_vector(mu_params_list)

        rho_vector = torch.nn.utils.parameters_to_vector(rho_params_list)

        self.threshhold = self.method(self.amount, mu_vector, rho_vector)

    def apply_to_params(self):
        print(f'Utilizing a threshhold of {self.threshhold} from {type(self.method).__name__}')

        for (module, mu_name), (_, rho_name) in zip(self.mu_list, self.rho_list):
            # Sanity Check, Should never trigger o_0
            assert module == _

            # Create Scores for threshhold, else just feed in mu
            if hasattr(self.method, 'get_scores'):
                scores = self.method.get_scores(module.get_parameter(mu_name), module.get_parameter(rho_name))
            else:
                scores = None

            symbol_flip = True if type(self.method).__name__ == 'PruneByRho' else False
                
            GlobalUnstructuredPrune.ThreshholdPrune.apply(module, mu_name, False, symbol_flip, self.threshhold, scores)
            GlobalUnstructuredPrune.ThreshholdPrune.apply(module, rho_name, True, symbol_flip, self.threshhold, scores)


def accuracyByPrune(prune_intervals:list, prune_method_list:list, test_loader, num_mc:int , model_path = None):
    x = np.array(prune_intervals)

    # Run Results for Just MU
    just_mus = []
    for interval in prune_intervals:
            model = torch.load(model_path) if model_path is not None else BNN(3)
            model.to(DEVICE)

            pruner = BaysianPruning(model)
            
            pruner.global_just_mu_prune(interval)

            _, accuracy, _ = bayesUtils.test_Bayes(model, test_loader, num_mc=10)

            just_mus.append(accuracy)

    fig = px.line(x=x, y=just_mus, template='plotly_dark')#, range_y=[0.5,1])
    # Cant add name trace ugh
    #'''
    for method in prune_method_list:
        method_accuracys = []
        for interval in prune_intervals:
            model = torch.load(model_path) if model_path is not None else BNN(3)
            model.to(DEVICE)

            pruner = BaysianPruning(model)

            glob = GlobalUnstructuredPrune(interval, pruner, method)

            glob.collect_theshhold()

            glob.apply_to_params()

            _, accuracy, _ = bayesUtils.test_Bayes(model, test_loader, num_mc=10)

            method_accuracys.append(accuracy)


        fig.add_scatter(x=x, y= method_accuracys, mode='lines')
        fig.data[-1].name = method.__name__
    #'''
    fig.update_layout(
                    showlegend=True,
                    xaxis_title="Prune Rate",
                    yaxis_title="Accuracy"
                    )
    fig.show()




model = torch.load('model_90_BNN.path')

#model = BNN(in_channels=3, in_feat= 32*32*3, out_feat= 10)

model.to(DEVICE)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)

# pruner = BaysianPruning(model)

# glob = GlobalUnstructuredPrune(0.5, pruner, PruneByMU)

# glob.collect_theshhold()

# glob.apply_to_params()



#pruner.global_mu_prune(0.90)

#pruner.global_rho_prune(0.50)

#-------- Test masks are the same-------
# print(model.fc1.mu_weight_mask)
# print()
# print(model.fc1.rho_weight_mask)

# x = torch.eq(model.fc1.mu_weight_mask, model.fc1.rho_weight_mask)

# print(torch.equal(model.fc1.mu_weight_mask, model.fc1.rho_weight_mask))
# print('% Same:', torch.flatten(x).count_nonzero() / torch.flatten(x).numel())

# print('values_where_mask_is_zero' , model.fc1.mu_weight_mask[x == 0])
#---------------------------------------

train_loader, val_loader, test_loader = loadData('CIFAR-10',batch_size= 200)
# bayesUtils.test_Bayes(model, test_loader, num_mc=10)

#accuracyByPrune([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9],[PruneByMU, PruneByRho, PruneByHyper, PruneByKL], test_loader=test_loader, num_mc=10, model_path='model_90_BNN.path')
#accuracyByPrune([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9],[PruneByHyper], test_loader=test_loader, num_mc=1, model_path='model_90_BNN.path')
