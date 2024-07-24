import torch
import numpy as np
from math import ceil
from torch.nn.utils import prune
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_rho
import plotly.express as px
import plotly.graph_objects as go
from bayesArchetectures import BNN, DNN
import bayesUtils
import model_utils
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
            

    def collect_all_parameters(self, collect_by: str, collect_bias= False):
        all_mu= []
        all_rho= []

        match collect_by:
            case 'mu':
                for module in self.model.children():
                    for name, parameter in module.named_parameters():
                        if ('mu' in name and collect_bias) or ('mu' in name and 'bias' not in name):
                            
                            all_mu.append((module, name))
                            all_rho.append((module, name.replace('mu', 'rho')))
            
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
        #print(all_mu, all_rho)

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


def prune_dnn(model: torch.nn.Module, amount: float):
    all_params = []
    for module in model.modules():
        if hasattr(module, 'weight'):
            all_params.append((module, 'weight'))

    prune.global_unstructured(
        all_params,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    for param in all_params:
        prune.remove(*param)


def raise_reapply_prune(model: torch.nn.Module, used_delta: float):

    def dnn_to_bnn_mu_mask(param: torch.nn.Parameter):
        mask = torch.ones_like(param)

        abs_weights = torch.abs(param)

        mask[abs_weights == 0] = 0

        return mask
   

    def dnn_to_bnn_rho_mask(param: torch.nn.Parameter):
        
        mask = torch.ones_like(param)

        abs_weights = torch.abs(param)

        mask[abs_weights == 0] = torch.inf

        return mask


    const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",
        "moped_enable": True,
        "moped_delta": used_delta,
    }

    dnn_to_bnn(model, const_bnn_prior_parameters)

    param_collector = BaysianPruning(model)

    mu_params, rho_params = param_collector.collect_all_parameters(collect_by='mu')

    for (module, mu_name), (_, rho_name) in zip(mu_params, rho_params):
        # Sanity Check, Should never trigger o_0
        assert module == _

        prune.custom_from_mask(module, mu_name, 
                               mask=dnn_to_bnn_mu_mask(getattr(module, mu_name)))
        
        prune.custom_from_mask(module, rho_name, 
                               mask=dnn_to_bnn_rho_mask(getattr(module, mu_name)))
        

def accuracy_by_prune(prune_intervals:list, prune_method_list:list, test_loader, num_mc:int , model_path = None):
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


    fig = go.Figure()
    fig.add_scatter(x=x, y=just_mus)
    fig.data[-1].name = 'Prune Only MU'


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

    fig.update_layout(
                showlegend=True,
                template='plotly_dark',
                xaxis_title="Prune Rate",
                yaxis_title="Accuracy",
                title = f"Survey of Bayesien Pruning Methods"
                )
    fig.show()


def compare_bnn_dnn(prune_intervals:list, test_loader, method, num_mc:int , dnn: DNN, bnn: BNN):
    x = np.array(prune_intervals)

    dnn_accs = []
    bnn_accs = []
    for interval in prune_intervals:
   
        dnn.to(DEVICE)
        bnn.to(DEVICE)

        dnn_collector = BaysianPruning(dnn)
        bnn_collector = BaysianPruning(bnn)

        dnn_pruner = GlobalUnstructuredPrune(interval, dnn_collector, method)
        bnn_pruner = GlobalUnstructuredPrune(interval, bnn_collector, method)

        dnn_pruner.collect_theshhold()
        bnn_pruner.collect_theshhold()

        dnn_pruner.apply_to_params()
        bnn_pruner.apply_to_params()

        print(bnn.fc1.mu_weight_mask)

        print(f'testing dnn:')
        _, dnn_accuracy, _ = bayesUtils.test_Bayes(dnn, test_loader,from_dnn=True, num_mc=num_mc)
        print(f'testing bnn:')
        _, bnn_accuracy, _ = bayesUtils.test_Bayes(bnn, test_loader, num_mc=num_mc)

        dnn_accs.append(dnn_accuracy)
        bnn_accs.append(bnn_accuracy)

    fig = go.Figure()

    fig.add_scatter(x=x, y=dnn_accs)
    fig.data[-1].name = 'From DNN'

    fig.add_scatter(x=x, y= bnn_accs)
    fig.data[-1].name = 'original BNN'

    fig.update_layout(
                    showlegend=True,
                    template='plotly_dark',
                    xaxis_title="Prune Rate",
                    yaxis_title="Accuracy",
                    title = f"Trained as BNN vs Raised from DNN | Method: {method.__name__}"
                    )
    
    fig.show()

train_loader, val_loader, test_loader = loadData('CIFAR-10',batch_size= 200)

#------TEST BNN Pruning-----------------
bnn = torch.load('model_90_BNN.path')

#model = BNN(in_channels=3, in_feat= 32*32*3, out_feat= 10)

bnn.to(DEVICE)

#model_parameters = filter(lambda p: p.requires_grad, model.parameters())
# params = sum([np.prod(p.size()) for p in model_parameters])
# print(params)


# pruner = BaysianPruning(bnn)

# glob = GlobalUnstructuredPrune(0.5, pruner, PruneByMU)

# glob.collect_theshhold()

# glob.apply_to_params()

# # print(model.fc1.mu_weight_mask)

# bayesUtils.test_Bayes(bnn, test_loader, from_dnn=False, num_mc=10)

#pruner.global_mu_prune(0.90)

#pruner.global_rho_prune(0.50)
#------ TEST DNN to BNN Pruning--------------
dnn = torch.load('other_model_90.path')

# prune_dnn(dnn, amount=0.5)

# #print(dnn.fc1.weight)

raise_reapply_prune(dnn, used_delta=0.00)

dnn.to(DEVICE)
#-------- Test masks are the same-------
# print(model.fc1.mu_weight_mask)
# print()
# print(model.fc1.rho_weight_mask)

# x = torch.eq(model.fc1.mu_weight_mask, model.fc1.rho_weight_mask)

# print(torch.equal(model.fc1.mu_weight_mask, model.fc1.rho_weight_mask))
# print('% Same:', torch.flatten(x).count_nonzero() / torch.flatten(x).numel())

# print('values_where_mask_is_zero' , model.fc1.mu_weight_mask[x == 0])
#---------------------------------------


# (train_loss, val_loss),(train_acc, test_acc), best_model_path = bayesUtils.train_Bayes(model=model,
#                                                                                         train_loader=train_loader,
#                                                                                         test_loader=test_loader,
#                                                                                         num_epochs=10,
#                                                                                         num_mc= 5,
#                                                                                         temperature= 1.0,
#                                                                                         lr = 0.0001,
#                                                                                         from_dnn=True,
#                                                                                         save=False,
#                                                                                         save_mode='accuracy')

#model_utils.test_fas_mnist(model,test_loader)
#bayesUtils.test_Bayes(model, test_loader, from_dnn=False, num_mc=10)

#accuracy_by_prune([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9],[PruneByMU, PruneByRho, PruneByHyper, PruneByKL], test_loader=test_loader, num_mc=10, model_path='model_90_BNN.path')
#accuracy_by_prune([0.01, 0.5],[PruneByHyper], test_loader=test_loader, num_mc=1, model_path='model_90_BNN.path')


#compare_bnn_dnn([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], test_loader, PruneByMU, 10, dnn, bnn)
#compare_bnn_dnn([0.0,0.5], test_loader, PruneByMU, 1, dnn, bnn)
