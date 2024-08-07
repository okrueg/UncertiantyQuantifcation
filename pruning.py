from matplotlib import pyplot as plt
import torch
import numpy as np
from math import ceil
from functools import partial
from torch.nn.utils import prune
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_rho
import plotly.express as px
import plotly.graph_objects as go
from bayesArchetectures import BNN, DNN
from wideresnet import WideResNet
import bayesUtils
import model_utils
from datasets import loadData
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

DEVICE = 'cpu'
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'

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
    overall_max = -1 * torch.inf
    overall_min = 1 * torch.inf
    set_min_max = False

    @classmethod
    def minMaxNorm(cls,t):
        if cls.set_min_max is False:
            cls.overall_max = torch.max(t)
            cls.overall_min = torch.min(t)
            cls.set_min_max = True

        minimum = cls.overall_min
        maximum = cls.overall_max

        return (t - minimum) / ((maximum - minimum) + 1e-9)


    def __init__(self, mu_weight= 0.75):
        super().__init__()
        #Check To see if mu weight is valid
        assert (mu_weight <= 1 and mu_weight >= 0)
        print(f'mu_weight: {mu_weight}')

        self.mu_weight = mu_weight
        self.rho_weight = 1 - mu_weight


    def __call__(self, amount, mu_tensor, rho_tensor):
        return self.calc_threshhold(amount, mu_tensor, rho_tensor)


    def get_scores(self, mu_tensor: torch.Tensor, rho_tensor: torch.Tensor):
        mu_abs = torch.abs(mu_tensor)

        mu_norm = PruneByHyper.minMaxNorm(mu_abs)

        mu_scores = mu_norm * self.mu_weight

        rho_norm = PruneByHyper.minMaxNorm(-1* rho_tensor)

        rho_scores = rho_norm * self.rho_weight

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

        kl = torch.log(sigma_p) - torch.log(sigma_q+ 1e-6) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 *(sigma_p**2)) - 0.5
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


class BayesParamCollector():

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
            

    def collect_weight_params(self, collect_by: str):
        #https://stackoverflow.com/questions/54846905/pytorch-get-all-layers-of-model
        def get_children(model: torch.nn.Module):

            children = list(model.children())
            flatt_children = []
            if children == []:

                return model
            else:

                for child in children:
                    try:
                        flatt_children.extend(get_children(child))
                    except TypeError:
                        flatt_children.append(get_children(child))
            return flatt_children
        

        all_mu= []
        all_rho= []
        match collect_by:
            case 'mu':
                for module in get_children(self.model):
                        for name, parameter in module.named_parameters():
                            if ('mu' in name and 'bias' not in name):
                                
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
        #print(all_mu)

        # all_mu = all_mu[0] 
        # all_rho = all_rho[0]

        return all_mu, all_rho


    def collect_bias_params(self, collect_by: str):
        all_mu= []
        all_rho= []

        match collect_by:
            case 'mu':
                for module in self.model.children():
                    for name, parameter in module.named_parameters():
                        if ('mu' in name and 'bias' in name):
                            
                            all_mu.append((module, name))
                            all_rho.append((module, name.replace('mu', 'rho')))
            
            case 'rho':
                for module in self.model.children():
                    for name, parameter in module.named_parameters():
                        if ('rho' in name and 'bias' in name):
                            
                            all_rho.append((module, name))
                            all_mu.append((module, name.replace('rho', 'mu')))
            
            case _:
                raise ValueError(f'Invalid collection type of {collect_by}')

    # Test collecting just 1 layers  
        # all_mu.clear()
        # all_mu.append((self.model.fc1, 'mu_bias'))

        # all_rho.clear()
        # all_rho.append((self.model.fc1, 'rho_bias'))
        #print(all_mu, all_rho)

        return all_mu, all_rho

    def global_just_mu_prune(self, amount: float):

        all_mu, _ = self.collect_weight_params('mu')

        prune.global_unstructured(
            all_mu,
            pruning_method=prune.L1Unstructured,
            amount=amount)
        
    
    def global_just_rho_prune(self, amount: float):
        _, all_rho = self.collect_weight_params('rho')

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
    

    def __init__(self, amount: float, pruner: BayesParamCollector, method, **kwargs) -> None:
        assert (amount <= 1 and amount >= 0), "Prune range must be [0,1]"

        self.amount = amount

        self.mu_list, self.rho_list = pruner.collect_weight_params('mu')

        self.method = method(**kwargs)

        self.threshhold = 0


    def collect_theshhold(self):

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

            scores = self.method.get_scores(module.get_parameter(mu_name), module.get_parameter(rho_name))

            symbol_flip = True if type(self.method).__name__ == 'PruneByRho' else False

            
                
            GlobalUnstructuredPrune.ThreshholdPrune.apply(module, mu_name, False, symbol_flip, self.threshhold, scores)
            GlobalUnstructuredPrune.ThreshholdPrune.apply(module, rho_name, True, symbol_flip, self.threshhold, scores)
            
            rho_param = module.get_parameter(rho_name+ '_orig')
            mask = module.get_buffer(rho_name+'_mask')
            rho_param.register_hook(partial(block_rho_grads, mask=mask))


def block_rho_grads(grad, mask):

    grad[mask != 1] = 0
    return grad


def add_safe_kl(model):
    def safe_kl_div(self, mu_q, sigma_q, mu_p, sigma_p, tiny_val = 1e-8):
            """
            UPDATED: Prevents Inf loss from log(0)
            """
            kl = torch.log(sigma_p) - torch.log(
                sigma_q + tiny_val) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 *
                                                            (sigma_p**2)) - 0.5
            return kl.mean()
    
    #OPTIONALISH SAFETY:
    for layer in model.modules():
        if hasattr(layer, "kl_loss"):
            # pylint: disable=no-value-for-parameter
            layer.kl_div = safe_kl_div.__get__(layer)


def remove_safe_kl(model):
    '''
    only needed for proper model saving
    '''
    for layer in model.modules():
        if hasattr(layer, "kl_div"):
            # pylint: disable=no-value-for-parameter
            delattr(layer, 'kl_div')


def prune_dnn(model: torch.nn.Module, amount: float, perminate:bool):
    all_params = []
    for module in model.modules():
        if hasattr(module, 'weight'):
            all_params.append((module, 'weight'))

    prune.global_unstructured(
        all_params,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    if perminate:
        for param in all_params:
            prune.remove(*param)


def reapply_prune(model: torch.nn.Module):


    def dnn_to_bnn_mu_mask(param: torch.nn.Parameter):
        mask = torch.ones_like(param)

        abs_weights = torch.abs(param)

        mask[abs_weights == 0] = 0

        return mask
   

    def dnn_to_bnn_rho_mask(param: torch.nn.Parameter):
        
        mask = torch.ones_like(param)

        abs_weights = torch.abs(param)

        mask[abs_weights == 0] = 1

        return mask


    param_collector = BayesParamCollector(model)

    mu_params, rho_params = param_collector.collect_weight_params(collect_by='mu')

    for (module, mu_name), (_, rho_name) in zip(mu_params, rho_params):
        # Sanity Check, Should never trigger o_0
        assert module == _
        mu_mask = dnn_to_bnn_mu_mask(getattr(module, mu_name))
        rho_mask = dnn_to_bnn_rho_mask(getattr(module, mu_name))

        prune.custom_from_mask(module, mu_name,
                               mask=mu_mask)
        
        prune.custom_from_mask(module, rho_name,
                               mask=rho_mask)
        
        rho_param = module.get_parameter(rho_name+ '_orig')
        mask = module.get_buffer(rho_name+'_mask')
        rho_param.register_hook(partial(block_rho_grads, mask=mask))


def compare_by_prune(prune_intervals:list, 
                      prune_method_list:list, 
                      just_mu_test:bool, 
                      test_loader,
                      pretune_epochs:int,
                      tune_epochs:int, 
                      num_mc:int,
                      to_bnn:bool,
                      orig_dnn:bool,   
                      model_path:str,
                      train_loader=None):
    x = np.array(prune_intervals)

    acc_fig = go.Figure()
    calib_fig = go.Figure()

    # Run Results for Just MU
    if just_mu_test:
        just_mus = []
        just_mus_calibrations= []
        for interval in prune_intervals:
                model = torch.load(model_path, map_location=DEVICE)

                if to_bnn:
                    const_bnn_prior_parameters = {
                                    "prior_mu": 0.0,
                                    "prior_sigma": 1.0,
                                    "posterior_mu_init": 0.0,
                                    "posterior_rho_init": -3.0,
                                    "type": "Reparameterization",
                                    "moped_enable": True,
                                    "moped_delta": 0.000,
                                    }
                    dnn_to_bnn(model, const_bnn_prior_parameters)

                    add_safe_kl(model)

                pruner = BayesParamCollector(model)
                
                pruner.global_just_mu_prune(interval)

                model.to(DEVICE)

                if tune_epochs > 0:
                    _ = bayesUtils.train_Bayes(model=model,
                                            train_loader=train_loader,
                                            test_loader=test_loader,
                                            num_epochs=tune_epochs,
                                            num_mc= 5,
                                            temperature= 1,
                                            lr = 0.001,
                                            from_dnn=orig_dnn,
                                            save=False,
                                            save_mode='accuracy',
                                            verbose=True)


                _, calibration, accuracy, _ = bayesUtils.test_Bayes(model, test_loader,from_dnn=orig_dnn, num_mc=num_mc)

                just_mus.append(accuracy)
                just_mus_calibrations.append(calibration)

        acc_fig.add_scatter(x=x, y=just_mus)
        acc_fig.data[-1].name = 'Prune Only MU'

        calib_fig.add_scatter(x=x, y=just_mus_calibrations)
        calib_fig.data[-1].name = 'Prune Only MU'

    for method in prune_method_list:
        method_accuracys = []
        method_calibrations = []
        print('memory',torch.mps.current_allocated_memory())
        for interval in prune_intervals:
            model = torch.load(model_path, map_location=DEVICE)

            if to_bnn:
                const_bnn_prior_parameters = {
                                "prior_mu": 0.0,
                                "prior_sigma": 1.0,
                                "posterior_mu_init": 0.0,
                                "posterior_rho_init": -3.0,
                                "type": "Reparameterization",
                                "moped_enable": True,
                                "moped_delta": 0.000,
                                }
                dnn_to_bnn(model, const_bnn_prior_parameters)

            add_safe_kl(model)

            # Tune the Uninitialized Rhos
            if pretune_epochs > 0:
                _ = bayesUtils.train_Bayes(model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    num_epochs=pretune_epochs,
                    num_mc= 5,
                    temperature= 1,
                    lr = 0.0005,
                    from_dnn=orig_dnn,
                    save=False,
                    save_mode='accuracy',
                    verbose=True)

            model.to(DEVICE)

            pruner = BayesParamCollector(model)

            glob = GlobalUnstructuredPrune(interval, pruner, method)

            glob.collect_theshhold()

            glob.apply_to_params()

            if tune_epochs > 0:
                _ = bayesUtils.train_Bayes(model=model,
                                        train_loader=train_loader,
                                        test_loader=test_loader,
                                        num_epochs=tune_epochs,
                                        num_mc= 5,
                                        temperature= 1,
                                        lr = 0.001,
                                        from_dnn=orig_dnn,
                                        save=False,
                                        save_mode='accuracy',
                                        verbose=True)

            _, calibration, accuracy, _ = bayesUtils.test_Bayes(model, test_loader,from_dnn=orig_dnn, num_mc=num_mc)

            method_accuracys.append(accuracy)
            method_calibrations.append(calibration)

        acc_fig.add_scatter(x=x, y= method_accuracys, mode='lines')
        acc_fig.data[-1].name = method.__name__

        calib_fig.add_scatter(x=x, y= method_calibrations, mode='lines')
        calib_fig.data[-1].name = method.__name__

    acc_fig.update_layout(
                showlegend=True,
                template='plotly_dark',
                yaxis_range=[0.5,1],
                xaxis_title="Prune Rate",
                yaxis_title="Accuracy",
                title = f"Survey of Bayesien Pruning Methods | Model: {type(model).__name__}"
                )
    
    calib_fig.update_layout(
            showlegend=True,
            template='plotly_dark',
            #yaxis_range=[0.0,0.4],
            xaxis_title="Prune Rate",
            yaxis_title="Callibration Error",
            title = f"Bayesien Pruning Methods Calibration Error | Model: {type(model).__name__}"
            )
    
    acc_fig.write_image(f"prune_method_survey_tune{tune_epochs}_accuracys.png")
    calib_fig.write_image(f"prune_method_survey_tune{tune_epochs}_calibrations.png")
    
    calib_fig.show()
    acc_fig.show()


def hyper_pruning_compare(prune_intervals:list, 
                            hyper_list:list, 
                            tune_epochs:int, 
                            num_mc:int,
                            from_dnn:bool,      
                            model_path:str,
                            test_loader, 
                            train_loader=None):
    x = np.array(prune_intervals)

    fig = go.Figure()

    for hyper in hyper_list:
        hyper_accuracys = []
        for interval in prune_intervals:
            model = torch.load(model_path)

            if from_dnn:
                const_bnn_prior_parameters = {
                                "prior_mu": 0.0,
                                "prior_sigma": 1.0,
                                "posterior_mu_init": 0.0,
                                "posterior_rho_init": -3.0,
                                "type": "Reparameterization",
                                "moped_enable": True,
                                "moped_delta": 0.000,
                                }
                dnn_to_bnn(model, const_bnn_prior_parameters)

            add_safe_kl(model)

            model.to(DEVICE)

            pruner = BayesParamCollector(model)

            glob = GlobalUnstructuredPrune(interval, pruner, PruneByHyper, mu_weight= hyper)

            glob.collect_theshhold()

            glob.apply_to_params()

            if tune_epochs > 0:
                _ = bayesUtils.train_Bayes(model=model,
                                        train_loader=train_loader,
                                        test_loader=test_loader,
                                        num_epochs=tune_epochs,
                                        num_mc= 5,
                                        temperature= 1,
                                        lr = 0.001,
                                        from_dnn=from_dnn,
                                        save=False,
                                        save_mode='accuracy',
                                        verbose=True)

            _, accuracy, _ = bayesUtils.test_Bayes(model, test_loader,from_dnn=from_dnn, num_mc=num_mc)

            hyper_accuracys.append(accuracy)

        fig.add_scatter(x=x, y= hyper_accuracys, mode='lines')
        fig.data[-1].name = hyper

    fig.update_layout(
                showlegend=True,
                template='plotly_dark',
                yaxis_range=[0.5,1],
                xaxis_title="Prune Rate",
                yaxis_title="Accuracy",
                title = f"PruneByHyper HyperParams | Model: {type(model).__name__}"
                )
    
    fig.write_image(f"hyper_survey_tune{tune_epochs}.png")
    fig.show()
    

def compare_bnn_dnn(prune_intervals:list, method, pretune_epochs:int, tune_epochs:int, num_mc:int, dnn_path:str, bnn_path:str, test_loader, train_loader=None):
    x = np.array(prune_intervals)
    # Storing untuned accuracies
    orig_dnn_accs = []
    from_dnn_accs = []
    bnn_accs = []

    # Storing untuned calibrations
    orig_dnn_calis = []
    from_dnn_calis = []
    bnn_calis = []

    # Storing tuned accuracies
    orig_dnn_tuned_accs= []
    from_dnn_tuned_accs = []
    bnn_tuned_accs = []

    # Storing tuned calibrations
    orig_dnn_tuned_calis= []
    from_dnn_tuned_calis = []
    bnn_tuned_calis = []

    for interval in prune_intervals:
        #load all them models
        orig_dnn = torch.load(dnn_path)
        from_dnn = torch.load(dnn_path)
        bnn = torch.load(bnn_path)

        const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",
        "moped_enable": True,
        "moped_delta": 0.000,
        }

        dnn_to_bnn(from_dnn, const_bnn_prior_parameters)

        add_safe_kl(from_dnn)
        add_safe_kl(bnn)

        from_dnn.to(DEVICE)
        orig_dnn.to(DEVICE)
        bnn.to(DEVICE)

        if pretune_epochs > 0:
            _ = bayesUtils.train_Bayes(model=from_dnn,
                                    train_loader=train_loader,
                                    test_loader=test_loader,
                                    num_epochs=pretune_epochs,
                                    num_mc= 5,
                                    temperature= 1,
                                    lr = 0.0005,
                                    from_dnn=True,
                                    save=False,
                                    save_mode='accuracy',
                                    verbose=True)
            
        from_dnn_collector = BayesParamCollector(from_dnn)
        bnn_collector = BayesParamCollector(bnn)


        from_dnn_pruner = GlobalUnstructuredPrune(interval, from_dnn_collector, method)
        bnn_pruner = GlobalUnstructuredPrune(interval, bnn_collector, method)
        

        from_dnn_pruner.collect_theshhold()
        bnn_pruner.collect_theshhold()

        from_dnn_pruner.apply_to_params()
        bnn_pruner.apply_to_params()

        prune_dnn(orig_dnn, amount=interval, perminate=False)

        print(f'testing Original DNN at {interval} before tuning:')
        _,orig_dnn_calibration, orig_dnn_accuracy, _ = model_utils.test_fas_mnist(orig_dnn, test_loader)

        print(f'testing From DNN at {interval} before tuning:')
        _,from_dnn_calibration, from_dnn_accuracy, _ = bayesUtils.test_Bayes(from_dnn, test_loader,from_dnn=True, num_mc=num_mc)

        print(f'testing DNN at {interval} before tuning:')
        _,bnn_calibration, bnn_accuracy, _ = bayesUtils.test_Bayes(bnn, test_loader,from_dnn=True, num_mc=num_mc)
        print()

        orig_dnn_accs.append(orig_dnn_accuracy)
        from_dnn_accs.append(from_dnn_accuracy)
        bnn_accs.append(bnn_accuracy)

        orig_dnn_calis.append(orig_dnn_calibration)
        from_dnn_calis.append(from_dnn_calibration)
        bnn_calis.append(bnn_calibration)

        if tune_epochs > 0:
            _ = bayesUtils.train_Bayes(model=bnn,
                                    train_loader=train_loader,
                                    test_loader=test_loader,
                                    num_epochs=tune_epochs,
                                    num_mc= 5,
                                    temperature= 1,
                                    lr = 0.001,
                                    from_dnn=True,
                                    save=False,
                                    save_mode='accuracy',
                                    verbose=True)

            _ = bayesUtils.train_Bayes(model=from_dnn,
                                    train_loader=train_loader,
                                    test_loader=test_loader,
                                    num_epochs=tune_epochs,
                                    num_mc= 5,
                                    temperature= 1,
                                    lr = 0.001,
                                    from_dnn=True,
                                    save=False,
                                    save_mode='accuracy',
                                    verbose=True)
            
            _ = model_utils.train_fas_mnist(model=orig_dnn,
                                    train_loader=train_loader,
                                    val_loader=val_loader,
                                    test_loader=test_loader,
                                    num_epochs=tune_epochs,
                                    activation_gamma=0,
                                    lr= 0.001,
                                    save=False,
                                    save_mode='accuracy',
                                    verbose=True)

            print(f'testing Original DNN at {interval} after tuning:')
            _,orig_dnn_calibration_tuned, orig_dnn_accuracy_tuned, _ = model_utils.test_fas_mnist(orig_dnn, test_loader)

            print(f'testing From DNN at {interval} after tuning:')
            _,from_dnn_calibration_tuned, from_dnn_accuracy_tuned, _ = bayesUtils.test_Bayes(from_dnn, test_loader,from_dnn=True, num_mc=num_mc)

            print(f'testing BNN at {interval} after tuning:')
            _,bnn_calibration_tuned, bnn_accuracy_tuned, _ = bayesUtils.test_Bayes(bnn, test_loader,from_dnn=True, num_mc=num_mc)
            print()

            orig_dnn_tuned_accs.append(orig_dnn_accuracy_tuned)
            from_dnn_tuned_accs.append(from_dnn_accuracy_tuned)
            bnn_tuned_accs.append(bnn_accuracy_tuned)

            orig_dnn_tuned_calis.append(orig_dnn_calibration_tuned)
            from_dnn_tuned_calis.append(from_dnn_calibration_tuned)
            bnn_tuned_calis.append(bnn_calibration_tuned)

    acc_fig = go.Figure()
    cali_fig = go.Figure()

    acc_fig.add_scatter(x=x, y=orig_dnn_accs, line=dict(color='darkgreen', width=4, dash='dash'))
    acc_fig.data[-1].name = 'Orig DNN'

    acc_fig.add_scatter(x=x, y=from_dnn_accs, line=dict(color='darkblue', width=3, dash='dash'))
    acc_fig.data[-1].name = 'From DNN'

    acc_fig.add_scatter(x=x, y= bnn_accs, line=dict(color='darkred', width=3, dash='dash'))
    acc_fig.data[-1].name = 'BNN'

    #Calibrations to figure
    cali_fig.add_scatter(x=x, y=orig_dnn_calis, line=dict(color='darkgreen', width=4, dash='dash'))
    cali_fig.data[-1].name = 'Orig DNN'

    cali_fig.add_scatter(x=x, y=from_dnn_calis, line=dict(color='darkblue', width=3, dash='dash'))
    cali_fig.data[-1].name = 'From DNN'

    cali_fig.add_scatter(x=x, y= bnn_calis, line=dict(color='darkred', width=3, dash='dash'))
    cali_fig.data[-1].name = 'BNN'

    if tune_epochs > 0:
        acc_fig.add_scatter(x=x, y=orig_dnn_tuned_accs, line=dict(color='green', width=3))
        acc_fig.data[-1].name = 'Tuned Orig DNN'

        acc_fig.add_scatter(x=x, y=from_dnn_tuned_accs, line=dict(color='blue', width=3))
        acc_fig.data[-1].name = 'Tuned From DNN'

        acc_fig.add_scatter(x=x, y= bnn_tuned_accs, line=dict(color='red', width=3))
        acc_fig.data[-1].name = 'Tuned BNN'

        #Tuned calibrations add to figure
        cali_fig.add_scatter(x=x, y=orig_dnn_tuned_calis, line=dict(color='green', width=3))
        cali_fig.data[-1].name = 'Tuned Orig DNN'

        cali_fig.add_scatter(x=x, y=from_dnn_tuned_calis, line=dict(color='blue', width=3))
        cali_fig.data[-1].name = 'Tuned From DNN'

        cali_fig.add_scatter(x=x, y= bnn_tuned_calis, line=dict(color='red', width=3))
        cali_fig.data[-1].name = 'Tuned BNN'

    acc_fig.update_layout(
                    showlegend=True,
                    template='plotly_dark',
                    yaxis_range=[0.7,1],
                    xaxis_title="Prune Rate",
                    yaxis_title="Accuracy",
                    title = f"Orig DNN vs Raised DNN vs BNN | Method: {method.__name__} | Tuning: {tune_epochs}| Accuracy"
                    )
    
    cali_fig.update_layout(
                showlegend=True,
                template='plotly_dark',
                xaxis_title="Prune Rate",
                yaxis_title="Calibration Error",
                title = f"Orig DNN vs Raised DNN vs BNN | Method: {method.__name__} | Tuning: {tune_epochs}| Calibrations"
                )
    
    acc_fig.write_image(f"bnn_dnn_comparison_tune{tune_epochs}_accuracys.png")
    acc_fig.show()

    cali_fig.write_image(f"bnn_dnn_comparison_tune{tune_epochs}_calibrations.png")
    cali_fig.show()


def kl_vs_mu_rho(model:BNN|DNN):
    pruner = BayesParamCollector(model)

    mu_list, rho_list = pruner.collect_weight_params('mu')

    mu_params_list = [ getattr(mu_param, name) for mu_param, name in mu_list]

    rho_params_list = [ getattr(rho_param, name) for rho_param, name in rho_list]

    mu_vector = torch.nn.utils.parameters_to_vector(mu_params_list[0])

    rho_vector = torch.nn.utils.parameters_to_vector(rho_params_list[0])

    sigma_tensor = PruneByKL.sigma_calc(rho_vector)

    kl_scores = PruneByKL.indv_kl(mu_q=mu_vector,sigma_q=sigma_tensor,mu_p=torch.zeros_like(mu_vector), sigma_p=torch.ones_like(sigma_tensor))


    fig = px.scatter_3d(x=mu_vector.detach().cpu().numpy(),y=sigma_tensor.detach().cpu().numpy(), z=kl_scores.detach().cpu().numpy())
    fig.update_layout(
                showlegend=True,
                template='plotly_dark',
                xaxis_title="Mu",
                yaxis_title="Sigma",
                title = "Mu vs Sigma vs KL"
                )
    fig.write_image("kl_vs_mu_vs_rho.png")
    fig.show()
    

train_loader, val_loader, test_loader = loadData('CIFAR-10',batch_size= 200)

#------TEST BNN Pruning-----------------
# bnn = torch.load('prune_test_models/BNN_90.path')
# dnn = torch.load('prune_test_models/DNN_90.path')
# #model_parameters = filter(lambda p: p.requires_grad, model.parameters())
# # params = sum([np.prod(p.size()) for p in model_parameters])
# # print(params)

# pruner = BayesParamCollector(bnn)
# pruner.collect_weight_params('mu')
# #pruner.global_just_mu_prune(0.1)
# glob = GlobalUnstructuredPrune(0.75, pruner, PruneByMU)

# glob.collect_theshhold()

# glob.apply_to_params()

# #print(bnn.fc1.mu_weight_mask)

# print(bnn.fc1.rho_weight_mask)
# print(bnn.fc1.rho_weight)

# bayes_test_results = bayesUtils.test_Bayes(bnn, test_loader, from_dnn=False, num_mc=5)

# test_results = model_utils.test_fas_mnist(dnn, test_loader)

# print(bayes_test_results[1])
# print(test_results[1])
# print(bnn.fc1.rho_weight)

# bayesUtils.test_Bayes(bnn, test_loader,num_mc=10, from_dnn=False)
#pruner.global_mu_prune(0.90)

#pruner.global_rho_prune(0.50)
#------ TEST DNN to BNN Pruning--------------
#wide = torch.load('WideResNet_90.path')
# wide = WideResNet(16, 10, 8)

# const_bnn_prior_parameters = {
# "prior_mu": 0.0,
# "prior_sigma": 1.0,
# "posterior_mu_init": 0.0,
# "posterior_rho_init": -3.0,
# "type": "Reparameterization",
# "moped_enable": True,
# "moped_delta": 0.000,
# }

# dnn_to_bnn(wide, const_bnn_prior_parameters)

# #add_safe_kl(wide)

# wide.to(DEVICE)

# _ = bayesUtils.train_Bayes(model=wide,
#                         train_loader=train_loader,
#                         test_loader=test_loader,
#                         num_epochs=90,
#                         num_mc= 5,
#                         temperature= 1,
#                         lr = 0.001,
#                         from_dnn=True,
#                         save=True,
#                         save_mode='accuracy',
#                         verbose=True)

#remove_safe_kl(wide)
# torch.save(wide, 'wide_working.path')

# wide = torch.load('wide_working.path')

#bayesUtils.test_Bayes(wide, test_loader,num_mc=1, from_dnn=True)
#model_utils.test_fas_mnist(wide,test_loader)

#-------- Test masks are the same-------
# print(model.fc1.mu_weight_mask)
# print()
# print(model.fc1.rho_weight_mask)

# x = torch.eq(model.fc1.mu_weight_mask, model.fc1.rho_weight_mask)

# print(torch.equal(model.fc1.mu_weight_mask, model.fc1.rho_weight_mask))
# print('% Same:', torch.flatten(x).count_nonzero() / torch.flatten(x).numel())

# print('values_where_mask_is_zero' , model.fc1.mu_weight_mask[x == 0])
#----------------------------Previous Run------------------------------
# hyper_pruning_compare([0.1, 0.5, 0.75, 0.9], [0.1, 0.5, 0.75, 0.9],
#                    test_loader=test_loader, train_loader=train_loader, 
#                    tune_epochs=10, num_mc=5, from_dnn=False, model_path='prune_test_models/BNN_90.path')

# compare_by_prune([0.001, 0.25, 0.5, 0.7, 0.8, 0.95],[PruneByMU, PruneByHyper, PruneByKL],just_mu_test=True,
#                    test_loader=val_loader, train_loader=val_loader, 
#                    tune_epochs=0, pretune_epochs=0, num_mc=1, orig_dnn= True, to_bnn=False, 
#                    model_path='prune_test_models/wide_90_BNN.path')
#-----------------------------CURRENT RUN--------------------------------

compare_bnn_dnn([0.5,0.7,0.9,0.95,0.98], PruneByMU, pretune_epochs=5, tune_epochs=10, num_mc=5,
                dnn_path='prune_test_models/WideResNet_90.path', bnn_path='prune_test_models/wide_90_BNN.path',
                test_loader=test_loader, train_loader=train_loader)

#------------------------------ TO RUN------------------------------------

# compare_bnn_dnn([0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], PruneByHyper, pretune_epochs=1, tune_epochs=1, num_mc=1,
#                 dnn_path='prune_test_models/DNN_90.path', bnn_path='prune_test_models/BNN_90.path',
#                 test_loader=val_loader, train_loader=val_loader)



