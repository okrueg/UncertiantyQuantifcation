import torch
from torch.nn.utils import prune
from bayesArchetectures import BNN
import bayesUtils
from datasets import loadData

class PruneByMU(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    mu_mask = torch.empty(0)
    mask1 = None
    mask2 = None

    def __init__(self, amount: float):
        super().__init__()
        self.amount = amount

    def compute_mask(self, t, default_mask):

        # For Second global pass provide the rho mask
        if PruneByMU.mu_mask.nelement() > 0:

            rho_mask = PruneByMU.mu_mask.clone()

            #print(rho_mask)

            #PruneByMU.mask2 = rho_mask.clone()
        
            rho_mask[rho_mask == 0] = torch.inf

            PruneByMU.mu_mask = torch.empty(0)

            #print(torch.equal(PruneByMU.mask1, PruneByMU.mask2))

            return PruneByMU.mu_mask
        
        else:
        
            print('mask')

            # Get the L1  of the weights
            t = t.abs()

            num_elements = t.numel()
            num_prune = int(self.amount * num_elements)
            
            _, topk_indicies = torch.topk(t, num_prune, largest=False)

            mask = default_mask

            mask[topk_indicies] = 0

            #mask.to('mps')

            #mask = torch.arange(t.numel(), device= 'mps')

            #mask = torch.rand_like(t)

            PruneByMU.mu_mask = mask.clone()

            #PruneByMU.mask1 = mask.clone()
            
            return PruneByMU.mu_mask

    
    def apply(self, module, name ):
        print('called')

        mask = self.compute_mask(module.get_parameter(name), default_mask=None)

        rho_mask = mask
        
        rho_mask[rho_mask == 0] = -1 * torch.inf

        print(rho_mask)

        if name == 'mu_weight':
            if hasattr(module, 'rho_weight'):

                # Apply masks
                prune.custom_from_mask(module, name='mu_weight', mask=mask)

                prune.custom_from_mask(module, name='rho_weight', mask=rho_mask)

            else:
                raise Exception


        if name == 'mu_kernel':
            if hasattr(module, 'rho_kernel'):

                # Apply masks
                prune.custom_from_mask(module, name='mu_kernel', mask=mask)

                prune.custom_from_mask(module, name='rho_kernel', mask=rho_mask)
            else:
                raise Exception


class PruneByRho(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    rho_mask = torch.empty(0)

    def __init__(self, amount: float):
        super().__init__()
        self.amount = amount

    def compute_mask(self, t, default_mask):

        # For Second global pass provide the mu mask
        if PruneByRho.rho_mask.nelement() > 0:

            mu_mask = PruneByRho.rho_mask
        
            mu_mask[mu_mask == torch.inf] = 0

            PruneByRho.rho_mask = torch.empty(0)

            return mu_mask

        # Get the L1  of the rho weights
        num_elements = t.numel()
        num_prune = int(self.amount * num_elements)
        
        topk_values, _ = torch.topk(t.view(-1), num_prune, largest=True)

        threshold = topk_values.min()

        mask = default_mask.clone()

        mask[t >= threshold] =  torch.inf

        PruneByRho.rho_mask = mask

        return mask


class PruneByHyper(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'
    mu_scores = torch.empty(0)

    mask = torch.empty(0)

    @staticmethod
    def minMaxNorm(t):

        min = torch.min(t)
        max = torch.max(t)

        return (t - min) / (max - min)
    

    def __init__(self, amount: float, mu_weight: float, isRho = False):
        super().__init__()
        self.amount = amount
        self.mu_weight = mu_weight
        self.isRho = isRho

    def compute_mask(self, t, default_mask):

        if PruneByHyper.mu_scores.nelement() == 0:


            t = torch.abs(t)

            mu_norm = PruneByHyper.minMaxNorm(t)

            mu_norm *= self.mu_weight

            PruneByHyper.mu_scores = mu_norm

            return default_mask
        
        elif PruneByHyper.mask.nelement() == 0:

            rho_norm = PruneByHyper.minMaxNorm(t)

            rho_norm = 1 - (rho_norm * (1- self.mu_weight))

            scores = PruneByHyper.mu_scores + rho_norm

            num_elements = t.numel()
            num_prune = int(self.amount * num_elements)

            mask = default_mask.clone()
            
            topk_values, _ = torch.topk(scores, num_prune, largest=False)

            threshold = topk_values.max()

            mask[scores <= threshold] = 0

            PruneByHyper.mask = mask

            return default_mask
        else:
            
            mask = PruneByHyper.mask.clone()

            #if self.isRho:
                #mask[mask == 0 ] = torch.inf

            return mask


class PruneByKL(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    mu_tensor = torch.empty(0)
    rho_tensor = torch.empty(0)


    mask = torch.empty(0)

    mask1 = torch.empty(0)
    mask2 = torch.empty(0)

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

    def __init__(self, amount: float, is_rho = False):
        super().__init__()
        
        self.amount = amount
        self.is_rho = is_rho


    def compute_mask(self, t, default_mask):

        if PruneByKL.mu_tensor.nelement() == 0:

            PruneByKL.mu_tensor = t

            return default_mask


        elif PruneByKL.rho_tensor.nelement() == 0:

            PruneByKL.rho_tensor = t

            return default_mask
        
        else:
            sigma_tensor = PruneByKL.sigma_calc(PruneByKL.rho_tensor)

            mu_prior = torch.ones_like(PruneByKL.mu_tensor)

            sigma_prior = PruneByKL.sigma_calc(-3 * torch.ones_like(PruneByKL.rho_tensor))

            kl_scores = PruneByKL.indv_kl(PruneByKL.mu_tensor, sigma_tensor, mu_prior, sigma_prior )

            num_elements = t.numel()
            num_prune = int(self.amount * num_elements)
            
            topk_values, _ = torch.topk(kl_scores.view(-1), num_prune, largest=False)

            threshold = topk_values.max()

            mask = default_mask.clone()

            mask[t <= threshold] = 0

            PruneByKL.mask1 = mask

            if self.is_rho:
                PruneByKL.mask2 = mask
                #mask[mask == 0 ] = torch.inf
                
            return mask


class BaysianPruning():
    def __init__(self, model: BNN):
        self.model = model


    def check_rho_zeros(self, all_rho):
        for x, name in all_rho:
            if torch.max(x.get_parameter(name)).item() > 0:
                raise ValueError(f'There is a postive rho found: {torch.max(x.get_parameter(name)).item()}')
            

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
            
        all_mu.clear()
        all_mu.append((self.model.fc1, 'mu_weight'))

        all_rho.clear()
        all_rho.append((self.model.fc1, 'rho_weight'))
        print(all_mu, all_rho)
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
    
    
    def global_both_by_mu_prune(self, amount: float):

        all_mu, all_rho = self.collect_all_parameters('mu')

        self.check_rho_zeros(all_rho)

        method = PruneByMU

        prune.global_unstructured(
            all_mu,
            pruning_method=method,
            amount=amount)
    
        prune.global_unstructured(
            all_rho,
            pruning_method=method,
            amount=amount)


    def global_both_by_rho_prune(self, amount: float):

        all_mu, all_rho = self.collect_all_parameters('rho')
                
        self.check_rho_zeros(all_rho)
        
        method = PruneByRho

        prune.global_unstructured(
            all_rho,
            pruning_method=method,
            amount=amount)
        
        prune.global_unstructured(
            all_mu,
            pruning_method=method,
            amount=amount)
    


    def global_hyper_both_prune(self, amount: float, mu_weight: float):
        all_mu, all_rho = self.collect_all_parameters('mu')

        self.check_rho_zeros(all_rho)

        method = PruneByHyper

        # Collect the scores (Just applies a default Mask)

        prune.global_unstructured(
            all_mu,
            pruning_method=method,
            amount=amount,
            mu_weight = mu_weight)
        
        prune.global_unstructured(
            all_rho,
            pruning_method=method,
            amount=amount,
            mu_weight = mu_weight)
        
        # Second pass actually applies the scores
        
        prune.global_unstructured(
            all_mu,
            pruning_method=method,
            amount=amount,
            mu_weight = mu_weight)
        
        prune.global_unstructured(
            all_rho,
            pruning_method=method,
            amount=amount,
            mu_weight = mu_weight,
            isRho = True)   
 

    def global_kl_prune(self, amount: float):
        all_mu, all_rho = self.collect_all_parameters('mu')

        self.check_rho_zeros(all_rho)

        method = PruneByKL

        # Collect the scores (Just applies a default Mask)

        prune.global_unstructured(
            all_mu,
            pruning_method=method,
            amount=amount)
        
        prune.global_unstructured(
            all_rho,
            pruning_method=method,
            amount=amount)
        
        # Second pass actually applies the scores
        
        prune.global_unstructured(
            all_mu,
            pruning_method=method,
            amount=amount)
        
        prune.global_unstructured(
            all_rho,
            pruning_method=method,
            amount=amount,
            is_rho = True)   


model = torch.load('model_90_BNN.path')

pruner = BaysianPruning(model)

train_loader, val_loader, test_loader = loadData('CIFAR-10',batch_size= 200)

#pruner.global_mu_prune(0.90)

#pruner.global_rho_prune(0.50)

pruner.global_both_by_mu_prune(0.90)

#pruner.global_both_by_rho_prune(0.9)

#pruner.global_hyper_both_prune(0.5, mu_weight=7.0)

#pruner.global_kl_prune(0.5)

print(model.fc1.mu_weight_mask)
print()
print(model.fc1.rho_weight_mask)

x = torch.eq(model.fc1.mu_weight_mask, model.fc1.rho_weight_mask)

print(torch.equal(model.fc1.mu_weight_mask, model.fc1.rho_weight_mask))
print(torch.flatten(x).count_nonzero())
print(torch.flatten(x).numel())

# print('fc2')
# print(model.fc2.mu_weight_mask)
# print()
# print(model.fc2.rho_weight_mask)


# x = model.fc1
# sigma_weight = PruneByKL.sigma_calc(x.rho_weight)
# y = PruneByKL.indv_kl(x.mu_weight, sigma_weight, x.prior_weight_mu, x.prior_weight_sigma)
# print(y.shape)
bayesUtils.test_Bayes(model, test_loader, num_mc=10)
