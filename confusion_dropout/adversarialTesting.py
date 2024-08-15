from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from matplotlib import pyplot as plt
import plotly.express as px
import numpy as np
import torch
from tqdm import tqdm

from shifted_dropout.datasets import loadData
from shifted_dropout.model_architectures import BasicCNN
from model_utils import train_fas_mnist, test_fas_mnist

reg_model = BasicCNN(num_classes= 10,
                              in_channels=3,
                              out_feature_size=2048,
                              use_reg_dropout=True,
                              dropout_prob=0.5,
                              num_drop_channels=2,
                              drop_certainty=1)

our_model = BasicCNN(num_classes= 10,
                              in_channels=3,
                              out_feature_size=2048,
                              use_reg_dropout=False,
                              dropout_prob=0.5,
                              num_drop_channels=2,
                              drop_certainty=1)

train_loader, val_loader, test_loader = loadData('CIFAR-10',batch_size= 100)


if 1==2:
    (train_loss, val_loss),(train_acc, test_acc), reg_model = train_fas_mnist(model=reg_model,
                                                                        train_loader=train_loader,
                                                                        val_loader=val_loader,
                                                                        test_loader=test_loader,
                                                                        num_epochs=25,
                                                                        save=True,
                                                                        save_mode='accuracy')

    (train_loss, val_loss),(train_acc, test_acc), our_model = train_fas_mnist(model=our_model,
                                                                        train_loader=train_loader,
                                                                        val_loader=val_loader,
                                                                        test_loader=test_loader,
                                                                        num_epochs=25,
                                                                        save=True,
                                                                        save_mode='accuracy')

reg_model = torch.load('model_25_True.path')
our_model = torch.load('model_25_False.path')

images, labels = next(iter(test_loader))

def testAdverserial(model: BasicCNN, images, labels):
    loss = torch.nn.MSELoss(reduction='none')
    model.dropout.test = False
    model.train()


    # throw data to the gpu
    images = images.to('mps')
    labels = labels.to('mps')
    
    # manualy set the y labels to use
    model.dropout.y = labels

    #generate the adversarial images
    adv_images = carlini_wagner_l2(model_fn=model, x=images, y= labels, n_classes=10, binary_search_steps= 10, max_iterations= 100)
    #adv_images = fast_gradient_method(model, images, 0.5, 2, 40, np.inf )

    # Predict both regular testing data and addversarial data
    reg_output = model(images)
    adv_output = model(adv_images)

    # Get regular images accuracy
    reg_predictions = torch.max(reg_output, dim=1)[1]
    reg_acc = torch.eq(labels, reg_predictions).int()
    reg_acc = torch.mean(reg_acc.to(torch.float))

    # Get the adversarial accuaracy
    adv_predictions = torch.max(adv_output, dim=1)[1]
    adv_acc = torch.eq(labels, adv_predictions).int()
    
    acc_colors = adv_acc

    adv_acc = torch.mean(adv_acc.to(torch.float))

    print(images.shape, adv_images.shape)

    print(f'Accuracy on regular: {100 * reg_acc}%')
    print(f'Accuracy on adversarial: {100 * adv_acc}%')



    distortions = sorted(torch.sum(((adv_images*255)-(images*255))**2,axis=(1,2,3)).detach().to('cpu')**.5)

    #print('Distortions',distortions)
    print(acc_colors.to('cpu').shape)

    #distortions = torch.mean(loss(adv_images, images), dim=(1,2,3)).to('cpu').detach()


    return np.array(distortions), np.array(acc_colors.to('cpu'))
    
reg_dist, reg_acc = testAdverserial(reg_model, images, labels)
ours_dist, our_acc = testAdverserial(our_model, images, labels)

x = np.arange(0, 100,1)
#x = np.arange(0,1,.01)

#print(np.array(reg_dist), x.shape)
success_rate = np.arange(0, 1, 0.01)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the first line
plt.plot(reg_dist, success_rate, label='Regular Dropout')

# Plot the second line
plt.plot(ours_dist, success_rate, label='Our Dropout')

# Add labels and title
plt.xlabel("Distortion (l2 distance)")
plt.ylabel("Success rate")
plt.title("Success Rate vs Distortion")

# Add legend
plt.legend()

# Show the plot
plt.show()

fig = px.scatter(x, y=reg_dist, color=reg_acc )
fig.add_scatter(x=x, y= ours_dist, mode='markers',marker=dict(color=our_acc), name= 'our Dropout')
#fig.show()
