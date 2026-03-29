# task.py
import torch

from train import Net, config_cuda, draw_accuracy_comparison_plot


def print_technical_justification():
    """
    GEN AI STATEMENT, DISCUSS HOW IT INITALLY SUGGESTED A FAR TOO BIG NETWORK (many many layers, and final channel size of like 1024)
    MASSIVELY OVERFITTING, AND EVALUATION SCORE STAYED REALLY LOW: ~15%
    DUE TO THIS I MANUALLY EDITED THE SIZE A LOT AND PLAYED AROUND WITH OUTPUT SIZES AND LAYER NUMBERS AND ADDING MAXPOOL ETC...
    THIS HAD A HUGE EFFECT; SO FUTURE IMPLEMENTATION COULD LACK AT IMPLEMENTING ARCHITECHURE SEARCH TO AUTOMATE THIS PROCESS, AS IT IS VERY TIME CONSUMING,
    COULD PROBABLY BE DONE BETTER ALGORITHMICALLY (Genetic Algorithms optimisation over architecture hyperparameters) THIS SEEMED
    TO HAVE THE LARGEST EFFECT ON PERFORMANCE, MORE SO THAN OPTIMISING OTHER HYPERPARAMETERS LIKE LEARNING RATE, BATCH SIZE, OR EPOCHS, OR OTHER REGULARISATION
    , WHICH HAD MORE MARGINAL EFFECTS
    """
    para_1 = "The clearest pattern in these results is the generalisation gap: training accuracy continues to rise while validation accuracy saturates and then declines. In the unregularised model, training accuracy increases from 9.13% to 92.41%, but validation accuracy only reaches a best value of 48.17% before falling slightly after. That leaves a final train–validation gap of 44.24 percentage points. By contrast, the regularised model reaches a lower final training accuracy of 83.45%, and a higher peak validation accuracy of 50.57%, with a smaller final gap of 32.88 percentage points. The mean gap also drops from about 0.174 for the unregularised model to 0.145 for the regularised model. This is exactly the pattern expected when regularisation reduces variance: the model fits the training set less aggressively and generalises better to unseen data.\n"
    para_2 = "On the bias–variance curve, the unregularised network sits too far on the low-bias, high-variance side. It has enough capacity to memorise features of the training set, which is why training accuracy keeps climbing even after validation performance has stopped improving. The regularised model shifts slightly towards higher bias but substantially lower variance. That trade-off is favourable here, because CIFAR-100 is a difficult 100-class problem with relatively small 32×32 images, so uncontrolled fitting of local texture or sample-specific noise is costly for generalisation.\n"
    para_3 = "This shift was produced by both explicit and implicit regularisation. The explicit regularisers were DropBlock and weight decay. Weight decay penalises large weights, discouraging sharp decision boundaries and reducing effective model complexity. DropBlock is stronger than standard dropout for convolutional maps because it removes contiguous spatial regions rather than isolated activations, which prevents the network from relying too heavily on one discriminative patch.\n"
    para_4 = "The chosen regularised configuration, weight_decay = 1e-3, drop_prob = 0.05, and block_size = 3, is sensible because it is strong enough to reduce co-adaptation without destroying too much information. A larger block or higher drop probability would likely push the model too far into underfitting, especially given the modest network size. These hyperparameters were selected via grid search over DropBlock and weight decay values (and learning rate and momentum), explicitly optimising for validation set performance. However, this introduces a risk of validation set overfitting (or evaluation set gaming), since the model selection process is directly conditioned on maximising validation accuracy. As a result, the reported validation performance may be optimistically biased. To properly assess generalisation, the final selected configuration should be evaluated on the held-out test set that was not used during hyperparameter tuning.\n"
    para_5 = "The optimiser also contributed implicit regularisation. Using SGD with momentum = 0.85 and learning rate = 0.01 (obtained from above mentioned grid search) biases training towards flatter minima compared with more adaptive methods such as Adam in many vision settings. Mini-batch SGD injects gradient noise because each update is computed on only part of the data. That stochasticity can prevent convergence to narrow, sharp minima that fit the training set very well but generalise poorly. Momentum smooths the noisy gradient trajectory while still preserving this beneficial stochastic effect. So even before adding DropBlock and weight decay, the optimiser itself already acted as a mild regulariser.\n"
    gen_statement = "One specific technical error was that when brainstorming network architectures with GenAI, it initially suggested a network that was far too large, with many extra layers and a final channel width around 1024. That dramatically increased capacity caused severe overfitting, and left evaluation accuracy around 15% while training accuracy would shoot up to high 90%, so the architecture had to be manually reduced and restructured with more appropriate pooling and channel sizes. Interestingly this had a huge effect on performance, jumping validation accuracy from 18% to around 45%. This may motivate a more in-depth architecture search. This could be combined with hyperparameter optimisation given that the models are quite quick to train and so wouldn’t be too time consuming to optimise this. However, I would need to stay careful not to game for validation accuracy too much, as possibly explicitly optimising for validation accuracy so strongly would lead to high validation accuracy, but substantially lower test accuracy.\n"

    output_string = f"{para_1}\n{para_2}\n{para_3}\n{para_4}\n{para_5}\n{gen_statement}"
    print("TECHNICAL JUSTIFICATION: \n\n")
    print(output_string)


if __name__ == "__main__":
    """
    main function to evaluate the trained model on a noisy test set and save a MixUp demo image.
    Also includes discussion points on the effects of MixUp and label smoothing on memorization and regularization.
    """
    batch_size = 128
    noise_std = 0.05
    unreg_model_path = "best_model_no_regularisation_task1.pt"
    reg_model_path = "best_model_regularisation_task1.pt"
    device, use_cuda = config_cuda()

    # load in regularised model
    reg_net = Net().to(device)
    state_dict = torch.load(reg_model_path, map_location=device)
    reg_net.load_state_dict(state_dict)

    # load in unregularised model
    unreg_net = Net().to(device)
    state_dict2 = torch.load(unreg_model_path, map_location=device)
    unreg_net.load_state_dict(state_dict2)

    """
    the following were the results I got from training the regularised and unregularised models,
    In order to not need to re-run train.py to plot the results, I have hardcoded the training and validation accuracies for both models here
    There was an option to save these accuracies during training, i.e. to a .csv or .txt but we were explicitly told not to rely on any external files
    other than train and task.py, hence the hardcoding...
    """
    series = [('unregularised train accuracy', [0.091275, 0.22605, 0.3269, 0.396025, 0.4467, 0.496475, 0.539275, 0.57505, 0.615275, 0.653125, 0.691625, 0.7344, 0.771675, 0.816575, 0.854325, 0.890925, 0.9241]), 
              ('unregularised val accuracy', [0.1555, 0.2616, 0.3084, 0.3651, 0.4062, 0.4314, 0.4478, 0.4504, 0.4689, 0.4644, 0.4791, 0.4729, 0.4779, 0.4817, 0.4744, 0.4626, 0.4811]), 
              ('regularised train accuracy', [0.091575, 0.222575, 0.31355, 0.388875, 0.4391, 0.4766, 0.516975, 0.547475, 0.5844, 0.613525, 0.64625, 0.67045, 0.6972, 0.731075, 0.75395, 0.782675, 0.80955, 0.834475]), 
              ('regularised val accuracy', [0.1598, 0.2081, 0.3044, 0.3471, 0.3968, 0.433, 0.3926, 0.4286, 0.4774, 0.4546, 0.4805, 0.4784, 0.4876, 0.4902, 0.5057, 0.4988, 0.4953, 0.4766])]
    
    draw_accuracy_comparison_plot(series, "generalization_gap.png",title="Training vs Validation Accuracy Regularised vs Unregularised")


    print_technical_justification()