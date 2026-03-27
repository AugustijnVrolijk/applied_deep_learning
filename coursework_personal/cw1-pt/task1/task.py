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
    para_1 = "The clearest pattern in these results is the generalisation gap: training accuracy continues to rise while validation accuracy saturates and then declines. In the unregularised model, training accuracy increases from 11.74% to 76.89%, but validation accuracy only reaches a best value of 41.48% before falling to 39.68%. That leaves a final train–validation gap of 37.21 percentage points. By contrast, the regularised model reaches a lower final training accuracy of 68.31%, but a higher peak validation accuracy of 44.66%, with a smaller final gap of 27.01 percentage points. The mean gap also drops from about 0.145 to 0.104. This is exactly the pattern expected when regularisation reduces variance: the model fits the training set less aggressively but generalises better to unseen data.\n"
    para_2 = "On the bias–variance curve, the unregularised network sits too far on the low-bias, high-variance side. It has enough capacity to memorise idiosyncratic features of the training set, which is why training accuracy keeps climbing even after validation performance has stopped improving. The regularised model shifts slightly towards higher bias but substantially lower variance. That trade-off is favourable here, because CIFAR-100 is a difficult 100-class problem with relatively small 32×32 images, so uncontrolled fitting of local texture or sample-specific noise is costly for generalisation.\n"
    para_3 = "This shift was produced by both explicit and implicit regularisation. The explicit regularisers were DropBlock and weight decay. Weight decay penalises large weights, discouraging sharp decision boundaries and reducing effective model complexity. DropBlock is stronger than standard dropout for convolutional maps because it removes contiguous spatial regions rather than isolated activations, which prevents the network from relying too heavily on one discriminative patch. Revised section with added technical caveat:\n"
    para_4 = "The chosen regularised configuration, weight_decay = 5e-4, drop_prob = 0.05, and block_size = 3, is sensible because it is strong enough to reduce co-adaptation without destroying too much information. A larger block or higher drop probability would likely push the model too far into underfitting, especially given the modest network size. These hyperparameters were selected via grid search over DropBlock and weight decay values, explicitly optimising for validation set performance. However, this introduces a risk of validation set overfitting (or evaluation set gaming), since the model selection process is directly conditioned on maximising validation accuracy. As a result, the reported validation performance may be optimistically biased. To properly assess generalisation, the final selected configuration should be evaluated on the held-out test set that was not used during hyperparameter tuning.\n"
    para_5 = "The optimiser also contributed implicit regularisation. Using SGD with momentum = 0.9 and learning rate = 0.05 biases training towards flatter minima compared with more adaptive methods such as Adam in many vision settings. Mini-batch SGD injects gradient noise because each update is computed on only part of the data. That stochasticity can prevent convergence to narrow, sharp minima that fit the training set very well but generalise poorly. Momentum smooths the noisy gradient trajectory while still preserving this beneficial stochastic effect. So even before adding DropBlock and weight decay, the optimiser itself already acted as a mild regulariser. The explicit regularisers then amplified that effect.\n"
    gen_statement = "One specific technical error was that when brainstorming network architectures with GenAI, it initially suggested a network that was far too large, with many extra layers and a final channel width around 1024. That dramatically increased capacity caused severe overfitting, and left evaluation accuracy around 15%, so the architecture had to be manually reduced and restructured with more appropriate pooling and channel sizes. Interestingly this had a huge effect on performance – 18% -44% difference in validation accuracy, which may motivate a more in-depth architecture search. This could be combined with hyperparameter optimisation given that the models are quite quick to train and so wouldn’t be too time consuming to optimise this. However, I would need to stay careful not to game for validation accuracy too much.\n"

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

    series = [('unregularised train accuracy', [0.1174, 0.252025, 0.352625, 0.436025, 0.50275, 0.556725, 0.621575, 0.678175, 0.725425, 0.76885]), 
     ('unregularised val accuracy', [0.1896, 0.2554, 0.3326, 0.3689, 0.3936, 0.3995, 0.4148, 0.4078, 0.4063, 0.3968]), 
     ('regularised train accuracy', [0.127575, 0.2604, 0.3482, 0.4138, 0.46455, 0.5072, 0.54505, 0.583425, 0.6083, 0.630375, 0.655025, 0.6831]), 
     ('regularised val accuracy', [0.1988, 0.2606, 0.3464, 0.3776, 0.4069, 0.4213, 0.4257, 0.4191, 0.4466, 0.4292, 0.4345, 0.413])]
    draw_accuracy_comparison_plot(series, "generalization_gap.png",title="Training vs Validation Accuracy Regularised vs Unregularised")

    print_technical_justification()