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

    """
    the following were the results I got from training the regularised and unregularised models,
    In order to not need to re-run train.py to plot the results, I have hardcoded the training and validation accuracies for both models here
    There was an option to save these accuracies during training, i.e. to a .csv or .txt but we were explicitly told not to rely on any external files
    other than train and task.py, hence the hardcoding...
    """
    series = [('unregularised train accuracy', [0.0958, 0.235775, 0.3259, 0.39305, 0.4479, 0.495075, 0.535725, 0.5767, 0.6168, 0.656725, 0.698325, 0.7321, 0.775625, 0.817575, 0.854675]), 
              ('unregularised val accuracy', [0.1535, 0.27, 0.3156, 0.338, 0.3574, 0.4274, 0.4374, 0.4428, 0.4712, 0.469, 0.4759, 0.487, 0.4782, 0.4763, 0.4818]), 
              ('regularised train accuracy', [0.092325, 0.223025, 0.310725, 0.376975, 0.43, 0.472, 0.50615, 0.539725, 0.573875, 0.60415, 0.63535, 0.66535, 0.692925, 0.72095, 0.75315, 0.77945, 0.804575, 0.828775, 0.8465, 0.870025, 0.88335]), 
              ('regularised val accuracy', [0.1365, 0.2503, 0.3279, 0.364, 0.3951, 0.4204, 0.4296, 0.4488, 0.4755, 0.4684, 0.4736, 0.482, 0.4634, 0.4871, 0.4983, 0.4908, 0.4943, 0.5045, 0.5027, 0.5019, 0.4993])]
    draw_accuracy_comparison_plot(series, "generalization_gap.png",title="Training vs Validation Accuracy Regularised vs Unregularised")


    print_technical_justification()