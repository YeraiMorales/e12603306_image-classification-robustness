# e12603306_image-classification-robustness
Applied Deep Learning project: Image Classification Under Noisy Conditions: CNNs, Data Augmentation and a Traditional Baseline

**Course:** Applied Deep Learning
**Topic:** Computer Vision / Image Classification  
**Project type:** Bring Your Own Method

## 1. Motivation

I am a Computer Science student from Barcelona, currently studying abroad as an exchange student. Computer Vision is an area I have been interested in for a while. At my home university, I took a Computer Vision course and worked on a licence plate recognition project. We were only allowed to use traditional Computer Vision techniques, which made me curious about how I could approach similar problems using Deep Learning.

I am studying the theory behind neural networks, and I have never trained one myself. For this course, I wanted to choose a Computer Vision project so that I could work in a field I already know a little about while learning the practical side of Deep Learning: building a model, training it, evaluating it and understanding why it makes mistakes.

I also want to compare the neural network with a more traditional approach. That connects this project to what I learned in my previous Computer Vision course and should help me see what Deep Learning does differently, rather than just using it because it is a newer technique.

## 2. Project Idea

Image classifiers can work well on clean images but struggle when the input is noisy, blurry or partly hidden. I want to investigate how different training strategies affect this problem.

I will train a small **Convolutional Neural Network (CNN)** to classify images from the CIFAR-10 dataset. After training a first model on clean images, I will continue training copies of it under different conditions and compare their performance on corrupted images. The main experiment will compare **random corruption augmentation** with **progressive corruption augmentation** (a simple form of curriculum learning).

I will also build a traditional image classification pipeline using **Histograms of Oriented Gradients (HOG)** and a **linear Support Vector Machine (SVM)**. This will give me a reference point for how a classical feature-extraction approach behaves under the same corruptions.

The aim is not to build the best CIFAR-10 classifier or to propose a completely new neural network. I want to understand the training process and carry out a fair experiment about robustness.

### Main question

**Does gradually increasing the difficulty of image corruptions during CNN fine-tuning produce a more robust classifier than applying corruptions in a random order?**

I also want to explore a second question: **How does the CNN compare with a traditional HOG + SVM classifier when the images are degraded?**

I expect corruption augmentation to improve performance on some degraded images, but I do not know whether introducing it progressively will be better than random augmentation. It is also possible that improving robustness reduces accuracy on clean images. Those trade-offs will be part of the analysis.

## 3. Project Type

I have chosen **Bring Your Own Method**.

The main methodological change will be in the CNN's training procedure: I will implement a schedule that starts with mild image corruptions and progressively increases their severity. I will compare it against random augmentation while keeping the CNN architecture, initial weights and training budget as similar as possible.

The HOG + SVM classifier is an **additional traditional baseline**, not the main contribution. It also introduces an element of *Beat the Classics*, but I will keep *Bring Your Own Method* as the official project type because the main investigation is about modifying how the neural network is trained.

## 4. Datasets

### CIFAR-10 - Training and clean-image evaluation

- **Source:** [CIFAR-10, University of Toronto](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Size:** 60,000 colour images (32 × 32 pixels)
- **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck
- **Official split:** 50,000 training images and 10,000 test images

I will use CIFAR-10 because it is well documented, small enough to work with on my laptop and suitable for trying different classification methods without collecting or labelling images myself.

I plan to divide the official training set into **45,000 training images and 5,000 validation images**, using a stratified split and a fixed random seed. The 10,000 official test images will only be used for the final evaluation. I may use a smaller training subset for early tests if training on CPU is too slow.

### CIFAR-10-C - Robustness evaluation

- **Source:** [CIFAR-10-C (Zenodo)](https://doi.org/10.5281/zenodo.2535967)
- **Content:** Corrupted versions of the CIFAR-10 test images, with five severity levels for each corruption type

I will focus on a limited selection of corruptions, such as Gaussian noise, blur, fog, JPEG compression and pixelation. This should be enough to study different types of image degradation without making the evaluation too large.

**CIFAR-10-C will be used only for final testing**, not to train the models or select hyperparameters. During training and validation, I will generate corruptions myself from the corresponding CIFAR-10 images. Some evaluation corruptions will be different from those used in training, to see whether any improvements generalise.

The same test examples and corruption levels will be used for both the CNNs and HOG + SVM.

## 5. Methodology

### 5.1. Traditional Computer Vision baseline: HOG + SVM

I will start with a classical pipeline:

1. Extract HOG descriptors from each image. HOG describes local edge orientations instead of learning image features directly from raw pixels.
2. Train a **linear SVM** using those descriptors and the CIFAR-10 class labels.
3. Use the validation set to choose reasonable HOG settings and the SVM regularisation parameter.
4. Evaluate the classifier on clean CIFAR-10 and the selected CIFAR-10-C corruptions.

This method does not involve a neural network or fine-tuning. It will give me a baseline to compare with the CNN models. Since CIFAR-10 images are only 32 × 32 pixels, HOG may not describe every class particularly well; that limitation will also be discussed in the results.

A linear SVM should be more manageable on a CPU than a large kernel SVM. If necessary, I will start with fewer images while checking how the baseline scales.

### 5.2. CNN: Initial training

I will implement a small CNN in **PyTorch**, with a few convolutional layers, ReLU activations, pooling and a final classification layer for the 10 CIFAR-10 classes. The exact size will be chosen after some initial tests so that training remains feasible on my laptop.

I will train this CNN **from scratch**, starting with randomly initialised weights and using the clean CIFAR-10 training images. This phase will help me get familiar with data loaders, the loss function, backpropagation, optimisers, training and validation loops, and model checkpoints.

Once the model has reached reasonable performance, I will save its weights. All three later CNN experiments will start from this same checkpoint.

### 5.3. CNN: Fine-tuning experiments

I will compare three ways of continuing the training of the base CNN:

| Experiment | How the model is fine-tuned | Why it is included |
| --- | --- | --- |
| **A - Clean** | Continue training on clean images | Check what happens simply from training for longer |
| **B - Random augmentation** | Add noise, blur or occasional occlusion with randomly selected intensities | Measure the effect of ordinary corruption augmentation |
| **C - Progressive augmentation** | Start with mild corruptions and increase their intensity over training | Test whether the order of difficulty makes a difference |

**Progressive augmentation is where curriculum learning comes into the project.** Curriculum learning means organising training examples or tasks from easier to harder. Here, I will apply that idea to the *severity of the image corruptions*, not to the CNN architecture: early fine-tuning epochs will use light distortions, followed by medium and then stronger ones.

To make B and C comparable, I will aim to keep the number of training steps, probability of augmentation and overall exposure to each severity level roughly equal. The main difference should be **when** the model sees each level of difficulty. I will choose the schedule using validation data, not CIFAR-10-C.

### 5.4. Comparing the approaches

In total, I plan to evaluate **four models**:

1. **HOG + linear SVM**, trained on clean images (traditional baseline).
2. **CNN A**, fine-tuned on clean images.
3. **CNN B**, fine-tuned with random corruption augmentation.
4. **CNN C**, fine-tuned with progressive corruption augmentation.

The CNN experiments will address the main question about training strategies. The HOG + SVM comparison will address the separate question of traditional methods versus Deep Learning. I will not assume that the traditional model and the CNNs have the same training cost or capacity, so performance and runtime will be reported separately.

For each model, I will measure:

- **Accuracy on clean images.**
- **Accuracy on corrupted images**, separated by corruption type and severity.
- **Drop in accuracy** from clean to corrupted images.
- **Average accuracy** across the selected corruption/severity combinations (not the full official CIFAR-10-C score).
- **Training time** and, if feasible, inference time on the same hardware.

I will present the results in tables and plots, especially accuracy versus corruption severity. If computation time allows, I will repeat the most important experiments using different random seeds. Otherwise, I will explain the limitation of comparing single runs.

A useful result does not necessarily mean that the progressive strategy wins: finding that it makes little difference, or that it helps with some corruptions but hurts on others, would also answer the research question.

## 6. Application to Present the Results

I plan to make a simple local application where a user can select an image, apply different degradations and compare the predictions of the trained models. I would also like to display the main plots from the experiments.

I may use a **React/Node.js frontend with a small Python API (FastAPI)** to run the models, since I already have some experience building applications around AI models. A simpler Python interface would also be fine if building the frontend starts taking time away from the training and experiments.

The application is mainly a way to demonstrate the results. It is not intended to be a production-ready image classifier.

## 7. Tools and Computing Resources

- **Python, PyTorch and torchvision:** CNN implementation, training and image transformations.
- **scikit-image and scikit-learn:** HOG feature extraction and the linear SVM classifier.
- **NumPy, pandas and Matplotlib:** Data handling and result analysis.
- **Node.js and FastAPI / Streamlit:** Application to show the results.
- **Git:** Version control and submission.

Some of these are not definitive and may change in case that I find a more suitable tool in the process. I expect to work primarily on my **laptop CPU**, without relying on a dedicated GPU or paid computing services. This is why I will use a relatively small CNN and limit the number of corruption types and experiments. I can reduce the training subset or the number of epochs if the first tests show that training takes too long. If the results on these limitations are too poor, I will consider the paid computing service option.

## 8. Work Breakdown and Time Estimates

These are the broad estimates for the full project (reality will most likely differ). The hours represent my expected active work; training time may vary depending on the laptop and may include additional waiting time.

| Task | Work planned | Hours |
| --- | --- | ---: |
| Literature review and planning | Read relevant papers and define the experiments and metrics | 3 |
| Dataset preparation | Download and inspect CIFAR-10; make train/validation split; prepare selected CIFAR-10-C files | 3 |
| Traditional baseline | Implement HOG extraction, train and validate the linear SVM | 5 |
| CNN implementation | Design a small CNN; implement data loaders, training loop, metrics and checkpoint saving | 5 |
| Initial CNN training | Train the model on clean CIFAR-10 and inspect learning curves | 4 |
| Corruption pipeline | Implement noise, blur, occlusion and severity controls for training and validation | 4 |
| Fine-tuning experiments | Run the clean, random augmentation and progressive augmentation branches | 9 |
| Evaluation and comparison | Evaluate all four models, create plots and analyse mistakes and trade-offs | 6 |
| Results application | Build a small interface for predictions, image corruptions and plots | 4 |
| Final report | Write up methods, experiments, results and limitations | 5 |
| Presentation | Prepare slides and a short presentation with demo | 3 |
| Repository cleanup | Improve code structure, write usage instructions and proofread the submission | 2 |
| **Total** | | **53 hours** |

If the training or implementation turns out to be more demanding than expected, I will prioritise the clean CNN, the traditional baseline and the **random-versus-progressive augmentation comparison**. I can limit the number of images, corruption types or repetitions, and keep the application simple.

## 9. Risks and Limitations

The main practical concern is **training time on CPU**. I will do small pilot runs first, then decide whether using all 45,000 training images is realistic. Keeping the model small should make the project manageable, though I cannot know the exact runtime yet.

Another concern is **fair comparison**. The CNNs should start from identical weights and use comparable fine-tuning budgets. For random and progressive augmentation, differences in the total number or severity of altered images could otherwise explain the results. I will record the settings and report any differences that remain.

There is also a risk of **overfitting the experiment to the benchmark**. I will use the validation split to decide settings and keep CIFAR-10-C for final testing. I will distinguish corruptions similar to the ones used in training from genuinely held-out types.

Finally, HOG + SVM and a CNN are different kinds of models, with different representational capacity. The traditional method may perform worse even on clean images. I will compare their absolute accuracy and robustness drop, without claiming that one controlled variable explains every difference.

## 10. Scientific Papers

1. **Hendrycks, D., & Dietterich, T. (2019).** *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.* International Conference on Learning Representations (ICLR).  
   https://arxiv.org/abs/1903.12261  
   This paper introduces the corruption benchmark behind CIFAR-10-C and motivates the evaluation part of my project.

2. **Hendrycks, D., Mu, N., Cubuk, E. D., Zoph, B., Gilmer, J., & Lakshminarayanan, B. (2020).** *AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty.* International Conference on Learning Representations (ICLR).  
   https://arxiv.org/abs/1912.02781  
   This is relevant to the idea that the way images are augmented during training can change a model's robustness.

3. **Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009).** *Curriculum Learning.* International Conference on Machine Learning (ICML).  
   https://doi.org/10.1145/1553374.1553380  
   The paper provides the basis for the progressive augmentation experiment: presenting easier examples before harder ones.

4. **Dalal, N., & Triggs, B. (2005).** *Histograms of Oriented Gradients for Human Detection.* IEEE Conference on Computer Vision and Pattern Recognition (CVPR).  
   https://doi.org/10.1109/CVPR.2005.177  
   This paper introduces HOG features and motivates the handcrafted-feature baseline. The original task was human detection; I will adapt the features to CIFAR-10 image classification.

### Dataset references

- **Krizhevsky, A. (2009).** *Learning Multiple Layers of Features from Tiny Images.* https://www.cs.toronto.edu/~kriz/cifar.html / https://huggingface.co/datasets/uoft-cs/cifar10
- **CIFAR-10-C dataset:** https://doi.org/10.5281/zenodo.2535967 (still might find a better one, this is the provisional)
