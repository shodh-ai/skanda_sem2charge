# A. FORWARD SURROGATE MODEL (3D-CNN for Battery Performance Prediction)

## Surrogate Modeling & Battery Simulation (5 sources)

1. **Transfer Learning Based Multi-fidelity Surrogate Model for Lithium-ion Battery Pack**
   - URL: http://cs230.stanford.edu/projects_fall_2022/reports/61.pdf
   - Key Methodology: CNN-LSTM architecture with transfer learning, uncertainty quantification for multi-fidelity data
   - Relevance: Direct application to battery surrogate modeling with state-of-charge sensitivity

2. **A surrogate-assisted uncertainty quantification and parameter optimization on a coupled electrochemical-thermal-aging model**
   - Authors: Alipour et al. (2023)
   - URL: https://www.sciencedirect.com/science/article/pii/S0378775323006493
   - Key Methodology: Surrogate model reduction of high-fidelity PDEs (DFN model), sensitivity analysis framework
   - Relevance: Uncertainty propagation in coupled physics models; parameter optimization methodology

3. **Physics-Informed Neural Network surrogate model for bypassing Blade Element Momentum theory in wind turbine aerodynamic load estimation**
   - Authors: Baisthakur et al. (2024)
   - URL: https://ui.adsabs.harvard.edu/abs/2024REne..22420122B/abstract
   - Key Methodology: PINNs incorporating physics-based constraints, 40x computational speedup
   - Relevance: Physics-informed approach to surrogate modeling; constraint integration patterns

4. **Surrogate Modeling of Lithium-Ion Battery Electrode**
   - Authors: Vijay et al. (2025)
   - URL: https://chemistry-europe.onlinelibrary.wiley.com/doi/full/10.1002/batt.202500433
   - Key Methodology: FEM + electrochemical model surrogate fusion
   - Relevance: Electrode-level property prediction from microstructure

5. **Physics-informed neural networks as surrogate models of discrete event simulations**
   - Authors: Donnelly et al. (2024)
   - URL: https://www.sciencedirect.com/science/article/pii/S0048969723074430
   - Key Methodology: Input-output pattern approximation using neural networks
   - Relevance: Surrogate model design principles for expensive simulators

---

## 3D CNN Architecture & Volumetric Processing (5 sources)

6. **3D Convolutional Neural Network (3D CNN) — A Guide for Engineers**
   - Source: NeuralConcept (2026)
   - URL: https://www.neuralconcept.com/post/3d-convolutional-neural-network-a-guide-for-engineers
   - Key Methodology: 3D convolution for CAD geometry, feature extraction hierarchy, CFD prediction
   - Relevance: Detailed 3D-CNN engineering implementation for voxel-based inputs

7. **A 3D convolutional neural network accurately predicts the permeability of fuel cell gas diffusion layer (GDL) materials**
   - Authors: Cawte et al. (2022)
   - URL: https://www.sciencedirect.com/science/article/abs/pii/S2352492822002261
   - Key Methodology: 3D CNN for transport property prediction from X-ray CT tomography data
   - Relevance: Direct application to porous materials property prediction from binary voxel grids

8. **3D-structure-attention graph neural network for crystals**
   - Authors: Lin et al. (2022)
   - URL: https://www.tandfonline.com/doi/abs/10.1080/00268976.2022.2077258
   - Key Methodology: Attention mechanism for 3D spatial structure, multi-scale property prediction
   - Relevance: Attention-weighted 3D structure representations; interpretability mechanisms

9. **Prediction of Energetic Material Properties from Electronic Structure using 3D CNN**
   - Authors: Casey et al. (2020)
   - URL: https://pubs.acs.org/doi/10.1021/acs.jcim.0c00259
   - Key Methodology: 3D electron density parsing, end-to-end feature learning from volumetric data
   - Relevance: 3D spatial data representation without manual descriptors

10. **Prediction of Characteristics Using a Convolutional Neural Network Based on Metamaterials Structure**
    - Authors: Zozyuk et al. (2023)
    - URL: http://www.iapress.org/index.php/soic/article/view/1707
    - Key Methodology: CNN-based structure-to-property mapping, experimental data integration
    - Relevance: Metamaterial design pattern applicable to electrode microstructure

---

## Physics-ML Frameworks & Implementation (4 sources)

11. **NVIDIA PhysicsNeMo: Physics-Informed Machine Learning Platform**
    - Source: NVIDIA Developer (2023)
    - URL: https://developer.nvidia.com/physicsnemo
    - Key Methodology: Physics-driven causality with automatic differentiation, neural operators, GNNs
    - Relevance: Industry-standard framework for physics-ML surrogate model development

12. **AI-Powered Simulation Tools for Surrogate Modeling: Siml.ai and PhysicsNeMo**
    - Source: NVIDIA Blog (2025)
    - URL: https://developer.nvidia.com/blog/ai-powered-simulation-tools-for-surrogate-modeling-engineering-workflows-with-siml-ai-and-nvid...
    - Key Methodology: Containerized training (SITE), 96% cost/time savings, visual model composition
    - Relevance: Practical deployment patterns for physics-based surrogate models

13. **Machine Learning-Driven Surrogate Models for Electrolytes**
    - Source: Michigan Tech Digital Repository
    - URL: https://digitalcommons.mtu.edu/cgi/viewcontent.cgi?article=2631&context=etdr
    - Key Methodology: Ensemble neural networks for MC simulation reduction
    - Relevance: Electrochemical property surrogate design patterns

14. **Application of Physics-Informed Neural Network to Cylinder Flow (Limited Data)**
    - Source: Wiley Online Library
    - URL: https://onlinelibrary.wiley.com/doi/full/10.1002/fld.70024
    - Key Methodology: PINN application with sparse observation data
    - Relevance: Low-data regime surrogate modeling approach

---

## SUPPORTING ARCHITECTURES & METHODOLOGIES

### Generative Model Alternatives (Complementary/Comparative) (5 sources)

28. **Microstructural Studies Using Generative Adversarial Networks (Wasserstein GAN)**
    - Source: ArXiv (2025)
    - URL: https://arxiv.org/html/2506.05860v1
    - Key Methodology: Wasserstein GAN for sharp microstructure generation, FEM property validation
    - Relevance: GAN alternative for high-fidelity microstructure synthesis; comparison baseline

29. **GAN-enabled Statistically Equivalent Virtual Microstructures (SliceGAN + Dream3D)**
    - Authors: Murgas et al. (2024)
    - URL: https://www.nature.com/articles/s41524-024-01219-4
    - Key Methodology: SliceGAN for 3D structure generation from 2D, hybrid with grain-packing algorithms
    - Relevance: 2D-to-3D generation pattern; polycrystalline morphology capture

30. **Designing complex architectured materials with GANs**
    - Authors: Mao et al. (2020)
    - URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC7182413/
    - Key Methodology: Crystallographic symmetry categorization, experience-free design, Hashin-Shtrikman bounds
    - Relevance: Symmetry-aware generation; materials property constraints

31. **Applications of generative adversarial networks in materials science (Review)**
    - Authors: Jiang et al. (2024)
    - URL: https://onlinelibrary.wiley.com/doi/full/10.1002/mgea.30
    - Key Methodology: Comprehensive GAN variants for materials design
    - Relevance: Comparative analysis framework; when to choose GANs vs. diffusion models

32. **Variational Autoencoders for Material Generation (Overview)**
    - Source: ScienceDirect Materials Science Topic (2024)
    - URL: https://www.sciencedirect.com/topics/materials-science/variational-autoencoder
    - Key Methodology: Latent space continuous representation, encoder-decoder structure, disentangled representations
    - Relevance: VAE latent space manipulation for guided generation; hybrid approaches

---

### Graph Neural Networks for Property Prediction (4 sources)

33. **SA-GNN: Material property prediction using Self-Attention Enhanced GNNs**
    - Source: AIP Advances (2024)
    - URL: https://pubs.aip.org/aip/adv/article/14/5/055033/3295158/SA-GNN-Prediction-of-material-properties-using
    - Key Methodology: Multi-head self-attention on graph nodes, crystal structure property prediction
    - Relevance: Attention mechanisms in material science; alternative to CNN for atomic structures

34. **Graph neural networks for materials science and chemistry (Review)**
    - Source: Nature Communications Materials (2022)
    - URL: https://www.nature.com/articles/s43246-022-00315-6
    - Key Methodology: GNN architectures for structured data, message passing framework
    - Relevance: Graph-based representation alternative for atomic/molecular systems

35. **Enhancing material property prediction with ensemble GNN (CGCNN)**
    - Source: ArXiv (2024)
    - URL: https://arxiv.org/abs/2407.18847
    - Key Methodology: Crystal Graph Convolutional NN, ensemble averaging for robustness
    - Relevance: Ensemble methodologies for property prediction robustness

36. **Scalable deeper graph neural networks for materials property prediction**
    - Source: NIH PMC (2022)
    - URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC9122959/
    - Key Methodology: Deep GNN scalability, materials discovery acceleration
    - Relevance: Scalability patterns for large material databases

---

### U-Net & Encoder-Decoder Architectures (4 sources)

37. **UNet++: Nested U-Net Architecture with Dense Skip Pathways**
    - Source: NIH PMC (2018)
    - URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC7329239/
    - Key Methodology: Deep supervision, nested skip connections, semantic gap reduction
    - Relevance: Advanced U-Net variant for diffusion denoiser backbone

38. **Medical Image Segmentation: Automatic Optimized U-Net (GA-UNet)**
    - Source: MDPI (2023)
    - URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC10533074/
    - Key Methodology: Genetic algorithm for architecture optimization, parameter efficiency
    - Relevance: U-Net architectural search patterns

39. **Medical image segmentation with UNet-based multi-scale context fusion**
    - Source: Nature Scientific Reports (2024)
    - URL: https://www.nature.com/articles/s41598-024-66585-x
    - Key Methodology: Attention U-Net, Transformer-U-Net integration, context fusion
    - Relevance: Attention integration in U-Net; multi-scale feature handling

40. **U-Net-Based Medical Image Segmentation (Review)**
    - Source: NIH PMC (2022)
    - URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC9033381/
    - Key Methodology: Encoder-decoder architecture fundamentals, skip connection mechanics
    - Relevance: Foundational U-Net design principles for 3D adaption

---

### Attention Mechanisms & Transformers (4 sources)

41. **Transformer Attention Mechanism in NLP**
    - Source: GeeksforGeeks (2024)
    - URL: https://www.geeksforgeeks.org/nlp/transformer-attention-mechanism-in-nlp/
    - Key Methodology: Scaled dot-product, multi-head attention, positional encoding
    - Relevance: Attention mechanism fundamentals; multi-scale feature focus

42. **How Attention Mechanism Works in Transformer Architecture (Visual)**
    - Source: YouTube (2025)
    - URL: https://www.youtube.com/watch?v=KMHkbXzHn7s
    - Key Methodology: Self-attention, causal attention, multi-head mechanics with visuals
    - Relevance: Intuitive understanding of attention for 3D spatial weighting

43. **Attention Mechanisms in Transformers (Comprehensive)**
    - Source: CloudThat (2024)
    - URL: https://www.cloudthat.com/resources/blog/attention-mechanisms-in-transformers
    - Key Methodology: Multi-head parallel processing, context window expansion
    - Relevance: Parallel attention processing for efficiency

44. **Transformer (Deep Learning) - Wikipedia**
    - Source: Wikipedia (2019)
    - URL: https://en.wikipedia.org/wiki/Transformer_(deep_learning)
    - Key Methodology: Transformer evolution, applications across domains
    - Relevance: Historical context and broader applications

---

### ResNet & 3D Convolutional Networks (4 sources)

45. **3D ResNet: Volumetric Deep Learning Architecture**
    - Source: Emergent Mind (2025)
    - URL: https://www.emergentmind.com/topics/3d-resnet
    - Key Methodology: 3D residual blocks, skip connections, volumetric downsampling
    - Relevance: 3D ResNet backbone for forward surrogate encoder

46. **3D CNN-Residual Neural Network for Medical Image Classification**
    - Source: IJETT Journal
    - URL: https://ijettjournal.org/Volume-70/Issue-10/IJETT-V70I10P236.pdf
    - Key Methodology: 3D ResNet architecture details, disease classification methodology
    - Relevance: 3D CNN implementation patterns for volumetric classification

47. **What is ResNet? 3D Visualizations**
    - Source: YouTube (2022)
    - URL: https://www.youtube.com/watch?v=nc7FzLiB_AY
    - Key Methodology: Skip connections solving vanishing gradient, deep network training
    - Relevance: Fundamental ResNet principles for deep network training

48. **Residual Networks (ResNet) - Deep Learning Fundamentals**
    - Source: GeeksforGeeks (2020)
    - URL: https://www.geeksforgeeks.org/deep-learning/residual-networks-resnet-deep-learning/
    - Key Methodology: ResNet variants, depth scaling, hyperparameter tuning
    - Relevance: ResNet design principles and depth selection

---

### Normalization & Training Techniques (6 sources)

49. **Batch Normalization: Accelerating Deep Network Training**
    - Authors: Ioffe & Szegedy (2015)
    - URL: https://arxiv.org/abs/1502.03167
    - Key Methodology: Internal covariate shift reduction, mini-batch statistics, learnable scale/shift
    - Relevance: Batch norm benefits for convergence speed and stability

50. **Build Better Deep Learning Models with Batch and Layer Normalization**
    - Source: Pinecone.io (2026)
    - URL: https://www.pinecone.io/learn/batch-layer-normalization/
    - Key Methodology: Batch vs. layer normalization comparison, moving averages, inference behavior
    - Relevance: When to choose batch vs. layer normalization in different architectures

51. **BatchNormalizationLayer - MATLAB Documentation**
    - Source: MathWorks (2024)
    - URL: https://www.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.batchnormalizationlayer.html
    - Key Methodology: Batch norm layer implementation, learnable gamma/beta parameters
    - Relevance: Implementation-level batch norm details

52. **Batch Normalization - Dive into Deep Learning**
    - Source: D2L.ai (2016)
    - URL: http://d2l.ai/chapter_convolutional-modern/batch-norm.html
    - Key Methodology: Fully connected vs. convolutional layer normalization
    - Relevance: Architecture-specific batch norm application

53. **What is Batch Normalization? - DeepChecks**
    - Source: DeepChecks (2022)
    - URL: https://www.deepchecks.com/glossary/batch-normalization/
    - Key Methodology: Batch norm regularization effects, initialization sensitivity reduction
    - Relevance: Batch norm as implicit regularizer for generalization

54. **Weighted Mean Squared Error Loss Implementation - PyTorch**
    - Source: YouTube (2025)
    - URL: https://www.youtube.com/watch?v=Mo5pRRIVboY
    - Key Methodology: WMSE for imbalanced datasets, sample weight application in DataLoader
    - Relevance: Implementation of weighted loss for multi-scale output balancing

---

### Loss Functions for Regression (3 sources)

55. **Weighted Mean Squared Error Loss - Keras Implementation**
    - Source: YouTube (2025)
    - URL: https://www.youtube.com/watch?v=75LYtN_-e6U
    - Key Methodology: Custom loss function in Keras, weight dictionary mapping
    - Relevance: Keras-based weighted loss implementation patterns

56. **Loss function for Linear regression in Machine Learning**
    - Source: GeeksforGeeks (2024)
    - URL: https://www.geeksforgeeks.org/machine-learning/loss-function-for-linear-regression/
    - Key Methodology: MSE, MAE, Huber loss, loss function selection criteria
    - Relevance: Loss function taxonomy and selection guidelines

57. **Linear regression: Loss - Google ML Crash Course**
    - Source: Google Developers (2026)
    - URL: https://developers.google.com/machine-learning/crash-course/linear-regression/loss
    - Key Methodology: L1/L2 loss formulations, RMSE, loss fundamentals
    - Relevance: Foundational loss function mathematics

---

### Weighted Loss Implementation (2 sources)

58. **Weighted Mean Squared Error PyTorch - StackOverflow**
    - Source: Stack Overflow (2022)
    - URL: https://stackoverflow.com/questions/74525618/weighted-mean-squared-error-pytorch-with-sample-weights-regression
    - Key Methodology: Loss weight expansion, training loop integration patterns
    - Relevance: PyTorch practical weighted MSE implementation

59. **Variational Autoencoders: Comprehensive Guide**
    - Source: DataCamp (2024)
    - URL: https://www.datacamp.com/tutorial/variational-autoencoders
    - Key Methodology: VAE architecture, reparameterization trick, continuous latent space
    - Relevance: VAE methodology for guided generation (complementary to diffusion)

---
