The objective of this project:                
1. Develop an EEG denoising framework that performs both artifact suppression and neural information preservation.
2. Design a Neural Information Preservation Index (NIPI) to evaluate retained neural information after denoising.

Introduction:

1. EEG signals are widely used in brain-computer interfaces, cognitive analysis, and clinical diagnosis, but are highly affected by artifacts such as EOG(Electrooculography), EMG(Electromyography), motion noise, and power-line interference.[1]
2. Conventional denoising methods mainly focus on artifact removal and often distort important neural information such as functional connectivity, phase synchrony, and ERP(event related potential) patterns.
3. Recent deep learning methods improve denoising performance; however, preservation of physiologically meaningful neural dynamics remains a major challenge.[2]
4. Therefore, this work aims to develop an adaptive EEG denoising framework that suppresses artifacts while preserving task-relevant neural information.

The existing approaches:
<img width="1725" height="741" alt="image" src="https://github.com/user-attachments/assets/73f5441f-b62e-43ac-b791-bc188c75042e" />

Literature Survey:
<img width="1931" height="872" alt="image" src="https://github.com/user-attachments/assets/c07586ae-ce8a-43be-b79c-73780a9bdf3a" />

References:
1. Y. Dong et al., "An Approach for EEG Denoising Based on Wasserstein Generative Adversarial Network," in IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 31, pp. 3524-3534, 2023, doi: 10.1109/TNSRE.2023.3309815.
2. J. Yin, A. Liu, C. Li, R. Qian and X. Chen, "A GAN Guided Parallel CNN and Transformer       Network for EEG Denoising," in IEEE Journal of Biomedical and Health Informatics, vol. 29, no. 6, pp. 3930-3941, June 2025, doi: 10.1109/JBHI.2023.3277596.
3. F. Taleb, M. Vasco, N. Rajabi, M. Björkman and D. Kragic, "Challenging Deep Learning Methods for EEG Signal Denoising under Data Corruption," 2024 46th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), Orlando, FL, USA, 2024, pp. 1-4, doi: 10.1109/EMBC53108.2024.10782132.
