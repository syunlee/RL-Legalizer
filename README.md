# RL-Legalizer
RL-Legalizer: Reinforcement Learning-based Mixed-Height Standard Cell Legalization

## Contact
This project is implemented by Sung-Yun Lee and Seonghyeon Park (Ph.D. Advisor: Prof. Seokhyeong Kang) \
Contact: syun.lee@postech.ac.kr

## Reference paper
Title: "RL-Legalizer: Reinforcement Learning-based Cell Priority Optimization in Mixed-Height Standard Cell Legalization" \
Authors: S.-Y. Lee, S. Park, D. Kim, M. Kim, T. P. Le and S. Kang \
Conference: 2023 26th IEEE/ACM Design, Automation and Test in Europe Conference \& Exhibition (DATE 2023) (to appear)

## Based legalization algorithm
[OpenDP: Open Source Detailed Placement Engine](https://github.com/sanggido/OpenDP) \
Reference paper: S. Do, M. Woo and S. Kang, "Fence-region-aware mixed-height standard cell legalization,” in IEEE/ACM GLSVLSI 2019 ([link](https://dl.acm.org/doi/10.1145/3299874.3318012)) \
This legalizer has been embedded in [OpenROAD](https://github.com/The-OpenROAD-Project/OpenROAD/tree/master/src/dpl)

## File description

`RLagent.py`: PyTorch RL agent (including agent, model, train algorithms, environment, etc.)

