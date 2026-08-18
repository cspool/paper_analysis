# Efficient 3D Gaussian Splatting with Axis-Shared Rasterization and Order-independent Transmittance

Zhican Wang\*, Guanghui He\*, Lingjun Gao<sup>‡</sup>, Dantong Liu<sup>†</sup>, Shell Xu Hu<sup>§</sup>, Chen Zhang\*, Zhuoran Song\*, Nicholas Lane<sup>†</sup>, and Hongxiang Fan<sup>‡</sup>

\*Shanghai Jiao Tong University {wang\_zhican, guanghui.he, chenzhang, songzhuoran}@sjtu.edu.cn

†University of Cambridge liudt921115@gmail.com, ndl32@cam.ac.uk

‡Imperial College London {lingjun.gao24, hongxiang.fan}@imperial.ac.uk

§Samsung AI shell.hu@samsung.com

Abstract-3D Gaussian Splatting (3DGS) has emerged as a powerful technique for novel view synthesis, combining highquality reconstruction with efficient rendering. It has been widely adopted in domains such as AR/VR, robotics, and autonomous driving. However, achieving real-time performance on resourceconstrained platforms remains challenging due to strict power and area budgets. Prior accelerators improve hardware performance but still overlook key inefficiencies, including insufficient rasterization efficiency, poor sorting scalability, and pipeline imbalance. This paper presents an architecture-algorithm codesign to address these challenges. First, we propose axis-shared rasterization, which precomputes and reuses common terms along the X- and Y-axes, reducing multiply-and-accumulate (MAC) operations by up to 38% while preserving high parallelism. Second, we develop a novel order-independent transmittance method that removes the need for explicit sorting by leveraging a lightweight Multilayer Perceptron (MLP) to directly approximate the transmittance of each Gaussian, enabling efficient  $\alpha$  blending with negligible quality loss. Third, we design a unified reconfigurable PE array that supports both rasterization and MLP inference, sustaining high utilization without costly sorting hardware. Our experiments demonstrate that our design preserves rendering quality while achieving a  $1.33 \sim 1,88 \times$  speedup over the state-of-the-art 3DGS accelerators. Our code is open source at https://github.com/WangZhican/ISCA26\_3DGS\_Acc.

