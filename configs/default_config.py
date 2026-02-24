def get_default_config():
    config = {
        # 训练模式:
        #   'standalone_cls'            — [A] 干净 CIFAR 训练 ResNet
        #   'standalone_cls_noisy'      — [B] 噪声 CIFAR 训练 ResNet
        #   'standalone_denoise'        — 单独训练 UNet 去噪
        #   'multitask_no_sidd'         — [C] 噪声CIFAR + 注入, 无SIDD
        #   'multitask_no_cifar_noise'  — [D] 干净CIFAR + 注入 + SIDD
        #   'multitask'                 — [M] 噪声CIFAR + 注入 + SIDD
        'mode': 'multitask',

        # 公共参数
        'epochs': 50,
        'seed': 1234,

        # CIFAR-10 分类
        'cifar_data_dir': './cifar10_data',
        'cifar_batch_size': 64,

        # SIDD 去噪
        'sidd_data_dir': './SIDD_Medium_Srgb/Data',
        'sidd_patch_size': 512,
        'sidd_batch_size': 4,

        # UNet 架构
        'unet_base_channels': 32,
        'unet_num_cab': 2,

        # 跨任务特征注入参数
        'noise_sigma_range': (0.05, 0.25),   # CIFAR 训练时添加的噪声范围
        'grad_scale_cls2unet': 0.1,          # cls_loss → UNet 梯度缩放因子
        'lambda_denoise_cifar': 1.0,         # CIFAR 去噪损失权重

        # 优化器
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'warmup_epochs': 5,

        # CIFAR-10-C 评估
        'cifar10c_dir': './CIFAR-10-C',
        'cifar10c_severities': [1, 3, 5],

        # 其他
        'num_workers': 4,
        'save_every': 10,
    }
    return config
