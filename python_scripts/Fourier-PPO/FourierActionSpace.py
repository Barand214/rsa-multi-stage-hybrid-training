import numpy as np

# ===================== 傅里叶动作空间 =====================
class FourierActionSpace:
    """
    傅里叶参数动作空间包装器
    """
    def __init__(self, n_servos=20, max_harmonics=20, T_max=10.0):
        self.n_servos = n_servos
        self.max_harmonics = max_harmonics
        self.T_max = T_max
        
        # 参数维度说明：
        # 1. n_harmonics: 1个离散参数 [1, max_harmonics]
        # 2. T: 1个连续参数 (0, T_max]
        # 3. A: n_servos * max_harmonics个参数 [-1, 1]
        # 4. ω: max_harmonics个参数 (0, ∞)
        # 5. φ: n_servos * max_harmonics个参数 [0, T]
        
        # 总连续参数维度
        self.continuous_param_dim = 1 + (n_servos * max_harmonics) + max_harmonics + (n_servos * max_harmonics)
        
    def sample_random(self):
        """随机采样参数（用于初始化）"""
        # n_harmonics: 离散值 [1, max_harmonics]
        n = np.random.randint(1, self.max_harmonics + 1)
        
        # T: 连续值 (0, T_max]
        T = np.random.uniform(0.1, self.T_max)
        
        # A_i ∈ [-1, 1]
        A = np.random.uniform(-1, 1, (self.n_servos, self.max_harmonics))
        
        # ω_i ∈ (0, ∞)，实际用(0, 20π]比较合理（对应频率0-10Hz）
        ω = np.random.uniform(0.1, 20 * np.pi, self.max_harmonics)
        
        # φ_i ∈ [0, T]
        φ = np.random.uniform(0, T, (self.n_servos, self.max_harmonics))
        
        return {
            'n': n,
            'T': T,
            'A': A,
            'ω': ω,
            'φ': φ
        }
    
    def get_fourier_curve(self, params, t):
        """计算在时间t的舵机角度"""
        n = params['n']
        T = params['T']
        A = params['A']
        ω = params['ω']
        φ = params['φ']
        
        # 确保t在合理范围内
        t = t % T  # 循环动作
        
        angles = np.zeros(self.n_servos)
        
        for servo_idx in range(self.n_servos):
            # f(t) = A_0/2 + Σ_{k=1}^{n-1} A_k * cos(ω_k*t - φ_k)
            # 注意：A[servo_idx, 0]对应A_0
            angle = A[servo_idx, 0] / 2.0
            
            for harmonic in range(1, n):
                angle += A[servo_idx, harmonic] * np.cos(
                    ω[harmonic] * t - φ[servo_idx, harmonic]
                )
            
            # 限制角度在合理范围内（假设舵机角度范围[-1, 1]对应实际角度范围）
            angles[servo_idx] = np.clip(angle, -1.0, 1.0)
        
        return angles
    
    def batch_get_fourier_curve(self, params, t_values):
        """批量计算多个时间点的舵机角度"""
        n = params['n']
        T = params['T']
        A = params['A']
        ω = params['ω']
        φ = params['φ']
        
        # t_values: [batch_size] 或标量
        t_values = np.array(t_values) % T
        
        if np.isscalar(t_values):
            t_values = np.array([t_values])
        
        batch_size = len(t_values)
        angles = np.zeros((batch_size, self.n_servos))
        
        for i, t in enumerate(t_values):
            for servo_idx in range(self.n_servos):
                angle = A[servo_idx, 0] / 2.0
                
                for harmonic in range(1, n):
                    angle += A[servo_idx, harmonic] * np.cos(
                        ω[harmonic] * t - φ[servo_idx, harmonic]
                    )
                
                angles[i, servo_idx] = np.clip(angle, -1.0, 1.0)
        
        return angles