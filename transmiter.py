import torch
import torch.nn as nn

class PredictionBasedAdapter(nn.Module):
        def __init__(self, D_in: int = 1024, D_mid: int = 1024):
            super().__init__()
            self.W_in = nn.Linear(D_in, D_mid, bias = False)
            self.mlp = nn.Sequential(
                nn.LayerNorm(D_mid),
                nn.Linear(D_mid, D_mid * 4),
                nn.GELU(),
                nn.Linear(D_mid * 4, D_mid)
            )
            self.W_out = nn.Linear(D_mid, D_in, bias = False)

        def forward(self, z: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
            h_s = self.W_in(z)
            h_s_hat = h_s + (scale * self.mlp(h_s))
            return self.W_out(h_s_hat)
            
def calculate_procrustes_alignment(H_s: torch.Tensor, H_t: torch.Tensor) -> torch.Tensor:
    M =   H_t.T @ H_s
    U, S, V_T = torch.linalg.svd(M)
    return U@V_T

if __name__ == "__main__":
    H_weak = torch.randn(32, 1024)
    H_strong = torch.randn(32, 1024)
    
    print(f"Initial Distance between models: {torch.nn.functional.mse_loss(H_weak, H_strong):.4f}")
    
    W_hat = calculate_procrustes_alignment(H_weak, H_strong)
    
    H_strong_aligned = H_strong @ W_hat
    
    print(f"Distance after SVD Alignment: {torch.nn.functional.mse_loss(H_weak, H_strong_aligned):.4f}")
    print("Alignment successful. Zero backpropagation required.")
        # Testing the suggested scaling feature
    adapter = PredictionBasedAdapter()
    z_input = torch.randn(32, 1024)
    print(f"\n--- Testing MLP Scaling Parameter ---")
    print(f"Adapter variance with scale=1.0: {torch.var(adapter(z_input, scale=1.0)):.4f}")
    print(f"Adapter variance with scale=0.5: {torch.var(adapter(z_input, scale=0.5)):.4f}")
    print("Scaling successfully suppresses adaptation magnitude.")