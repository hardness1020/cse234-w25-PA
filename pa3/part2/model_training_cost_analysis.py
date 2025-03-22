import argparse
import json
import math

def model_training_cost_analysis_llama(model_config_path):
    #TODO you code here.
    with open(model_config_path, 'r') as f:
        config = json.load(f)
        b = 1 
        s = config['max_sequence_length']
        n = config['num_attention_heads']
        h = config['hidden_size']
        i = config['intermediate_size']
        l = config['num_hidden_layers']
        v = config['vocab_size']
        
        # Calculate total parameters
        embeding_layer = v*h + s*h  # word embeddings + positional embeddings
        transformer_layer = l * (h + 4*h*h + h + 3*h*i)
        total_params = embeding_layer + transformer_layer
        
        # Calculate TFLOPs for a single Transformer layers
        W_att = 3 * 2*b*s*h*h
        P = 2*b*s*s*h + 3*b*s*s*n
        PV = 2*b*s*s*h
        AW = 2*b*s*h*h
        W_gate = 2*b*s*h*i
        W_up = 2*b*s*h*i
        W_down = 2*b*s*h*i
        tflops_layer_TF = (W_att + P + PV + AW + W_gate + W_up + W_down) / 1e12
        
        # Calculate peak memory for a single transformer layer cost in GB (fp16)
        params = h + 4*h*h + h + 3*h*i
        GB = 1024**3
        params_memory_GB = 2 * params / GB
        optimizer_state_memory_GB = (2+2+2+2+2) * params / GB  # fixed fp16 precision training
        gradients_memory_GB = 2 * params / GB
        activations_memory_GB = 2 * b*s*h / GB  # checkpoint rematerialization
        peak_memory_GB = params_memory_GB + optimizer_state_memory_GB + \
            gradients_memory_GB + activations_memory_GB
    return total_params, tflops_layer_TF, peak_memory_GB

def model_training_cost_analysis_deepseek(model_config_path):
    with open(model_config_path, 'r') as f:
        config = json.load(f)
        b = 1  # batch size
        s = config['max_position_embeddings']
        n = config['num_attention_heads']
        h = config['hidden_size']
        i = config['moe_intermediate_size']
        l = config['num_hidden_layers']
        v = config['vocab_size']
        num_experts = config['n_routed_experts']
        num_shared_experts = config['n_shared_experts']
        num_selected = config['num_experts_per_tok']
        head_dim = config['qk_rope_head_dim']  # latent attention
        
        # 1. Total Parameters (MoE adaptation)
        embedding_params = v * h + s * head_dim  # word embeddings + positional embeddings
        attn_params = (3 * h * head_dim) + (h * h)  # QKV to head_dim + output
        moe_params = (num_experts * 3 * h * i) + (h * num_experts)  # experts + router
        shared_expert_params = num_shared_experts * 3 * h * i  # shared experts
        layer_norm_params = 4 * h  # 2 norms * 2 params each
        params_per_layer = attn_params + moe_params + shared_expert_params + layer_norm_params
        
        transformer_params = l * params_per_layer
        total_params = embedding_params + transformer_params
        
        
        # 2. Calculate TFLOPs for a single Transformer layers
        # Attention FLOPs (same as Llama)
        W_att = 3 * 2*b*s*h*head_dim
        P = 2*b*s*s*head_dim + 3*b*s*s*n
        PV = 2*b*s*s*head_dim
        AW = 2*b*s*h*head_dim
        flops_att = W_att + P + PV + AW
        
        # MoE-specific: router + selected experts + shared experts
        router = 2 * b * s * h * num_experts  # Routing computation
        experts = num_selected * 3 * 2 * b * s * h * i  # Selected experts
        shared = num_shared_experts * 3 * 2 * b * s * h * i  # Shared experts
        
        flops_moe = router + experts + shared
        tflops_layer_TF = (flops_att + flops_moe) / 1e12
        
        
        # 3. Calculate peak memory for a single transformer layer cost in GB
        GB = 1024 ** 3
        params_memory_GB = 1 * params_per_layer / GB  # fp8 parameters
        optimizer_memory_GB = (4+2+1+2+2) * params_per_layer / GB  # fp8-fp16-fp32 mixed precision
        gradients_memory_GB = 2 * params_per_layer / GB  # fp16 gradients
        activations_memory_GB = (1*b*s*h + 1*b*s*num_selected*i) / GB  # attention + MoE (fp8)
        peak_memory_GB = params_memory_GB + optimizer_memory_GB + \
            gradients_memory_GB + activations_memory_GB
    return total_params, tflops_layer_TF, peak_memory_GB

def get_optimal_N_D_from_cost(cost_budget):
    """
    cost_budget:  a monetary training budget (in dollars)
    Returns:
        N: Optimal total model parameters (in absolute numbers)
        D: Optimal number of training tokens (in absolute numbers)
        training_budget_flops: Effective total training FLOPs (in FLOPs)
        best_gpu: name of the selected GPU (one of 'A100', 'V100', 'T4')
    """
    # 1. Select the best GPU and calculate the maximum N*D (k) by the budget and the selected GPU
    gpus = [
        {'name': 'A100', 'cost_per_hour': 4.0, 'flops': 312},
        {'name': 'V100', 'cost_per_hour': 2.5, 'flops': 125},
        {'name': 'T4', 'cost_per_hour': 1.0, 'flops': 65}
    ]

    max_k = -1  # N*D
    best_gpu_info = None

    for gpu in gpus:
        numerator = cost_budget * gpu['flops'] * 1e12 * 0.4 * 3600
        denominator = 6 * gpu['cost_per_hour']  # training a transformer model requires 
                                                # about 6 FLOPs per parameter per token
        k = numerator / denominator
        if k > max_k:
            max_k = k
            best_gpu_info = gpu
    
    # 2. Compute optimal N and D by solving the scaling law under the constraint k (N*D)
    #    Use the method of Lagurange multipliers and take partial derivatives to get the following equation
    ratio = (406.4 * 0.37) / (410.7 * 0.29)
    term = ratio * (max_k ** 0.29)
    exponent = 1.0 / (0.34 + 0.29)
    N = term ** exponent
    D = max_k / N
    
    # 3. Compute traing Flops. For a transformer model, the training FLOPs are roughly 6*N*D
    training_budget_flops = 6 * max_k

    return round(N), round(D), training_budget_flops, best_gpu_info['name']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training cost analysis')
    parser.add_argument('--model_config', type=str, help='Path to model config file')
    parser.add_argument('--training_budget', type=float, default=None, help='Training budget')
    args = parser.parse_args()

    if args.model_config:
        if 'deepseek' in args.model_config:
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_deepseek(args.model_config)
        elif 'llama' in args.model_config:
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_llama(args.model_config)
        else:
            print('Unknown LLM Type!')
            exit()
        print(f"Number of parameters: {num_parameters}")
        print(f"Number of TFLOPs: {num_flops}")
        print(f"Peak memory cost: {memory_cost} GBs")

    if args.training_budget:    
        N, D, training_budget_flops, best_gpu = get_optimal_N_D_from_cost(args.training_budget)
        print(f"best_gpu: {best_gpu}")
        print(f"training_budget_flops: {training_budget_flops}")
        print(f"Optimal N: {N}")
        print(f"Optimal D: {D}")

    