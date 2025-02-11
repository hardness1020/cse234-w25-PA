from typing import Any, Dict, List
import torch
from auto_diff import *

class MatMulLayerNormOp(Op):
    """Fused matrix multiplication and layer normalization operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        normalized_shape: List[int], 
        eps: float = 1e-5
    ) -> Node:
        """
        Args:
            node_A: The first input node.
            node_B: The second input node.
            normalized_shape: The shape of the normalization axes.
            eps: The epsilon value to avoid division by zero.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "normalized_shape": normalized_shape,
                "eps": eps
            },
            name=f"MatMulLayerNorm({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and layer normalization result."""
        assert len(input_values) == 2
        matmul_result = torch.matmul(input_values[0], input_values[1])
        layer_norm_result = torch.nn.functional.layer_norm(
            matmul_result, node.attrs["normalized_shape"], eps=node.attrs["eps"]
        )
        return layer_norm_result
    
    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        A, B = node.inputs
        normalized_shape = node.attrs["normalized_shape"]
        eps = node.attrs["eps"]
        
        C = matmul(A, B)
        D = layernorm(C, normalized_shape, eps)
        
        grad_C = layernorm.gradient(D, output_grad)[0]
        grad_A, grad_B = matmul.gradient(C, grad_C)
        return [grad_A, grad_B]
        

class MatMulSoftmaxOp(Op):
    """Fused matrix multiplication and softmax operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        dim: int = -1
    ) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "dim": dim
            },
            name=f"MatMulSoftmax({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and softmax result."""
        assert len(input_values) == 2
        matmul_result = torch.matmul(input_values[0], input_values[1])
        softmax_result = torch.nn.functional.softmax(matmul_result, dim=node.attrs["dim"])
        return softmax_result

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        # First compute the forward pass result we need for softmax gradient
        A, B = node.inputs
        dim = node.attrs["dim"]
        
        C = matmul(A, B)
        D = softmax(C, dim)
        
        # Compute the gradient of the softmax
        grad_C = softmax.gradient(D, output_grad)[0]
        grad_A, grad_B = matmul.gradient(C, grad_C)
        
        return [grad_A, grad_B]

# Create global instances of the fused ops
matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()