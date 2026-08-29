from dataclasses import dataclass
import torch as t
from nnsight.intervention.envoy import Envoy
from collections import namedtuple
from dictionary_learning import AutoEncoder, JumpReluAutoEncoder
from dictionary_learning.dictionary import IdentityDict
from typing import Literal, Optional
from huggingface_hub import list_repo_files
from tqdm import tqdm
import os

DICT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/dictionaries"

@dataclass(frozen=True)
class Submodule:
    name: str
    submodule: Envoy
    layer_idx: Optional[int] = None
    pre_ln: Optional[Envoy] = None
    post_ln: Optional[Envoy] = None

    def __hash__(self):
        return hash(self.name)
    
    @property
    def path(self):
        return self.submodule.config._name_or_path

    @property
    def is_attn(self):
        return 'attn' in self.name
    
    @property
    def is_mlp(self):
        return 'mlp' in self.name
    
    @property
    def is_resid(self):
        return 'resid' in self.name
    
    @property
    def is_head(self):
        return 'lm_head' == self.name
    
    @property
    def device(self):
        return self.submodule.device
    
    @property
    def config(self):
        return self.submodule.config

    def get_in_activation(self):
        inp = self.submodule.input
        return inp
    
    def get_pre_ln_in_activation(self):
        if self.pre_ln is not None:
            inp = self.pre_ln.input
        else:
            inp = self.submodule.input
        return inp
    
    def set_in_activation(self, x):
        self.submodule.input = x
    
    def get_out_activation(self):
        if self.post_ln is not None:
            out = self.post_ln.output
        else:
            out = self.submodule.output
        if isinstance(out, (tuple, list)):
            return out[0]
        else:
            return out

    def set_out_activation(self, x):
        if self.post_ln is not None:
            submodule = self.post_ln
        else:
            submodule = self.submodule
        if isinstance(submodule.output, (tuple, list)):
            submodule.output[0][:] = x
        else:
            submodule.output[:] = x

    def stop_grad(self):
        if isinstance(self.submodule.output, (tuple, list)):
            self.submodule.output[0].grad[:] = t.zeros_like(self.submodule.output[0])
        else:
            self.submodule.output.grad[:] = t.zeros_like(self.submodule.output)

    def get_qkv_inputs_separate(self):
        """Returns three SEPARATE tensors for Q, K, V gradient tracking"""
        assert self.is_attn, f"get_qkv_inputs_separate only valid for attention, got {self.name}"
        shared_input = self.submodule.q_proj.input
        # Clone to create separate graph nodes
        q_in = shared_input.clone()
        k_in = shared_input.clone()
        v_in = shared_input.clone()
        return q_in, k_in, v_in
    
    def set_qkv_inputs_separate(self, q_in, k_in, v_in):
        """
        Replace Q/K/V projections to use separate input tensors.
        Call this AFTER get_qkv_inputs_separate and setting requires_grad.
        """
        assert self.is_attn, f"set_qkv_inputs_separate only valid for attention, got {self.name}"
        attn = self.submodule
        
        # Manually compute Q, K, V
        Q = t.matmul(q_in, attn.q_proj.weight.T)
        K = t.matmul(k_in, attn.k_proj.weight.T)
        V = t.matmul(v_in, attn.v_proj.weight.T)
        
        if attn.q_proj.bias is not None:
            Q = Q + attn.q_proj.bias
        if attn.k_proj.bias is not None:
            K = K + attn.k_proj.bias
        if attn.v_proj.bias is not None:
            V = V + attn.v_proj.bias
        
        attn.q_proj.output = Q
        attn.k_proj.output = K
        attn.v_proj.output = V
        return Q, K, V
    
    def get_per_head_outputs(self):
        assert self.is_attn
        attn = self.submodule
        
        W_O = attn.o_proj.weight
        d_model = W_O.shape[0]
        total_head_dim = W_O.shape[1]
        d_head = attn.head_dim
        n_heads = attn.config.num_attention_heads
        
        z_flat = attn.o_proj.input
        
        # Now z_flat should be (batch, seq, n_heads * d_head)
        z = z_flat.reshape(z_flat.shape[0], z_flat.shape[1], n_heads, d_head)
        W_O_heads = W_O.reshape(d_model, n_heads, d_head)
        
        head_outputs = t.einsum('bsnh,dnh->bsnd', z, W_O_heads)
        return head_outputs
    
    def replace_attn_output_with_head_sum(self, head_outputs):
        """
        Replace o_proj output with sum of head_outputs.
        This routes gradients through the per-head tensor.
        This has small rounding errors, so do not use. 
        Can calculate head grads manually with get_head_grads instead.
        """
        assert self.is_attn
        attn = self.submodule
        new_output = head_outputs.sum(dim=2)
        if attn.o_proj.bias is not None:
            new_output = new_output + attn.o_proj.bias
        attn.o_proj.output = new_output
        return new_output
    
    def get_head_grads(self, output_grad):
        """
        Given gradient w.r.t. attention output (batch, seq, d_model),
        return gradient w.r.t. per-head outputs (batch, seq, n_heads, d_model).
        
        Since output = sum(head_outputs, dim=2), grad is just broadcasted.
        """
        n_heads = self.submodule.config.num_attention_heads
        
        # (batch, seq, d_model) -> (batch, seq, 1, d_model) -> (batch, seq, n_heads, d_model)
        return output_grad.unsqueeze(2).expand(-1, -1, n_heads, -1)
    
    def handle_post_attn_mlp_norm(self, model, acts):
        def get_norm_stats(x, ln_eps):
            return t.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + ln_eps)
        def get_norm_v(raw_in_v, ln_stats, ln_weight):
            in_v = raw_in_v.float() * ln_stats
            in_v = in_v * (1.0 + ln_weight.float())
            return in_v.type_as(raw_in_v)
        if self.is_head:
            return acts
        if self.path.startswith("google/gemma-2-"): # has a post norm after both attn and mlp
            layer_idx = int(self.name.split('_')[1])
            resid = model.model.layers[layer_idx]

            if self.is_attn:
                post_ln = resid.post_attention_layernorm
                inp = post_ln.input
                stats = get_norm_stats(inp, post_ln.eps)
                acts_norm = get_norm_v(acts, stats.unsqueeze(dim=-1), post_ln.weight)
                return acts_norm
            elif self.is_mlp:
                post_ln = resid.post_feedforward_layernorm
                inp = post_ln.input
                stats = get_norm_stats(inp, post_ln.eps)
                acts_norm = get_norm_v(acts, stats, post_ln.weight)
                return acts_norm
            else:
                return acts
        else:
            return acts


SubmoduleStash = namedtuple("SubmoduleStash", ["embed", "attns", "mlps", "resids", "lm_head"])


def _load_qwen_submodules(
    model,
    separate_by_type: bool = False,
):
    attns = []
    mlps = []
    resids = []
    embed = Submodule(
        name="embed",
        submodule=model.model.embed_tokens,
    )
    """This does not have a post ln before adding to residual stream"""
    for i, layer in tqdm(
        enumerate(model.model.layers),
        total=len(model.model.layers),
        desc="Loading Model Submodules",
    ):
        attns.append(
            attn := Submodule(
                name=f"attn_{i}", submodule=layer.self_attn, pre_ln=layer.input_layernorm, layer_idx=i
            )
        )
        mlps.append(
            mlp := Submodule(
                name=f"mlp_{i}", submodule=layer.mlp, pre_ln=layer.post_attention_layernorm, layer_idx=i
            )
        )
        resids.append(
            resid := Submodule(name=f"resid_{i}", submodule=layer
            )
        )
    
    lm_head = Submodule(name="lm_head", submodule=model.lm_head, pre_ln=model.model.norm)

    if separate_by_type:
        return SubmoduleStash(embed, attns, mlps, resids, lm_head)
    else:
        submodules = [embed] + [
            x
            for layer_dictionaries in zip(attns, mlps, resids)
            for x in layer_dictionaries
        ] + [lm_head]
        return submodules

def _load_gemma_submodules(
    model,
    separate_by_type: bool = False,
):
    """This has a pre and post ln before adding to residual stream"""
    attns = []
    mlps = []
    resids = []
    embed = Submodule(
        name="embed",
        submodule=model.model.embed_tokens,
    )

    for i, layer in tqdm(
        enumerate(model.model.layers),
        total=len(model.model.layers),
        desc="Loading Model Submodules",
    ):
        attns.append(
            attn := Submodule(
                name=f"attn_{i}", submodule=layer.self_attn, pre_ln=layer.input_layernorm, post_ln=layer.post_attention_layernorm, layer_idx=i
            )
        )
        mlps.append(
            mlp := Submodule(
                name=f"mlp_{i}", submodule=layer.mlp, pre_ln=layer.pre_feedforward_layernorm, post_ln=layer.post_feedforward_layernorm, layer_idx=i
            )
        )
        resids.append(
            resid := Submodule(name=f"resid_{i}", submodule=layer
            )
        )
    
    lm_head = Submodule(name="lm_head", submodule=model.lm_head, pre_ln=model.model.norm)

    if separate_by_type:
        return SubmoduleStash(embed, attns, mlps, resids, lm_head)
    else:
        submodules = [embed] + [
            x
            for layer_dictionaries in zip(attns, mlps, resids)
            for x in layer_dictionaries
        ] + [lm_head]
        return submodules
    
def _load_llama_submodules(
    model,
    separate_by_type: bool = False,
):

    attns = []
    mlps = []
    resids = []
    embed = Submodule(
        name="embed",
        submodule=model.model.embed_tokens,
    )

    for i, layer in tqdm(
        enumerate(model.model.layers),
        total=len(model.model.layers),
        desc="Loading Model Submodules",
    ):
        attns.append(
            attn := Submodule(
                name=f"attn_{i}", submodule=layer.self_attn, pre_ln=layer.input_layernorm, layer_idx=i
            )
        )
        mlps.append(
            mlp := Submodule(
                name=f"mlp_{i}", submodule=layer.mlp, pre_ln=layer.post_attention_layernorm, layer_idx=i
            )
        )
        resids.append(
            resid := Submodule(name=f"resid_{i}", submodule=layer
            )
        )
    
    lm_head = Submodule(name="lm_head", submodule=model.lm_head, pre_ln=model.model.norm)

    if separate_by_type:
        return SubmoduleStash(embed, attns, mlps, resids, lm_head)
    else:
        submodules = [embed] + [
            x
            for layer_dictionaries in zip(attns, mlps, resids)
            for x in layer_dictionaries
        ] + [lm_head]
        return submodules

def load_submodules(
    model,
    separate_by_type: bool = True,
):
    model_name = model.config._name_or_path

    if model_name.startswith("google/gemma"):
        return _load_gemma_submodules(model, separate_by_type=separate_by_type)
    elif model_name.startswith("meta-llama"):
        return _load_llama_submodules(model, separate_by_type=separate_by_type)
    elif model_name.startswith("Qwen"):
        return _load_qwen_submodules(model, separate_by_type=separate_by_type)
    else:
        raise ValueError(f'invalid model: {model}')
    
def format_neg_circuit_components(
    neg_nodes: list[tuple[str, int]],
    neg_edges: list[tuple[str, int, str, str]],
    submodules: SubmoduleStash,
):
    def map_name_to_submodule(name):
        for submodule in submodules:
            if submodule.name in name:
                return submodule
        raise ValueError('submodule not found')
    
    # handle nodes
    nodes_to_test = []
    for pair in neg_nodes:
        name = pair[0]
        submodule = map_name_to_submodule(name)
        if submodule.is_attn:
            nodes_to_test.append(submodule, pair[1])
        else:
            nodes_to_test.append(submodule, None)
    
    # handle edges
    edges_to_test = []

    
#### TESTING ####

def test_submodule_shapes(model, submodule):
    """Test that activations and gradients have consistent shapes [batch, seq_len, dim]"""
    # Get baseline output
    with model.trace("test input"):
        out1 = model.output.logits.sum().save()
    
    # Test gradient shape in separate trace
    with model.trace("test input"):
        acts = submodule.get_activation().save()
        grads = submodule.get_activation().grad.save()
        out2 = model.output.logits.sum().save()
        out2.backward()

    assert t.allclose(out1.value, out2.value), f"Submodule {submodule.name}: Output should be the same before and after grabbing acts/grads"
    assert len(acts.shape) == 3, f"Submodule {submodule.name}: Expected 3D tensor [batch, seq_len, dim], got shape {acts.shape}"
    assert grads.shape == acts.shape, f"Submodule {submodule.name}: Gradient shape {grads.shape} doesn't match activation shape {acts.shape}"

def test_activation_intervention(model, submodule):
    """Test that zeroing activations changes the output"""
    # Get baseline output
    with model.trace("test input"):
        acts = submodule.get_activation().save()
        out1 = model.output.logits.sum().save()
    
    # Test intervention in separate trace
    with model.trace("test input"):
        submodule.set_activation(t.randn_like(acts.value))
        out2 = model.output.logits.sum().save()
    
    assert not t.allclose(out1.value, out2.value), f"Submodule {submodule.name}: Output should change after modifying activation"

def test_gradient_intervention(model, upstream_submodule, downstream_submodule):
    """Test that stopping gradients affects upstream gradients"""
    # Get baseline gradients
    with model.trace("test input"):
        grads1 = upstream_submodule.get_activation().grad.save()
        out1 = model.output.logits.sum().save()
        out1.backward()
    
    # Test intervention
    with model.trace("test input"):
        grads2 = upstream_submodule.get_activation().grad.save()
        downstream_submodule.stop_grad()
        out2 = model.output.logits.sum().save()
        out2.backward()
    
    assert t.allclose(out1.value, out2.value), f"Submodule {upstream_submodule.name}: Output should be same before and after stopping gradients"
    assert not t.allclose(grads1.value, grads2.value), f"Submodule {upstream_submodule.name}: Gradients should change after stopping grad flow in {downstream_submodule.name}"

def run_tests(model):
    """Run all tests for the Submodule class"""
    submodules, _ = load_submodules(model)  # Use neurons=True to avoid loading dictionaries
    
    # Test shapes on different types of submodules
    print("Testing submodule shapes...")
    test_cases = [
        submodules[0],  # embed
        submodules[1],  # first attn
        submodules[2],  # first mlp
        submodules[3],  # first resid
        submodules[-3], # last mlp
        submodules[-1], # last resid
    ]
    for submodule in test_cases:
        print(f"Testing shapes for {submodule.name}...")
        test_submodule_shapes(model, submodule)
    print("Shape tests passed!")
    
    # Test activation interventions
    print("\nTesting activation interventions...")
    for submodule in test_cases:
        print(f"Testing activation intervention for {submodule.name}...")
        test_activation_intervention(model, submodule)
    print("Activation intervention tests passed!")
    
    # Test gradient interventions
    print("\nTesting gradient interventions...")
    gradient_test_pairs = [
        (submodules[0], submodules[2]),   # attn_0 -> resid_0
        (submodules[1], submodules[2]),   # mlp_0 -> resid_0
        (submodules[-2], submodules[-1]), # last mlp -> last resid
    ]
    for upstream, downstream in gradient_test_pairs:
        print(f"Testing gradient intervention from {upstream.name} to {downstream.name}...")
        test_gradient_intervention(model, upstream, downstream)
    print("Gradient intervention tests passed!")
    
    print("\nAll tests passed!")

def test_attn_in(model, submodules):
    q_clone, k_clone, v_clone = {}, {}, {}
    q_orig, k_orig, v_orig = {}, {}, {}

    with model.trace(t.tensor([[27, 30]])):
        for submodule in submodules:
            if submodule.name.startswith("attn"):
                q_in, k_in, v_in = submodule.get_qkv_inputs_separate()
                q_in.requires_grad_()
                k_in.requires_grad_()
                v_in.requires_grad_()

                Q, K, V = submodule.set_qkv_inputs_separate(q_in, k_in, v_in)

                q_clone[submodule.name] = Q.save()
                k_clone[submodule.name] = K.save()
                v_clone[submodule.name] = V.save()

    with model.trace(t.tensor([[27, 30]])):
        for submodule in submodules:
            if submodule.name.startswith("attn"):
                q_orig[submodule.name] = submodule.submodule.q_proj.output.save()
                k_orig[submodule.name] = submodule.submodule.k_proj.output.save()
                v_orig[submodule.name] = submodule.submodule.v_proj.output.save()

    for name1, name2 in zip(q_clone, q_orig):
        assert(t.allclose(q_clone[name1], q_orig[name2]))
    for name1, name2 in zip(k_clone, k_orig):
        assert(t.allclose(k_clone[name1], k_orig[name2]))
    for name1, name2 in zip(v_clone, v_orig):
        assert(t.allclose(v_clone[name1], v_orig[name2]))

def test_attn_out(model, submodules):
    n_heads = model.config.num_attention_heads
    d_model = model.config.hidden_size
    print(d_model, n_heads)

    head_outs = {}
    attn_outs_clone = {}
    attn_outs_orig = {}
    outs_orig = {}
    with model.trace(t.tensor([[27, 30]])):
        for submodule in submodules:
            if submodule.name.startswith("attn"):
                head_out = submodule.get_per_head_outputs()

                head_out.requires_grad_().retain_grad()
                # attn = submodule.submodule
                # attn_in = attn.o_proj.input
                # W_O = attn.o_proj.weight
                # new_out = t.matmul(attn_in, W_O.T)
                # if attn.o_proj.bias is not None:
                #     new_out = new_out + attn.o_proj.bias

                new_out = submodule.replace_attn_output_with_head_sum(head_out)
                head_outs[submodule.name] = head_out.save()
                attn_outs_clone[submodule.name] = new_out.save()
            
        logits = model.lm_head.output
        print(logits.shape)
        # loss = logits[:, -1, :10].sum()
        # loss.backward()

    with model.trace(t.tensor([[27, 30]])):
        for submodule in submodules:
            if submodule.name.startswith("attn"):
                attn = submodule.submodule
                out1 = attn.o_proj.output
                out1[0].requires_grad_().retain_grad()
                out2 = attn.output
                out2[0].requires_grad_().retain_grad()
                attn_outs_orig[submodule.name] = out1.save()
                outs_orig[submodule.name] = out2.save()
        logits = model.lm_head.output
        print(logits.shape)
        # loss = logits[:, -1, :10].sum()
        # loss.backward()

    for name in attn_outs_clone:
        clone_val = attn_outs_clone[name]
        orig_val = attn_outs_orig[name]
        print(clone_val.shape, clone_val[0,-1])
        print(orig_val.shape, orig_val[0,-1])
        assert t.allclose(orig_val, clone_val, atol=1e-4) # need very forgiving bounds
        assert t.allclose(orig_val, clone_val, atol=1e-1, rtol=1e-2) # need very forgiving bounds
    
    for name in outs_orig:
        assert t.allclose(attn_outs_orig[name], outs_orig[name][0])
        # assert t.allclose(attn_outs_orig[name].grad, outs_orig[name][0].grad)

    for submodule in submodules:
        if submodule.name.startswith("attn"):
            head_outs[submodule.name].grad = submodule.get_head_grads(outs_orig[submodule.name][0].grad)


if __name__ == "__main__":
    from nnsight import LanguageModel
    model = LanguageModel("google/gemma-2-2b-it", dispatch=True, device_map="auto", dtype=t.float32)
    submodules = load_submodules(model, separate_by_type=False)
    test_attn_in(model, submodules)
    test_attn_out(model, submodules)

    print('passed test')
    


    # print("Running tests for gemma-2-2b-it")
    # run_tests(model)



