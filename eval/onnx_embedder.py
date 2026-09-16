"""An int8 ONNX embedder that ReasonGraph can use (any object with encode() works). This is how a big embedder
like multilingual-e5-large is run on a CPU box: export the transformer to ONNX, dynamic-int8 quantize, and pool in
numpy. Exported files are cached under eval/.onnx_cache/ on first use (they are large, so they are not committed).

Selected via REASONGRAPH_EMBED_MODEL="onnx-int8:intfloat/multilingual-e5-large". model_name is set to the source
id so ReasonGraph applies the model family's query/passage prefixes (e5) automatically. Mean pooling + L2 (the e5
convention). Requires onnx, onnxruntime, onnxscript, torch, sentence-transformers, transformers.
"""
import os

import numpy as np

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".onnx_cache")


def _export(hf_id, fp32_path):
    import torch
    import torch.nn as nn
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(hf_id, device="cpu").eval()
    transformer, tok = st[0].auto_model, st.tokenizer

    class HS(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    enc = tok(["a short trace sentence", "a longer trace sentence for the onnx export step goes right here now"],
              return_tensors="pt", padding=True, truncation=True, max_length=64)
    batch = torch.export.Dim("batch", min=1, max=512)
    seq = torch.export.Dim("seq", min=2, max=256)
    with torch.no_grad():
        torch.onnx.export(HS(transformer).eval(), (enc["input_ids"], enc["attention_mask"]), fp32_path,
                          input_names=["input_ids", "attention_mask"], output_names=["last_hidden_state"],
                          dynamic_shapes=({0: batch, 1: seq}, {0: batch, 1: seq}), opset_version=18, dynamo=True)


class OnnxInt8Embedder:
    def __init__(self, hf_id, threads=None):
        import onnxruntime as ort
        from onnxruntime.quantization import quantize_dynamic, QuantType
        from transformers import AutoTokenizer
        self.model_name = hf_id
        os.makedirs(CACHE, exist_ok=True)
        safe = hf_id.replace("/", "__")
        fp32 = os.path.join(CACHE, f"{safe}_fp32.onnx")
        int8 = os.path.join(CACHE, f"{safe}_int8.onnx")
        if not os.path.exists(int8):
            if not os.path.exists(fp32):
                _export(hf_id, fp32)
            quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
        self.tok = AutoTokenizer.from_pretrained(hf_id)
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        so.enable_cpu_mem_arena = False  # dynamic-shape inference leaks the CPU arena otherwise
        self.sess = ort.InferenceSession(int8, so, providers=["CPUExecutionProvider"])

    def encode(self, texts, **kw):
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        enc = self.tok(batch, return_tensors="np", padding=True, truncation=True, max_length=256)
        lhs = self.sess.run(["last_hidden_state"],
                            {"input_ids": enc["input_ids"].astype(np.int64),
                             "attention_mask": enc["attention_mask"].astype(np.int64)})[0]
        mask = enc["attention_mask"][:, :, None].astype(np.float32)
        v = (lhs * mask).sum(1) / np.clip(mask.sum(1), 1e-9, None)  # mean pool
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n == 0] = 1.0
        v = v / n
        return v[0] if single else v
