import mlx.core as mx
from mlx import nn

from mflux.models.seedvr2.model.seedvr2_transformer.rms_norm import RMSNorm
from mflux.models.seedvr2.model.seedvr2_transformer.rope import RoPEModule
from mflux.models.seedvr2.model.seedvr2_transformer.window import WindowPartitioner


class MMAttention(nn.Module):
    def __init__(
        self,
        vid_dim: int,
        txt_dim: int,
        heads: int = 20,
        head_dim: int = 128,
        qk_bias: bool = False,
        qk_norm_eps: float = 1e-5,
        rope_dim: int = 128,
        rope_freqs_for: str = "lang",
        text_rope_freqs_for: str | None = None,
        rope_on_text: bool = True,
        text_attention_mode: str = "window_pool",
        shared_weights: bool = False,
        window: tuple[int, int, int] = (4, 3, 3),
        shift: bool = False,
    ):
        super().__init__()
        if text_attention_mode not in {"window_pool", "global_text"}:
            raise ValueError(f"Unsupported SeedVR2 text attention mode: {text_attention_mode!r}")
        self.shared_weights = shared_weights
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.window = window
        self.shift = shift
        self.rope_on_text = rope_on_text
        self.text_attention_mode = text_attention_mode

        inner_dim = heads * head_dim

        self.proj_qkv_vid = nn.Linear(vid_dim, 3 * inner_dim, bias=qk_bias)
        self.proj_out_vid = nn.Linear(inner_dim, vid_dim, bias=True)
        self.norm_q_vid = RMSNorm(head_dim, eps=qk_norm_eps)
        self.norm_k_vid = RMSNorm(head_dim, eps=qk_norm_eps)

        if shared_weights:
            self.proj_qkv_txt = self.proj_qkv_vid
            self.proj_out_txt = self.proj_out_vid
            self.norm_q_txt = self.norm_q_vid
            self.norm_k_txt = self.norm_k_vid
        else:
            self.proj_qkv_txt = nn.Linear(txt_dim, 3 * inner_dim, bias=qk_bias)
            self.proj_out_txt = nn.Linear(inner_dim, txt_dim, bias=True)
            self.norm_q_txt = RMSNorm(head_dim, eps=qk_norm_eps)
            self.norm_k_txt = RMSNorm(head_dim, eps=qk_norm_eps)

        self.rope = RoPEModule(dim=rope_dim, freqs_for=rope_freqs_for, text_freqs_for=text_rope_freqs_for)

    def __call__(self, vid, txt, vid_shape, txt_shape):
        B, L, Bt, Lt = vid.shape[0], vid.shape[1], txt.shape[0], txt.shape[1]

        qkv_vid = self.proj_qkv_vid(vid.reshape(-1, vid.shape[-1])).reshape(-1, 3, self.heads, self.head_dim)
        qkv_txt = self.proj_qkv_txt(txt.reshape(-1, txt.shape[-1])).reshape(-1, 3, self.heads, self.head_dim)

        partitioner = WindowPartitioner(vid_shape, self.window, self.shift)
        q_vid_full = self.norm_q_vid(qkv_vid[:, 0])
        k_vid_full = self.norm_k_vid(qkv_vid[:, 1])
        v_vid_full = qkv_vid[:, 2]
        q_txt = self.norm_q_txt(qkv_txt[:, 0])
        k_txt = self.norm_k_txt(qkv_txt[:, 1])
        v_txt = qkv_txt[:, 2]

        counts, txt_len = partitioner.window_counts, txt_shape[:, 0]
        q_vid = partitioner.partition(q_vid_full)
        k_vid = partitioner.partition(k_vid_full)
        v_vid = partitioner.partition(v_vid_full)
        qkv_t_rep = self._repeat_text_for_windows(mx.stack([q_txt, k_txt, v_txt], axis=1), txt_len, counts)
        q_txt_rep, k_txt_rep, v_txt_rep = qkv_t_rep[:, 0], qkv_t_rep[:, 1], qkv_t_rep[:, 2]
        if self.rope_on_text:
            q_vid, k_vid, q_txt_rep, k_txt_rep = self.rope(
                vid_q=q_vid,
                vid_k=k_vid,
                vid_shape=partitioner.window_shapes,
                txt_q=q_txt_rep,
                txt_k=k_txt_rep,
                txt_shape=txt_shape[MMAttention._window_batch_ids(counts)],
            )
        else:
            q_vid, k_vid = self.rope(
                vid_q=q_vid,
                vid_k=k_vid,
                vid_shape=partitioner.window_shapes,
            )

        vid_lens = mx.prod(partitioner.window_shapes, axis=1)
        window_txt_lens = MMAttention._window_text_lengths(txt_len, counts)
        window_batch_ids = MMAttention._window_batch_ids(counts)

        if self.text_attention_mode == "window_pool":
            vid_out, txt_out = self._window_joint_attention(
                q_txt_rep=q_txt_rep,
                k_txt_rep=k_txt_rep,
                v_txt_rep=v_txt_rep,
                vid_lens=vid_lens,
                window_txt_lens=window_txt_lens,
                window_batch_ids=window_batch_ids,
                counts=counts,
                q_vid=q_vid,
                k_vid=k_vid,
                v_vid=v_vid,
            )
        else:
            vid_out = self._window_video_attention_only(
                q_txt_rep=q_txt_rep,
                k_txt_rep=k_txt_rep,
                v_txt_rep=v_txt_rep,
                vid_lens=vid_lens,
                window_txt_lens=window_txt_lens,
                q_vid=q_vid,
                k_vid=k_vid,
                v_vid=v_vid,
            )
            txt_out = self._global_text_attention(
                q_txt=q_txt,
                k_txt=k_txt,
                v_txt=v_txt,
                k_vid_full=k_vid_full,
                v_vid_full=v_vid_full,
                vid_shape=vid_shape,
                txt_shape=txt_shape,
            )

        return (
            self.proj_out_vid(partitioner.reverse(vid_out)).reshape(B, L, -1),
            self.proj_out_txt(txt_out).reshape(Bt, Lt, -1),
        )

    def _window_joint_attention(
        self,
        *,
        q_txt_rep: mx.array,
        k_txt_rep: mx.array,
        v_txt_rep: mx.array,
        vid_lens: mx.array,
        window_txt_lens: mx.array,
        window_batch_ids: mx.array,
        counts: list[int],
        q_vid: mx.array,
        k_vid: mx.array,
        v_vid: mx.array,
    ) -> tuple[mx.array, mx.array]:
        vid_parts: list[mx.array] = []
        txt_parts_by_batch: list[list[mx.array]] = [[] for _ in counts]
        vid_offset = 0
        txt_offset = 0

        for window_index in range(int(window_txt_lens.shape[0])):
            vid_len = int(vid_lens[window_index])
            txt_len = int(window_txt_lens[window_index])
            batch_index = int(window_batch_ids[window_index])

            q = MMAttention._concat_tokens(
                q_vid[vid_offset : vid_offset + vid_len],
                q_txt_rep[txt_offset : txt_offset + txt_len],
            )
            k = MMAttention._concat_tokens(
                k_vid[vid_offset : vid_offset + vid_len],
                k_txt_rep[txt_offset : txt_offset + txt_len],
            )
            v = MMAttention._concat_tokens(
                v_vid[vid_offset : vid_offset + vid_len],
                v_txt_rep[txt_offset : txt_offset + txt_len],
            )
            attention = self._run_attention(q=q, k=k, v=v)
            vid_parts.append(attention[:vid_len])
            txt_parts_by_batch[batch_index].append(attention[vid_len:])

            vid_offset += vid_len
            txt_offset += txt_len

        pooled_txt = [mx.stack(parts).mean(axis=0) for parts in txt_parts_by_batch]
        return (
            mx.concatenate(vid_parts, axis=0).reshape(-1, self.heads * self.head_dim),
            mx.concatenate(pooled_txt, axis=0).reshape(-1, self.heads * self.head_dim),
        )

    def _window_video_attention_only(
        self,
        *,
        q_txt_rep: mx.array,
        k_txt_rep: mx.array,
        v_txt_rep: mx.array,
        vid_lens: mx.array,
        window_txt_lens: mx.array,
        q_vid: mx.array,
        k_vid: mx.array,
        v_vid: mx.array,
    ) -> mx.array:
        vid_parts: list[mx.array] = []
        vid_offset = 0
        txt_offset = 0

        for window_index in range(int(window_txt_lens.shape[0])):
            vid_len = int(vid_lens[window_index])
            txt_len = int(window_txt_lens[window_index])

            q = MMAttention._concat_tokens(
                q_vid[vid_offset : vid_offset + vid_len],
                q_txt_rep[txt_offset : txt_offset + txt_len],
            )
            k = MMAttention._concat_tokens(
                k_vid[vid_offset : vid_offset + vid_len],
                k_txt_rep[txt_offset : txt_offset + txt_len],
            )
            v = MMAttention._concat_tokens(
                v_vid[vid_offset : vid_offset + vid_len],
                v_txt_rep[txt_offset : txt_offset + txt_len],
            )
            attention = self._run_attention(q=q, k=k, v=v)
            vid_parts.append(attention[:vid_len])

            vid_offset += vid_len
            txt_offset += txt_len

        return mx.concatenate(vid_parts, axis=0).reshape(-1, self.heads * self.head_dim)

    def _global_text_attention(
        self,
        *,
        q_txt: mx.array,
        k_txt: mx.array,
        v_txt: mx.array,
        k_vid_full: mx.array,
        v_vid_full: mx.array,
        vid_shape: mx.array,
        txt_shape: mx.array,
    ) -> mx.array:
        vid_lengths = mx.prod(vid_shape, axis=1).tolist()
        txt_lengths = txt_shape[:, 0].tolist()
        vid_offsets = MMAttention._offsets(vid_lengths)
        txt_offsets = MMAttention._offsets(txt_lengths)

        outputs: list[mx.array] = []
        for batch_index, (vid_offset, vid_len, txt_offset, txt_len) in enumerate(
            zip(vid_offsets, vid_lengths, txt_offsets, txt_lengths)
        ):
            del batch_index
            attention = self._run_attention(
                q=q_txt[txt_offset : txt_offset + txt_len],
                k=MMAttention._concat_tokens(
                    k_vid_full[vid_offset : vid_offset + vid_len],
                    k_txt[txt_offset : txt_offset + txt_len],
                ),
                v=MMAttention._concat_tokens(
                    v_vid_full[vid_offset : vid_offset + vid_len],
                    v_txt[txt_offset : txt_offset + txt_len],
                ),
            )
            outputs.append(attention)
        return mx.concatenate(outputs, axis=0).reshape(-1, self.heads * self.head_dim)

    def _run_attention(
        self,
        *,
        q: mx.array,
        k: mx.array,
        v: mx.array,
    ) -> mx.array:
        attention = mx.fast.scaled_dot_product_attention(
            q[None].transpose(0, 2, 1, 3),
            k[None].transpose(0, 2, 1, 3),
            v[None].transpose(0, 2, 1, 3),
            scale=self.scale,
        )
        return attention.transpose(0, 2, 1, 3).squeeze(0)

    @staticmethod
    def _repeat_text_for_windows(txt, txt_len, counts):
        B, L = len(counts), int(txt_len[0])
        txt = txt.reshape(B, L, *txt.shape[1:])
        return txt[MMAttention._window_batch_ids(counts)].reshape(-1, *txt.shape[2:])

    @staticmethod
    def _window_text_lengths(txt_len: mx.array, counts: list[int]) -> mx.array:
        return txt_len[MMAttention._window_batch_ids(counts)]

    @staticmethod
    def _window_batch_ids(counts: list[int]) -> mx.array:
        # Gather rather than mx.repeat: MLX only repeats by a scalar, so a per-batch
        # count list has to be expanded into explicit row indices.
        return mx.array([b for b, count in enumerate(counts) for _ in range(count)], dtype=mx.int32)

    @staticmethod
    def _concat_tokens(left: mx.array, right: mx.array) -> mx.array:
        return mx.concatenate([left, right], axis=0)

    @staticmethod
    def _offsets(lengths: list[int]) -> list[int]:
        offsets: list[int] = []
        current = 0
        for length in lengths:
            offsets.append(current)
            current += int(length)
        return offsets
