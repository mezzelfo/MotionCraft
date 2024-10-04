import torch
import torch.nn.functional as F
from einops import rearrange, einsum


class MultipleCrossFrameAttnProcessor:
    """
    This is a custom attention processor. It supports:
    - Cross frame attention.
        - Every frame attends to itself (this option is selected by default and it is the same as the standard attention).
        - Every frame attends to the first frame.
        - Every frame attends to the previous frame. The first frame attends to itself.
    - Attention history.

    Args:
        video_length: The number of frames in the video. It should automaigcally works for guidance.
        should_record_history: Whether to record the attention-map history.
    """

    def __init__(self, video_length=2):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        # Stuff for cross frame attention
        self.video_length = video_length
        # This is the actual batch size, other than the frames and the text guidance.
        # For example, should be a multiple of num_images_per_prompt
        self.true_batch_size = 1
        # By default, every frame attends just to itself. Hence the cross frame attention is disabled.
        self._frame_idx_to_attend = [[i] for i in range(video_length)]

        # Stuff for attention history
        self.should_record_history = False
        self.attention_history = []
        self.filter_latent_dimension = 32

    @property
    def frame_idx_to_attend(self):
        return self._frame_idx_to_attend

    @frame_idx_to_attend.setter
    def frame_idx_to_attend(self, new_frame_idx_to_attend):
        assert len(new_frame_idx_to_attend) == self.video_length
        # Now new_frame_idx_to_attend is a list of lists.
        # Sublist i-th contains the indices of the frames that the i-th frame should attend to.
        # We require that each sublist contains the same number of elements
        # and that each element is a valid frame index.
        for frame_idx in range(self.video_length):
            to_attend = new_frame_idx_to_attend[frame_idx]
            if isinstance(to_attend, int):
                new_frame_idx_to_attend[frame_idx] = [to_attend]
            assert len(to_attend) == len(new_frame_idx_to_attend[0])
            for frame_idx_to_attend in new_frame_idx_to_attend[frame_idx]:
                assert 0 <= frame_idx_to_attend < self.video_length
        self._frame_idx_to_attend = new_frame_idx_to_attend

    def each_frame_attends_to_itself(self):
        self.frame_idx_to_attend = [[i] for i in range(self.video_length)]

    def each_frame_attends_to_first_frame(self):
        self.frame_idx_to_attend = [[0] for _ in range(self.video_length)]

    def each_frame_attends_to_previous_frame(self):
        self.frame_idx_to_attend = [[max(0, i - 1)] for i in range(self.video_length)]

    def each_frame_attends_to_all(self):
        self.frame_idx_to_attend = [list(range(self.video_length))] * self.video_length

    def reset(self):
        self.attention_history = []

    def get_attention_history(self):
        assert self.should_record_history
        return torch.stack(self.attention_history)

    def __call__(
        self,
        attn,
        image_tokens,
        encoder_hidden_states=None,  # This should be renamed to text_tokens
        attention_mask=None,
    ):
        text_tokens = encoder_hidden_states  # Rename this variable for clarity
        is_image_text_attention = not (text_tokens is None)

        ############################ Project tokens to query, key, value
        if not is_image_text_attention:
            batch_size, sequence_length, _ = image_tokens.shape
            query = attn.to_q(image_tokens)
            key = attn.to_k(image_tokens)
            value = attn.to_v(image_tokens)
        else:
            batch_size, sequence_length, _ = text_tokens.shape
            if attn.norm_cross:
                text_tokens = attn.norm_encoder_hidden_states(text_tokens)
            query = attn.to_q(image_tokens)
            key = attn.to_k(text_tokens)
            value = attn.to_v(text_tokens)

        ############################ Prepare attention mask
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        ############################ Rearrange query, key, value to support cross-frame attention
        # OSS: "tokens" dim can differ between query, key, and value, when using text-to-image attention.
        query = rearrange(
            query,
            "(batch guidance frames) tokens dim -> batch guidance frames tokens dim",
            batch=self.true_batch_size,
            frames=self.video_length,
        )
        key = rearrange(
            key,
            "(batch guidance frames) tokens dim -> batch guidance frames tokens dim",
            batch=self.true_batch_size,
            frames=self.video_length,
        )
        value = rearrange(
            value,
            "(batch guidance frames) tokens dim -> batch guidance frames tokens dim",
            batch=self.true_batch_size,
            frames=self.video_length,
        )

        # check guidance dimensions
        assert query.shape[1] in [1, 2]

        ############################ Perform cross-frame attention
        if not is_image_text_attention:
            keys = [[] for _ in range(self.video_length)]
            values = [[] for _ in range(self.video_length)]
            for i in range(self.video_length):
                for f in self.frame_idx_to_attend[i]:
                    keys[i].append(key[:, :, f])
                    values[i].append(value[:, :, f])
                keys[i] = torch.cat(keys[i], dim=-2)
                values[i] = torch.cat(values[i], dim=-2)

            key = torch.stack(keys, dim=-3)  # batch guidance frames tokens dim
            value = torch.stack(values, dim=-3)  # batch guidance frames tokens dim

        ############################ Reshape for multi-head attention
        # We reshape the tensors to 4 dims to support advanced SDP backends
        # "Both fused kernels requires query, key and value to be 4 dimensional"
        pre_rearrange_args = {
            'pattern' : "batch guidance frames tokens (heads innerdim) -> (batch guidance frames) heads tokens innerdim",
            'heads' : attn.heads
        }
        query = rearrange(query,**pre_rearrange_args)
        key = rearrange(key,**pre_rearrange_args)
        value = rearrange(value,**pre_rearrange_args)

        ############################ Perform multi-head attention
        post_image_tokens = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )  # (batch guidance frames) heads tokens innerdim

        post_rearrange_args = {
            'pattern' : "(batch guidance frames) heads tokens innerdim -> batch guidance frames heads tokens innerdim",
            'batch' : self.true_batch_size,
            'frames' : self.video_length
        }
        query = rearrange(post_image_tokens,**post_rearrange_args)
        key = rearrange(post_image_tokens,**post_rearrange_args)
        value = rearrange(post_image_tokens,**post_rearrange_args)
        post_image_tokens = rearrange(post_image_tokens,**post_rearrange_args)
        ############################ Record attention history
        if self.should_record_history and is_image_text_attention:
            side = int(query.shape[-2] ** 0.5)
            if side == self.filter_latent_dimension:
                attention = einsum(
                    query,
                    key,
                    "batch guidance frames heads img innerdim, batch guidance frames heads text innerdim -> batch guidance frames heads img text",
                )
                attention = attention / (query.shape[-1] ** 0.5)
                attention = F.softmax(attention, dim=-1)

                attention = rearrange(
                    attention,
                    "batch guidance frames heads (H W) text -> batch guidance frames H W heads text",
                    H=side,
                    W=side,
                )
                self.attention_history.append(attention)

        ############################ Rearrange and finalize
        post_image_tokens = rearrange(
            post_image_tokens,
            "batch guidance frames heads tokens innerdim -> (batch guidance frames) tokens (heads innerdim)",
            heads=attn.heads,
        )
        post_image_tokens = post_image_tokens.to(query.dtype)

        # linear proj
        post_image_tokens = attn.to_out[0](post_image_tokens)
        # dropout
        post_image_tokens = attn.to_out[1](post_image_tokens)
        return post_image_tokens


def get_attention_processor(video_length, crossframe_attn, should_record_history=False):

    attn = MultipleCrossFrameAttnProcessor(video_length=video_length)

    if crossframe_attn == "disabled":
        attn.each_frame_attends_to_itself()
    elif crossframe_attn == "first":
        attn.each_frame_attends_to_first_frame()
    elif crossframe_attn == "previous":
        attn.each_frame_attends_to_previous_frame()
    elif crossframe_attn == "all":
        attn.each_frame_attends_to_all()
    else:
        raise ValueError(f"Invalid crossframe_attn: {crossframe_attn}")

    attn.should_record_history = should_record_history

    return attn


def get_attention_processor_from_pattern(pattern: str, should_record_history=False):
    try:
        pattern = pattern.replace("'", "").replace('"', "")
        pattern_list = eval(pattern)
        assert type(pattern_list) == list
        assert [type(i) == list for i in pattern_list]
        assert [type(n) == int for i in pattern_list for n in i]
    except:
        raise RuntimeError(f"Invalid pattern <{pattern}> for crossframe attention")

    video_length = len(pattern_list)
    attn = MultipleCrossFrameAttnProcessor(video_length=video_length)
    attn.should_record_history = should_record_history
    attn.frame_idx_to_attend = pattern_list
    return attn
