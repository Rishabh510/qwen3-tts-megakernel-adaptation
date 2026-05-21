# Beginner Glossary

This explains the GPU/model terms using questions first.

## Kernel

Question: when Python asks the GPU to do neural-network work, what exactly runs
on the GPU?

Answer: a CUDA kernel.

Here, kernel does not mean the operating-system kernel. It means a GPU function.
The CPU launches it, and many GPU threads run it in parallel.

## Megakernel

Question: why not let PyTorch launch normal kernels?

Answer: PyTorch usually launches many small GPU operations. A megakernel fuses a
large amount of work into one GPU program. That reduces launch overhead and can
keep data moving in a more controlled way.

In this project, the megakernel runs one autoregressive decode step for a
Qwen-style transformer.

## Decode Step

Question: what does a language model do during generation?

Answer: it repeatedly predicts the next token.

```text
current context -> predict next token -> append token -> repeat
```

One pass through that loop is a decode step.

## Shape

Question: why do we keep saying "target shape"?

Answer: neural-network weights are arrays. The CUDA kernel is written assuming
specific array sizes. If those sizes do not match, the math is wrong or the
program crashes.

For this project, the key shape question is:

```text
Does Qwen3-TTS talker look like Qwen3-0.6B internally?
```

For the 0.6B TTS talker, mostly yes.

## Hidden Size

Question: how wide is one token's internal representation?

Answer: hidden size.

Here it is `1024`, meaning each token becomes a vector of 1024 numbers inside
the transformer.

## Layers

Question: how many repeated transformer blocks does the model have?

Answer: layers.

The talker has `28` layers. Each layer updates the token representation using
attention and MLP computation.

## Attention Heads

Question: why split attention into heads?

Answer: each head can focus on different relationships between tokens.

For this model:

- `16` query heads ask questions.
- `8` key/value heads store information to attend to.
- Each head has dimension `128`.

## Head Dim

Question: how many numbers does one attention head use?

Answer: head dimension.

Here it is `128`. Since there are `16` query heads:

```text
16 heads * 128 numbers = 2048 query projection size
```

## Intermediate Size

Question: inside each transformer layer, why is there a bigger temporary vector?

Answer: the MLP expands the hidden vector, transforms it, then shrinks it back.

Here the intermediate size is `3072`.

## bf16

Question: why not use normal 32-bit floats?

Answer: bf16, or bfloat16, uses half the memory per number. That matters because
this kernel is mostly limited by how fast the GPU can read model weights.

bf16 is common for inference because it is smaller and faster while still being
accurate enough for many neural networks.

## Codec / Codebook

Question: if TTS produces audio, why are there tokens?

Answer: Qwen3-TTS represents speech as discrete audio codes first. A speech
tokenizer/vocoder later turns those codes into waveform audio.

A codebook is like a dictionary of possible audio-code values.

## Talker Decoder

Question: which Qwen3-TTS component should the megakernel accelerate?

Answer: the talker decoder.

The talker predicts the main codec token for each audio frame. The target
explicitly says this is the target, not the codebook generator.

## Code Predictor / Subtalker

Question: what happens after the talker predicts the first code?

Answer: another smaller model predicts the remaining codebook values for that
audio frame.

The reference implementation accelerates this too. We can consider that after
the talker path is proven, but the primary target is the talker decoder.

## RoPE And Theta

Question: how does the model know token order?

Answer: position encoding.

Qwen-style models use RoPE, rotary position embedding. `theta` controls the
frequency scale used in that position math.

Original Qwen3-0.6B megakernel used `theta=10000`. Qwen3-TTS 0.6B config uses
`theta=1000000`. If this does not match, the model can lose track of sequence
positions and output quality/EOS behavior can degrade.

## Vocab Size

Question: how many possible next tokens can the model choose from?

Answer: vocabulary size.

Original Qwen3 text model:

```text
151936 possible text tokens
```

Qwen3-TTS talker:

```text
3072 possible codec tokens
```

This is one of the easiest kernel changes: the output scan should cover 3072
rows, not 151936.
