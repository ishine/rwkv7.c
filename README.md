# rwkv7.c

*Inspired by [llama2.c](https://github.com/karpathy/llama2.c).*

Inference RWKV v7 in **pure C**.

Check [`classic`](https://github.com/KevlarKanou/rwkv7.c/tree/classic) branch for the pure RNN version.

## Tested model

```
RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth
RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth
RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth
RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth
rwkv7-g1-0.1b-20250307-ctx4096.pth
rwkv7-g1-0.4b-20250324-ctx4096.pth
rwkv7-g1-1.5b-20250429-ctx4096.pth
rwkv7-g1-2.9b-20250519-ctx4096.pth
rwkv7-g0-7.2b-20250722-ctx4096.pth
rwkv7a-g1b-0.1b-20250819-ctx4096.pth
```

## Build

```shell
make # make blas/avx/neon
```
- `make blas`: Use BLAS as linear algebra backend, here we use [OpenBLAS](http://www.openmathlib.org/OpenBLAS/) by default.

- `make avx`/`make neon`: Simple linear algebra backends written using SIMD instruction sets, lacking performance optimization but free of dependencies.

## Usage

``` shell
wget "https://huggingface.co/BlinkDL/rwkv-7-world/resolve/main/RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth" -O model.pth
python ./utils/export.py ./model.pth ./model.bin # pip install torch rwkv

# default params: temperature = 1.0, top-p = 0.7, presence_penalty = 0.1, frequency_penalty = 0.2, max_dec_len = 10240
# generate mode
./rwkv7 ./model.bin -i "Once upon a time," --temperature 1.3 --top-p 0.8 --presence_penalty 0.4 --frequency_penalty 0.5
# chat mode
./rwkv7 ./model.bin --chat -i "Where is the capital of France?" --temperature 0.8 --top-p 0.2 --presence_penalty 0.1 --frequency_penalty 0.2
# reasoner mode (rwkv7-g1)
./rwkv7 ./model-g1.bin --reasoner -i "What is RWKV?" # use default params
# benchmark mode
./rwkv7 ./model.bin --bench # try to decrease chunk_size if stack overflow

```

## TODO
- DeepEmbedAttention
- FP16 support, mainly on ARM NEON
- Model quantization

## Benchmark
model: 0.1B, p512g128, chunk_size=64
| Platform                      | Backend                               | Prefill t/s   | Decode t/s    |
| ------------------------------|---------------------------------------|---------------|---------------|
| **AMD R5-4600H, DDR4-3200**   | -                                     | 12.91 ±0.01   | 8.07 ±0.00    |
|                               | `avx`                                 | 60.39 ±0.12   | 28.29 ±0.06   |
|                               | openblas 0.3.26, `OMP_NUM_THREADS=1`  | 310.53 ±1.16  | 30.49 ±0.17   |
| **Intel i9-9820X, DDR4-2133** | -                                     | 11.53 ±0.02   | 7.03 ±0.01    |
|                               | `avx`                                 | 69.66 ±0.05   | 23.84 ±0.01   |
|                               | openblas 0.3.26, `OMP_NUM_THREADS=1`  | 355.67 ±2.81  | 22.99 ±0.02   |
| **Rockchip RK3588, LPDDR4x**  | -                                     | 12.15 ±0.02   | 7.41 ±0.01    |
|                               | `neon`                                | 44.38 ±0.02   | 22.54 ±0.02   |
|                               | openblas 0.3.26, `OMP_NUM_THREADS=1`  | 128.03 ±0.80  | 21.97 ±0.82   |
