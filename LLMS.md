# LLMS.md — KAIT, for LLM agents

> This file is written for an LLM agent (or a developer handing this repo
> to one). It is the complete, factual overview of the KAIT library: what
> it is, how to install it, the full API with exact signatures, the model
> format, the numeric guarantees, and the traps. Read this file first;
> you do not need any other document to use KAIT correctly. If this file
> and the code disagree, the code is right — fix this file.

## 1. What KAIT is

KAIT is a Kotlin library that runs TAIT-trained `.npz` models on-device.
Pure Kotlin/JVM, no Android dependency, no server, no account, no API
key, no network I/O of its own: the model file is loaded from an
`InputStream` the caller supplies (a file, an asset, or bytes downloaded
by the caller) and executed locally.

One `.npz`, three runtimes:

- **TAIT** trains the model (Python / NumPy, on a phone) —
  <https://github.com/muzammilsoft/tait>
- **KAIT** runs the model on the phone and in any JVM app (Kotlin) —
  this repo
- **JAIT** runs the model in the browser (TypeScript) —
  <https://github.com/muzammilsoft/jait>

KAIT is a line-by-line port of TAIT's `tait/core/model.py` (MiniGPT
forward), `tait/core/tokenizer.py` (byte-level BPE) and
`tait/core/reasoning.py` (think/answer split). It is not a general ML
framework: it runs exactly one architecture (a single transformer block,
no LayerNorm — TAIT has none) on exactly one file format (TAIT `.npz`,
see §5).

The Android app built on this library is KAIT Studio
(<https://github.com/muzammilsoft/KAIT-Studio>), which runs TinyBili on
a real phone. This repo is the library alone — no UI, no Android code.

## 2. Install

KAIT is **not on Maven Central**. Two ways to use it:

**1. As a Gradle module (recommended for Android apps).** Clone this
repo next to your project and add it as a composite build:

```kotlin
// settings.gradle.kts
includeBuild("kait") // path to a checkout of github.com/muzammilsoft/kait

// app/build.gradle.kts
dependencies {
    implementation("com.kgsoft.kait.studio:kait:0.1.0")
}
```

In an Android app you can also add the module directly
(`include(":kait")` in `settings.gradle.kts`, then
`implementation(project(":kait"))`) — that is how KAIT Studio consumes
its engine.

**2. As a JAR.** Build it and depend on the file:

```sh
./gradlew jar   # build/libs/kait-0.1.0.jar
```

Build requirements: JDK 17, Gradle 9.3.1 (via the included wrapper —
`./gradlew`), Kotlin 2.4.21. No Android SDK is needed to build or test
the library itself. Runtime dependencies: none (only the Kotlin
stdlib). Test dependencies: kotlin-test-junit5, JUnit Jupiter 5.10.3.

## 3. Quick start

```kotlin
import com.kgsoft.kait.studio.engine.KaitEngine
import com.kgsoft.kait.studio.engine.GenerationConfig
import java.io.FileInputStream

val model = FileInputStream("tinyGPT0.1.npz").use { KaitEngine.load(it) }
val out = model.generate("مرحبا", GenerationConfig(temperature = 0.7f))
println(out.answer)
println("tokens=${out.newTokens} time=${out.millis}ms")
```

Load from a URL by downloading the bytes first (the library never does
network I/O itself):

```kotlin
import java.net.URL
val bytes = URL("https://huggingface.co/muzammilsoft/tinybili/resolve/main/tinyGPT0.1.npz").readBytes()
val model = KaitEngine.load(bytes.inputStream())
```

## 4. Full API reference

Everything is in package `com.kgsoft.kait.studio.engine` (source in
`src/main/kotlin/com/kgsoft/kait/studio/engine/`).

### Loading

```kotlin
object KaitEngine {
    fun load(input: InputStream): LoadedModel
}
```

- Throws `ModelLoadException(message)` when the file is missing data,
  inconsistent, or malformed (missing weights, missing BPE merge data,
  wrong shapes, inconsistent pad ids, bad `.npy` entries). A non-TAIT
  file is rejected, never silently misread.
- `load` reads the whole stream; the caller owns closing it (use
  `InputStream.use { }`).

### Model

```kotlin
class LoadedModel internal constructor(...) {
    val vocabSize: Int
    val dModel: Int
    val dFf: Int
    val blockSize: Int
    val sepId: Int
    val endId: Int
    val padId: Int

    fun encode(text: String): IntArray
    fun decode(ids: IntArray): String
    fun forwardProbs(ids: IntArray): FloatArray
    fun forward(ids: IntArray, validMask: BooleanArray): Array<FloatArray>
    fun generate(prompt: String, config: GenerationConfig = GenerationConfig(), seed: Long? = null): GeneratedText
}
```

- `forward(ids, validMask)` runs the full forward pass and returns
  per-position probabilities as `Array<FloatArray>` (row `t` =
  distribution after token `t`). `validMask[i] == false` marks a
  padding position. Requires `ids.size <= blockSize` and every id in
  `[0, vocabSize)`, otherwise it throws `IllegalArgumentException`.
- `forwardProbs(ids)` returns the probabilities at the last position
  only, as a fresh `FloatArray`. It first truncates to the last
  `blockSize` ids, so longer inputs are safe. This is the call
  generation uses.
- `encode` / `decode` delegate to the tokenizer (§4, Tokenizer).
  `decode` skips SEP, END and pad ids — it is for display text.

### Generation

```kotlin
data class GenerationConfig(
    val maxNewTokens: Int = 60,
    val temperature: Float = 0.7f,
    val topK: Int = 8,
    val topP: Float = 0.9f,
)

data class GeneratedText(
    val reasoning: String,  // text between <think> and </think>, or "" if none
    val answer: String,     // text after </think> (or the whole text), inside <answer> tags if present
    val raw: String,        // full generated text (prompt excluded, SEP/END stripped) before tag extraction
    val newTokens: Int,     // new tokens sampled (including END if it stopped the run)
    val millis: Long,       // wall time of the generation loop in milliseconds
)

fun extractReasoning(text: String): Pair<String, String>  // (reasoning, answer)
```

- `generate()` follows TAIT's `MiniGPT.generate` step for step:
  ids = encode(prompt) + SEP; each step takes the last `blockSize` ids,
  gets the last-position probs, and samples (see the pipeline below);
  stops at END or after `config.maxNewTokens` new tokens. The returned
  `raw` is the generated text with the prompt and the SEP/END framing
  removed (like TAIT's `generate_reply`); `reasoning` / `answer` are
  split from it by `extractReasoning`.
- Sampling pipeline (in `Generation.kt`, `sampleDistribution` — internal):
  (1) pad probability set to 0, renormalise; (2) temperature power
  `p = probs ** (1 / max(T, 1e-6))`, renormalise; (3) top-k, renormalise;
  (4) top-p, renormalise; (5) draw one token with a cumulative draw.
  If the distribution sums to 0 at any step, generation stops
  (`sampleDistribution` returns null).
- **RNG difference (honest):** Python samples with NumPy's
  `default_rng`; Kotlin uses `kotlin.random.Random` (seeded with the
  caller's seed, or `Random.Default` when `seed` is null). The
  distribution at every step is identical; the sampled token for a
  given seed is not — NumPy and Kotlin simply have different RNGs.
  Never compare sampled outputs by seed across the two. Compare with
  top-5 probabilities or greedy decoding instead. The same seed IS
  repeatable inside Kotlin (verified by test).
- **Defaults are TAIT's, not JAIT's:** `maxNewTokens` 60,
  `temperature` 0.7, `topK` 8, `topP` 0.9. Do not copy JAIT's defaults
  (64 / 0.8 / 40 / 0.95) into KAIT code.

### Tokenizer

```kotlin
class Tokenizer internal constructor(
    val vocab: List<ByteArray>,              // vocab[id] = raw bytes of that token; empty for the 3 special ids
    val merges: Map<Pair<Int, Int>, Int>,    // (a, b) -> new id, in file order
    val sepId: Int, val endId: Int, val padId: Int,
) {
    val vocabSize: Int
    fun encode(text: String): IntArray
    fun decode(ids: IntArray): String                    // skips SEP, END and pad — for display
    fun decodeRaw(ids: IntArray): String                 // keeps SEP (\u0001) and END (\u0002), skips pad only
    companion object {
        fun fromNpz(mergeA: IntArray, mergeB: IntArray, mergeId: IntArray,
                    sepId: Int, endId: Int, padId: Int, vocabSize: Int): Tokenizer
    }
}
```

- Byte-level BPE, rebuilt from the `.npz` merge arrays exactly as TAIT
  does. Vocab 0..255 are the single bytes; merge triples are walked **in
  file order** (a later merge may use an earlier merged token); the
  check `256 + merges + 3 == vocabSize` is enforced, otherwise loading
  fails with `ModelLoadException`.
- Encoding is TAIT's rule: text to UTF-8 bytes, then repeatedly merge
  the adjacent pair with the **lowest** merge id until no known pair
  remains. A literal SEP (`\u0001`) or END (`\u0002`) character in the
  input becomes the single special id instead of being byte-encoded.
- `decode` concatenates vocab bytes and decodes as UTF-8, skipping
  SEP/END/pad. `decodeRaw` keeps SEP/END in place — the same behaviour
  as TAIT's `BPETokenizer.decode`; `LoadedModel.generate` uses it
  internally so the SEP/END framing can be stripped afterwards.

### NPZ / NPY reader

```kotlin
fun readNpz(input: InputStream): Map<String, Npy>   // key = entry name without the ".npy" suffix
fun parseNpy(bytes: ByteArray): Npy

data class Npy(
    val shape: List<Int>,
    val descr: String,
    val floats: FloatArray?,
    val ints: IntArray?,
    val longs: LongArray?,
) {
    val count: Int                                   // product of shape, or 1 for a scalar
    fun scalarLong(): Long
    fun requireFloats(name: String): FloatArray
    fun requireInts(name: String): IntArray
}
```

- An `.npz` is a ZIP of `.npy` files (STORED or DEFLATE — both are
  inflated by `ZipInputStream`). Only `.npy` version 1 is supported.
  Supported dtypes: `<f4` / `|f4` → `floats`; `<f8` → `floats`
  (float64 narrowed to `Float` — TAIT writes float32 for wte/wpe and
  float64 for the rest, and KAIT computes in float32 throughout);
  `<i4` / `|i4` → `ints` (BPE merges); `<i8` → `longs` (scalars).
  Exactly one of `floats` / `ints` / `longs` is non-null.
- Fortran-order arrays are refused with a clear error (TAIT always
  writes C order). A truncated file, bad magic, unsupported version or
  unsupported descr throws `ModelLoadException` with the entry name.
- **Android note (kept from the source):** `readNpz` uses
  `InputStream.readBytes()`, never `readAllBytes()` — the latter is a
  Java 9 API that Android only provides from API 33 and crashes with
  `NoSuchMethodError` on older phones (seen on Android 10 / API 29).
  Do not "modernise" this call.

### Reasoning split

```kotlin
fun extractReasoning(text: String): Pair<String, String>
```

Exactly like TAIT's `extract_generation` (`tait/core/reasoning.py`):
if `<think>` is present, the text between it and `</think>` is the
reasoning and everything after `</think>` is the answer (if there is no
`</think>`, everything after `<think>` is the answer and the reasoning
is empty); if `<answer>` is present in the answer, the text after it —
up to `</answer>` if present — is the answer; with no tags at all, the
reasoning is empty and the answer is the whole text. Both parts are
trimmed.

## 5. The model format

A TAIT `.npz` is self-contained: no `vocab.json` is ever needed. It
contains:

| Arrays | Dtype in file | Meaning |
|---|---|---|
| `wte`, `wpe` | float32 | token + position embeddings |
| `Wq`, `Wk`, `Wv`, `Wo` | float64 | attention weights |
| `W1`, `b1`, `W2`, `b2` | float64 | MLP weights/biases |
| `Whead`, `bhead` | float64 | output head |
| `merge_a`, `merge_b`, `merge_id` | int32 | BPE merges |
| `sep_id`, `end_id`, `tok_pad_id`, `block_size`, `pad_id` | int scalars | special ids + limits |

Architecture: a single transformer block with **no LayerNorm** (TAIT
has none). `h0 = wte[x] + wpe[pos]`; causal single-head attention with
`-1e9` masking (`scores = Q @ K^T / sqrt(D)`, masked where `j > i` or
the key is padding); `h1 = h0 + attn @ Wo`; **ReLU** MLP
(`ff1 = relu(h1 @ W1 + b1)`, `ff2 = ff1 @ W2 + b2`); `h2 = h1 + ff2`;
`logits = h2 @ Whead + bhead`; row-wise softmax with a `+1e-9`
denominator (same as TAIT). Note: the MLP activation is ReLU, not
GELU — an older KAIT Studio doc said GELU; that was wrong.

Reference model (the one the tests use): TinyBili,
<https://huggingface.co/muzammilsoft/tinybili> (`tinyGPT0.1.npz`,
1,072,206 bytes): vocab 1003, `dModel` 64, `dFf` 128, `blockSize` 64,
SEP 1000, END 1001, PAD 1002, 744 merges, trained on 6,666 examples
for 50 epochs. TinyBili is a tiny experimental model — its answers
are often garbled. Present it honestly as an experimental model,
never as a strong one.

## 6. Numeric guarantees

- KAIT computes in **float32** throughout (the JVM `Float` type).
  float64 weights are narrowed to `Float` when the `.npz` is loaded.
  That is why the Python difference is ~5.6e-9 instead of ~1e-16 —
  far inside the 1e-3 gate, and the top-5 and argmax are identical.
- Softmax is numerically stable (subtract the row max, divide by
  `sum + 1e-9`), matching TAIT.
- The embedding addition is done in float32 (both `wte` and `wpe`
  are float32 in the file), matching TAIT's float32 embedding step.
- Verification gate (see §8): same top-5 next-token ids, same argmax,
  max probability difference below 1e-3. Measured on the test model:
  max difference **5.6e-9**.

## 7. Performance and allocation

- `forward()` allocates fresh arrays on every call (each `matmul`
  allocates its result, plus the intermediate `h0`/`h1`/`h2`, the
  attention matrix, and the per-row softmax results). There is **no
  scratch reuse yet** — unlike JAIT, whose `forward()` allocates zero
  arrays after its buffer-reuse fix. For TinyBili-sized models this is
  fine on a phone (KAIT Studio v0.1.2 is verified on a real phone), but
  be aware of GC pressure if you call `forward()` in a tight loop or on
  a much larger model. A scratch-reuse optimisation is a possible
  future change, not a present guarantee.
- `generate()` calls `forwardProbs()` once per generated token (each
  call is one full forward pass over the current context, truncated to
  the last `blockSize` tokens). Generation time is dominated by these
  forward passes; `GeneratedText.millis` reports the wall time of the
  loop.
- There is no batching, no GPU, no NNAPI. One sequence at a time, on
  the CPU, in pure Kotlin.

## 8. Verification

```sh
./gradlew test   # build + parity gate (JDK 17)
```

`src/test/kotlin/.../EngineTest.kt` loads the bundled test model
(`src/test/resources/tiny.npz` — an untrained MiniGPT saved by TAIT:
vocab 303, `dModel` 16, `dFf` 32, `blockSize` 53, 12 merges) and checks
it against the real Python reference (`ref_last.npy`, the
last-position probabilities of TAIT's Python forward on prompt
"Hello", ids [72, 274, 282]):

1. forward probabilities match Python: same top-5 ids, same argmax,
   max difference < 1e-3 (measured 5.6e-9);
2. encode/decode round-trips (Arabic and English, including mixed
   text and SEP/END handling);
3. reasoning extraction (`<think>` / `<answer>` cases);
4. a model with its BPE merge data removed is rejected with
   `ModelLoadException`;
5. generation is deterministic for the same seed, and the
   reasoning/answer split is consistent with `raw`.

CI (`.github/workflows/ci.yml`) runs `./gradlew test` on every push
and pull request (Ubuntu, Temurin JDK 17).

## 9. Recipes

**Run a prompt and print the reply** — see §3.

**Load a model from a file and print its shape:**

```kotlin
val model = FileInputStream("model.npz").use { KaitEngine.load(it) }
println("V=${model.vocabSize} D=${model.dModel} dFf=${model.dFf} block=${model.blockSize}")
```

**Get only the answer (no reasoning):**

```kotlin
val out = model.generate(prompt)
println(out.answer)   // out.reasoning holds the <think> text when the model wrote any
```

**Deterministic (repeatable) generation:**

```kotlin
val out = model.generate(prompt, GenerationConfig(temperature = 0.7f), seed = 42L)
```

**Check the next-token distribution yourself:**

```kotlin
val ids = model.encode("Hello")
val probs = model.forwardProbs(ids)
val top5 = probs.indices.sortedByDescending { probs[it] }.take(5)
```

**Verify a new TAIT model file behaves like Python:** load it with
`KaitEngine.load`, run `forwardProbs` on a fixed prompt, and compare
the top-5 ids and argmax with Python's forward on the same prompt and
model. Do not compare sampled outputs by seed (§4, RNG difference).

**Use it in an Android app:** add the module (see §2), load the model
on a background thread (generation blocks — never call it on the main
thread), and post `GeneratedText.answer` back to the UI. KAIT Studio
is the reference for this pattern.

## 10. Constraints — do not do these

- Do not compare sampled outputs to Python by seed (different RNGs).
- Do not copy JAIT's generation defaults (64 / 0.8 / 40 / 0.95) into
  KAIT code — KAIT's defaults are TAIT's (60 / 0.7 / 8 / 0.9).
- Do not present TinyBili (or any tiny TAIT demo model) as a strong
  or general model. It is experimental; say so.
- Do not add a server, an API key, an account, or any tracking. The
  whole point is that everything runs locally.
- Do not change the sampling order or the `+1e-9` softmax denominator
  — they are copied from Python and the verification gate depends on them.
- Do not replace `InputStream.readBytes()` with `readAllBytes()` in
  `readNpz` — it crashes on Android API < 33 (see §4).
- Do not claim KAIT is affiliated with Bilibili, Hugging Face, or any
  model host. It is an independent open-source project by Khartoum Ghoul.

## 11. Links

- KAIT (this repo): <https://github.com/muzammilsoft/kait>
- KAIT Studio (Android app built on it): <https://github.com/muzammilsoft/KAIT-Studio>
- TAIT (trainer): <https://github.com/muzammilsoft/tait>
- JAIT (browser runner): <https://github.com/muzammilsoft/jait>
- TinyBili (reference model): <https://huggingface.co/muzammilsoft/tinybili>
- License: MIT (see `LICENSE`).
