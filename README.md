# KAIT

**KAIT is a Kotlin library that runs TAIT-trained `.npz` models on-device.**
Pure Kotlin/JVM, no Android dependency, no server, no account, no API key.
The model file is loaded from the device (or downloaded once) and executed
locally — your prompts never leave the device.

Arabic: [README.ar.md](README.ar.md) · 简体中文: [README.zh-CN.md](README.zh-CN.md)

One `.npz`, three runtimes:

- **TAIT** trains the model (Python / NumPy, on a phone) —
  <https://github.com/muzammilsoft/tait>
- **KAIT** runs the model on the phone and in any JVM app (Kotlin) — this repo
- **JAIT** runs the model in the browser (TypeScript) —
  <https://github.com/muzammilsoft/jait>

KAIT Studio, the Android app that runs TinyBili on a phone, is built on
this library: <https://github.com/muzammilsoft/KAIT-Studio>

> **For LLM agents:** [`LLMS.md`](LLMS.md) is the complete document for
> you — full API, model format, numeric guarantees, recipes and traps.
> Read it first; you do not need any other document to use KAIT correctly.

## Quick start

```kotlin
import com.kgsoft.kait.studio.engine.KaitEngine
import com.kgsoft.kait.studio.engine.GenerationConfig
import java.io.FileInputStream

val model = FileInputStream("tinyGPT0.1.npz").use { KaitEngine.load(it) }
val out = model.generate("مرحبا", GenerationConfig(temperature = 0.7f))
println(out.answer)   // the reply; out.reasoning is "" when the model wrote no <think> tags
```

Load from a URL by downloading the bytes first (the library never does
network I/O itself):

```kotlin
import java.net.URL
val bytes = URL("https://huggingface.co/muzammilsoft/tinybili/resolve/main/tinyGPT0.1.npz").readBytes()
val model = KaitEngine.load(bytes.inputStream())
```

## Use it in your project

KAIT is a pure Kotlin/JVM library (Gradle, Kotlin 2.4.21, JVM 17). It is
not published to Maven Central yet. Two ways to use it:

**1. As a Gradle module (recommended for Android apps).** Clone this repo
next to your project and add it as a composite build:

```kotlin
// settings.gradle.kts
includeBuild("kait") // path to a checkout of github.com/muzammilsoft/kait

// app/build.gradle.kts
dependencies {
    implementation("com.kgsoft.kait.studio:kait:0.1.0")
}
```

**2. As a JAR.** Build it and depend on the file:

```sh
./gradlew jar   # build/libs/kait-0.1.0.jar
```

In an Android app, add the module to `settings.gradle.kts` and use
`implementation(project(":kait"))` — that is exactly how KAIT Studio
consumes its engine.

## API (essentials)

```kotlin
KaitEngine.load(input: InputStream): LoadedModel   // throws ModelLoadException on a bad file

LoadedModel {
    val vocabSize: Int; val dModel: Int; val dFf: Int
    val blockSize: Int; val sepId: Int; val endId: Int; val padId: Int
    fun encode(text: String): IntArray
    fun decode(ids: IntArray): String
    fun forwardProbs(ids: IntArray): FloatArray            // last-position probabilities
    fun forward(ids: IntArray, validMask: BooleanArray): Array<FloatArray>
    fun generate(prompt: String,
                 config: GenerationConfig = GenerationConfig(),
                 seed: Long? = null): GeneratedText
}

GenerationConfig(maxNewTokens = 60, temperature = 0.7f, topK = 8, topP = 0.9f)
GeneratedText(reasoning, answer, raw, newTokens, millis)
extractReasoning(text: String): Pair<String, String>  // <think>/<answer> split
```

`generate()` returns the reply split into `reasoning` (text between
`<think>` and `</think>`) and `answer` (the rest, inside `<answer>` tags
if present). TinyBili was trained on plain Q&A with no reasoning format,
so its `reasoning` is empty and `answer` is the whole reply. Full API,
sampling order and the model format are in [`LLMS.md`](LLMS.md).

## Verification

```sh
./gradlew test   # build + the parity gate (JDK 17)
```

The tests load a tiny TAIT model (`src/test/resources/tiny.npz`) and
check it against the real Python reference (`ref_last.npy`):

- forward probabilities: same top-5 next-token ids, same argmax, max
  difference **5.6e-9** (gate: below 1e-3);
- encode/decode round-trips (Arabic and English);
- a model missing its BPE merge data is rejected with a clear error;
- generation is deterministic for the same seed.

CI runs `./gradlew test` on every push and pull request.

## Honest notes

- KAIT computes in **float32** (float64 weights are narrowed to `Float`
  on load). That is why the Python difference is ~5.6e-9 instead of
  ~1e-16 — far inside the 1e-3 gate, and the top-5 and argmax are
  identical.
- Sampling uses Kotlin's `Random`, not NumPy's. The same seed gives a
  *different but repeatable* sample in Kotlin than in Python — never
  compare sampled outputs by seed across the two. Compare top-5
  probabilities or greedy decoding instead.
- `forward()` allocates fresh arrays on every call (no scratch reuse
  yet). For TinyBili-sized models on a phone this is fine — KAIT Studio
  v0.1.2 is verified on a real phone — but be aware of GC pressure if
  you call it in a tight loop on a large model.
- TinyBili is a tiny experimental model (6,666 examples, 50 epochs).
  Its answers are often garbled. Present it honestly as experimental,
  never as a strong model.

## License

MIT — see [`LICENSE`](LICENSE). Copyright (c) 2026 Khartoum Ghoul.
