# KAIT

**KAIT 是一个 Kotlin 库，可在设备上运行 TAIT 训练的 `.npz` 模型。**
纯 Kotlin/JVM，无 Android 依赖、无服务器、无账号、无 API 密钥。
模型文件从设备加载（或下载一次）并在本地执行——您的提示永远不会
离开设备。

English: [README.md](README.md) · العربية: [README.ar.md](README.ar.md)

一个 `.npz`，三种运行时：

- **TAIT** 训练模型（Python / NumPy，在手机上）——
  <https://github.com/muzammilsoft/tait>
- **KAIT** 在手机和任何 JVM 应用中运行模型（Kotlin）——本仓库
- **JAIT** 在浏览器中运行模型（TypeScript）——
  <https://github.com/muzammilsoft/jait>

KAIT Studio 是在手机上运行 TinyBili 的 Android 应用，基于本库构建：
<https://github.com/muzammilsoft/KAIT-Studio>

> **对于 LLM 智能体：**[`LLMS.md`](LLMS.md) 是为您准备的完整文档——
> 完整 API、模型格式、数值保证、示例与陷阱。请先阅读它；正确使用
> KAIT 不需要任何其他文档。

## 快速开始

```kotlin
import com.kgsoft.kait.studio.engine.KaitEngine
import com.kgsoft.kait.studio.engine.GenerationConfig
import java.io.FileInputStream

val model = FileInputStream("tinyGPT0.1.npz").use { KaitEngine.load(it) }
val out = model.generate("مرحبا", GenerationConfig(temperature = 0.7f))
println(out.answer)   // the reply; out.reasoning is "" when the model wrote no <think> tags
```

通过先下载字节从 URL 加载（本库本身不进行任何网络 I/O）：

```kotlin
import java.net.URL
val bytes = URL("https://huggingface.co/muzammilsoft/tinybili/resolve/main/tinyGPT0.1.npz").readBytes()
val model = KaitEngine.load(bytes.inputStream())
```

## 在您的项目中使用

KAIT 是纯 Kotlin/JVM 库（Gradle、Kotlin 2.4.21、JVM 17）。尚未发布
到 Maven Central。有两种使用方式：

**1. 作为 Gradle 模块（推荐用于 Android 应用）。**将本仓库克隆到
项目旁边，并添加为复合构建：

```kotlin
// settings.gradle.kts
includeBuild("kait") // path to a checkout of github.com/muzammilsoft/kait

// app/build.gradle.kts
dependencies {
    implementation("com.kgsoft.kait.studio:kait:0.1.0")
}
```

**2. 作为 JAR 文件。**构建它并依赖该文件：

```sh
./gradlew jar   # build/libs/kait-0.1.0.jar
```

在 Android 应用中，将模块添加到 `settings.gradle.kts` 并使用
`implementation(project(":kait"))`——KAIT Studio 正是这样使用其引擎的。

## API（要点）

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

`generate()` 返回拆分为 `reasoning`（`<think>` 与 `</think>` 之间的
文本）和 `answer`（其余部分，如有则位于 `<answer>` 标签内）的回复。
TinyBili 是在没有推理格式的普通问答上训练的，因此其 `reasoning`
为空，`answer` 即为完整回复。完整 API、采样顺序和模型格式见
[`LLMS.md`](LLMS.md)。

## 验证

```sh
./gradlew test   # build + the parity gate (JDK 17)
```

测试加载一个微型 TAIT 模型（`src/test/resources/tiny.npz`）并对照
真实的 Python 参考（`ref_last.npy`）进行检查：

- 前向概率：前 5 个下一个 token 的 ID 相同、argmax 相同、最大差异
  为 **5.6e-9**（关卡：小于 1e-3）；
- 编码/解码往返（阿拉伯语和英语）；
- 缺少 BPE 合并数据的模型会被明确报错并拒绝；
- 同一随机种子的生成是确定性的。

CI 在每次推送和拉取请求时运行 `./gradlew test`。

## 诚实说明

- KAIT 以 **float32** 计算（float64 权重在加载时收窄为 `Float`）。
  这就是 Python 差异约为 5.6e-9 而非 1e-16 的原因——远在 1e-3 关卡
  之内，且前 5 名和 argmax 完全相同。
- 采样使用 Kotlin 的 `Random`，而非 NumPy。同一随机种子在 Kotlin
  中得到与 Python *不同但可重复* 的采样——绝不要按随机种子比较两者
  的采样输出。请改为比较前 5 概率或贪心解码。
- `forward()` 每次调用都会分配新数组（尚无暂存区复用）。对于手机上
  TinyBili 大小的模型，这没有问题——KAIT Studio v0.1.2 已在真机上
  验证——但如果您在大型模型上于紧密循环中调用它，请注意垃圾回收压力。
- TinyBili 是小型实验模型（6,666 个样本、50 轮）。其回答常常是乱码。
  请诚实地将其展示为实验模型，绝不要说它是强大的模型。

## 许可证

MIT——见 [`LICENSE`](LICENSE)。版权所有 (c) 2026 Khartoum Ghoul。
