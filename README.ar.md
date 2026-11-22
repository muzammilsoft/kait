# KAIT

**KAIT مكتبة Kotlin تشغّل نماذج `.npz` المدرَّبة بـ TAIT على الجهاز.**
Kotlin خالص / JVM، بلا اعتماد على أندرويد، بلا خادم، بلا حساب، بلا
مفتاح API. يُحمَّل ملف النموذج من الجهاز (أو يُنزَّل مرة واحدة)
ويُنفَّذ محليا — موجهاتك لا تغادر الجهاز أبدا.

English: [README.md](README.md) · 简体中文: [README.zh-CN.md](README.zh-CN.md)

ملف `.npz` واحد، ثلاثة مشغلات:

- **TAIT** يدرّب النموذج (Python / NumPy، على الهاتف) —
  <https://github.com/muzammilsoft/tait>
- **KAIT** تشغّل النموذج على الهاتف وفي أي تطبيق JVM (Kotlin) — هذا المستودع
- **JAIT** تشغّل النموذج في المتصفح (TypeScript) —
  <https://github.com/muzammilsoft/jait>

KAIT Studio، تطبيق أندرويد الذي يشغّل TinyBili على الهاتف، مبني على
هذه المكتبة: <https://github.com/muzammilsoft/KAIT-Studio>

> **للوكلاء (LLM):** ملف [`LLMS.md`](LLMS.md) هو المستند الكامل لك —
> واجهة البرمجة الكاملة، وصيغة النموذج، والضمانات العددية، والوصفات،
> والفخاخ. اقرأه أولا؛ لا تحتاج إلى أي مستند آخر لاستخدام KAIT بشكل
> صحيح.

## بداية سريعة

```kotlin
import com.kgsoft.kait.studio.engine.KaitEngine
import com.kgsoft.kait.studio.engine.GenerationConfig
import java.io.FileInputStream

val model = FileInputStream("tinyGPT0.1.npz").use { KaitEngine.load(it) }
val out = model.generate("مرحبا", GenerationConfig(temperature = 0.7f))
println(out.answer)   // the reply; out.reasoning is "" when the model wrote no <think> tags
```

التحميل من رابط بتنزيل البايتات أولا (المكتبة لا تقوم بأي شبكة بنفسها):

```kotlin
import java.net.URL
val bytes = URL("https://huggingface.co/muzammilsoft/tinybili/resolve/main/tinyGPT0.1.npz").readBytes()
val model = KaitEngine.load(bytes.inputStream())
```

## استخدمها في مشروعك

KAIT مكتبة Kotlin/JVM خالصة (Gradle، Kotlin 2.4.21، JVM 17). ليست
منشورة على Maven Central بعد. طريقتان لاستخدامها:

**1. كوحدة Gradle (موصى بها لتطبيقات أندرويد).** استنسخ هذا المستودع
بجانب مشروعك وأضفه كبناء مركب:

```kotlin
// settings.gradle.kts
includeBuild("kait") // path to a checkout of github.com/muzammilsoft/kait

// app/build.gradle.kts
dependencies {
    implementation("com.kgsoft.kait.studio:kait:0.1.0")
}
```

**2. كملف JAR.** ابنِها واعتمد على الملف:

```sh
./gradlew jar   # build/libs/kait-0.1.0.jar
```

في تطبيق أندرويد، أضف الوحدة إلى `settings.gradle.kts` واستخدم
`implementation(project(":kait"))` — هكذا بالضبط يستهلك KAIT Studio
محركه.

## واجهة البرمجة (الأساسيات)

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

تعيد `generate()` الرد مقسما إلى `reasoning` (النص بين `<think>`
و`</think>`) و`answer` (الباقي، داخل وسوم `<answer>` إن وجدت).
دُرِّب TinyBili على أسئلة وأجوبة عادية بلا صيغة استدلال، فـ `reasoning`
فارغ و`answer` هو الرد كله. واجهة البرمجة الكاملة وترتيب السحب
وصيغة النموذج في [`LLMS.md`](LLMS.md).

## التحقق

```sh
./gradlew test   # build + the parity gate (JDK 17)
```

تحمّل الاختبارات نموذج TAIT صغيرا (`src/test/resources/tiny.npz`)
وتتحقق منه مقابل المرجع الحقيقي في Python (`ref_last.npy`):

- احتمالات التمريرة: نفس أعلى 5 رموز تالية، ونفس argmax، وأقصى فرق
  **5.6e-9** (البوابة: أقل من 1e-3)؛
- ذهاب وعودة الترميز/فك الترميز (عربي وإنجليزي)؛
- نموذج تنقصه بيانات دمج BPE يُرفض بخطأ واضح؛
- التوليد حتمي للبذرة نفسها.

يشغّل CI الأمر `./gradlew test` عند كل دفع وطلب سحب.

## ملاحظات صريحة

- تحسب KAIT بـ **float32** (أوزان float64 تُضيَّق إلى `Float` عند
  التحميل). لهذا فرق Python نحو 5.6e-9 لا 1e-16 — داخل البوابة 1e-3
  بكثير، وأعلى 5 وargmax متطابقان.
- يستخدم السحب `Random` في Kotlin، لا في NumPy. البذرة نفسها تعطي
  عينة *مختلفة لكن قابلة للتكرار* في Kotlin عنها في Python — لا
  تقارن مخرجات مسحوبة بالبذرة بين الاثنين أبدا. قارن أعلى 5 احتمالات
  أو فك التشفير الجشع بدلا من ذلك.
- تخصّص `forward()` مصفوفات جديدة في كل استدعاء (بلا إعادة استخدام
  مؤقت بعد). لنماذج بحجم TinyBili على الهاتف هذا مقبول — KAIT Studio
  v0.1.2 مختبر على هاتف حقيقي — لكن انتبه لضغط جامع القمامة إن
  استدعيتها في حلقة ضيقة على نموذج كبير.
- TinyBili نموذج تجريبي صغير (6,666 مثالا، 50 حقبة). إجاباته غالبا
  مشوشة. اعرضه بصراحة كتجريبي، لا كنموذج قوي أبدا.

## الرخصة

MIT — انظر [`LICENSE`](LICENSE). حقوق النشر (c) 2026 Khartoum Ghoul.
