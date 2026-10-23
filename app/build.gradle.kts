plugins {
    alias(libs.plugins.android.application)
}

android {
    // namespace 只决定 R / BuildConfig 等生成类的 Java 包名，和安装包的包名（applicationId）无关，没跟着改
    namespace = "com.example.helloworld"
    compileSdk = 36

    defaultConfig {
        // 安装包的包名，与 iOS 版的 Bundle ID 一致（区分大小写）。Google Play 不接受 com.example 开头的包名；上架后不能再改。
        // 和以前的 com.example.helloworld 是两个 app：老用户装新版不会覆盖旧版，需要手动卸载旧版
        applicationId = "com.aibei.Xcup"
        minSdk = 26
        // Google Play 从 2026-08-31 起要求新应用和更新 target Android 16（API 36）
        targetSdk = 36
        versionCode = 127
        versionName = "1.2.7"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

dependencies {

    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.activity)
    implementation(libs.constraintlayout)

    // LiteRT：TensorFlow Lite 改名后的同一套运行时，org.tensorflow.lite.Interpreter 的用法不变。
    // 原来的 org.tensorflow:tensorflow-lite:2.13.0 的 .so 只按 4 KB 对齐，Google Play 要求 target 35+ 的 app
    // 支持 16 KB 内存页；LiteRT 1.4.0 起 .so 按 16 KB 对齐
    implementation("com.google.ai.edge.litert:litert:1.4.0")
    // 如果需要更好的权限管理，可以添加（可选）：
    implementation("pub.devrel:easypermissions:3.0.0")
    // 单元测试 & Android 测试
    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)
}



