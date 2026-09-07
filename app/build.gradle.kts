plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.example.helloworld"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.helloworld"
        minSdk = 26
        targetSdk = 35
        versionCode = 124
        versionName = "1.2.4"

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

    // AndroidX UI 组件
    implementation("androidx.appcompat:appcompat:1.3.1")
    implementation("androidx.constraintlayout:constraintlayout:2.1.1")
    implementation("com.google.android.material:material:1.4.0")
    // TensorFlow Lite 核心库
    implementation("org.tensorflow:tensorflow-lite:2.13.0")
    // 如果需要更好的权限管理，可以添加（可选）：
    implementation("pub.devrel:easypermissions:3.0.0")
    // Media3 / ExoPlayer（仅"网页视频"模式使用，离线/在线模式仍走原本的 MediaExtractor 路径）
    implementation("androidx.media3:media3-exoplayer:1.4.1")
    implementation("androidx.media3:media3-exoplayer-hls:1.4.1")
    implementation("androidx.media3:media3-datasource:1.4.1")
    implementation("androidx.media3:media3-ui:1.4.1")
    // Cronet 数据源：让网页视频模式的 ExoPlayer 用 Chromium 网络栈拉流，
    // TLS/HTTP 指纹与 WebView 一致，过 missav CDN（surrit.com）的 Cloudflare 机器人校验。
    // cronet-embedded 内嵌 Chromium 网络栈（约 +8MB），不依赖 Google Play 服务，任何设备可用。
    implementation("androidx.media3:media3-datasource-cronet:1.4.1")
    // 版本对齐 media3-datasource-cronet 依赖的 cronet-api，避免版本冲突/重复类。
    implementation("org.chromium.net:cronet-embedded:113.5672.61")
    // 单元测试 & Android 测试
    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)
}



