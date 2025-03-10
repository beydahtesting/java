plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.example.myapplication"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.myapplication"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

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
    kotlinOptions {
        jvmTarget = "11"
    }
}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.constraintlayout)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    implementation 'com.itextpdf:itextpdf:5.5.13.3'
    implementation 'org.apache.poi:poi:5.2.3'
    implementation 'org.apache.poi:poi-ooxml:5.2.3'
    implementation 'androidx.exifinterface:exifinterface:1.3.6'
    implementation libs.appcompat
    implementation libs.material
    implementation libs.activity
    implementation libs.constraintlayout
    implementation libs.core.ktx
    testImplementation libs.junit
    androidTestImplementation libs.ext.junit
    androidTestImplementation libs.espresso.core

    // OpenCV for Android – if you have the OpenCV library module imported:
     implementation project(':OpenCV')
    // Alternatively, if you prefer using Maven dependency (uncomment the next line):
    // implementation 'org.opencv:opencv-android:4.5.3'

    // Apache POI for Excel file handling
    implementation 'org.apache.poi:poi:5.4.0'
    implementation 'org.apache.poi:poi-ooxml:5.4.0'

    // JSON handling library
    implementation 'org.json:json:20250107'

    // PhotoView for zoomable image display
    implementation 'com.github.chrisbanes:PhotoView:2.3.0'

    // OkHttp for networking (recommended for Gemini API calls)
    implementation 'com.squareup.okhttp3:okhttp:4.12.0'
}