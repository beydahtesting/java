package com.example.mymcqscannerapp;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import com.google.android.material.button.MaterialButton;
import org.opencv.android.OpenCVLoader;

public class MainActivity extends BaseActivity {
    private static final String TAG = "MainActivity";
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Initialize OpenCV before using any OpenCV code.
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "Unable to load OpenCV");
        } else {
            Log.i(TAG, "OpenCV loaded successfully");
        }
        setContentView(R.layout.activity_main);
        setUpToolbar();

        MaterialButton btnOnline = findViewById(R.id.btn_online);
        MaterialButton btnOffline = findViewById(R.id.btn_offline);

        btnOnline.setOnClickListener(v -> {
            AppMode.setOnlineMode(true);
            startActivity(new Intent(MainActivity.this, TeacherImageActivity.class));
        });

        btnOffline.setOnClickListener(v -> {
            AppMode.setOnlineMode(false);
            startActivity(new Intent(MainActivity.this, TeacherImageActivity.class));
        });
    }

    @Override
    protected Intent getForwardIntent() {
        return null; // No forward arrow action from MainActivity.
    }
}
