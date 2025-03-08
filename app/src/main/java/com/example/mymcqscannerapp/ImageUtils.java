package com.example.mymcqscannerapp;

import android.graphics.Bitmap;
import java.io.ByteArrayOutputStream;

public class ImageUtils {
    // Compress bitmap to JPEG with the specified quality.
    public static byte[] compressToJPEG(Bitmap bitmap, int quality) {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, quality, baos);
        return baos.toByteArray();
    }
}
