package com.example.mymcqscannerapp;

import android.graphics.Bitmap;

public class ImageCache {
    private static ImageCache instance;
    private Bitmap studentImage;
    private Bitmap teacherImage;

    private ImageCache() { }

    public static synchronized ImageCache getInstance() {
        if (instance == null) {
            instance = new ImageCache();
        }
        return instance;
    }

    public Bitmap getStudentImage() {
        return studentImage;
    }

    public void setStudentImage(Bitmap studentImage) {
        this.studentImage = studentImage;
    }

    public Bitmap getTeacherImage() {
        return teacherImage;
    }

    public void setTeacherImage(Bitmap teacherImage) {
        this.teacherImage = teacherImage;
    }
}
